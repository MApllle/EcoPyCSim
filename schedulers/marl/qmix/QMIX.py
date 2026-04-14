"""QMIX（Monotonic Value Function Factorisation）多智能体协调器

核心思想（Rashid et al., 2018）：
    Q_tot(s, a) = MixNet([Q_1(o_1,a_1), ..., Q_N(o_N,a_N)], s)

    混合网络权重由超网络从全局状态 s 生成，并约束为非负（abs()），
    保证 ∂Q_tot/∂Q_i ≥ 0（单调性），从而确保分散执行的最优性。

训练阶段（集中式）：
    Q_tot = MixNet([Q_i(o_i, a_i)], s)
    Q_tot_target = r_tot + γ · MixNet_target([max_{a'_i} Q_i_target(o'_i)], s')
    loss = MSE(Q_tot, Q_tot_target)
    ——联合 loss 反向传播，更新所有 Q_i 网络及混合网络

执行阶段（分散式）：
    每个 agent 独立 argmax Q_i(o_i)
    （单调性保证此分散最优 ⟺ Q_tot 最优）

与 VDN 的关系：
    VDN 是 QMIX 的特例（混合网络退化为求和）：
    VDN: Q_tot = Σ Q_i
    QMIX: Q_tot = MixNet([Q_i], s)  （更强的表达能力，利用全局状态）

接口与 VDN / IDQN 完全对齐：
    add / select_action / learn / update_target / save / load
"""

import logging
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from schedulers.marl.maddpg.Buffer import Buffer
from schedulers.marl.qmix.QMIXAgent import QMIXAgent
from schedulers.marl.qmix.QMixer import QMixer


def _setup_logger(filename: str) -> logging.Logger:
    logger = logging.getLogger(f"qmix_{filename}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename, mode='w')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    return logger


class QMIX:
    """QMIX 协调器：集中训练，分散执行。

    Args:
        dim_info:  {agent_id: {'obs_shape': {key: shape, ...}, 'action_dim': int}}
                   与 VDN / IDQN / MADDPG 相同的格式，直接来自 set_env()
        capacity:  回放缓冲区容量
        batch_size: 每次学习的 minibatch 大小
        lr:        Adam 优化器学习率
        res_dir:   结果保存目录（模型 + 日志）
        embed_dim: 混合网络隐层维度，默认 32
        device:    torch.device；None 时自动选 CUDA/CPU
    """

    def __init__(self, dim_info: dict, capacity: int, batch_size: int,
                 lr: float, res_dir: str, embed_dim: int = 32, device=None):
        self.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.batch_size = batch_size
        self.capacity = capacity
        self.res_dir = res_dir
        self.logger = _setup_logger(os.path.join(res_dir, 'qmix.log'))

        # ── 每个 agent 的 obs_dim（用于拼接全局状态） ─────────────────────
        self.obs_dims: dict[str, int] = {
            agent_id: int(sum(np.prod(shape) for shape in info['obs_shape'].values()))
            for agent_id, info in dim_info.items()
        }
        state_dim = sum(self.obs_dims.values())
        self.dim_info = dim_info

        # ── 每个 agent 独立的 Q-network（无独立 optimizer） ──────────────
        self.agents: dict[str, QMIXAgent] = {}
        self.buffers: dict[str, Buffer] = {}

        for agent_id, info in dim_info.items():
            obs_dim = self.obs_dims[agent_id]
            act_dim = info['action_dim']
            self.agents[agent_id] = QMIXAgent(obs_dim, act_dim, self.device)
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim=1, device=self.device)

        # ── 全局状态缓冲区（与 per-agent Buffer 共享同一环形指针） ────────
        # state = concat([flat_obs_agent1, flat_obs_agent2, ...])，按 sorted(agent_id)
        self.state_buf      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self._buf_index = 0   # 环形缓冲区指针（与各 Buffer._index 同步）
        self._buf_size  = 0   # 当前填充量

        # ── 混合网络（在线 + 目标） ────────────────────────────────────────
        self.mixer = QMixer(
            n_agents=len(dim_info),
            state_dim=state_dim,
            embed_dim=embed_dim,
        ).to(self.device)

        self.target_mixer = deepcopy(self.mixer)
        for p in self.target_mixer.parameters():
            p.requires_grad = False

        # ── 联合优化器：管理所有 agent Q-net + 在线混合网络的参数 ──────────
        self.all_params = (
            [p for agent in self.agents.values() for p in agent.q_net.parameters()]
            + list(self.mixer.parameters())
        )
        self.optimizer = Adam(self.all_params, lr=lr)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def flatten_obs(obs_dict: dict) -> np.ndarray:
        """将 dict 观测拼接为 1D numpy 数组（键排序，与 VDN 相同）。"""
        parts = []
        for key in sorted(obs_dict.keys()):
            arr = obs_dict[key]
            parts.extend(arr.flatten() if isinstance(arr, np.ndarray) else [arr])
        return np.array(parts, dtype=np.float32)

    def _build_state(self, obs: dict) -> np.ndarray:
        """将所有 agent 的 dict 观测拼接为全局状态向量。

        按 sorted(agent_id) 保证拼接顺序确定性。
        """
        return np.concatenate([
            self.flatten_obs(obs[aid]) for aid in sorted(obs.keys())
        ])

    # ------------------------------------------------------------------
    # 经验存储
    # ------------------------------------------------------------------

    def add(self, obs: dict, action: dict, reward: dict,
            next_obs: dict, done: dict):
        """将一步转移存入各 agent 缓冲区及全局状态缓冲区。"""
        # 全局状态
        state      = self._build_state(obs)
        next_state = self._build_state(next_obs)
        self.state_buf[self._buf_index]      = state
        self.next_state_buf[self._buf_index] = next_state
        self._buf_index = (self._buf_index + 1) % self.capacity
        self._buf_size  = min(self._buf_size + 1, self.capacity)

        # 各 agent 局部经验
        for agent_id in obs:
            flat_o  = self.flatten_obs(obs[agent_id])
            flat_no = self.flatten_obs(next_obs[agent_id])
            self.buffers[agent_id].add(
                flat_o,
                np.array([action[agent_id]], dtype=np.float32),  # (1,) float
                reward[agent_id],
                flat_no,
                done[agent_id],
            )

    # ------------------------------------------------------------------
    # 动作选取（分散执行）
    # ------------------------------------------------------------------

    def select_action(self, obs: dict, epsilon: float = 0.0) -> dict:
        """
        Args:
            obs:     {agent_id: obs_dict}
            epsilon: epsilon-greedy 探索概率
        Returns:
            {agent_id: int}
        """
        actions = {}
        for agent_id, o in obs.items():
            flat_o  = self.flatten_obs(o)
            obs_t   = torch.from_numpy(flat_o).unsqueeze(0).float().to(self.device)
            actions[agent_id] = self.agents[agent_id].select_action(obs_t, epsilon)
        return actions

    # ------------------------------------------------------------------
    # 集中训练
    # ------------------------------------------------------------------

    def learn(self, batch_size: int, gamma: float):
        """QMIX 集中式联合更新。

        关键步骤：
        1. 所有 agent Buffer 与全局状态缓冲区用**相同索引**同步采样
        2. 计算 Q_tot = MixNet([Q_i(o_i, a_i)], s)
        3. 计算 Q_tot_target = r_tot + γ · MixNet_target([max Q_i_target(o'_i)], s')
        4. loss = MSE(Q_tot, Q_tot_target) ——单次反向传播更新所有 Q_i 和 MixNet
        """
        # 数据量不足时跳过
        min_size = min(self._buf_size, min(len(b) for b in self.buffers.values()))
        if min_size < batch_size:
            return

        # ── 同步采样 ─────────────────────────────────────────────────────
        indices = np.random.choice(min_size, size=batch_size, replace=False)
        samples = {
            agent_id: buf.sample(indices)
            for agent_id, buf in self.buffers.items()
        }
        # 全局状态（原始 float，不做归一化）
        state      = torch.from_numpy(self.state_buf[indices]).float().to(self.device)
        next_state = torch.from_numpy(self.next_state_buf[indices]).float().to(self.device)

        # ── 计算当前 Q_tot（通过混合网络） ────────────────────────────────
        q_agent_list = []
        for agent_id, (obs_b, actions_b, _, _, _) in samples.items():
            q_i = self.agents[agent_id].q_values(obs_b).gather(
                1, actions_b.long()
            ).squeeze(1)                                    # (B,)
            q_agent_list.append(q_i)

        q_agents = torch.stack(q_agent_list, dim=1)         # (B, n_agents)
        q_tot = self.mixer(q_agents, state)                  # (B,)

        # ── 计算 TD target（no_grad） ─────────────────────────────────────
        with torch.no_grad():
            q_next_list = []
            r_tot = None
            for agent_id, (_, _, rewards, next_obs_b, dones) in samples.items():
                q_next_i = self.agents[agent_id].target_q_values(next_obs_b).max(dim=1)[0]
                q_next_list.append(q_next_i)
                r_tot = rewards if r_tot is None else r_tot + rewards

            q_next_agents = torch.stack(q_next_list, dim=1)              # (B, n_agents)
            q_tot_next    = self.target_mixer(q_next_agents, next_state)  # (B,)

            dones_flag    = next(iter(samples.values()))[4]               # (B,)
            q_tot_target  = r_tot + gamma * q_tot_next * (1.0 - dones_flag)

        # ── 联合反向传播 ──────────────────────────────────────────────────
        loss = F.mse_loss(q_tot, q_tot_target)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.all_params, 0.5)
        self.optimizer.step()

        self.logger.info(f'joint loss: {loss.item():.6f}')

    # ------------------------------------------------------------------
    # 目标网络软更新
    # ------------------------------------------------------------------

    def update_target(self, tau: float):
        """对所有 agent Q-net 及混合网络的目标网络做软更新。"""
        for agent in self.agents.values():
            agent.soft_update(tau)
        # 混合网络目标网络软更新
        for src, dst in zip(self.mixer.parameters(), self.target_mixer.parameters()):
            dst.data.copy_(tau * src.data + (1.0 - tau) * dst.data)

    # ------------------------------------------------------------------
    # 模型保存 / 加载
    # ------------------------------------------------------------------

    def save(self, reward=None):
        """将各 agent Q-net、target Q-net 及混合网络参数保存到 res_dir/model.pt。"""
        torch.save(
            {
                'agents': {
                    name: {
                        'q_net':        agent.q_net.state_dict(),
                        'target_q_net': agent.target_q_net.state_dict(),
                    }
                    for name, agent in self.agents.items()
                },
                'mixer':        self.mixer.state_dict(),
                'target_mixer': self.target_mixer.state_dict(),
            },
            os.path.join(self.res_dir, 'model.pt'),
        )

    @classmethod
    def load(cls, dim_info: dict, file: str, capacity: int,
             batch_size: int, lr: float, embed_dim: int = 32,
             device=None) -> 'QMIX':
        """从保存的 model.pt 恢复 QMIX 实例（用于评估或续训）。"""
        instance = cls(dim_info, capacity, batch_size, lr,
                       os.path.dirname(file), embed_dim=embed_dim, device=device)
        data = torch.load(file, map_location=instance.device)
        for agent_id, agent in instance.agents.items():
            agent.q_net.load_state_dict(data['agents'][agent_id]['q_net'])
            agent.target_q_net.load_state_dict(data['agents'][agent_id]['target_q_net'])
        instance.mixer.load_state_dict(data['mixer'])
        instance.target_mixer.load_state_dict(data['target_mixer'])
        return instance
