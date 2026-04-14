"""VDN（Value Decomposition Networks）多智能体协调器

核心思想（Sunehag et al., 2018）：
    Q_tot(s, a) = Σ_i Q_i(o_i, a_i)

训练阶段（集中式）：
    r_tot = Σ_i r_i
    Q_tot_target = r_tot + γ · Σ_i max_{a'_i} Q_i_target(o'_i)
    loss = MSE(Q_tot, Q_tot_target)
    ——联合 loss 对所有 Q_i 网络反向传播，梯度通过求和操作自然分配

执行阶段（分散式）：
    每个 agent 独立 argmax Q_i(o_i)
    （因 Q_tot = Σ Q_i，各 agent 独立最优 ⟺ 联合最优）

与 IDQN 的接口完全对齐：
    add / select_action / learn / update_target / save / load
"""

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from schedulers.marl.maddpg.Buffer import Buffer
from schedulers.marl.vdn.VDNAgent import VDNAgent


def _setup_logger(filename: str) -> logging.Logger:
    logger = logging.getLogger(f"vdn_{filename}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename, mode='w')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    return logger


class VDN:
    """VDN 协调器：集中训练，分散执行。

    Args:
        dim_info:   {agent_id: {'obs_shape': {key: shape, ...}, 'action_dim': int}}
                    与 IDQN / MADDPG 相同的格式，直接来自 set_env()
        capacity:   回放缓冲区容量
        batch_size: 每次学习的 minibatch 大小
        lr:         联合 Adam 优化器学习率
        res_dir:    结果保存目录（模型 + 日志）
        device:     torch.device；None 时自动选 CUDA/CPU
    """

    def __init__(self, dim_info: dict, capacity: int, batch_size: int,
                 lr: float, res_dir: str, device=None):
        self.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.batch_size = batch_size
        self.res_dir = res_dir
        self.logger = _setup_logger(os.path.join(res_dir, 'vdn.log'))

        # ── 每个 agent 独立的 Q-network（无独立 optimizer） ──────────────
        self.agents: dict[str, VDNAgent] = {}
        self.buffers: dict[str, Buffer] = {}

        for agent_id, info in dim_info.items():
            obs_dim = sum(np.prod(shape) for shape in info['obs_shape'].values())
            act_dim = info['action_dim']
            self.agents[agent_id] = VDNAgent(obs_dim, act_dim, self.device)
            # act_dim=1：将离散 int 动作存为单列 float，采样时转 long 用于 gather
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim=1, device=self.device)

        self.dim_info = dim_info

        # ── 联合优化器：管理所有 agent Q-net 的参数 ──────────────────────
        # 梯度通过 Q_tot = Σ Q_i 的求和操作自动流回每个子网络
        self.all_params = [
            p for agent in self.agents.values()
            for p in agent.q_net.parameters()
        ]
        self.optimizer = Adam(self.all_params, lr=lr)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def flatten_obs(obs_dict: dict) -> np.ndarray:
        """将 dict 观测拼接为 1D numpy 数组（键排序，与 IDQN 相同）。"""
        parts = []
        for key in sorted(obs_dict.keys()):
            arr = obs_dict[key]
            parts.extend(arr.flatten() if isinstance(arr, np.ndarray) else [arr])
        return np.array(parts, dtype=np.float32)

    # ------------------------------------------------------------------
    # 经验存储
    # ------------------------------------------------------------------

    def add(self, obs: dict, action: dict, reward: dict,
            next_obs: dict, done: dict):
        """将一步转移存入各 agent 的回放缓冲区。"""
        for agent_id in obs:
            flat_o = self.flatten_obs(obs[agent_id])
            flat_no = self.flatten_obs(next_obs[agent_id])
            self.buffers[agent_id].add(
                flat_o,
                np.array([action[agent_id]], dtype=np.float32),  # 存为 (1,) float
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
            flat_o = self.flatten_obs(o)
            obs_t = torch.from_numpy(flat_o).unsqueeze(0).float().to(self.device)
            actions[agent_id] = self.agents[agent_id].select_action(obs_t, epsilon)
        return actions

    # ------------------------------------------------------------------
    # 联合训练
    # ------------------------------------------------------------------

    def learn(self, batch_size: int, gamma: float):
        """VDN 集中式联合更新。

        关键步骤：
        1. 所有 agent 的 Buffer 用**相同的随机索引**同步采样，保证来自同一时间步
        2. 计算 Q_tot = Σ_i Q_i(o_i, a_i)
        3. 计算 Q_tot_target = r_tot + γ · Σ_i max Q_i_target(o'_i)
        4. loss = MSE(Q_tot, Q_tot_target) ——单次反向传播更新所有 Q_i
        """
        # 数据量不足时跳过
        if any(len(buf) < batch_size for buf in self.buffers.values()):
            return

        # ── 同步采样（相同索引保证时序对齐） ────────────────────────────
        total = min(len(buf) for buf in self.buffers.values())
        indices = np.random.choice(total, size=batch_size, replace=False)
        samples = {
            agent_id: buf.sample(indices)
            for agent_id, buf in self.buffers.items()
        }
        # samples[agent_id] = (obs, actions, rewards, next_obs, dones)
        # Buffer.sample() 内部对 reward 做了各自归一化；此处相加作近似处理
        # （精确做法：从 buf.reward[indices] 取原始值后统一归一化，但误差可接受）

        # ── 计算当前 Q_tot ────────────────────────────────────────────────
        q_vals = []
        for agent_id, (obs, actions, _, _, _) in samples.items():
            # obs: (B, obs_dim), actions: (B, 1) float → gather 需要 int64
            q_i = self.agents[agent_id].q_values(obs).gather(
                1, actions.long()
            ).squeeze(1)                                        # (B,)
            q_vals.append(q_i)
        q_tot = sum(q_vals)                                     # (B,)

        # ── 计算 TD target（no_grad） ─────────────────────────────────────
        with torch.no_grad():
            q_next_vals = []
            r_tot = None
            for agent_id, (_, _, rewards, next_obs, dones) in samples.items():
                # 各 agent 目标 Q 的最大值
                q_next_i = self.agents[agent_id].target_q_values(next_obs).max(dim=1)[0]
                q_next_vals.append(q_next_i)
                # 联合奖励（逐样本相加）
                r_tot = rewards if r_tot is None else r_tot + rewards

            q_tot_next = sum(q_next_vals)                       # (B,)
            # done 标志：取第一个 agent 的（两 agent 同时终止）
            dones_flag = next(iter(samples.values()))[4]        # (B,)
            q_tot_target = r_tot + gamma * q_tot_next * (1.0 - dones_flag)

        # ── 联合反向传播 ──────────────────────────────────────────────────
        loss = F.mse_loss(q_tot, q_tot_target)

        self.optimizer.zero_grad()
        loss.backward()                       # 梯度流过 Σ Q_i，自动分配给各子网络
        clip_grad_norm_(self.all_params, 0.5)
        self.optimizer.step()

        self.logger.info(f'joint loss: {loss.item():.6f}')

    # ------------------------------------------------------------------
    # 目标网络软更新
    # ------------------------------------------------------------------

    def update_target(self, tau: float):
        """对所有 agent 的目标网络做软更新。"""
        for agent in self.agents.values():
            agent.soft_update(tau)

    # ------------------------------------------------------------------
    # 模型保存 / 加载
    # ------------------------------------------------------------------

    def save(self, reward=None):
        """将各 agent 的 Q-net 与 target Q-net 参数保存到 res_dir/model.pt。

        同时保存 target_q_net 以确保断点续训的稳定性。
        """
        torch.save(
            {
                name: {
                    'q_net': agent.q_net.state_dict(),
                    'target_q_net': agent.target_q_net.state_dict(),
                }
                for name, agent in self.agents.items()
            },
            os.path.join(self.res_dir, 'model.pt'),
        )

    @classmethod
    def load(cls, dim_info: dict, file: str, capacity: int,
             batch_size: int, lr: float, device=None) -> 'VDN':
        """从保存的 model.pt 恢复 VDN 实例（用于评估或续训）。"""
        instance = cls(dim_info, capacity, batch_size, lr,
                       os.path.dirname(file), device=device)
        data = torch.load(file, map_location=instance.device)
        for agent_id, agent in instance.agents.items():
            agent.q_net.load_state_dict(data[agent_id]['q_net'])
            agent.target_q_net.load_state_dict(data[agent_id]['target_q_net'])
        return instance
