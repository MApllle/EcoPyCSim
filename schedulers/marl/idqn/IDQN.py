"""Independent DQN (IDQN) — multi-agent 协调器

设计对比 MADDPG：
- MADDPG 有集中式 Critic，输入全局 obs + 全局 act
- IDQN 每个 agent 只看自己的 obs，独立做 Q-learning（无 centralized critic）
- 接口（add / select_action / learn / update_target / save / load）与 MADDPG 对齐，
  方便在实验脚本里直接替换
"""

import logging
import os

import numpy as np
import torch

from schedulers.marl.maddpg.Buffer import Buffer   # 复用已有 Buffer
from schedulers.marl.idqn.DQNAgent import DQNAgent


def _setup_logger(filename: str) -> logging.Logger:
    logger = logging.getLogger(f"idqn_{filename}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename, mode='w')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    return logger


class IDQN:
    """Independent DQN：每个 agent 各自独立训练，无集中式 Critic。

    Args:
        dim_info:   {agent_id: {'obs_shape': {key: shape, ...}, 'action_dim': int}}
                    与 MADDPG 相同的格式，直接来自 set_env()
        capacity:   回放缓冲区容量
        batch_size: 每次学习的 minibatch 大小
        lr:         Q-net 学习率
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
        self.logger = _setup_logger(os.path.join(res_dir, 'idqn.log'))

        self.agents: dict[str, DQNAgent] = {}
        self.buffers: dict[str, Buffer] = {}

        for agent_id, info in dim_info.items():
            obs_dim = sum(np.prod(shape) for shape in info['obs_shape'].values())
            act_dim = info['action_dim']
            self.agents[agent_id] = DQNAgent(obs_dim, act_dim, lr, self.device)
            # act_dim=1：将离散 int 动作存为单列 float，采样时转 long 用于 gather
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim=1, device=self.device)

        self.dim_info = dim_info

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def flatten_obs(obs_dict: dict) -> np.ndarray:
        """将 dict 观测拼接为 1D numpy 数组（键排序，与 MADDPG 相同）。"""
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
    # 动作选取
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
    # 学习
    # ------------------------------------------------------------------

    def learn(self, batch_size: int, gamma: float):
        """各 agent 独立执行一步 DQN 更新。"""
        for agent_id, agent in self.agents.items():
            buf = self.buffers[agent_id]
            if len(buf) < batch_size:
                continue  # 数据不足，跳过

            total = len(buf)
            indices = np.random.choice(total, size=min(batch_size, total), replace=False)
            obs, actions, rewards, next_obs, dones = buf.sample(indices)
            # actions: (B, 1) float → (B, 1) int64
            loss = agent.learn(obs, actions.long(), rewards, next_obs, dones, gamma)
            self.logger.info(f'{agent_id} loss: {loss:.6f}')

    def update_target(self, tau: float):
        """对所有 agent 的目标网络做软更新。"""
        for agent in self.agents.values():
            agent.soft_update(tau)

    # ------------------------------------------------------------------
    # 模型保存 / 加载
    # ------------------------------------------------------------------

    def save(self, reward=None):
        """将各 agent 的 Q-net 参数保存到 res_dir/model.pt。"""
        torch.save(
            {name: agent.q_net.state_dict() for name, agent in self.agents.items()},
            os.path.join(self.res_dir, 'model.pt'),
        )

    @classmethod
    def load(cls, dim_info: dict, file: str, capacity: int,
             batch_size: int, lr: float, device=None) -> 'IDQN':
        """从保存的 model.pt 恢复 IDQN 实例（用于评估）。"""
        instance = cls(dim_info, capacity, batch_size, lr,
                       os.path.dirname(file), device=device)
        data = torch.load(file, map_location=instance.device)
        for agent_id, agent in instance.agents.items():
            agent.q_net.load_state_dict(data[agent_id])
            agent.target_q_net.load_state_dict(data[agent_id])
        return instance
