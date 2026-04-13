"""QMIX 单体 Agent —— 只包含 Q-network，不含独立优化器。

与 VDNAgent 架构完全相同：
- Q-network 和 target Q-network（软更新）
- select_action（epsilon-greedy 分散执行）
- q_values / target_q_values（纯前向接口，供 QMIX 协调器调用）
- 无独立优化器 —— 优化器由 QMIX 协调器统一管理

与 VDN 的区别体现在协调器层面：
- VDN：Q_tot = Σ Q_i（线性分解）
- QMIX：Q_tot = MixNet([Q_1,...,Q_N], s)（由全局状态条件化的单调混合网络）
"""

from copy import deepcopy

import torch
import torch.nn as nn


class MLPNetwork(nn.Module):
    """与 VDN / IDQN / MADDPG 保持相同的 2 层 MLP 结构，方便公平对比。"""

    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self._init)

    @staticmethod
    def _init(m):
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)


class QMIXAgent:
    """单个 agent 的 Q-network，不含独立优化器。

    QMIX 中所有 agent 的 Q-network 参数由协调器（QMIX）统一放入一个 Adam optimizer，
    通过混合网络的联合损失 MSE(Q_tot, target) 同步更新，实现集中训练。
    执行时每个 agent 仍独立 argmax Q_i（分散执行）。

    与 VDNAgent 架构完全一致 —— QMIX 与 VDN 的核心区别在混合网络（QMixer），
    而非单体 agent 网络。
    """

    def __init__(self, obs_dim: int, act_dim: int, device: torch.device):
        self.act_dim = act_dim
        self.device = device

        self.q_net = MLPNetwork(obs_dim, act_dim).to(device)
        self.target_q_net = deepcopy(self.q_net)

        # 目标网络不参与梯度计算
        for p in self.target_q_net.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # 前向接口（供 QMIX 协调器在 learn() 中调用）
    # ------------------------------------------------------------------

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        """返回所有动作的 Q 值，shape (B, act_dim)。"""
        return self.q_net(obs)

    def target_q_values(self, obs: torch.Tensor) -> torch.Tensor:
        """返回目标网络的 Q 值，shape (B, act_dim)。"""
        return self.target_q_net(obs)

    # ------------------------------------------------------------------
    # 分散执行：epsilon-greedy 动作选取
    # ------------------------------------------------------------------

    def select_action(self, obs_tensor: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Args:
            obs_tensor: shape (1, obs_dim)，已在正确 device 上
            epsilon:    随机探索概率 [0, 1]
        Returns:
            int 离散动作索引
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(self.act_dim, (1,)).item()
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)   # (1, act_dim)
        return q_values.argmax(dim=1).item()

    # ------------------------------------------------------------------
    # 目标网络软更新
    # ------------------------------------------------------------------

    def soft_update(self, tau: float):
        """θ_target ← τ·θ + (1-τ)·θ_target"""
        for src, dst in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            dst.data.copy_(tau * src.data + (1.0 - tau) * dst.data)
