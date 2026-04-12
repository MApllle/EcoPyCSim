from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam


class MLPNetwork(nn.Module):
    """与 MADDPG 保持相同的 2 层 MLP 结构，方便公平对比。"""

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


class DQNAgent:
    """单个 agent 的独立 DQN。

    与 MADDPG Agent 的核心区别：
    - 只有 Q-net（无 Critic），Q-net 直接输出每个离散动作的 Q 值
    - 完全不依赖其他 agent 的观测/动作（Independent 的含义）
    - 使用 epsilon-greedy 探索代替 Gumbel-Softmax
    """

    def __init__(self, obs_dim: int, act_dim: int, lr: float, device: torch.device):
        self.act_dim = act_dim
        self.device = device

        self.q_net = MLPNetwork(obs_dim, act_dim).to(device)
        self.target_q_net = deepcopy(self.q_net)
        self.optimizer = Adam(self.q_net.parameters(), lr=lr)

        # 目标网络不参与梯度计算
        for p in self.target_q_net.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # 动作选取（epsilon-greedy）
    # ------------------------------------------------------------------

    def select_action(self, obs_tensor: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Args:
            obs_tensor: shape (1, obs_dim)，已在正确 device 上
            epsilon: 随机探索概率 [0, 1]
        Returns:
            int 离散动作
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(self.act_dim, (1,)).item()
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)          # (1, act_dim)
        return q_values.argmax(dim=1).item()

    # ------------------------------------------------------------------
    # 学习步
    # ------------------------------------------------------------------

    def learn(
        self,
        obs: torch.Tensor,        # (B, obs_dim)
        actions: torch.Tensor,    # (B, 1)  int64
        rewards: torch.Tensor,    # (B,)
        next_obs: torch.Tensor,   # (B, obs_dim)
        dones: torch.Tensor,      # (B,)
        gamma: float,
    ) -> float:
        """执行一步 DQN 更新，返回 loss 值（float）。"""
        # 当前 Q(s, a)
        q_pred = self.q_net(obs).gather(1, actions.long()).squeeze(1)  # (B,)

        # 目标 Q = r + γ * max_{a'} Q_target(s', a')
        with torch.no_grad():
            q_next = self.target_q_net(next_obs).max(dim=1)[0]         # (B,)
            q_target = rewards + gamma * q_next * (1.0 - dones)

        loss = F.mse_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # 目标网络软更新
    # ------------------------------------------------------------------

    def soft_update(self, tau: float):
        """θ_target ← τ·θ + (1-τ)·θ_target"""
        for src, dst in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            dst.data.copy_(tau * src.data + (1.0 - tau) * dst.data)
