"""QMIX 混合网络（Monotonic Mixing Network）

核心思想（Rashid et al., 2018）：
    Q_tot(s, a) = MixNet([Q_1(o_1,a_1), ..., Q_N(o_N,a_N)], s)

关键约束 —— 单调性（Monotonicity）：
    ∂Q_tot / ∂Q_i ≥ 0，对所有 i 成立

    这保证了分散执行的最优性：
        argmax_a Q_tot ⟺ (argmax_{a_1} Q_1, ..., argmax_{a_N} Q_N)

实现方式：
    混合网络的权重矩阵由超网络（hypernetwork）从全局状态 s 生成，
    并通过 abs() 约束为非负，从而满足单调性。
    偏置项不受约束（保留更强的表达能力）。

网络结构：
    超网络 hyper_w1 : state → weights of layer 1  (n_agents × embed_dim)  [非负]
    超网络 hyper_w2 : state → weights of layer 2  (embed_dim × 1)         [非负]
    超网络 hyper_b1 : state → bias of layer 1     (embed_dim,)             [无约束]
    超网络 hyper_b2 : state → bias of layer 2     (1,)                     [无约束]

前向计算：
    w1     = |hyper_w1(s)|                          (B, n_agents, embed_dim)
    b1     = hyper_b1(s)                            (B, 1, embed_dim)
    hidden = ELU( Q_agents @ w1 + b1 )              (B, 1, embed_dim)
    w2     = |hyper_w2(s)|                          (B, embed_dim, 1)
    b2     = hyper_b2(s)                            (B, 1, 1)
    Q_tot  = hidden @ w2 + b2                       (B,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    """QMIX 混合网络：将 N 个 agent 的 Q 值通过状态条件化的单调函数融合为 Q_tot。

    Args:
        n_agents:  agent 数量（本框架中为 2）
        state_dim: 全局状态维度（所有 agent 观测拼接后的维度）
        embed_dim: 混合网络隐层维度，默认 32（原论文默认值）
    """

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        # 超网络：生成第一层权重  state → (n_agents × embed_dim)
        # 使用两层 MLP 给超网络足够表达能力
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim),
        )

        # 超网络：生成第二层权重  state → (embed_dim × 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),      # embed_dim * 1 = embed_dim
        )

        # 超网络：生成第一层偏置（无非负约束）
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # 超网络：生成第二层偏置（无非负约束，使用小网络增加灵活性）
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, q_agents: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """单调混合前向计算。

        Args:
            q_agents: (B, n_agents)  — 各 agent 当前动作对应的 Q 值
            state:    (B, state_dim) — 全局状态（所有 agent 观测拼接）
        Returns:
            q_tot:    (B,)           — 混合后的联合 Q 值
        """
        B = q_agents.size(0)

        # q_agents 变形为行向量：(B, 1, n_agents)
        q = q_agents.unsqueeze(1)

        # ── 第一层 ────────────────────────────────────────────────────────
        # abs() 保证权重非负 → 单调性约束
        w1 = self.hyper_w1(state).abs().view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.embed_dim)
        # (B, 1, n_agents) @ (B, n_agents, embed_dim) + (B, 1, embed_dim)
        hidden = F.elu(torch.bmm(q, w1) + b1)   # (B, 1, embed_dim)

        # ── 第二层 ────────────────────────────────────────────────────────
        w2 = self.hyper_w2(state).abs().view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        # (B, 1, embed_dim) @ (B, embed_dim, 1) + (B, 1, 1)
        q_tot = (torch.bmm(hidden, w2) + b2).view(B)   # (B,)

        return q_tot
