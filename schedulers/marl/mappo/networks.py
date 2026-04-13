"""
MAPPO Actor / Critic 网络。
简化自 on-policy-main r_actor_critic.py：
  - 去除 gym.Space 依赖，直接接受 obs_dim / act_dim 整数
  - 去除 CNN 分支（环境使用向量观察）
  - 默认不使用 RNN（use_rnn=False），可选开启
  - 使用离散动作（Categorical）
"""
import torch
import torch.nn as nn

from schedulers.marl.mappo.utils import check, init


# ── 基础 MLP ──────────────────────────────────────────────────────────────────

def _make_mlp(input_dim: int, hidden_size: int, num_layers: int = 1,
              use_orthogonal: bool = True, use_relu: bool = False) -> nn.Module:
    active = nn.ReLU() if use_relu else nn.Tanh()
    init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
    gain = nn.init.calculate_gain('relu' if use_relu else 'tanh')

    def _init(m):
        return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

    layers = [_init(nn.Linear(input_dim, hidden_size)), active, nn.LayerNorm(hidden_size)]
    for _ in range(num_layers):
        layers += [_init(nn.Linear(hidden_size, hidden_size)), active, nn.LayerNorm(hidden_size)]
    return nn.Sequential(*layers)


# ── 可选 GRU 层 ───────────────────────────────────────────────────────────────

class _GRULayer(nn.Module):
    """单层 GRU，含 LayerNorm 输出。"""

    def __init__(self, input_dim: int, hidden_size: int, use_orthogonal: bool = True):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=1, batch_first=False)
        for name, p in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(p, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(p) if use_orthogonal else nn.init.xavier_uniform_(p)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, hx, masks):
        # x: (B, H), hx: (B, 1, H), masks: (B, 1)
        hx = hx * masks.unsqueeze(-1)          # reset hidden on episode boundary
        hx_t = hx.transpose(0, 1).contiguous() # (1, B, H)
        out, hx_new = self.gru(x.unsqueeze(0), hx_t)
        out = self.norm(out.squeeze(0))
        return out, hx_new.transpose(0, 1)      # (B, H), (B, 1, H)


# ── 离散动作头 ────────────────────────────────────────────────────────────────

class _CategoricalHead(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, use_orthogonal: bool = True):
        super().__init__()
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
        self.linear = init(nn.Linear(input_dim, action_dim),
                           init_method, lambda x: nn.init.constant_(x, 0), gain=0.01)

    def forward(self, x, available_actions=None):
        logits = self.linear(x)
        if available_actions is not None:
            logits[available_actions == 0] = -1e10
        return torch.distributions.Categorical(logits=logits)


# ── Actor ─────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """
    分散执行的 Actor 网络（每个 agent 独立实例）。

    Args:
        obs_dim:    局部观察展平后的维度
        act_dim:    离散动作数
        hidden_size: 隐层宽度
        num_layers:  MLP 隐层数（不含输入层）
        use_rnn:    是否使用 GRU
        device:     计算设备
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64,
                 num_layers: int = 1, use_rnn: bool = False,
                 use_orthogonal: bool = True, device=torch.device('cpu')):
        super().__init__()
        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.base = _make_mlp(obs_dim, hidden_size, num_layers, use_orthogonal)
        if use_rnn:
            self.rnn = _GRULayer(hidden_size, hidden_size, use_orthogonal)
        self.act_head = _CategoricalHead(hidden_size, act_dim, use_orthogonal)
        self.to(device)

    def forward(self, obs, rnn_hx=None, masks=None, available_actions=None,
                deterministic=False):
        """
        Returns:
            actions:        (B,)  int tensor
            action_log_probs: (B, 1)
            rnn_hx:         (B, 1, H) or None
        """
        obs = check(obs).to(**self.tpdv)
        feat = self.base(obs)

        if self.use_rnn:
            rnn_hx = check(rnn_hx).to(**self.tpdv)
            masks = check(masks).to(**self.tpdv)
            feat, rnn_hx = self.rnn(feat, rnn_hx, masks)

        dist = self.act_head(feat, available_actions)
        actions = dist.mode if deterministic else dist.sample()  # (B,) or scalar
        # Ensure shape (B,) to avoid scalar issues
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
        log_probs = dist.log_prob(actions).unsqueeze(-1)          # (B, 1)
        return actions, log_probs, rnn_hx

    def evaluate_actions(self, obs, actions, rnn_hx=None, masks=None,
                         available_actions=None, active_masks=None):
        """
        Returns:
            action_log_probs: (B, 1)
            dist_entropy:     scalar
        """
        obs = check(obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        feat = self.base(obs)

        if self.use_rnn:
            rnn_hx = check(rnn_hx).to(**self.tpdv)
            masks = check(masks).to(**self.tpdv)
            feat, _ = self.rnn(feat, rnn_hx, masks)

        dist = self.act_head(feat, available_actions)
        log_probs = dist.log_prob(actions.squeeze(-1).long()).unsqueeze(-1)  # (B, 1)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
            entropy = (dist.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
        else:
            entropy = dist.entropy().mean()

        return log_probs, entropy


# ── Critic ────────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """
    集中式 Critic（所有 agent 共用）。输入为全局（集中）观察。

    Args:
        cent_obs_dim: 所有 agent 局部观察拼接后的维度
        hidden_size:  隐层宽度
        num_layers:   MLP 隐层数
        use_rnn:      是否使用 GRU
        device:       计算设备
    """

    def __init__(self, cent_obs_dim: int, hidden_size: int = 64,
                 num_layers: int = 1, use_rnn: bool = False,
                 use_orthogonal: bool = True, device=torch.device('cpu')):
        super().__init__()
        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.base = _make_mlp(cent_obs_dim, hidden_size, num_layers, use_orthogonal)
        if use_rnn:
            self.rnn = _GRULayer(hidden_size, hidden_size, use_orthogonal)

        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
        self.v_out = init(nn.Linear(hidden_size, 1),
                         init_method, lambda x: nn.init.constant_(x, 0))
        self.to(device)

    def forward(self, cent_obs, rnn_hx=None, masks=None):
        """
        Returns:
            values:  (B, 1)
            rnn_hx:  (B, 1, H) or None
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        feat = self.base(cent_obs)

        if self.use_rnn:
            rnn_hx = check(rnn_hx).to(**self.tpdv)
            masks = check(masks).to(**self.tpdv)
            feat, rnn_hx = self.rnn(feat, rnn_hx, masks)

        values = self.v_out(feat)
        return values, rnn_hx
