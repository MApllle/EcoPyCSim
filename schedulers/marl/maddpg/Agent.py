from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam


class HeterogeneousCritic(nn.Module):
    """Attention-based global critic that explicitly models task-resource heterogeneity (A3).

    The critic parses the concatenated global observation into:
    - task features  : [task_cpu, task_ram, task_deadline]  (3-dim)
    - resource matrix: shape (num_servers, 2) — [cpu_util, efficiency_tier] per server

    A single cross-attention layer lets the task query each resource, producing a
    heterogeneity-aware representation before the final Q-value MLP.

    Parameters
    ----------
    global_obs_dim       : total flattened observation dimension across all agents
    global_act_dim       : total action dimension across all agents
    num_servers          : number of physical servers (= farms × servers_per_farm)
    task_feature_indices : tuple of 3 ints — indices into global obs for
                           (task_cpu, task_ram, task_deadline)
    wall_time_idx        : index into global obs for wall_time
    d_model              : internal embedding dimension (default 32)
    n_heads              : number of attention heads (default 2)
    """

    def __init__(
        self,
        global_obs_dim: int,
        global_act_dim: int,
        num_servers: int,
        task_feature_indices: Tuple[int, int, int],
        wall_time_idx: int,
        d_model: int = 32,
        n_heads: int = 2,
    ):
        super().__init__()
        self.num_servers          = num_servers
        self.task_feature_indices = list(task_feature_indices)
        self.wall_time_idx        = wall_time_idx

        task_dim     = len(task_feature_indices)   # 3
        resource_dim = 2                            # [cpu_util, efficiency_tier]

        self.task_encoder     = nn.Linear(task_dim, d_model)
        self.resource_encoder = nn.Linear(resource_dim, d_model)
        # task (query) attends over resources (key / value)
        self.cross_attn       = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.context_encoder  = nn.Linear(1 + global_act_dim, d_model)

        self.output_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            m.bias.data.fill_(0.01)

    def forward(self, global_obs: Tensor, global_actions: Tensor) -> Tensor:
        # global_obs      : [B, global_obs_dim]
        # global_actions  : [B, global_act_dim]

        # --- Task features ---
        task_feat = global_obs[:, self.task_feature_indices]  # [B, 3]
        task_emb  = F.relu(self.task_encoder(task_feat))      # [B, d]

        # --- Resource features: first num_servers cols = cpu_utils,
        #     next num_servers cols = efficiency_tiers (server_farm obs layout) ---
        n = self.num_servers
        cpu_utils  = global_obs[:, :n]            # [B, n]
        eff_tiers  = global_obs[:, n:2 * n]       # [B, n]
        res_feat   = torch.stack([cpu_utils, eff_tiers], dim=-1)  # [B, n, 2]
        res_emb    = F.relu(self.resource_encoder(res_feat))      # [B, n, d]

        # --- Cross-attention ---
        attn_out, _ = self.cross_attn(
            task_emb.unsqueeze(1),   # query  [B, 1, d]
            res_emb,                  # key    [B, n, d]
            res_emb,                  # value  [B, n, d]
        )
        attn_out = attn_out.squeeze(1)  # [B, d]

        # --- Context: wall_time + joint actions ---
        wall_time   = global_obs[:, self.wall_time_idx:self.wall_time_idx + 1]  # [B, 1]
        ctx         = torch.cat([wall_time, global_actions], dim=-1)
        ctx_emb     = F.relu(self.context_encoder(ctx))                          # [B, d]

        combined = torch.cat([task_emb, attn_out, ctx_emb], dim=-1)  # [B, 3d]
        return self.output_mlp(combined).squeeze(-1)                  # [B]


class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(
        self,
        obs_dim,
        act_dim,
        global_obs_dim,
        actor_lr,
        critic_lr,
        device,
        attention_critic_kwargs: Optional[dict] = None,
    ):
        self.actor = MLPNetwork(obs_dim, act_dim).to(device)

        if attention_critic_kwargs is not None:
            self.critic = HeterogeneousCritic(**attention_critic_kwargs).to(device)
            self._use_attention = True
        else:
            # Standard MLP critic: input = concat(all_obs, all_actions)
            self.critic = MLPNetwork(global_obs_dim, 1).to(device)
            self._use_attention = False

        self.actor_optimizer  = Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor  = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
        # but as mention in the doc, it may be removed in the future, so i implement it myself
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def action(self, obs, model_out=False):
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])

        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        # action = self.gumbel_softmax(logits)
        action = F.gumbel_softmax(logits, hard=True)
        if model_out:
            return action, logits
        return action

    def target_action(self, obs):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])

        with torch.no_grad():
            logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
            # action = self.gumbel_softmax(logits)
            action = F.gumbel_softmax(logits, hard=True)
        return action.squeeze(0).detach()

    def _critic_forward(self, net, state_list: List[Tensor], act_list: List[Tensor]) -> Tensor:
        if self._use_attention:
            global_obs  = torch.cat(state_list, dim=1)
            global_acts = torch.cat(act_list,   dim=1)
            return net(global_obs, global_acts)
        x = torch.cat(state_list + act_list, 1)
        return net(x).squeeze(1)

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]) -> Tensor:
        return self._critic_forward(self.critic, state_list, act_list)

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]) -> Tensor:
        return self._critic_forward(self.target_critic, state_list, act_list)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)
