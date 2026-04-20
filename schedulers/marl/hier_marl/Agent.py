from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.optim import Adam

from schedulers.marl.maddpg.Agent import MLPNetwork


class LocalAgent:
    """局部调度器 agent，critic 仅看自己的 (obs + act)。"""

    def __init__(self, obs_dim, act_dim, actor_lr, critic_lr, device):
        self.device = device
        critic_in = obs_dim + act_dim

        self.actor = MLPNetwork(obs_dim, act_dim).to(device)
        self.critic = MLPNetwork(critic_in, 1).to(device)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)

    def action(self, obs, model_out=False):
        logits = self.actor(obs)
        action = F.gumbel_softmax(logits, hard=True)
        return (action, logits) if model_out else action

    def target_action(self, obs):
        with torch.no_grad():
            logits = self.target_actor(obs)
            action = F.gumbel_softmax(logits, hard=True)
        return action.detach()

    def critic_value(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.critic(x).squeeze(1)

    def target_critic_value(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.target_critic(x).squeeze(1)

    def update_actor(self, loss):
        self.actor_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_opt.step()

    def update_critic(self, loss):
        self.critic_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_opt.step()

    def soft_update(self, tau=0.01):
        for p, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


class GlobalAgent:
    """全局调度器 agent，critic 接收自己 + 被选中 local 的 (obs+act)。

    Critic 输入维度 = global_obs_dim + global_act_dim + local_obs_dim + local_act_dim，
    与 local agent 总数 N 无关，是邻域 critic 可扩展性的关键。
    """

    def __init__(self, global_obs_dim, global_act_dim,
                 local_obs_dim, local_act_dim,
                 actor_lr, critic_lr, device):
        self.device = device
        critic_in = global_obs_dim + global_act_dim + local_obs_dim + local_act_dim

        self.actor = MLPNetwork(global_obs_dim, global_act_dim).to(device)
        self.critic = MLPNetwork(critic_in, 1).to(device)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)

    def action(self, obs, model_out=False):
        logits = self.actor(obs)
        action = F.gumbel_softmax(logits, hard=True)
        return (action, logits) if model_out else action

    def target_action(self, obs):
        with torch.no_grad():
            logits = self.target_actor(obs)
            action = F.gumbel_softmax(logits, hard=True)
        return action.detach()

    def critic_value(self, g_obs, g_act, l_obs, l_act):
        x = torch.cat([g_obs, g_act, l_obs, l_act], dim=1)
        return self.critic(x).squeeze(1)

    def target_critic_value(self, g_obs, g_act, l_obs, l_act):
        x = torch.cat([g_obs, g_act, l_obs, l_act], dim=1)
        return self.target_critic(x).squeeze(1)

    def update_actor(self, loss):
        self.actor_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_opt.step()

    def update_critic(self, loss):
        self.critic_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_opt.step()

    def soft_update(self, tau=0.01):
        for p, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
