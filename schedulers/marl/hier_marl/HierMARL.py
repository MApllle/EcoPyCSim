import os

import numpy as np
import torch
import torch.nn.functional as F

from schedulers.marl.hier_marl.Agent import GlobalAgent, LocalAgent
from schedulers.marl.hier_marl.Buffer import GlobalBuffer, LocalBuffer


class HierMARL:
    def __init__(self, dim_info, capacity, batch_size,
                 actor_lr, critic_lr, res_dir, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        g_info = dim_info["global"]
        local_keys = sorted([k for k in dim_info if k.startswith("local_")])
        self.num_locals = len(local_keys)
        l_info = dim_info[local_keys[0]]

        g_obs_dim = int(sum(np.prod(s) for s in g_info["obs_shape"].values()))
        g_act_dim = g_info["action_dim"]
        l_obs_dim = int(sum(np.prod(s) for s in l_info["obs_shape"].values()))
        l_act_dim = l_info["action_dim"]

        self.g_act_dim = g_act_dim
        self.l_act_dim = l_act_dim

        self.global_agent = GlobalAgent(
            g_obs_dim, g_act_dim, l_obs_dim, l_act_dim,
            actor_lr, critic_lr, self.device)
        self.local_agents = {
            f"local_{i}": LocalAgent(l_obs_dim, l_act_dim, actor_lr, critic_lr, self.device)
            for i in range(self.num_locals)
        }

        self.global_buffer = GlobalBuffer(
            capacity, g_obs_dim, g_act_dim, l_obs_dim, l_act_dim, self.device)
        self.local_buffers = {
            lid: LocalBuffer(capacity, l_obs_dim, l_act_dim, self.device)
            for lid in self.local_agents
        }

        self.batch_size = batch_size
        self.res_dir = res_dir

    # ── helpers ──────────────────────────────────────────────────────────

    def _flat(self, obs_dict):
        return np.concatenate(
            [obs_dict[k].flatten() for k in sorted(obs_dict.keys())]
        ).astype(np.float32)

    def _onehot(self, action_int, dim):
        v = np.zeros(dim, dtype=np.float32)
        v[int(action_int)] = 1.0
        return v

    # ── action selection ──────────────────────────────────────────────────

    def select_action(self, obs):
        actions = {}
        with torch.no_grad():
            g_t = torch.from_numpy(self._flat(obs["global"])).unsqueeze(0).to(self.device)
            actions["global"] = self.global_agent.action(g_t).squeeze(0).argmax().item()
            for lid, agent in self.local_agents.items():
                l_t = torch.from_numpy(self._flat(obs[lid])).unsqueeze(0).to(self.device)
                actions[lid] = agent.action(l_t).squeeze(0).argmax().item()
        return actions

    # ── experience storage ────────────────────────────────────────────────

    def add(self, obs, action, reward, next_obs, done, selected_farm_id):
        sel_lid = f"local_{selected_farm_id}"

        g_obs_f  = self._flat(obs["global"])
        g_act_oh = self._onehot(action["global"], self.g_act_dim)
        g_next_f = self._flat(next_obs["global"])
        l_obs_f  = self._flat(obs[sel_lid])
        l_act_oh = self._onehot(action[sel_lid], self.l_act_dim)
        l_next_f = self._flat(next_obs[sel_lid])

        self.global_buffer.add(
            g_obs_f, g_act_oh, reward["global"], g_next_f, done["global"],
            selected_farm_id, l_obs_f, l_act_oh, l_next_f,
        )
        self.local_buffers[sel_lid].add(
            l_obs_f, l_act_oh, reward[sel_lid], l_next_f, done[sel_lid],
        )

    # ── learning (Algorithm 1 + 2) ────────────────────────────────────────

    def learn(self, gamma):
        if len(self.global_buffer) >= self.batch_size:
            batch = self.global_buffer.sample(self.batch_size)
            self._update_global(batch, gamma)

        for lid, buf in self.local_buffers.items():
            if len(buf) >= self.batch_size:
                indices = np.random.choice(len(buf), self.batch_size, replace=False)
                batch = buf.sample(indices)
                self._update_local(lid, batch, gamma)

    def _update_global(self, batch, gamma):
        g_o, g_a, r, g_no, d, sel_id, l_o, l_a, l_no = batch
        # sel_id is a numpy int64 array (not moved to device)

        with torch.no_grad():
            next_g_a = self.global_agent.target_action(g_no)
            next_l_a = torch.zeros_like(l_a)
            for i in range(self.num_locals):
                mask = sel_id == i
                if mask.any():
                    mask_t = torch.from_numpy(mask).to(self.device)
                    next_l_a[mask_t] = self.local_agents[f"local_{i}"].target_action(
                        l_no[mask_t]
                    )
            target_q = r + gamma * (1.0 - d) * \
                self.global_agent.target_critic_value(g_no, next_g_a, l_no, next_l_a)

        curr_q = self.global_agent.critic_value(g_o, g_a, l_o, l_a)
        critic_loss = F.mse_loss(curr_q, target_q)
        self.global_agent.update_critic(critic_loss)

        g_action, g_logits = self.global_agent.action(g_o, model_out=True)
        actor_loss = -self.global_agent.critic_value(g_o, g_action, l_o, l_a).mean()
        actor_loss_pse = torch.pow(g_logits, 2).mean()
        self.global_agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)

    def _update_local(self, lid, batch, gamma):
        obs, action, reward, next_obs, done = batch
        agent = self.local_agents[lid]

        with torch.no_grad():
            next_a = agent.target_action(next_obs)
            target_q = reward + gamma * (1.0 - done) * agent.target_critic_value(next_obs, next_a)

        curr_q = agent.critic_value(obs, action)
        critic_loss = F.mse_loss(curr_q, target_q)
        agent.update_critic(critic_loss)

        action_new, logits = agent.action(obs, model_out=True)
        actor_loss = -agent.critic_value(obs, action_new).mean()
        actor_loss_pse = torch.pow(logits, 2).mean()
        agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)

    # ── target network soft update ─────────────────────────────────────────

    def update_target(self, tau):
        self.global_agent.soft_update(tau)
        for agent in self.local_agents.values():
            agent.soft_update(tau)

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, reward):
        torch.save(
            {
                'global': self.global_agent.actor.state_dict(),
                **{lid: agent.actor.state_dict() for lid, agent in self.local_agents.items()}
            },
            os.path.join(self.res_dir, 'model.pt')
        )

    @classmethod
    def load(cls, dim_info, file, capacity, batch_size, actor_lr, critic_lr, device=None):
        instance = cls(dim_info, capacity, batch_size, actor_lr, critic_lr,
                       os.path.dirname(file), device=device)
        data = torch.load(file, map_location=instance.device)
        instance.global_agent.actor.load_state_dict(data['global'])
        for lid, agent in instance.local_agents.items():
            agent.actor.load_state_dict(data[lid])
        return instance
