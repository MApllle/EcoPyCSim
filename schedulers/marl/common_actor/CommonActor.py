import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from schedulers.marl.maddpg.Agent import MLPNetwork
from schedulers.marl.maddpg.Buffer import Buffer


class CommonActor:
    """
    Common-Actor baseline: N+1 independent actors with ONE shared global critic.
    Critic input = concat(all obs) + concat(all actions), i.e. no neighborhood restriction.
    Used to contrast against HierMARL and show the cost of full-state centralization.
    """

    def __init__(self, dim_info, capacity, batch_size,
                 actor_lr, critic_lr, res_dir, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Ordered agent list: global first, then local_0 … local_N
        self.agent_ids = (
            ["global"] + sorted([k for k in dim_info if k.startswith("local_")])
        )

        # Per-agent obs/act dims (flatten dict-obs to 1-D)
        self.obs_dims = {
            aid: int(sum(np.prod(s) for s in dim_info[aid]["obs_shape"].values()))
            for aid in self.agent_ids
        }
        self.act_dims = {aid: dim_info[aid]["action_dim"] for aid in self.agent_ids}

        global_obs_dim = sum(self.obs_dims.values())
        global_act_dim = sum(self.act_dims.values())
        self.global_obs_dim = global_obs_dim
        self.global_act_dim = global_act_dim

        print(
            f"[CommonActor] critic input dim = "
            f"{global_obs_dim} (obs) + {global_act_dim} (act) = "
            f"{global_obs_dim + global_act_dim}"
        )

        # Per-agent actors and their targets
        self.actors = {
            aid: MLPNetwork(self.obs_dims[aid], self.act_dims[aid]).to(self.device)
            for aid in self.agent_ids
        }
        self.target_actors = {aid: deepcopy(net) for aid, net in self.actors.items()}
        self.actor_optims = {
            aid: Adam(net.parameters(), lr=actor_lr)
            for aid, net in self.actors.items()
        }

        # ONE shared critic + its target
        critic_in = global_obs_dim + global_act_dim
        self.critic = MLPNetwork(critic_in, 1).to(self.device)
        self.target_critic = deepcopy(self.critic)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        # Per-agent replay buffers (same indices sampled together)
        self.buffers = {
            aid: Buffer(capacity, self.obs_dims[aid], self.act_dims[aid], self.device)
            for aid in self.agent_ids
        }

        self.batch_size = batch_size
        self.res_dir = res_dir

    # ── helpers ────────────────────────────────────────────────────────────

    def _flat(self, obs_dict):
        return np.concatenate(
            [obs_dict[k].flatten() for k in sorted(obs_dict.keys())]
        ).astype(np.float32)

    def _onehot(self, action_int, dim):
        v = np.zeros(dim, dtype=np.float32)
        v[int(action_int)] = 1.0
        return v

    def _global_tensor(self, obs_dict, act_dict):
        """Concatenate per-agent obs and act tensors into global vectors."""
        obs_cat = torch.cat([obs_dict[aid] for aid in self.agent_ids], dim=1)
        act_cat = torch.cat([act_dict[aid] for aid in self.agent_ids], dim=1)
        return torch.cat([obs_cat, act_cat], dim=1)

    # ── action selection ───────────────────────────────────────────────────

    def select_action(self, obs):
        actions = {}
        with torch.no_grad():
            for aid in self.agent_ids:
                obs_t = torch.from_numpy(
                    self._flat(obs[aid])
                ).unsqueeze(0).to(self.device)
                logits = self.actors[aid](obs_t)
                action = F.gumbel_softmax(logits, hard=True)
                actions[aid] = action.squeeze(0).argmax().item()
        return actions

    # ── experience storage ─────────────────────────────────────────────────

    def add(self, obs, action, reward, next_obs, done):
        for aid in self.agent_ids:
            self.buffers[aid].add(
                self._flat(obs[aid]),
                self._onehot(action[aid], self.act_dims[aid]),
                reward[aid],
                self._flat(next_obs[aid]),
                done[aid],
            )

    # ── learning ───────────────────────────────────────────────────────────

    def learn(self, gamma):
        min_size = min(len(buf) for buf in self.buffers.values())
        if min_size < self.batch_size:
            return

        indices = np.random.choice(min_size, self.batch_size, replace=False)
        samples = {aid: self.buffers[aid].sample(indices) for aid in self.agent_ids}

        obs_t    = {aid: samples[aid][0] for aid in self.agent_ids}
        act_t    = {aid: samples[aid][1] for aid in self.agent_ids}
        rew_t    = {aid: samples[aid][2] for aid in self.agent_ids}
        next_t   = {aid: samples[aid][3] for aid in self.agent_ids}
        done_t   = {aid: samples[aid][4] for aid in self.agent_ids}

        # Team reward: mean across all agents
        mean_rew = torch.stack(list(rew_t.values()), dim=0).mean(dim=0)
        # All agents reach done together
        done_flag = done_t["global"]

        # ── critic update ──────────────────────────────────────────────────
        with torch.no_grad():
            next_act_t = {}
            for aid in self.agent_ids:
                logits = self.target_actors[aid](next_t[aid])
                next_act_t[aid] = F.gumbel_softmax(logits, hard=True)

            next_global = torch.cat(
                [torch.cat([next_t[a] for a in self.agent_ids], dim=1),
                 torch.cat([next_act_t[a] for a in self.agent_ids], dim=1)],
                dim=1,
            )
            target_q = mean_rew + gamma * (1.0 - done_flag) * \
                self.target_critic(next_global).squeeze(1)

        curr_global = self._global_tensor(obs_t, act_t)
        curr_q = self.critic(curr_global).squeeze(1)
        critic_loss = F.mse_loss(curr_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()

        # ── actor updates (freeze critic to avoid accumulating its grads) ──
        for p in self.critic.parameters():
            p.requires_grad_(False)

        for aid in self.agent_ids:
            logits = self.actors[aid](obs_t[aid])
            new_act = F.gumbel_softmax(logits, hard=True)

            # Build global action with this agent's new action, others detached
            acts_for_critic = {
                a: (new_act if a == aid else act_t[a].detach())
                for a in self.agent_ids
            }
            actor_input = self._global_tensor(
                {a: obs_t[a].detach() for a in self.agent_ids},
                acts_for_critic,
            )
            actor_loss = -self.critic(actor_input).squeeze(1).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()

            self.actor_optims[aid].zero_grad()
            (actor_loss + 1e-3 * actor_loss_pse).backward()
            torch.nn.utils.clip_grad_norm_(self.actors[aid].parameters(), 0.5)
            self.actor_optims[aid].step()

        for p in self.critic.parameters():
            p.requires_grad_(True)

    # ── target network soft update ─────────────────────────────────────────

    def update_target(self, tau):
        for aid in self.agent_ids:
            for p, tp in zip(
                self.actors[aid].parameters(),
                self.target_actors[aid].parameters(),
            ):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    # ── persistence ────────────────────────────────────────────────────────

    def save(self, reward):
        torch.save(
            {aid: self.actors[aid].state_dict() for aid in self.agent_ids},
            os.path.join(self.res_dir, "model.pt"),
        )

    @classmethod
    def load(cls, dim_info, file, capacity, batch_size, actor_lr, critic_lr, device=None):
        instance = cls(
            dim_info, capacity, batch_size, actor_lr, critic_lr,
            os.path.dirname(file), device=device,
        )
        data = torch.load(file, map_location=instance.device)
        for aid in instance.agent_ids:
            instance.actors[aid].load_state_dict(data[aid])
        return instance
