"""
Lagrangian-constrained MADDPG (A1).

Formulation
-----------
  Reward  : r_t = energy_saving_reward  (minimize energy)
  Constraint: E[Σ c_t] ≤ budget,  where c_t = 1 if task rejected else 0
  Lagrangian objective: L(π, λ) = E[Σ r_t] - λ · (E[Σ c_t] - budget)

λ is updated via dual gradient ascent after each learn() call:
  λ ← max(0,  λ + η_λ · (avg_cost - budget))

The Lagrangian-adjusted per-step reward fed to the critic is:
  r_lagr = r_energy - λ · c_t
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from schedulers.marl.maddpg.Buffer import Buffer
from schedulers.marl.maddpg.MADDPG import MADDPG, setup_logger


class LagrangianBuffer(Buffer):
    """Replay buffer that stores an extra scalar constraint-cost per transition."""

    def __init__(self, capacity, obs_dim, act_dim, device):
        super().__init__(capacity, obs_dim, act_dim, device)
        self.cost = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, cost, next_obs, done):
        idx = self._index
        # Call parent with dummy reward so index bookkeeping stays in one place.
        # We override the stored reward right after.
        super().add(obs, action, reward, next_obs, done)
        self.cost[idx] = cost

    def sample(self, indices):
        # Return raw (un-normalised) reward + cost so the caller can normalise
        # the Lagrangian-adjusted reward jointly.
        obs      = torch.from_numpy(self.obs[indices]).float().to(self.device)
        action   = torch.from_numpy(self.action[indices]).float().to(self.device)
        reward   = torch.from_numpy(self.reward[indices]).float().to(self.device)
        cost     = torch.from_numpy(self.cost[indices]).float().to(self.device)
        next_obs = torch.from_numpy(self.next_obs[indices]).float().to(self.device)
        done     = torch.from_numpy(self.done[indices]).float().to(self.device)
        return obs, action, reward, cost, next_obs, done


class LagrangianMADDPG(MADDPG):
    """MADDPG with a shared Lagrangian multiplier for a rejection-rate constraint."""

    def __init__(
        self,
        dim_info,
        capacity,
        batch_size,
        actor_lr,
        critic_lr,
        res_dir,
        constraint_budget: float = 0.3,
        lambda_lr: float = 0.005,
        lambda_init: float = 1.0,
        device=None,
        use_attention_critic: bool = False,
        num_servers: int = None,
    ):
        super().__init__(
            dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir,
            use_attention_critic=use_attention_critic,
            num_servers=num_servers,
            device=device,
        )
        # Replace plain buffers with Lagrangian buffers
        for agent_id, info in dim_info.items():
            obs_dim = sum(
                int(np.prod(s)) for s in info['obs_shape'].values()
            )
            act_dim = info['action_dim']
            self.buffers[agent_id] = LagrangianBuffer(
                capacity, obs_dim, act_dim, self.device
            )

        self.lambda_param       = lambda_init
        self.lambda_lr          = lambda_lr
        self.constraint_budget  = constraint_budget
        self.logger = setup_logger(os.path.join(res_dir, 'lagrangian_maddpg.log'))

    # ------------------------------------------------------------------
    # Override add() to accept an extra per-step constraint cost signal
    # ------------------------------------------------------------------
    def add(self, obs, action, energy_reward, constraint_cost, next_obs, done):
        """
        Parameters
        ----------
        energy_reward    : dict[agent_id → float]  pure energy-saving reward
        constraint_cost  : float  1.0 if task was rejected this step, else 0.0
        """
        for agent_id in obs.keys():
            flat_o      = self.flatten_obs(obs[agent_id])
            flat_next_o = self.flatten_obs(next_obs[agent_id])
            assert flat_o.shape[0] == self.buffers[agent_id].obs.shape[1]
            self.buffers[agent_id].add(
                flat_o,
                action[agent_id],
                energy_reward[agent_id],
                constraint_cost,
                flat_next_o,
                done[agent_id],
            )

    # ------------------------------------------------------------------
    # Override sample() to return cost alongside other transitions
    # ------------------------------------------------------------------
    def sample(self, batch_size):
        total_num = len(next(iter(self.buffers.values())))
        if total_num < batch_size:
            batch_size = total_num
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        obs, act, reward, cost, next_obs, done, next_act = {}, {}, {}, {}, {}, {}, {}
        for agent_id in self.buffers.keys():
            o, a, r, c, n_o, d = self.buffers[agent_id].sample(indices)
            obs[agent_id]      = o
            act[agent_id]      = a
            reward[agent_id]   = r
            cost[agent_id]     = c
            next_obs[agent_id] = n_o
            done[agent_id]     = d
            next_act[agent_id] = self.agents[agent_id].target_action(n_o)

        return obs, act, reward, cost, next_obs, done, next_act

    # ------------------------------------------------------------------
    # Override learn() to use Lagrangian-adjusted reward
    # ------------------------------------------------------------------
    def learn(self, batch_size, gamma):
        total_cost = 0.0

        for agent_id, agent in self.agents.items():
            obs, act, reward, cost, next_obs, done, next_act = self.sample(batch_size)

            # Lagrangian-adjusted reward
            lagr_reward = reward[agent_id] - self.lambda_param * cost[agent_id]
            # Normalise after combination
            lagr_reward = (lagr_reward - lagr_reward.mean()) / (lagr_reward.std() + 1e-7)

            # Critic update
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))
            next_target  = agent.target_critic_value(
                list(next_obs.values()), list(next_act.values())
            )
            target_value = lagr_reward + gamma * next_target * (1 - done[agent_id])
            critic_loss  = F.mse_loss(critic_value, target_value.detach())
            agent.update_critic(critic_loss)

            # Actor update
            action, logits = agent.action(obs[agent_id], model_out=True)
            act[agent_id]  = action
            actor_loss     = -agent.critic_value(list(obs.values()), list(act.values())).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)

            total_cost += cost[agent_id].mean().item()

        # Dual gradient ascent on λ (shared across agents)
        avg_cost = total_cost / max(len(self.agents), 1)
        self.lambda_param = max(
            0.0,
            self.lambda_param + self.lambda_lr * (avg_cost - self.constraint_budget),
        )
