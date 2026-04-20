import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from env.cloud_scheduling_hier import CloudSchedulingEnvHier
from schedulers.marl.hier_marl.HierMARL import HierMARL


def build_dim_info(env):
    dim_info = {}
    for agent_id in env.agents:
        obs_space = env.observation_space(agent_id)
        obs_shape = {key: space.shape for key, space in obs_space.spaces.items()}
        action_dim = env.action_space(agent_id).n
        dim_info[agent_id] = {'obs_shape': obs_shape, 'action_dim': action_dim}
    return dim_info


parser = argparse.ArgumentParser(description='Evaluate a trained HierMARL model')
parser.add_argument('--model',   type=str, required=True, help='path to model.pt')
parser.add_argument('--jobs',    type=int, default=300)
parser.add_argument('--farms',   type=int, default=30)
parser.add_argument('--servers', type=int, default=210)
parser.add_argument('--seed',    type=int, default=42069)
args = parser.parse_args()

env = CloudSchedulingEnvHier(
    num_jobs=args.jobs,
    num_server_farms=args.farms,
    num_servers=args.servers,
)
env.reset()
dim_info = build_dim_info(env)

hier = HierMARL.load(
    dim_info, args.model,
    capacity=1000, batch_size=256,
    actor_lr=3e-4, critic_lr=3e-4,
)

# ── HierMARL rollout ──────────────────────────────────────────────────────

print("=== HierMARL evaluation ===")
obs, info = env.reset(seed=args.seed)

prices     = []
wall_times = []
step       = 0

while True:
    action = hier.select_action(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = {a: terminated[a] or truncated[a] for a in env.agents}

    g_info = info["global"]
    wall_times.append(g_info["wall_time"])
    prices.append(g_info["price"])

    step += 1
    obs = next_obs

    if all(done.values()):
        break

sum_price      = float(np.sum(prices))
average_price  = float(np.mean(prices))
end_wall_time  = wall_times[-1] if wall_times else 0.0

print(f"Steps:                {step}")
print(f"Rejected tasks:       {env.rejected_tasks_count}")
print(f"Completed jobs:       {env.num_completed_jobs}")
print(f"Average energy price: {average_price:.4f}")
print(f"Total energy price:   {sum_price:.4f}")
print(f"End wall time:        {end_wall_time:.2f}")
print(f"Jain's fairness:      {g_info.get('jains_fairness', 'N/A')}")
print(f"HER:                  {g_info.get('her', 'N/A')}")
print(f"Active server ratio:  {g_info.get('active_server_ratio', 'N/A')}")

# ── Random baseline ───────────────────────────────────────────────────────

print("\n=== Random baseline ===")
obs, info = env.reset(seed=args.seed)

rnd_prices     = []
rnd_wall_times = []
rnd_step       = 0

while True:
    action = {a: env.action_space(a).sample() for a in env.agents}
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = {a: terminated[a] or truncated[a] for a in env.agents}

    g_info = info["global"]
    rnd_wall_times.append(g_info["wall_time"])
    rnd_prices.append(g_info["price"])

    rnd_step += 1
    obs = next_obs

    if all(done.values()):
        break

print(f"Steps:                {rnd_step}")
print(f"Rejected tasks:       {env.rejected_tasks_count}")
print(f"Completed jobs:       {env.num_completed_jobs}")
print(f"Average energy price: {float(np.mean(rnd_prices)):.4f}")
print(f"Total energy price:   {float(np.sum(rnd_prices)):.4f}")
print(f"End wall time:        {rnd_wall_times[-1] if rnd_wall_times else 0.0:.2f}")

# ── Plot ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(range(step), prices)
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('Energy Cost')
axes[0, 0].set_title('HierMARL — Cost per Step')

axes[0, 1].plot(wall_times, prices)
axes[0, 1].set_xlabel('Wall Time')
axes[0, 1].set_ylabel('Energy Cost')
axes[0, 1].set_title('HierMARL — Cost vs Time')

axes[1, 0].plot(range(rnd_step), rnd_prices)
axes[1, 0].set_xlabel('Step')
axes[1, 0].set_ylabel('Energy Cost')
axes[1, 0].set_title('Random — Cost per Step')

axes[1, 1].plot(rnd_wall_times, rnd_prices)
axes[1, 1].set_xlabel('Wall Time')
axes[1, 1].set_ylabel('Energy Cost')
axes[1, 1].set_title('Random — Cost vs Time')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(args.model), 'eval_plot.png')
plt.savefig(out_path)
print(f"\n图表已保存到 {out_path}")
plt.show()
