import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from schedulers.marl.maddpg.MADDPG import MADDPG
from schedulers.marl.maddpg.LagrangianMADDPG import LagrangianMADDPG

from env import cloud_scheduling_v0

# ── Feature toggles (set via environment variables) ──────────────────────────
# A1: USE_LAGRANGIAN=1  →  reward = -energy, constraint = rejection rate ≤ budget
# A2: USE_SHAPING=1     →  add γΦ(s')−Φ(s) potential shaping to env reward
# A3: USE_ATTENTION=1   →  replace MLP critic with cross-attention critic
# USE_LAGRANGIAN = os.getenv("USE_LAGRANGIAN", "0") == "1"
# USE_SHAPING    = os.getenv("USE_SHAPING",    "0") == "1"
# USE_ATTENTION  = os.getenv("USE_ATTENTION",  "0") == "1"

USE_LAGRANGIAN = False
USE_SHAPING    = False
USE_ATTENTION  = False

# Lagrangian hyper-params (only used when USE_LAGRANGIAN=1)
CONSTRAINT_BUDGET = float(os.getenv("CONSTRAINT_BUDGET", "0.3"))
LAMBDA_LR         = float(os.getenv("LAMBDA_LR",         "0.005"))
LAMBDA_INIT       = float(os.getenv("LAMBDA_INIT",       "1.0"))


def set_env(num_jobs, num_server_farms, num_servers, use_shaping, shaping_gamma):
  env = cloud_scheduling_v0.CloudSchedulingEnv(
    num_jobs, num_server_farms, num_servers,
    use_potential_shaping=use_shaping,
    shaping_gamma=shaping_gamma,
  )

  env.reset()

  _dim_info = {}
  for agent_id in env.agents:
    obs_space  = env.observation_space(agent_id)
    obs_shape  = {key: space.shape for key, space in obs_space.spaces.items()}
    action_dim = env.action_space(agent_id).n
    _dim_info[agent_id] = {'obs_shape': obs_shape, 'action_dim': action_dim}

  return env, _dim_info


num_jobs         = 20
num_server_farms = 4
num_servers      = 2

episode_num      = int(os.getenv("EPISODES", "1000"))
random_steps     = max(int(num_jobs * 2), int(num_jobs * episode_num * 0.1))
learn_iterval    = 5
capacity         = 50_000
batch_size       = 256
actor_lr         = 0.0003
critic_lr        = 0.0003
gamma            = 0.9
tau              = 0.02
eps_start        = 1.0
eps_end          = 0.05
eps_decay_steps  = max(int(num_jobs * episode_num * 0.9), 1)

# Variant tag for result directory
variant_tag = "_".join(filter(None, [
  "lagrangian" if USE_LAGRANGIAN else "",
  "shaping"    if USE_SHAPING    else "",
  "attention"  if USE_ATTENTION  else "",
])) or "baseline"

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
env_dir = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  'results',
  f'maddpg_{variant_tag}_{timestamp}',
)
os.makedirs(env_dir, exist_ok=True)
script_snapshot_path = os.path.join(env_dir, os.path.basename(__file__))
shutil.copy2(os.path.abspath(__file__), script_snapshot_path)
print(f"本次实验输出目录: {env_dir}")
print(f"变体: {variant_tag}  "
      f"(lagrangian={USE_LAGRANGIAN}, shaping={USE_SHAPING}, attention={USE_ATTENTION})")
print(f"已保存训练脚本快照: {script_snapshot_path}")

reward_file_path = os.path.join(env_dir, 'reward.txt')
env, dim_info = set_env(
  num_jobs=num_jobs,
  num_server_farms=num_server_farms,
  num_servers=num_servers,
  use_shaping=USE_SHAPING,
  shaping_gamma=gamma,
)

total_servers = num_server_farms * num_servers  # used by attention critic

if USE_LAGRANGIAN:
  maddpg = LagrangianMADDPG(
    dim_info,
    capacity=capacity,
    batch_size=batch_size,
    actor_lr=actor_lr,
    critic_lr=critic_lr,
    res_dir=env_dir,
    constraint_budget=CONSTRAINT_BUDGET,
    lambda_lr=LAMBDA_LR,
    lambda_init=LAMBDA_INIT,
    use_attention_critic=USE_ATTENTION,
    num_servers=total_servers if USE_ATTENTION else None,
  )
else:
  maddpg = MADDPG(
    dim_info,
    capacity=capacity,
    batch_size=batch_size,
    actor_lr=actor_lr,
    critic_lr=critic_lr,
    res_dir=env_dir,
    use_attention_critic=USE_ATTENTION,
    num_servers=total_servers if USE_ATTENTION else None,
  )

agent_num = env.num_agents
episode_rewards = {agent_id: np.zeros(episode_num) for agent_id in env.agents}
global_step = 0

for episode in range(episode_num):
  obs, info = env.reset()
  agent_reward = {agent_id: 0 for agent_id in env.agents}

  step = 0
  while env.agents:
    step += 1
    global_step += 1
    if global_step <= random_steps:
      epsilon = 1.0
    else:
      decay_progress = min(1.0, (global_step - random_steps) / eps_decay_steps)
      epsilon = eps_start - (eps_start - eps_end) * decay_progress

    if np.random.rand() < epsilon:
      action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
    else:
      action = maddpg.select_action(obs)
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env.agents}
    
    if USE_LAGRANGIAN:
      sf_info      = info.get("server_farm", {}) if isinstance(info, dict) else {}
      is_sched     = sf_info.get("is_scheduling_step", False)
      energy_rwd   = sf_info.get("energy_reward",   0.0) if is_sched else 0.0
      c_cost       = sf_info.get("constraint_cost", 0.0) if is_sched else 0.0
      energy_rwds  = {aid: energy_rwd for aid in env.agents}
      maddpg.add(obs, action, energy_rwds, c_cost, next_obs, done)
    else:
      maddpg.add(obs, action, reward, next_obs, done)

    for agent_id, r in reward.items():
      agent_reward[agent_id] += r

    obs = next_obs

    if global_step > random_steps and global_step % learn_iterval == 0:
      maddpg.learn(batch_size, gamma)
      maddpg.update_target(tau)
    
    # Check if all agents are done
    if all(done.values()):
      break

  with open(reward_file_path, "a") as file:
    for agent_id, r in agent_reward.items():
      episode_rewards[agent_id][episode] = r
    server_farm_reward = agent_reward['server_farm']
    server_reward = agent_reward['server']
    sum_reward = sum(agent_reward.values())
    avg_reward = sum_reward / max(step, 1)
    sf_info = info.get("server_farm", {}) if isinstance(info, dict) else {}
    rejected_tasks = sf_info.get("rejected_tasks_count", 0)
    completed_jobs = len(sf_info.get("completed_job_ids", []))
    wall_time = sf_info.get("wall_time", 0)
    lambda_val = getattr(maddpg, 'lambda_param', None)
    lambda_str = f", lambda={lambda_val:.4f}" if lambda_val is not None else ""
    file.write(
      f"episode={episode + 1}, "
      f"steps={step}, "
      f"epsilon={epsilon:.4f}, "
      f"server_farm_reward={server_farm_reward:.4f}, "
      f"server_reward={server_reward:.4f}, "
      f"episode_total_reward={sum_reward:.4f}, "
      f"avg_reward_per_step={avg_reward:.4f}, "
      f"rejected_tasks={rejected_tasks}, "
      f"completed_jobs={completed_jobs}, "
      f"wall_time={wall_time}"
      f"{lambda_str}\n"
    )

  lambda_display = f"  λ={maddpg.lambda_param:.3f}" if USE_LAGRANGIAN else ""
  print(
    f"[MADDPG] episode {episode + 1:3d}/{episode_num}  "
    f"eps={epsilon:.3f}  "
    f"server_farm={server_farm_reward:8.4f}  "
    f"server={server_reward:8.4f}  "
    f"sum={sum_reward:8.4f}  "
    f"avg_step={avg_reward:8.4f}  "
    f"reject={rejected_tasks:4d}  "
    f"done_jobs={completed_jobs:4d}"
    f"{lambda_display}"
  )

  maddpg.save(episode_rewards)

  # 每 100 轮保存一次检查点
  if (episode + 1) % 100 == 0:
    ckpt_dir = os.path.join(env_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy(
      os.path.join(env_dir, 'model.pt'),
      os.path.join(ckpt_dir, f'model_ep{episode + 1}.pt'),
    )
    print(f"  [checkpoint] ep{episode + 1} 已保存到 {ckpt_dir}/model_ep{episode + 1}.pt")

def get_running_reward(arr: np.ndarray, window=100):
  """calculate the running reward, i.e., average of last 'window' elements from rewards"""
  running_reward = np.zeros_like(arr)
  for i in range(len(arr)):
    start_index = max(0, i - window + 1)
    running_reward[i] = np.mean(arr[start_index:i + 1])
  return running_reward

fig, ax = plt.subplots()
x = range(1, episode_num + 1)
for agent_id, reward in episode_rewards.items():
  ax.plot(x, reward, label=f'{agent_id} reward')
  ax.plot(x, get_running_reward(reward), linestyle='--', label=f'{agent_id} running reward')
ax.legend()
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
title = 'MADDPG performance'
ax.set_title(title)
plt.savefig(os.path.join(env_dir, title))
