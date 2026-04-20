import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from env.cloud_scheduling_hier import CloudSchedulingEnvHier
from schedulers.marl.common_actor.CommonActor import CommonActor


def build_dim_info(env):
    dim_info = {}
    for agent_id in env.agents:
        obs_space = env.observation_space(agent_id)
        obs_shape = {key: space.shape for key, space in obs_space.spaces.items()}
        action_dim = env.action_space(agent_id).n
        dim_info[agent_id] = {'obs_shape': obs_shape, 'action_dim': action_dim}
    return dim_info


def get_running_reward(arr, window=100):
    running = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        running[i] = np.mean(arr[start:i + 1])
    return running


num_jobs         = int(os.getenv("NUM_JOBS",   "50"))
num_server_farms = int(os.getenv("NUM_FARMS",  "5"))
num_servers      = int(os.getenv("NUM_SERVERS","30"))  # must be divisible by num_server_farms

episode_num      = int(os.getenv("EPISODES", "1000"))
random_steps     = max(int(num_jobs * 2), int(num_jobs * episode_num * 0.1))
learn_interval   = 5
capacity         = 50_000
batch_size       = 256
actor_lr         = 3e-4
critic_lr        = 3e-4
gamma            = 0.9
tau              = 0.02
eps_start        = 1.0
eps_end          = 0.05
eps_decay_steps  = max(int(num_jobs * episode_num * 0.9), 1)

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
env_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results',
    f'common_actor_{timestamp}'
)
os.makedirs(env_dir, exist_ok=True)
print(f"本次实验输出目录: {env_dir}")

env = CloudSchedulingEnvHier(
    num_jobs=num_jobs,
    num_server_farms=num_server_farms,
    num_servers=num_servers,
)
env.reset()
dim_info = build_dim_info(env)

agent = CommonActor(
    dim_info,
    capacity=capacity,
    batch_size=batch_size,
    actor_lr=actor_lr,
    critic_lr=critic_lr,
    res_dir=env_dir,
)

reward_file_path = os.path.join(env_dir, 'reward.txt')

global_step = 0
episode_global_rewards = np.zeros(episode_num)
episode_local_rewards  = np.zeros((episode_num, num_server_farms))

for episode in range(episode_num):
    obs, info = env.reset()
    agent_reward = {aid: 0.0 for aid in env.agents}
    step = 0

    while True:
        step += 1
        global_step += 1

        if global_step <= random_steps:
            epsilon = 1.0
        else:
            decay_progress = min(1.0, (global_step - random_steps) / eps_decay_steps)
            epsilon = eps_start - (eps_start - eps_end) * decay_progress

        if np.random.rand() < epsilon:
            action = {aid: env.action_space(aid).sample() for aid in env.agents}
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = {a: terminated[a] or truncated[a] for a in env.agents}

        agent.add(obs, action, reward, next_obs, done)

        for a, r in reward.items():
            agent_reward[a] += r

        obs = next_obs

        if global_step > random_steps and global_step % learn_interval == 0:
            agent.learn(gamma)
            agent.update_target(tau)

        if all(done.values()):
            break

    global_r = agent_reward["global"]
    local_rs  = [agent_reward[f"local_{i}"] for i in range(num_server_farms)]
    sum_r     = global_r + sum(local_rs)
    avg_r     = sum_r / max(step, 1)

    g_info         = info.get("global", {})
    rejected_tasks = g_info.get("rejected_tasks_count", 0)
    completed_jobs = len(g_info.get("completed_job_ids", []))
    wall_time      = g_info.get("wall_time", 0)

    episode_global_rewards[episode] = global_r
    for i, lr in enumerate(local_rs):
        episode_local_rewards[episode, i] = lr

    local_str = " ".join(f"local_{i}={lr:.4f}" for i, lr in enumerate(local_rs))
    with open(reward_file_path, "a") as f:
        f.write(
            f"episode={episode + 1}, steps={step}, epsilon={epsilon:.4f}, "
            f"global_reward={global_r:.4f}, {local_str}, "
            f"sum_reward={sum_r:.4f}, avg_reward_per_step={avg_r:.4f}, "
            f"rejected_tasks={rejected_tasks}, completed_jobs={completed_jobs}, "
            f"wall_time={wall_time}\n"
        )

    print(
        f"[CommonActor] ep {episode + 1:3d}/{episode_num}  "
        f"eps={epsilon:.3f}  "
        f"global={global_r:8.4f}  "
        f"locals_avg={np.mean(local_rs):8.4f}  "
        f"sum={sum_r:8.4f}  "
        f"avg_step={avg_r:.4f}  "
        f"reject={rejected_tasks:4d}  "
        f"done_jobs={completed_jobs:4d}"
    )

    agent.save(episode_global_rewards)

    if (episode + 1) % 100 == 0:
        ckpt_dir = os.path.join(env_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        shutil.copy(
            os.path.join(env_dir, 'model.pt'),
            os.path.join(ckpt_dir, f'model_ep{episode + 1}.pt'),
        )
        print(f"  [checkpoint] ep{episode + 1} 已保存到 {ckpt_dir}/model_ep{episode + 1}.pt")


fig, axes = plt.subplots(2, 1, figsize=(10, 8))
x = range(1, episode_num + 1)

axes[0].plot(x, episode_global_rewards, label='global reward')
axes[0].plot(x, get_running_reward(episode_global_rewards), '--', label='running avg')
axes[0].set_ylabel('Reward')
axes[0].set_title('CommonActor — Global Agent')
axes[0].legend()

avg_local = episode_local_rewards.mean(axis=1)
axes[1].plot(x, avg_local, label='avg local reward')
axes[1].plot(x, get_running_reward(avg_local), '--', label='running avg')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Reward')
axes[1].set_title('CommonActor — Local Agents (average)')
axes[1].legend()

plt.tight_layout()
title = 'CommonActor performance'
plt.savefig(os.path.join(env_dir, title))
