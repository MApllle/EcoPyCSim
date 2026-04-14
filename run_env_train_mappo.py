"""
MAPPO 原生训练脚本
==================
Multi-Agent Proximal Policy Optimization（MAPPO）：
  集中式 Critic + 分散式 Actor，on-policy PPO 更新。

与 off-policy 算法（QMIX/VDN）的核心区别：
  - 每个 episode 收集完整轨迹（T 步），再统一计算 GAE 回报；
  - PPO 在同一批数据上进行 ppo_epoch 轮更新（无经验回放）；
  - 学习率与 epsilon 等概念不同，无 random_steps 预热。

超参数设计尽量与其他脚本对齐，便于公平对比。

用法：
    python run_env_train_mappo.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from env import cloud_scheduling_v0
from schedulers.marl.mappo.MAPPO import MAPPO


# ── 环境与维度信息 ────────────────────────────────────────────────────────────

def set_env(num_jobs, num_server_farms, num_servers):
    env = cloud_scheduling_v0.CloudSchedulingEnv(
        num_jobs, num_server_farms, num_servers
    )
    env.reset()

    dim_info = {}
    for agent_id in env.agents:
        obs_space = env.observation_space(agent_id)
        dim_info[agent_id] = {
            'obs_shape': {key: space.shape for key, space in obs_space.spaces.items()},
            'action_dim': env.action_space(agent_id).n,
        }
    return env, dim_info


# ── 超参数 ────────────────────────────────────────────────────────────────────

num_jobs         = 300
num_server_farms = 30
num_servers      = 210

episode_num      = 10

# MAPPO 专属超参
episode_length   = num_jobs      # 每 episode 恰好处理完所有 job
num_mini_batch   = 4             # PPO 小批量数
ppo_epoch        = 10            # 每次 rollout 后 PPO 更新轮数
lr               = 5e-4
gamma            = 0.99
gae_lambda       = 0.95
clip_param       = 0.2
entropy_coef     = 0.01
value_loss_coef  = 1.0
max_grad_norm    = 10.0
hidden_size      = 64
use_valuenorm    = True

# ── 结果目录 ──────────────────────────────────────────────────────────────────

res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'mappo')
os.makedirs(res_dir, exist_ok=True)
reward_file_path = os.path.join(res_dir, 'reward.txt')
open(reward_file_path, 'w').close()

# ── 初始化 ────────────────────────────────────────────────────────────────────

env, dim_info = set_env(num_jobs, num_server_farms, num_servers)

mappo = MAPPO(
    dim_info       = dim_info,
    episode_length = episode_length,
    num_mini_batch = num_mini_batch,
    lr             = lr,
    res_dir        = res_dir,
    hidden_size    = hidden_size,
    ppo_epoch      = ppo_epoch,
    clip_param     = clip_param,
    value_loss_coef= value_loss_coef,
    entropy_coef   = entropy_coef,
    max_grad_norm  = max_grad_norm,
    use_valuenorm  = use_valuenorm,
    gamma          = gamma,
    gae_lambda     = gae_lambda,
)

episode_rewards = {agent_id: np.zeros(episode_num) for agent_id in env.agents}


# ── 训练循环（On-Policy） ──────────────────────────────────────────────────────

for episode in range(episode_num):
    obs, info = env.reset()
    agent_reward = {agent_id: 0.0 for agent_id in env.agents}

    # 设置 buffer 初始帧（step=0 时的 obs）
    mappo.set_initial_obs(obs)

    last_obs = obs
    last_dones = {aid: False for aid in env.agents}

    for step in range(episode_length):
        # 1. 采样动作、log prob、Critic 值
        actions, log_probs, values, flat_obs, cent_obs = mappo.collect(obs)

        # 2. 环境交互
        if not env.agents:
            break
        next_obs, reward, terminated, truncated, info = env.step(actions)

        done = {
            agent_id: terminated.get(agent_id, False) or truncated.get(agent_id, False)
            for agent_id in mappo.agent_ids
        }

        # 3. 写入 buffer
        # 此处传入的 flat_obs / cent_obs 是当前步的观察（即 step 对应的初始状态）
        # buffer 内部将其存入 step+1 位（next obs），由 set_initial_obs 负责 step=0
        next_flat = {aid: mappo.flatten_obs(next_obs[aid])
                     for aid in mappo.agent_ids if aid in next_obs}
        # 如 agent 已离线（not in next_obs），复用上一步的 obs
        for aid in mappo.agent_ids:
            if aid not in next_flat:
                next_flat[aid] = flat_obs[aid]

        next_cent = mappo._cent_obs(next_flat)

        mappo.insert(step, next_flat, next_cent,
                     actions, log_probs, values, reward, done)

        for agent_id, r in reward.items():
            agent_reward[agent_id] += r

        obs = next_obs
        last_obs = next_obs
        last_dones = done

        if all(done.values()):
            break

    # 4. 计算 GAE 回报
    mappo.compute_returns(last_obs, last_dones)

    # 5. PPO 更新
    train_info = mappo.learn()

    # 6. 重置 buffer（复制末帧到初始帧）
    mappo.buffer.after_update()

    # ── 记录 ─────────────────────────────────────────────────────────────────
    for agent_id, r in agent_reward.items():
        episode_rewards[agent_id][episode] = r

    sum_reward = sum(agent_reward.values())
    avg_reward = sum_reward / max(step + 1, 1)

    with open(reward_file_path, 'a') as f:
        f.write(
            f"episode={episode + 1}, "
            f"steps={step + 1}, "
            + ", ".join(f"{aid}_reward={agent_reward[aid]:.4f}"
                        for aid in mappo.agent_ids)
            + f", episode_total_reward={sum_reward:.4f}"
            + f", avg_reward_per_step={avg_reward:.4f}"
            + f", policy_loss={train_info['policy_loss']:.4f}"
            + f", value_loss={train_info['value_loss']:.4f}\n"
        )
    print(
        f"[MAPPO] episode {episode + 1:3d}/{episode_num}  "
        + "  ".join(f"{aid}={agent_reward[aid]:8.4f}" for aid in mappo.agent_ids)
        + f"  sum={sum_reward:8.4f}"
        + f"  π_loss={train_info['policy_loss']:.4f}"
        + f"  v_loss={train_info['value_loss']:.4f}"
    )

    mappo.save(episode_rewards)

print(f"\n训练完成，模型已保存到 {res_dir}/model.pt")


# ── 学习曲线绘图 ──────────────────────────────────────────────────────────────

def get_running_reward(arr: np.ndarray, window: int = 5) -> np.ndarray:
    running = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        running[i] = np.mean(arr[start:i + 1])
    return running


fig, ax = plt.subplots()
x = range(1, episode_num + 1)
for agent_id, rewards in episode_rewards.items():
    ax.plot(x, rewards, label=f'{agent_id}')
    ax.plot(x, get_running_reward(rewards), linestyle='--',
            label=f'{agent_id} (running avg)')
ax.legend()
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('MAPPO Training Performance')
fig.savefig(os.path.join(res_dir, 'MAPPO_performance.png'))
print(f"学习曲线已保存到 {res_dir}/MAPPO_performance.png")
