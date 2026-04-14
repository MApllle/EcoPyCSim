"""
QMIX 训练脚本
=============
QMIX（Rashid et al., 2018）：两个 agent 通过单调混合网络共享联合 Q 值训练
（集中训练、分散执行）。

    Q_tot = MixNet([Q_server_farm, Q_server], s)

混合网络的权重由超网络从全局状态 s 条件化生成，并约束为非负（单调性），
保证分散 argmax 的全局最优性。相比 VDN（Q_tot = Σ Q_i），QMIX 能利用
全局状态进行非线性、更丰富的值分解。

超参数与 run_env_train_vdn.py 保持完全一致，方便公平对比实验。

用法：
    python run_env_train_qmix.py
    # 或使用 uv：
    uv run python run_env_train_qmix.py
"""

import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from env import cloud_scheduling_v0
from schedulers.marl.qmix.QMIX import QMIX


# ── 环境 & 维度信息 ──────────────────────────────────────────────────────────

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


# ── 超参数（与 VDN / IDQN 脚本完全一致，保证公平对比） ─────────────────────────

num_jobs         = 300
num_server_farms = 30
num_servers      = 210

episode_num      = 1000
random_steps     = int(num_jobs * 0.1)   # 前 30 步纯随机探索
learn_interval   = 5
capacity         = int(1e6)
batch_size       = 1024
lr               = 0.0005
gamma            = 0.9
tau              = 0.1
embed_dim        = 32                    # 混合网络隐层维度（QMIX 新增超参）

# Epsilon-greedy 衰减：在前半段 episodes 的所有步内线性衰减
eps_start        = 1.0
eps_end          = 0.01
eps_decay_steps  = num_jobs * episode_num * 0.5   # 1500 步后达到最小值

# ── 结果目录 ─────────────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
res_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results',
    f'qmix_{timestamp}',
)
os.makedirs(res_dir, exist_ok=True)
reward_file_path = os.path.join(res_dir, 'reward.txt')
print(f"本次实验输出目录: {res_dir}")

# ── 初始化 ───────────────────────────────────────────────────────────────────

env, dim_info = set_env(num_jobs, num_server_farms, num_servers)

qmix = QMIX(
    dim_info   = dim_info,
    capacity   = capacity,
    batch_size = batch_size,
    lr         = lr,
    res_dir    = res_dir,
    embed_dim  = embed_dim,
)

episode_rewards = {agent_id: np.zeros(episode_num) for agent_id in env.agents}
global_step = 0   # 用于 epsilon 衰减计算


# ── 训练循环 ─────────────────────────────────────────────────────────────────

for episode in range(episode_num):
    obs, info = env.reset()
    agent_reward = {agent_id: 0.0 for agent_id in env.agents}
    step = 0

    while env.agents:
        step        += 1
        global_step += 1

        # 计算当前 epsilon
        if global_step <= random_steps:
            epsilon = 1.0
        else:
            decay_progress = min(1.0, (global_step - random_steps) / eps_decay_steps)
            epsilon = eps_start - (eps_start - eps_end) * decay_progress

        action = qmix.select_action(obs, epsilon=epsilon)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = {
            agent_id: terminated[agent_id] or truncated[agent_id]
            for agent_id in env.agents
        }

        qmix.add(obs, action, reward, next_obs, done)

        for agent_id, r in reward.items():
            agent_reward[agent_id] += r

        obs = next_obs

        if global_step > random_steps and global_step % learn_interval == 0:
            qmix.learn(batch_size, gamma)
            qmix.update_target(tau)

        if all(done.values()):
            break

    # 记录本 episode 奖励
    for agent_id, r in agent_reward.items():
        episode_rewards[agent_id][episode] = r

    sum_reward = sum(agent_reward.values())
    avg_reward = sum_reward / max(step, 1)

    with open(reward_file_path, 'a') as f:
        f.write(
            f"episode={episode + 1}, "
            f"steps={step}, "
            f"epsilon={epsilon:.4f}, "
            f"server_farm_reward={agent_reward['server_farm']:.4f}, "
            f"server_reward={agent_reward['server']:.4f}, "
            f"episode_total_reward={sum_reward:.4f}, "
            f"avg_reward_per_step={avg_reward:.4f}\n"
        )

    print(
        f"[QMIX] episode {episode + 1:3d}/{episode_num}  "
        f"eps={epsilon:.3f}  "
        f"server_farm={agent_reward['server_farm']:8.4f}  "
        f"server={agent_reward['server']:8.4f}  "
        f"sum={sum_reward:8.4f}  "
        f"avg_step={avg_reward:8.4f}"
    )

    qmix.save(episode_rewards)

    # 每 100 轮保存一次检查点
    if (episode + 1) % 10 == 0:
        ckpt_dir = os.path.join(res_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        shutil.copy(
            os.path.join(res_dir, 'model.pt'),
            os.path.join(ckpt_dir, f'model_ep{episode + 1}.pt'),
        )
        print(f"  [checkpoint] ep{episode + 1} 已保存到 {ckpt_dir}/model_ep{episode + 1}.pt")

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
ax.set_title('QMIX Training Performance')
fig.savefig(os.path.join(res_dir, 'QMIX_performance.png'))
print(f"学习曲线已保存到 {res_dir}/QMIX_performance.png")
