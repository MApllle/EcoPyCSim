# IDQN Baseline Implementation Plan

## Context

EcoPyCSim 已有 MADDPG（集中式 Critic）和五种启发式 baseline（random、round-robin 等）。为了证明 MADDPG 的 centralized training 有价值，需要增加 **Independent DQN (IDQN)** 作为对比 baseline：两个 agent 各自独立维护 Q-网络，无共享 Critic，不使用全局观测。

---

## 文件结构

**新建文件**（4个）：
```
schedulers/marl/idqn/__init__.py          # 空文件
schedulers/marl/idqn/DQNAgent.py          # 单 agent DQN（Q网络 + 目标网络 + 学习步）
schedulers/marl/idqn/IDQN.py              # 多 agent 协调器，接口对齐 MADDPG
run_env_train_idqn.py                     # 训练入口，镜像 run_env_train_maddpg.py
```

**修改文件**（1个）：
```
exp1_compare_baselines.py                 # 新增 "idqn" 策略，加载已训练模型评估
```

---

## 关键设计

### DQNAgent（`schedulers/marl/idqn/DQNAgent.py`）

复用 MADDPG 的 `MLPNetwork(obs_dim → act_dim)`，输出每个离散动作的 Q 值。

```
Q-net:     MLPNetwork(obs_dim, act_dim, hidden_dim=64)
Target-net: deepcopy(Q-net)，软更新
Optimizer: Adam(lr=lr)
```

核心方法：
- `select_action(obs_tensor, epsilon)` → 返回 int 动作（epsilon-greedy）
- `learn(batch)` → DQN Bellman 更新：
  ```python
  q = Q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
  with torch.no_grad():
      q_target = reward + gamma * Q_target(next_obs).max(1)[0] * (1 - done)
  loss = F.mse_loss(q, q_target)
  ```
- `soft_update(tau)` → 目标网络软更新

### DQN 专用 Buffer

复用 `schedulers/marl/maddpg/Buffer.py`，以 `act_dim=1` 存储整数动作（float 存储，采样后 `.long()` 转换）。存入时：`buffer.add(flat_obs, np.array([action_int]), reward, flat_next_obs, done)`

### IDQN 协调器（`schedulers/marl/idqn/IDQN.py`）

```python
class IDQN:
    def __init__(self, dim_info, capacity, batch_size, lr, res_dir, device=None)
    def flatten_obs(obs_dict) → np.ndarray     # 同 MADDPG，sorted key 拼接
    def add(obs, action, reward, next_obs, done)
    def select_action(obs, epsilon=0.0) → dict  # 各 agent 独立 argmax
    def learn(batch_size, gamma)                # 各 agent 独立 DQN 更新
    def update_target(tau)
    def save(reward)
    @classmethod load(...)
```

**与 MADDPG 的关键区别**：
- Critic 输入仅为 agent 自己的观测（`obs_dim` 而非 `global_obs_dim`）
- `learn()` 中不交换其他 agent 的观测/动作
- 无 actor/critic 分离，只有 Q-net

### 超参数（`run_env_train_idqn.py`）

与 MADDPG 保持相同规模以公平对比：
```python
num_jobs = 300, num_server_farms = 30, num_servers = 210
episode_num = 10
random_steps = num_jobs * 0.1   # 30步随机探索
learn_interval = 5
capacity = int(1e6)
batch_size = 1024
lr = 0.0005                     # DQN 只有一个网络，只需一个 lr
gamma = 0.9
tau = 0.1
eps_start = 1.0                 # epsilon 从 1.0 线性衰减
eps_end = 0.01
eps_decay_steps = num_jobs * episode_num * 0.5
```

结果保存到 `results/idqn/`（模型 `model.pt`，日志 `idqn.log`）

### exp1_compare_baselines.py 修改

在 `BaselineEvaluator.get_actions()` 中新增分支：
```python
elif strategy == "idqn":
    flat_o = flatten_obs(obs)
    obs_t = torch.from_numpy(flat_o).unsqueeze(0).float().to(device)
    with torch.no_grad():
        q_vals = idqn.agents[agent].q_net(obs_t)
    actions[agent] = q_vals.argmax(dim=1).item()
```

在 `run_experiment()` 中接受可选 `idqn_model` 参数；在 `__main__` 里先加载模型再运行 `"idqn"` 策略。

---

## 关键复用点

| 复用来源 | 路径 | 复用内容 |
|---|---|---|
| MLPNetwork | `schedulers/marl/maddpg/Agent.py:76` | Q网络结构 |
| Buffer | `schedulers/marl/maddpg/Buffer.py` | 回放缓冲区（act_dim=1） |
| flatten_obs | `schedulers/marl/maddpg/MADDPG.py:66` | obs dict → 1D array |
| set_env / dim_info | `run_env_train_maddpg.py:8` | 环境初始化和维度信息构建 |
| 训练循环结构 | `run_env_train_maddpg.py:64` | episode/step 循环框架 |

---

## 验证方式

1. 运行训练：`python run_env_train_idqn.py` → 检查 `results/idqn/idqn.log` 和奖励曲线
2. 对比实验：`python exp1_compare_baselines.py` → 查看 IDQN vs 启发式 vs（可选）MADDPG 的总能耗和 rejection rate
3. 预期结果：IDQN > heuristics（能学到策略），但 < MADDPG（无集中式 Critic，协调不足）
