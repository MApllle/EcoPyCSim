# VDN 实现计划：适配 EcoPyCSim 框架

## Context

EcoPyCSim 已实现 MAPPO（on-policy，centralized critic）、IDQN（独立 Q-learning）、MADDPG（集中式 Critic + 连续动作）。用户希望增加 **VDN（Value Decomposition Networks）**，这是一种基于 Q-learning 的 CTDE（集中训练、分散执行）算法。

**VDN 与 IDQN 的核心区别**：IDQN 各 agent 完全独立优化各自奖励；VDN 使用全局联合奖励联合训练，损失函数基于 `Q_tot = Σ Q_i`，梯度通过求和操作流回所有 Q-network。

---

## 算法设计

VDN 数学公式：

```
Q_tot(s, a) = Q_sf(o_sf, a_sf) + Q_s(o_s, a_s)
r_tot = r_sf + r_s
Q_tot_target = r_tot + γ · [max_{a'_sf} Q_sf_target(o_sf') + max_{a'_s} Q_s_target(o_s')]
loss = MSE(Q_tot, Q_tot_target)
```

执行阶段：每个 agent 仍独立 `argmax Q_i`（数学等价于 `argmax Q_tot`）。

---

## 关键文件

### 参考文件（只读）
- `schedulers/marl/idqn/DQNAgent.py` — VDNAgent 的基础
- `schedulers/marl/idqn/IDQN.py` — VDN 协调器结构参考
- `schedulers/marl/maddpg/Buffer.py` — 复用现有 Buffer
- `run_env_train_idqn.py` — 训练脚本结构参考

### 新建文件
1. `schedulers/marl/vdn/__init__.py`（空）
2. `schedulers/marl/vdn/VDNAgent.py`
3. `schedulers/marl/vdn/VDN.py`
4. `run_env_train_vdn.py`

---

## 实现步骤

### Step 1：`schedulers/marl/vdn/__init__.py`
空文件，与 IDQN 对齐。

### Step 2：`schedulers/marl/vdn/VDNAgent.py`

从 `DQNAgent.py` 改造：
- **保留**：`MLPNetwork`（2 层 MLP, Xavier init）、`target_q_net`、`select_action`（epsilon-greedy）、`soft_update`
- **删除**：`self.optimizer`（无独立优化器）、`learn()` 方法（提升至协调器）
- **新增**：`q_values(obs)` → `(B, act_dim)` 和 `target_q_values(obs)` → `(B, act_dim)` 两个纯前向方法

```python
class VDNAgent:
    def __init__(self, obs_dim, act_dim, device):
        self.q_net = MLPNetwork(obs_dim, act_dim).to(device)
        self.target_q_net = deepcopy(self.q_net)
        for p in self.target_q_net.parameters():
            p.requires_grad = False
    
    def select_action(self, obs_tensor, epsilon=0.0) -> int: ...
    def q_values(self, obs) -> Tensor: ...        # 用于训练时计算 Q_tot
    def target_q_values(self, obs) -> Tensor: ... # 用于计算 TD target
    def soft_update(self, tau): ...
```

### Step 3：`schedulers/marl/vdn/VDN.py`

**初始化**：
```python
self.agents = {id: VDNAgent(obs_dim, act_dim, device) for id, ... in dim_info}
self.buffers = {id: Buffer(capacity, obs_dim, act_dim=1, device) for id in ...}
# 联合优化器：收集所有 Q-net 的参数
all_params = [p for a in self.agents.values() for p in a.q_net.parameters()]
self.optimizer = Adam(all_params, lr=lr)
self.all_params = all_params  # 保存引用供梯度裁剪
```

**`add()`、`select_action()`、`update_target()`、`flatten_obs()`**：与 IDQN 完全相同。

**`learn(batch_size, gamma)` 关键逻辑**：
```python
def learn(self, batch_size, gamma):
    # 1. 检查数据量
    if any(len(buf) < batch_size for buf in self.buffers.values()):
        return
    
    # 2. 同一批索引同步采样
    total = min(len(buf) for buf in self.buffers.values())
    indices = np.random.choice(total, size=batch_size, replace=False)
    samples = {id: buf.sample(indices) for id, buf in self.buffers.items()}
    
    # 3. 计算当前 Q_tot
    q_vals = []
    for agent_id, (obs, actions, _, _, _) in samples.items():
        q_i = self.agents[agent_id].q_values(obs).gather(1, actions.long()).squeeze(1)
        q_vals.append(q_i)
    q_tot = sum(q_vals)  # (B,)
    
    # 4. 计算 TD target（no_grad）
    with torch.no_grad():
        q_next_vals = []
        for agent_id, (_, _, _, next_obs, _) in samples.items():
            q_next_i = self.agents[agent_id].target_q_values(next_obs).max(dim=1)[0]
            q_next_vals.append(q_next_i)
        q_tot_next = sum(q_next_vals)  # (B,)
        
        # 联合奖励（各 agent Buffer 已各自归一化，直接相加作近似）
        r_tot = sum(r for _, _, r, _, _ in samples.values())  # (B,)
        dones = list(samples.values())[0][4]  # 取任一 agent 的 done 标志
        q_tot_target = r_tot + gamma * q_tot_next * (1.0 - dones)
    
    # 5. 联合反向传播
    loss = F.mse_loss(q_tot, q_tot_target)
    self.optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(self.all_params, 0.5)
    self.optimizer.step()
    self.logger.info(f'joint loss: {loss.item():.6f}')
```

**`save()` / `load()`**：同时保存 `q_net` 和 `target_q_net`（确保断点续训稳定性）：
```python
def save(self, reward=None):
    torch.save(
        {name: {'q_net': a.q_net.state_dict(), 'target_q_net': a.target_q_net.state_dict()}
         for name, a in self.agents.items()},
        os.path.join(self.res_dir, 'model.pt')
    )
```

### Step 4：`run_env_train_vdn.py`

镜像 `run_env_train_idqn.py`，改动：
- 导入 `from schedulers.marl.vdn.VDN import VDN`
- 实例化 `vdn = VDN(...)`
- 结果目录改为 `results/vdn`
- 日志/图表标题改为 `VDN`
- 超参数与 IDQN 完全一致（保证可对比）

---

## 不同动作空间的处理

`server_farm`（30 个动作）和 `server`（7 个动作/集群）维度不同，但 VDN 中：
- `gather()` 后每个 agent 输出 `(B, 1)`，相加天然对齐
- 无需任何特殊处理

---

## 验证方法

1. **运行训练脚本**：`python run_env_train_vdn.py`，观察 loss 下降、奖励趋势
2. **对比 IDQN**：对比 `results/vdn/reward.txt` 与 `results/idqn/reward.txt` 中的 episode 奖励
3. **梯度验证**（可选）：在 `learn()` 中打印各 Q-net 参数的梯度范数，确认两个网络都在被更新
4. **模型加载测试**：运行 `VDN.load()` 后执行一次 `select_action()`，确认不报错


# 运行结果

实现完成
新增了 4 个文件，实现了 VDN（Value Decomposition Networks）算法：

新文件
文件	说明
schedulers/marl/vdn/__init__.py	包初始化（空文件）
schedulers/marl/vdn/VDNAgent.py	单体 Q-network，无独立 optimizer
schedulers/marl/vdn/VDN.py	VDN 协调器，实现联合训练
run_env_train_vdn.py	训练脚本，超参数与 IDQN 一致
VDN vs IDQN 的核心区别
IDQN：每个 agent 各自用自己的奖励独立计算 TD loss，梯度完全隔离。

VDN：Q_tot = Q_sf + Q_server，使用联合奖励 r_tot = r_sf + r_server 计算统一的 TD loss，一次 backward() 让梯度通过求和操作同时流回两个 Q-network，实现协同学习。

运行

uv run python run_env_train_vdn.py
结果保存在 results/vdn/，可与 results/idqn/ 直接对比。