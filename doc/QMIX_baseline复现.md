# QMIX Implementation Plan for EcoPyCSim

## Context

EcoPyCSim already has IDQN, VDN, MADDPG (off-policy value-based/actor-critic), and MAPPO (on-policy). QMIX (Rashid et al., 2018) fills the gap between VDN's linear decomposition and full centralized methods: it uses a **monotonic mixing network** conditioned on global state, enabling non-linear value factorization while keeping decentralized execution correct.

The 2-agent environment (`server_farm`, `server`) maps naturally to QMIX: global state = concat of both agents' flattened observations, individual Q-networks identical to VDN.

---

## Files to Create (5 total)

| File | Role |
|------|------|
| `schedulers/marl/qmix/__init__.py` | Empty module init |
| `schedulers/marl/qmix/QMIXAgent.py` | Individual Q-network (copy of VDNAgent, same architecture) |
| `schedulers/marl/qmix/QMixer.py` | **Core addition**: monotonic mixing network w/ hypernetworks |
| `schedulers/marl/qmix/QMIX.py` | Coordinator (VDN pattern + state buffer + mixer) |
| `run_env_train_qmix.py` | Training entry point (copy of run_env_train_vdn.py, swap names) |

---

## Algorithm Details

### Individual Q-networks (QMIXAgent)
Identical to `VDNAgent`: `MLPNetwork(obs_dim → 64 → 64 → act_dim)`, no independent optimizer, same `q_values()`, `target_q_values()`, `select_action(epsilon)`, `soft_update(tau)`.

### QMixer (`schedulers/marl/qmix/QMixer.py`)

Monotonic mixing: Q_tot = MixNet([Q_1, Q_2], state)

```
Hypernetworks (conditioned on global state s):
  hyper_w1: Linear(state_dim, embed_dim) → ReLU → Linear(embed_dim, n_agents*embed_dim)
  hyper_w2: Linear(state_dim, embed_dim) → ReLU → Linear(embed_dim, embed_dim)
  hyper_b1: Linear(state_dim, embed_dim)       # unconstrained
  hyper_b2: Linear(state_dim, embed_dim) → ReLU → Linear(embed_dim, 1)  # unconstrained

Forward(q_agents: (B, N), state: (B, state_dim)):
  w1 = abs(hyper_w1(state)).view(B, N, embed_dim)     # non-negative weights
  b1 = hyper_b1(state).view(B, 1, embed_dim)
  hidden = ELU( bmm(q_agents.unsqueeze(1), w1) + b1 ) # (B, 1, embed_dim)
  w2 = abs(hyper_w2(state)).view(B, embed_dim, 1)      # non-negative weights
  b2 = hyper_b2(state).view(B, 1, 1)
  q_tot = (bmm(hidden, w2) + b2).view(B)               # (B,)
```

`embed_dim=32` (default). `abs()` applied to weight **outputs** at forward time (not parameters), so gradients flow through normally.

### QMIX Coordinator (`schedulers/marl/qmix/QMIX.py`)

**Additions over VDN:**

1. **State buffer** (parallel arrays, ring-buffer synchronized with per-agent Buffers):
   ```python
   self.state_buf      = np.zeros((capacity, state_dim), dtype=np.float32)
   self.next_state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
   self._buf_index = 0
   self._buf_size  = 0
   ```
   State = `np.concatenate([flatten_obs(obs[aid]) for aid in sorted(obs)])` — `sorted()` ensures deterministic agent order.

2. **Mixer** (online + target, soft-updated in `update_target()`):
   ```python
   self.mixer        = QMixer(n_agents, state_dim, embed_dim).to(device)
   self.target_mixer = deepcopy(self.mixer)  # frozen, no grad
   ```

3. **Optimizer** covers agent Q-nets + online mixer params:
   ```python
   self.all_params = [p for agent in ... for p in agent.q_net.parameters()] \
                   + list(self.mixer.parameters())
   self.optimizer = Adam(self.all_params, lr=lr)
   ```

**`learn()` differences from VDN:**
- Sample `state`/`next_state` tensors from `state_buf`/`next_state_buf` with same `indices`
- `q_tot = self.mixer(stack(q_agents, dim=1), state)` — replaces `sum(q_vals)`
- `q_tot_next = self.target_mixer(stack(q_next_agents, dim=1), next_state)` — replaces `sum(q_next_vals)`

**`update_target()` additions:**
```python
for src, dst in zip(self.mixer.parameters(), self.target_mixer.parameters()):
    dst.data.copy_(tau * src.data + (1.0 - tau) * dst.data)
```

**`save()` format:**
```python
{'agents': {name: {'q_net': ..., 'target_q_net': ...}},
 'mixer': ..., 'target_mixer': ...}
```
Uses nested `'agents'` key to avoid collision with `'mixer'` key (unlike VDN which uses agent IDs at top level).

### Training Script (`run_env_train_qmix.py`)
Exact copy of `run_env_train_vdn.py` with:
- Import: `from schedulers.marl.qmix.QMIX import QMIX`
- Constructor: `QMIX(..., embed_dim=32)`
- All paths: `results/vdn` → `results/qmix`
- All labels: `"VDN"` → `"QMIX"`, `"vdn"` → `"qmix"`

All hyperparameters kept identical to VDN for fair comparison.

---

## Critical References

- Base VDN pattern: [schedulers/marl/vdn/VDN.py](schedulers/marl/vdn/VDN.py)
- Agent template: [schedulers/marl/vdn/VDNAgent.py](schedulers/marl/vdn/VDNAgent.py)
- Shared Buffer: [schedulers/marl/maddpg/Buffer.py](schedulers/marl/maddpg/Buffer.py)
- Training template: [run_env_train_vdn.py](run_env_train_vdn.py)

---

## Verification

Run training script with `uv run`:
```bash
cd /home/maple/Projects/school_homeworks/EcoPyCSim
uv run python run_env_train_qmix.py
```

Check:
1. No import errors on startup
2. Training loop runs (prints per-episode rewards)
3. `results/qmix/model.pt` saved with keys `agents`, `mixer`, `target_mixer`
4. `results/qmix/qmix.log` contains `joint loss:` entries
5. `results/qmix/QMIX_performance.png` generated at end


# 运行结果
QMIX 已成功实现并通过完整训练验证。以下是所有新增文件的总结：

新增文件
文件	说明
schedulers/marl/qmix/init.py	空模块 init
schedulers/marl/qmix/QMIXAgent.py	单体 Q-network，与 VDNAgent 完全相同
schedulers/marl/qmix/QMixer.py	核心：单调混合网络 + 超网络
schedulers/marl/qmix/QMIX.py	协调器（VDN 模式 + 全局状态缓冲 + 混合网络）
run_env_train_qmix.py	训练入口，超参与 VDN 一致
算法关键点
VDN vs QMIX 的核心差异：

VDN：Q_tot = Q_farm + Q_server（线性求和）
QMIX：Q_tot = MixNet([Q_farm, Q_server], s)，混合网络权重由超网络从全局状态 s 生成
单调性约束（∂Q_tot/∂Q_i ≥ 0）： 超网络权重输出经 abs() 约束为非负，保证分散 argmax 与联合最优等价。

全局状态： s = concat(flat_obs_server_farm, flat_obs_server)，存入独立的环形缓冲区（与 per-agent Buffer 同步）。

优化器： 单个 Adam 同时管理所有 agent Q-net 参数 + 在线混合网络参数，目标混合网络通过软更新维护。

训练输出已保存到 results/qmix/：model.pt、qmix.log、reward.txt、QMIX_performance.png。
