# Plan: Port MAPPO to Native EcoPyCSim Implementation

## Context

The current MAPPO implementation relies on the external `on-policy-main` subrepository (from the open-source `on-policy` framework), which has a complex infrastructure: argparse-based config, VecEnv wrappers, multi-thread runners, and gym.Space-dependent network construction. This is structurally inconsistent with the native algorithms (IDQN, QMIX, VDN, MADDPG) in `schedulers/marl/`, which follow a clean, simple interface using plain constructors and direct PettingZoo env interaction.

Goal: re-implement MAPPO natively so the `on-policy-main/` dependency can be removed, and MAPPO fits the existing project patterns.

---

## Key Differences: On-Policy vs Off-Policy Interface

**Off-policy algorithms** (QMIX/VDN/MADDPG pattern — `run_env_train_vdn.py`):
```
for step: select_action -> env.step -> buffer.add -> learn (if enough data)
```

**On-policy MAPPO** requires a different loop:
```
for episode:
    for step in episode_length:
        collect_action -> env.step -> buffer.insert
    compute_returns (GAE)
    for ppo_epoch: mini_batch PPO update
    buffer.after_update (reset for next rollout)
```
This means `run_env_train_mappo.py` will have a slightly different structure than the VDN/QMIX scripts, but the MAPPO class itself will have a similar constructor/save/load interface.

---

## Files to Create

### 1. `schedulers/marl/mappo/utils.py` (~80 lines)
Ported utility functions from `on-policy-main/onpolicy/utils/`:
- `huber_loss(e, d)`, `mse_loss(e)` — from `utils/util.py`
- `get_gard_norm(params)` — from `utils/util.py`
- `ValueNorm` class — from `utils/valuenorm.py`
- `check(x)` helper (numpy→tensor) — from `algorithms/utils/util.py`

### 2. `schedulers/marl/mappo/networks.py` (~180 lines)
Simplified Actor/Critic networks ported from `on-policy-main/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py`:
- Remove gym.Space / `get_shape_from_obs_space` dependency → accept flat `obs_dim: int` directly
- Remove CNN branch (environment uses vector observations)
- Keep: MLPBase inline (2-layer MLP), optional GRU, ACTLayer for discrete actions
- `Actor(obs_dim, act_dim, hidden_size, use_rnn, device)`
- `Critic(cent_obs_dim, hidden_size, use_rnn, device)`

### 3. `schedulers/marl/mappo/buffer.py` (~200 lines)
Simplified trajectory buffer ported from `on-policy-main/onpolicy/utils/shared_buffer.py`:
- Remove multi-thread (`n_rollout_threads=1` assumed) and gym.Space dependencies
- Accept flat `obs_dim`, `cent_obs_dim`, `act_dim` integers
- Keep: GAE / n-step return computation (`compute_returns`)
- Keep: `feed_forward_generator` for mini-batch sampling
- `RolloutBuffer(num_agents, obs_dim, cent_obs_dim, act_dim, episode_length, hidden_size, gamma, gae_lambda, ...)`

### 4. `schedulers/marl/mappo/MAPPO.py` (~280 lines)
Main coordinator class — interface analogous to MADDPG:
- Constructor: `MAPPO(dim_info, episode_length, batch_size, lr, res_dir, hidden_size, ppo_epoch, clip_param, ...)`
  - `dim_info`: same format as other algorithms `{agent_id: {'obs_shape': ..., 'action_dim': ...}}`
  - Internally flattens obs dict → `obs_dim`, constructs `cent_obs_dim = sum(obs_dim_i)`
- Methods:
  - `flatten_obs(obs_dict)` — same as in MADDPG
  - `select_action(obs, deterministic=False)` → `{agent_id: action}`
  - `get_values(cent_obs)` → `{agent_id: value}`
  - `insert(obs, cent_obs, actions, action_log_probs, values, rewards, dones)` — store one step in buffer
  - `compute_returns(next_cent_obs)` — GAE computation
  - `learn()` → `train_info dict` — PPO multi-epoch update
  - `save(reward=None)`, `load(cls, ...)`

Internally contains:
- `self.actor` / `self.critic` (networks from `networks.py`)
- `self.buffer` (RolloutBuffer)
- `self.actor_optimizer`, `self.critic_optimizer`
- PPO update logic ported from `on-policy-main/onpolicy/algorithms/r_mappo/r_mappo.py`

### 5. `run_env_train_mappo.py` (~150 lines)
New training script using the on-policy loop:
```python
for episode in range(episode_num):
    obs, info = env.reset()
    for step in range(episode_length):
        action, log_prob, value = mappo.select_action_with_info(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        mappo.insert(obs, cent_obs, action, log_prob, value, reward, done)
        obs = next_obs
    mappo.compute_returns(next_cent_obs)
    train_info = mappo.learn()
    mappo.save(episode_rewards)
```

---

## Critical Files to Reference During Implementation

| Source (on-policy-main) | Purpose |
|---|---|
| `onpolicy/algorithms/r_mappo/r_mappo.py` | PPO update logic (lines 91-224) |
| `onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py` | Actor/Critic network structure |
| `onpolicy/utils/shared_buffer.py` | Buffer + GAE computation |
| `onpolicy/utils/util.py` | Loss functions, get_gard_norm |
| `onpolicy/utils/valuenorm.py` | ValueNorm class |
| `onpolicy/algorithms/utils/util.py` | `check()`, `init()` helpers |

| Source (native) | Purpose |
|---|---|
| `schedulers/marl/maddpg/MADDPG.py` | Interface template (`flatten_obs`, `select_action`, `save/load`) |
| `run_env_train_vdn.py` | Training script template |

---

## Scope Estimate

| File | Est. Lines | Complexity |
|---|---|---|
| `utils.py` | ~80 | Low — direct port |
| `networks.py` | ~180 | Medium — remove gym.Space, inline MLP |
| `buffer.py` | ~200 | Medium — simplify multi-thread, keep GAE |
| `MAPPO.py` | ~280 | High — integrate all components, adapt interface |
| `run_env_train_mappo.py` | ~150 | Medium — on-policy loop pattern |
| **Total** | **~890** | |

---

## Verification

1. Run `python run_env_train_mappo.py` and confirm training starts without import errors
2. Verify loss values (policy_loss, value_loss) decrease over first 50 episodes
3. Compare final reward/cost against existing MAPPO results from `on-policy-main` runner
4. Run existing `exp1_compare_baselines.py` to confirm MAPPO is included and compatible
5. Confirm `on-policy-main/` is no longer imported anywhere after the port
