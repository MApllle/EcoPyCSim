"""
实验二：异构环境消融实验
目标：证明 MARL 算法在 CPU 异构场景下的适应性（创新点）

变量：
  server_mix  : balanced / legacy / modern
  hetero_flag : True / False
算法（评测模式，需已训练好的权重）：
  传统基线 energy_greedy / round_robin / least_loaded
  MARL: idqn / mappo / qmix / vdn / maddpg (权重存在则评测，否则跳过)

输出：results/exp2_heterogeneity_ablation.csv
"""
import csv
import os

import numpy as np
import torch

from env.cloud_scheduling import CloudSchedulingEnv

# Server proportion presets (must match make_server_farms.py definitions)
SERVER_MIXES = {
    "balanced": {"old": 0.33, "mid": 0.34, "new": 0.33},
    "legacy":   {"old": 0.60, "mid": 0.30, "new": 0.10},
    "modern":   {"old": 0.10, "mid": 0.20, "new": 0.70},
}

NUM_JOBS    = 100
NUM_FARMS   = 5
NUM_SERVERS = 50
NUM_SEEDS   = 5
RESULTS_DIR = "results"


# ── Helper ──────────────────────────────────────────────────────────────────

def _flatten_obs(obs_dict):
    parts = []
    for key in sorted(obs_dict.keys()):
        arr = obs_dict[key]
        parts.extend(arr.flatten() if isinstance(arr, np.ndarray) else [arr])
    return np.array(parts, dtype=np.float32)


def _run_episode(strategy_name, agent_obj, env, seed=None):
    obs, infos = env.reset(seed=seed)
    total_price = 0
    step_count  = 0
    last_info   = {}

    rr_farm = 0
    rr_srv  = 0

    while env.agents:
        if env.all_jobs_complete:
            break

        actions = {}
        for agent, ob in obs.items():
            n = env.action_space(agent).n

            if strategy_name == "round_robin":
                if agent == "server_farm":
                    actions[agent] = rr_farm % n; rr_farm += 1
                else:
                    actions[agent] = rr_srv  % n; rr_srv  += 1

            elif strategy_name == "least_loaded":
                loads = ob.get("cpus_utilization", [])
                if agent == "server_farm":
                    actions[agent] = int(np.argmin([np.mean(f) for f in loads]))
                else:
                    actions[agent] = int(np.argmin(loads))

            elif strategy_name == "energy_greedy":
                loads = ob["cpus_utilization"]
                tiers = ob["efficiency_tiers"]
                if agent == "server_farm":
                    avg_l = [np.mean(f) for f in loads]
                    avg_t = [np.mean(t) for t in tiers]
                    costs = [(1.0 / (avg_t[i] + 1e-6)) * (avg_l[i] + 0.1) for i in range(len(avg_l))]
                else:
                    costs = [(1.0 / (tiers[i] + 1e-6)) * (loads[i] + 0.1) for i in range(len(loads))]
                actions[agent] = int(np.argmin(costs))

            elif strategy_name == "idqn":
                flat = _flatten_obs(ob)
                t    = torch.from_numpy(flat).unsqueeze(0).float().to(agent_obj.device)
                with torch.no_grad():
                    q = agent_obj.agents[agent].q_net(t)
                actions[agent] = q.argmax(dim=1).item()

            elif strategy_name == "mappo":
                pass  # handled outside loop

            else:
                actions[agent] = env.action_space(agent).sample()

        if strategy_name == "mappo":
            actions, *_ = agent_obj.collect(obs, deterministic=True)

        try:
            obs, rewards, terminations, truncations, infos = env.step(actions)
            sf = infos.get("server_farm", {})
            total_price += sf.get("price", 0)
            step_count  += 1
            last_info    = sf
            if any(terminations.values()) or any(truncations.values()):
                break
        except AssertionError:
            break

    completed = len(last_info.get("completed_job_ids", set()))
    rejected  = last_info.get("rejected_tasks_count", 0)
    return {
        "steps":               step_count,
        "total_price":         round(total_price, 4),
        "eet":                 round(total_price / max(completed, 1), 4),
        "rejected_tasks":      rejected,
        "completed_jobs":      completed,
        "wall_time":           last_info.get("wall_time", 0),
        "jains_fairness":      last_info.get("jains_fairness", 0),
        "active_server_ratio": last_info.get("active_server_ratio", 0),
        "her":                 last_info.get("her", 0),
    }


def _load_marl_agents():
    """Attempt to load all trained MARL models. Returns dict name→agent_obj."""
    agents = {}
    base = os.path.dirname(os.path.abspath(__file__))

    # IDQN
    path = os.path.join(base, "results", "idqn", "model.pt")
    if os.path.exists(path):
        try:
            from schedulers.marl.idqn.IDQN import IDQN
            _env = CloudSchedulingEnv(NUM_JOBS, NUM_FARMS, NUM_SERVERS)
            _env.reset()
            dim_info = {a: {"obs_shape": {k: sp.shape for k, sp in _env.observation_space(a).spaces.items()},
                            "action_dim": _env.action_space(a).n} for a in _env.agents}
            _env.close()
            agents["idqn"] = IDQN.load(dim_info=dim_info, file=path, capacity=1, batch_size=1, lr=5e-4)
            print(f"  已加载 IDQN: {path}")
        except Exception as e:
            print(f"  IDQN 加载失败: {e}")

    # MAPPO
    path = os.path.join(base, "results", "mappo", "model.pt")
    if os.path.exists(path):
        try:
            from schedulers.marl.mappo.MAPPO import MAPPO
            _env = CloudSchedulingEnv(NUM_JOBS, NUM_FARMS, NUM_SERVERS)
            _env.reset()
            dim_info = {a: {"obs_shape": {k: sp.shape for k, sp in _env.observation_space(a).spaces.items()},
                            "action_dim": _env.action_space(a).n} for a in _env.agents}
            _env.close()
            agents["mappo"] = MAPPO.load(dim_info=dim_info, file=path, episode_length=NUM_JOBS,
                                          num_mini_batch=1, lr=5e-4, hidden_size=64)
            print(f"  已加载 MAPPO: {path}")
        except Exception as e:
            print(f"  MAPPO 加载失败: {e}")

    # QMIX
    path = os.path.join(base, "results", "qmix", "model.pt")
    if os.path.exists(path):
        try:
            from schedulers.marl.qmix.QMIX import QMIX
            _env = CloudSchedulingEnv(NUM_JOBS, NUM_FARMS, NUM_SERVERS)
            _env.reset()
            dim_info = {a: {"obs_shape": {k: sp.shape for k, sp in _env.observation_space(a).spaces.items()},
                            "action_dim": _env.action_space(a).n} for a in _env.agents}
            _env.close()
            agents["qmix"] = QMIX.load(dim_info=dim_info, file=path, capacity=1, batch_size=1, lr=5e-4)
            print(f"  已加载 QMIX: {path}")
        except Exception as e:
            print(f"  QMIX 加载失败: {e}")

    # VDN
    path = os.path.join(base, "results", "vdn", "model.pt")
    if os.path.exists(path):
        try:
            from schedulers.marl.vdn.VDN import VDN
            _env = CloudSchedulingEnv(NUM_JOBS, NUM_FARMS, NUM_SERVERS)
            _env.reset()
            dim_info = {a: {"obs_shape": {k: sp.shape for k, sp in _env.observation_space(a).spaces.items()},
                            "action_dim": _env.action_space(a).n} for a in _env.agents}
            _env.close()
            agents["vdn"] = VDN.load(dim_info=dim_info, file=path, capacity=1, batch_size=1, lr=5e-4)
            print(f"  已加载 VDN: {path}")
        except Exception as e:
            print(f"  VDN 加载失败: {e}")

    # MADDPG
    path = os.path.join(base, "results", "maddpg", "model.pt")
    if os.path.exists(path):
        try:
            from schedulers.marl.maddpg.MADDPG import MADDPG
            _env = CloudSchedulingEnv(NUM_JOBS, NUM_FARMS, NUM_SERVERS)
            _env.reset()
            dim_info = {a: {"obs_shape": {k: sp.shape for k, sp in _env.observation_space(a).spaces.items()},
                            "action_dim": _env.action_space(a).n} for a in _env.agents}
            _env.close()
            agents["maddpg"] = MADDPG.load(dim_info=dim_info, file=path, capacity=1, batch_size=1,
                                             actor_lr=1e-4, critic_lr=1e-4)
            print(f"  已加载 MADDPG: {path}")
        except Exception as e:
            print(f"  MADDPG 加载失败: {e}")

    return agents


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("加载已训练模型...")
    marl_agents = _load_marl_agents()

    # Traditional baselines always available
    traditional = ["energy_greedy", "least_loaded", "round_robin"]
    marl_names  = list(marl_agents.keys())
    all_strategies = traditional + marl_names

    all_rows = []

    for mix_name, proportions in SERVER_MIXES.items():
        for use_hetero in [True, False]:
            label = f"{mix_name}_hetero={'on' if use_hetero else 'off'}"
            print(f"\n=== 条件: {label} ===")

            for strategy in all_strategies:
                agent_obj = marl_agents.get(strategy, None)
                seed_results = []

                for seed in range(NUM_SEEDS):
                    env = CloudSchedulingEnv(
                        num_jobs=NUM_JOBS,
                        num_server_farms=NUM_FARMS,
                        num_servers=NUM_SERVERS,
                        use_heterogeneity=use_hetero,
                        server_proportions=proportions,
                    )
                    try:
                        r = _run_episode(strategy, agent_obj, env, seed=seed)
                        seed_results.append(r)
                    except Exception as e:
                        print(f"  {strategy} seed={seed} 出错: {e}")
                    finally:
                        env.close()

                if not seed_results:
                    continue

                metrics = list(seed_results[0].keys())
                row = {
                    "server_mix":    mix_name,
                    "use_hetero":    use_hetero,
                    "strategy":      strategy,
                }
                for m in metrics:
                    vals = [r[m] for r in seed_results]
                    row[f"{m}_mean"] = round(float(np.mean(vals)), 4)
                    row[f"{m}_std"]  = round(float(np.std(vals)),  4)

                print(
                    f"  {strategy:15} | 价格: {row['total_price_mean']:.2f}±{row['total_price_std']:.2f} "
                    f"| HER: {row['her_mean']:.4f}±{row['her_std']:.4f} "
                    f"| 拒绝: {row['rejected_tasks_mean']:.1f}"
                )
                all_rows.append(row)

    if all_rows:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "exp2_heterogeneity_ablation.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n消融实验结果已保存至 {csv_path}")
