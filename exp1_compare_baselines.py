import csv
import os

import numpy as np
import torch

from env.cloud_scheduling import CloudSchedulingEnv


NUM_JOBS = 50
NUM_FARMS = 2
NUM_SERVERS = 6
NUM_SEEDS = 5

# ── 指定各算法模型路径（相对于本文件目录）────────────────────────────────────
# 可填目录（默认拼接 model.pt），也可直接填模型文件路径（如 checkpoints/model_ep10.pt）
# 设为 None 则跳过该算法
MODEL_DIRS = {
    "idqn":   "results/idqn",
    "mappo":  "results/mappo",
    "qmix":   "results/qmix",
    "vdn":    "results/vdn",
    "maddpg": "results/maddpg",
}

# ── 指定要运行的策略（注释掉不需要的行即可）──────────────────────────────────
STRATEGIES = [
    # "random",
    # "round_robin",
    # "least_loaded",
    # "best_fit",
    # "energy_greedy",
    # "idqn",
    # "vdn",
    # "qmix",
    "mappo",
    "maddpg",
]
# ─────────────────────────────────────────────────────────────────────────────


def _build_dim_info(num_jobs=NUM_JOBS, num_farms=NUM_FARMS, num_servers=NUM_SERVERS):
    env = CloudSchedulingEnv(num_jobs=num_jobs, num_server_farms=num_farms, num_servers=num_servers)
    env.reset()
    dim_info = {
        agent_id: {
            "obs_shape": {
                key: space.shape
                for key, space in env.observation_space(agent_id).spaces.items()
            },
            "action_dim": env.action_space(agent_id).n,
        }
        for agent_id in env.agents
    }
    env.close()
    return dim_info


def _safe_mean(values):
    return round(float(np.mean(values)), 4) if values else 0.0


def _flatten_obs(obs_dict: dict) -> np.ndarray:
    """将 dict 观测拼接为 1D numpy 数组（键排序，与 IDQN/MADDPG 相同）。"""
    parts = []
    for key in sorted(obs_dict.keys()):
        arr = obs_dict[key]
        parts.extend(arr.flatten() if isinstance(arr, np.ndarray) else [arr])
    return np.array(parts, dtype=np.float32)


class BaselineEvaluator:
    def __init__(self, idqn=None, mappo=None, qmix=None, vdn=None, maddpg=None):
        self.rr_count_farm = 0
        self.rr_count_server = 0
        self.qmix = qmix
        self.vdn = vdn
        self.maddpg = maddpg
        self.idqn = idqn    # 可选：传入已加载的 IDQN 实例
        self.mappo = mappo  # 可选：传入已加载的 MAPPO 实例

    def get_actions(self, obs_dict, strategy, env):
        """适配 ParallelEnv：一次性为所有活动的 Agent 生成动作"""
        actions = {}

        for agent, obs in obs_dict.items():
            action_space = env.action_space(agent)

            if strategy == "random":
                actions[agent] = action_space.sample()

            elif strategy == "round_robin":
                if agent == "server_farm":
                    actions[agent] = self.rr_count_farm % action_space.n
                    self.rr_count_farm += 1
                else:
                    actions[agent] = self.rr_count_server % action_space.n
                    self.rr_count_server += 1

            elif strategy == "least_loaded":
                all_loads = obs.get('cpus_utilization', [])
                if agent == "server_farm":
                    # all_loads 是二维数组
                    avg_loads = [np.mean(f) for f in all_loads]
                    actions[agent] = int(np.argmin(avg_loads))
                else:
                    # all_loads 是一维数组
                    actions[agent] = int(np.argmin(all_loads))

            elif strategy == "best_fit":
                all_loads = obs['cpus_utilization']
                task_req = obs['task_cpu'][0]

                best_idx = -1
                min_remaining = 1.1

                # 如果是 server_farm 级别，all_loads 是二维的，我们取各 farm 平均负载
                if agent == "server_farm":
                    loads_to_check = [np.mean(f) for f in all_loads]
                else:
                    loads_to_check = all_loads

                for i, load in enumerate(loads_to_check):
                    remaining = 1.0 - load
                    if remaining >= task_req and remaining < min_remaining:
                        min_remaining = remaining
                        best_idx = i

                actions[agent] = int(best_idx) if best_idx != -1 else action_space.sample()

            elif strategy == "energy_greedy":
                all_loads = obs['cpus_utilization']
                tiers = obs['efficiency_tiers']

                if agent == "server_farm":
                    # farm 级别取平均值
                    avg_loads = [np.mean(f) for f in all_loads]
                    avg_tiers = [np.mean(t) for t in tiers]
                    costs = [(1.0 / (avg_tiers[i] + 1e-6)) * (avg_loads[i] + 0.1) for i in range(len(avg_loads))]
                else:
                    costs = [(1.0 / (tiers[i] + 1e-6)) * (all_loads[i] + 0.1) for i in range(len(all_loads))]

                actions[agent] = int(np.argmin(costs))

            elif strategy == "idqn":
                if self.idqn is None:
                    raise ValueError("strategy='idqn' 需要在 BaselineEvaluator(idqn=...) 传入已训练的 IDQN 实例")
                flat_o = _flatten_obs(obs)
                obs_t = torch.from_numpy(flat_o).unsqueeze(0).float().to(self.idqn.device)
                with torch.no_grad():
                    q_vals = self.idqn.agents[agent].q_net(obs_t)   # (1, act_dim)
                actions[agent] = q_vals.argmax(dim=1).item()

            elif strategy == "vdn":
                if self.vdn is None:
                    raise ValueError("strategy='vdn' requires a loaded VDN instance")
                actions = self.vdn.select_action(obs_dict, epsilon=0.0)
                break

            elif strategy == "qmix":
                if self.qmix is None:
                    raise ValueError("strategy='qmix' requires a loaded QMIX instance")
                actions = self.qmix.select_action(obs_dict, epsilon=0.0)
                break

            elif strategy == "maddpg":
                if self.maddpg is None:
                    raise ValueError("strategy='maddpg' requires a loaded MADDPG instance")
                actions = self.maddpg.select_action(obs_dict)
                break

            else:
                actions[agent] = action_space.sample()

        return actions

    def get_actions_mappo(self, obs_dict: dict) -> dict:
        """MAPPO 集中调用：一次性为所有 agent 生成确定性动作。"""
        if self.mappo is None:
            raise ValueError("strategy='mappo' 需要在 BaselineEvaluator(mappo=...) 传入已训练的 MAPPO 实例")
        actions, _, _, _, _ = self.mappo.collect(obs_dict, deterministic=True)
        return actions

def run_experiment(strategy_name, idqn=None, mappo=None, qmix=None, vdn=None, maddpg=None, seed=None):
    eval_env = CloudSchedulingEnv(num_jobs=NUM_JOBS, num_server_farms=NUM_FARMS, num_servers=NUM_SERVERS)
    observations, infos = eval_env.reset(seed=seed)

    evaluator = BaselineEvaluator(idqn=idqn, mappo=mappo, qmix=qmix, vdn=vdn, maddpg=maddpg)
    total_price = 0
    step_count = 0
    last_info = {}
    jains_series = []
    asr_series = []

    while eval_env.agents:
        if eval_env.all_jobs_complete:
            break

        if strategy_name == "mappo":
            actions = evaluator.get_actions_mappo(observations)
        else:
            actions = evaluator.get_actions(observations, strategy_name, eval_env)

        try:
            observations, rewards, terminations, truncations, infos = eval_env.step(actions)

            sf_info = infos.get("server_farm", {})
            total_price += sf_info.get("price", 0)
            step_count += 1
            last_info = sf_info
            if "jains_fairness" in sf_info:
                jains_series.append(sf_info["jains_fairness"])
            if "active_server_ratio" in sf_info:
                asr_series.append(sf_info["active_server_ratio"])

            if any(terminations.values()) or any(truncations.values()):
                break
        except AssertionError:
            break

    rejected = last_info.get("rejected_tasks_count", 0)
    wall_time = last_info.get("wall_time", 0)
    jains    = _safe_mean(jains_series)
    asr      = _safe_mean(asr_series)
    her      = last_info.get("her", 0)
    completed = len(last_info.get("completed_job_ids", set()))

    eet = round(total_price / max(completed, 1), 4)

    print(
        f"【实验结果】策略: {strategy_name:15} | 步数: {step_count:5} | "
        f"价格: {total_price:.2f} | EET: {eet:.4f} | "
        f"拒绝任务: {rejected:4} | 完成作业: {completed:4} | "
        f"Jain(mean): {jains:.4f} | ASR(mean): {asr:.4f} | HER(final): {her:.4f}"
    )
    eval_env.close()

    return {
        "strategy": strategy_name,
        "steps": step_count,
        "total_price": round(total_price, 4),
        "eet": eet,
        "rejected_tasks": rejected,
        "completed_jobs": completed,
        "wall_time": wall_time,
        "jains_fairness": jains,
        "active_server_ratio": asr,
        "her": her,
    }

def _load_marl_agents():
    agents = {}
    base = os.path.dirname(os.path.abspath(__file__))
    dim_info = _build_dim_info()
    model_exts = {".pt", ".pth", ".ckpt"}
    candidates = {}
    for name, configured_path in MODEL_DIRS.items():
        if configured_path is None:
            continue
        resolved = configured_path if os.path.isabs(configured_path) else os.path.join(base, configured_path)
        # 兼容两种配置：目录（自动补 model.pt）或文件（直接使用）
        if os.path.splitext(resolved)[1].lower() in model_exts:
            candidates[name] = resolved
        else:
            candidates[name] = os.path.join(resolved, "model.pt")

    for name, path in candidates.items():
        if not os.path.exists(path):
            continue
        try:
            if name == "idqn":
                from schedulers.marl.idqn.IDQN import IDQN

                agents[name] = IDQN.load(dim_info=dim_info, file=path, capacity=1, batch_size=1, lr=5e-4)
            elif name == "mappo":
                from schedulers.marl.mappo.MAPPO import MAPPO

                agents[name] = MAPPO.load(
                    dim_info=dim_info,
                    file=path,
                    episode_length=NUM_JOBS,
                    num_mini_batch=4,
                    lr=5e-4,
                    hidden_size=64,
                )
            elif name == "qmix":
                from schedulers.marl.qmix.QMIX import QMIX

                agents[name] = QMIX.load(
                    dim_info=dim_info,
                    file=path,
                    capacity=1,
                    batch_size=1,
                    lr=5e-4,
                    embed_dim=32,
                )
            elif name == "vdn":
                from schedulers.marl.vdn.VDN import VDN

                agents[name] = VDN.load(dim_info=dim_info, file=path, capacity=1, batch_size=1, lr=5e-4)
            elif name == "maddpg":
                from schedulers.marl.maddpg.MADDPG import MADDPG

                agents[name] = MADDPG.load(
                    dim_info=dim_info,
                    file=path,
                    capacity=1,
                    batch_size=1,
                    actor_lr=5e-4,
                    critic_lr=5e-4,
                )
            print(f"Loaded {name.upper()} model: {path}")
        except Exception as e:
            print(f"Skip {name}: failed to load model from {path} ({e})")

    return agents


def _run_main():
    marl_agents = _load_marl_agents()
    marl_names = {"idqn", "vdn", "qmix", "mappo", "maddpg"}
    strategies = [s for s in STRATEGIES if s not in marl_names or s in marl_agents]

    all_results = []
    for s in strategies:
        print(f"\nRunning strategy: {s} ({NUM_SEEDS} seeds)...")
        seed_results = []
        for seed in range(NUM_SEEDS):
            try:
                result = run_experiment(
                    s,
                    idqn=marl_agents.get("idqn"),
                    mappo=marl_agents.get("mappo"),
                    qmix=marl_agents.get("qmix"),
                    vdn=marl_agents.get("vdn"),
                    maddpg=marl_agents.get("maddpg"),
                    seed=seed,
                )
                seed_results.append(result)
            except Exception as e:
                print(f"Run failed for {s} seed={seed}: {e}")

        if seed_results:
            metrics = [
                "steps",
                "total_price",
                "eet",
                "rejected_tasks",
                "completed_jobs",
                "wall_time",
                "jains_fairness",
                "active_server_ratio",
                "her",
            ]
            agg = {"strategy": s}
            for metric in metrics:
                vals = [r[metric] for r in seed_results]
                agg[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
                agg[f"{metric}_std"] = round(float(np.std(vals)), 4)
            all_results.append(agg)

    if all_results:
        os.makedirs("results", exist_ok=True)
        csv_path = os.path.join("results", "exp1_baseline_comparison.csv")
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    _run_main()
    raise SystemExit(0)
    # ── 尝试加载已训练的 IDQN 模型 ─────────────────────────────────────────
    idqn_instance = None
    idqn_model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results', 'idqn', 'model.pt'
    )
    if os.path.exists(idqn_model_path):
        from schedulers.marl.idqn.IDQN import IDQN
        # 用与训练时相同的环境规模构建 dim_info
        _tmp_env = CloudSchedulingEnv(num_jobs=100, num_server_farms=5, num_servers=50)
        _tmp_env.reset()
        _dim_info = {
            agent_id: {
                'obs_shape': {
                    key: space.shape
                    for key, space in _tmp_env.observation_space(agent_id).spaces.items()
                },
                'action_dim': _tmp_env.action_space(agent_id).n,
            }
            for agent_id in _tmp_env.agents
        }
        _tmp_env.close()
        idqn_instance = IDQN.load(
            dim_info   = _dim_info,
            file       = idqn_model_path,
            capacity   = 1,       # eval 模式不需要 buffer
            batch_size = 1,
            lr         = 0.0005,
        )
        print(f"已加载 IDQN 模型：{idqn_model_path}")
    else:
        print(f"未找到 IDQN 模型（{idqn_model_path}），跳过 idqn 策略。")
        print("提示：先运行 python run_env_train_idqn.py 完成训练。")

    # ── 尝试加载已训练的 MAPPO 模型 ─────────────────────────────────────────
    mappo_instance = None
    mappo_model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results', 'mappo', 'model.pt'
    )
    if os.path.exists(mappo_model_path):
        try:
            from schedulers.marl.mappo.MAPPO import MAPPO
            _tmp_env = CloudSchedulingEnv(num_jobs=100, num_server_farms=5, num_servers=50)
            _tmp_env.reset()
            _mappo_dim_info = {
                agent_id: {
                    'obs_shape': {
                        key: space.shape
                        for key, space in _tmp_env.observation_space(agent_id).spaces.items()
                    },
                    'action_dim': _tmp_env.action_space(agent_id).n,
                }
                for agent_id in _tmp_env.agents
            }
            _tmp_env.close()
            mappo_instance = MAPPO.load(
                dim_info       = _mappo_dim_info,
                file           = mappo_model_path,
                episode_length = 100,   # eval env num_jobs
                num_mini_batch = 1,
                lr             = 5e-4,
                hidden_size    = 64,
            )
            print(f"已加载 MAPPO 模型：{mappo_model_path}")
        except Exception as e:
            print(f"加载 MAPPO 模型失败（{e}），跳过 mappo 策略。")
            print("提示：先运行 python run_env_train_mappo.py 完成训练，且训练时 env 规模需与评测一致。")
    else:
        print(f"未找到 MAPPO 模型（{mappo_model_path}），跳过 mappo 策略。")
        print("提示：先运行 python run_env_train_mappo.py 完成训练。")

    # ── 运行所有策略 ──────────────────────────────────────────────────────
    strategies = ["random", "round_robin", "least_loaded", "best_fit", "energy_greedy"]
    if idqn_instance is not None:
        strategies.append("idqn")
    if mappo_instance is not None:
        strategies.append("mappo")

    NUM_SEEDS = 5
    all_results = []

    for s in strategies:
        print(f"\n正在运行策略: {s} ({NUM_SEEDS} seeds)...")
        seed_results = []
        for seed in range(NUM_SEEDS):
            try:
                result = run_experiment(
                    s,
                    idqn=idqn_instance if s == "idqn" else None,
                    mappo=mappo_instance if s == "mappo" else None,
                    seed=seed,
                )
                seed_results.append(result)
            except Exception as e:
                print(f"运行 {s} seed={seed} 时出错: {e}")
                import traceback
                traceback.print_exc()

        if seed_results:
            # Aggregate across seeds
            metrics = ["steps", "total_price", "eet", "rejected_tasks",
                       "completed_jobs", "wall_time", "jains_fairness",
                       "active_server_ratio", "her"]
            agg = {"strategy": s}
            for m in metrics:
                vals = [r[m] for r in seed_results]
                agg[f"{m}_mean"] = round(float(np.mean(vals)), 4)
                agg[f"{m}_std"]  = round(float(np.std(vals)),  4)
            all_results.append(agg)

    # Write aggregated results to CSV
    if all_results:
        os.makedirs("results", exist_ok=True)
        csv_path = os.path.join("results", "exp1_baseline_comparison.csv")
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n结果已保存至 {csv_path}")
