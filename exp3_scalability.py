"""
实验3：可扩展性 / 调度与仿真开销（体系结构课程对齐）

测量内容
--------
1. 固定策略（均匀随机动作，与 main.py 一致）下：
   - 环境构建 + reset 耗时
   - 单 episode 墙钟时间、步数、累计能耗（仿真内 price 之和）
   - 每步平均耗时：动作采样 vs env.step（近似分解）
2. 可选：多进程并行 rollout 墙钟时间，估算采样并行加速比（对比单进程总时间）

用法
----
  python exp3_scalability.py --repeats 5
  python exp3_scalability.py --scale-mode servers_only --repeats 5
  python exp3_scalability.py --scale-mode all --repeats 5
  python exp3_scalability.py --parallel-benchmark --parallel-workers 2,4,8
  python exp3_scalability.py --parallel-episodes-sweep --parallel-jobs 20

输出
----
  results/exp3_scalability_<mode>.csv / .png   （mode: mixed | servers_only | farms_only）
  （--parallel-benchmark）exp3_parallel_rollout.csv / .png
  （--parallel-episodes-sweep）exp3_parallel_vs_episodes.csv / .png

规模模式（--scale-mode）
------------------------
  mixed         — 默认多点 (farms, total_servers) 组合
  servers_only  — 固定 farm 数，只扫总台数（隔离「规模」变量）
  farms_only    — 固定总台数，只扫 farm 数
  all           — 依次跑上述三种，写出三份 CSV/PNG
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from components.model_scripts.make_server_farms import PROPORTION_PRESETS
from env.cloud_scheduling_v0 import CloudSchedulingEnv

# ── 默认规模网格：(num_server_farms, total_servers) —— 与 env 一致，num_servers 为全集群总台数
DEFAULT_SCALE_GRID = [
    (2, 10),
    (3, 18),
    (5, 25),
    (5, 50),
    (8, 80),
    (10, 100),
]

QUICK_SCALE_GRID = [
    (2, 10),
    (3, 18),
    (5, 25),
]

# 固定 farm 数，只增加总台数（每 farm 平均台数随之增加）
SERVERS_SWEEP_FIXED_FARMS = 5
SERVERS_SWEEP_TOTALS = [10, 25, 50, 80, 100]

# 固定总台数，只增加 farm 数（每 farm 平均台数随之减少）
FARMS_SWEEP_FIXED_TOTAL = 50
FARMS_SWEEP_COUNTS = [2, 5, 10]


def build_scale_grid(
    mode: str,
    quick: bool,
    fixed_farms: int,
    fixed_total_servers: int,
) -> tuple[str, list[tuple[int, int]]]:
    """返回 (逻辑模式名, [(farms, total_servers), ...])。"""
    if quick and mode in ("mixed", "all"):
        return "mixed", list(QUICK_SCALE_GRID)

    if mode == "mixed":
        return "mixed", list(DEFAULT_SCALE_GRID)

    if mode == "servers_only":
        pts = [(fixed_farms, n) for n in (SERVERS_SWEEP_TOTALS if not quick else [10, 25, 50])]
        return "servers_only", pts

    if mode == "farms_only":
        tot = fixed_total_servers
        farms_list = FARMS_SWEEP_COUNTS if not quick else [2, 5]
        pts = [(f, tot) for f in farms_list]
        return "farms_only", pts

    raise ValueError(f"unknown scale mode: {mode}")


def _make_env(
    num_jobs: int,
    num_server_farms: int,
    num_servers: int,
    seed: int,
    use_heterogeneity: bool,
    hetero_weight: float,
    proportion_key: str,
) -> CloudSchedulingEnv:
    proportions = PROPORTION_PRESETS[proportion_key]
    return CloudSchedulingEnv(
        num_jobs=num_jobs,
        num_server_farms=num_server_farms,
        num_servers=num_servers,
        use_heterogeneity=use_heterogeneity,
        hetero_weight=hetero_weight,
        server_proportions=proportions,
    )


def run_episode_profiled(env: CloudSchedulingEnv, seed: int) -> dict[str, Any]:
    """跑完一个 episode，返回时间与能耗分解。"""
    t_construct0 = time.perf_counter()
    # 构造已在调用方完成；此处从 reset 计时
    obs, infos = env.reset(seed=seed)
    t_after_reset = time.perf_counter()

    total_energy = 0.0
    steps = 0
    sum_action_s = 0.0
    sum_step_s = 0.0
    asr_samples = []
    jain_samples = []

    while env.agents:
        t0 = time.perf_counter()
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        t1 = time.perf_counter()
        obs, rewards, terminations, truncations, infos = env.step(actions)
        t2 = time.perf_counter()

        sum_action_s += t1 - t0
        sum_step_s += t2 - t1
        steps += 1

        sf_info = infos.get("server_farm", {})
        total_energy += float(sf_info.get("price", 0.0))
        if "active_server_ratio" in sf_info:
            asr_samples.append(float(sf_info["active_server_ratio"]))
        if "jains_fairness" in sf_info:
            jain_samples.append(float(sf_info["jains_fairness"]))

        if all(terminations.values()):
            env.close()
            break

    t_end = time.perf_counter()
    wall_episode_s = t_end - t_after_reset
    reset_s = t_after_reset - t_construct0

    return {
        "reset_wall_s": reset_s,
        "episode_wall_s": wall_episode_s,
        "steps": steps,
        "total_energy_cost": total_energy,
        "sum_action_s": sum_action_s,
        "sum_step_s": sum_step_s,
        "mean_active_server_ratio": float(statistics.mean(asr_samples)) if asr_samples else 0.0,
        "mean_jains_fairness": float(statistics.mean(jain_samples)) if jain_samples else 0.0,
    }


def benchmark_scale(
    num_jobs: int,
    num_farms: int,
    total_servers: int,
    base_seed: int,
    repeats: int,
    use_heterogeneity: bool,
    hetero_weight: float,
    proportion_key: str,
) -> dict[str, Any]:
    """同一规模重复 repeats 次，返回均值与标准差。"""
    rows = []
    for r in range(repeats):
        seed = base_seed + r
        t0 = time.perf_counter()
        env = _make_env(
            num_jobs, num_farms, total_servers, seed,
            use_heterogeneity, hetero_weight, proportion_key,
        )
        t1 = time.perf_counter()
        construct_s = t1 - t0

        prof = run_episode_profiled(env, seed)
        prof["construct_s"] = construct_s
        prof["total_init_s"] = construct_s + prof["reset_wall_s"]
        rows.append(prof)

    def mean_std(key: str) -> tuple[float, float]:
        vals = [row[key] for row in rows]
        if len(vals) < 2:
            return float(vals[0]), 0.0
        return statistics.mean(vals), statistics.stdev(vals)

    m_ep, s_ep = mean_std("episode_wall_s")
    m_st, s_st = mean_std("steps")
    m_en, s_en = mean_std("total_energy_cost")
    m_init, s_init = mean_std("total_init_s")
    m_act, _ = mean_std("sum_action_s")
    m_step, _ = mean_std("sum_step_s")
    m_asr, s_asr = mean_std("mean_active_server_ratio")
    m_jain, s_jain = mean_std("mean_jains_fairness")

    ms_per_rep = [
        (row["episode_wall_s"] / row["steps"] * 1000.0) if row["steps"] > 0 else 0.0
        for row in rows
    ]
    if len(ms_per_rep) < 2:
        ms_std = 0.0
    else:
        ms_std = statistics.stdev(ms_per_rep)

    ms_per_step = (m_ep / m_st * 1000.0) if m_st > 0 else 0.0
    ms_action_per_step = (m_act / m_st * 1000.0) if m_st > 0 else 0.0
    ms_env_per_step = (m_step / m_st * 1000.0) if m_st > 0 else 0.0

    return {
        "num_server_farms": num_farms,
        "total_servers": total_servers,
        "servers_per_farm_avg": total_servers / num_farms,
        "mean_episode_wall_s": m_ep,
        "std_episode_wall_s": s_ep,
        "mean_steps": m_st,
        "std_steps": s_st,
        "mean_total_energy": m_en,
        "std_total_energy": s_en,
        "mean_init_s": m_init,
        "std_init_s": s_init,
        "mean_active_server_ratio": m_asr,
        "std_active_server_ratio": s_asr,
        "mean_jains_fairness": m_jain,
        "std_jains_fairness": s_jain,
        "ms_per_step_mean": ms_per_step,
        "std_ms_per_step": ms_std,
        "ms_action_per_step": ms_action_per_step,
        "ms_env_step_per_step": ms_env_per_step,
    }


def _ensure_results_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def plot_scalability(
    rows: list[dict[str, Any]],
    out_png: str,
    *,
    scale_mode: str = "mixed",
) -> None:
    # 图中使用英文标签，避免部分环境下默认字体缺中文导致乱码（报告内可配中文说明）。
    if scale_mode == "farms_only":
        xs = [r["num_server_farms"] for r in rows]
        x_labels = [f"{r['num_server_farms']}f/{r['total_servers']}srv" for r in rows]
        x_title = "Number of server farms (fixed total servers)"
    else:
        xs = [r["total_servers"] for r in rows]
        x_labels = [f"{r['num_server_farms']}f x {r['total_servers']}srv" for r in rows]
        if scale_mode == "servers_only":
            x_title = "Total servers (fixed number of farms)"
        else:
            x_title = "Total servers (cluster-wide)"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.errorbar(
        xs,
        [r["mean_episode_wall_s"] for r in rows],
        yerr=[r["std_episode_wall_s"] for r in rows],
        fmt="o-",
        capsize=3,
    )
    ax.set_xlabel(x_title)
    ax.set_ylabel("Episode wall time (s)")
    ax.set_title("Simulator: one episode wall time vs scale")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    yerr_ms = [r.get("std_ms_per_step", 0.0) for r in rows]
    ax.errorbar(
        xs,
        [r["ms_per_step_mean"] for r in rows],
        yerr=yerr_ms,
        fmt="s-",
        color="C1",
        capsize=3,
    )
    ax.set_xlabel(x_title)
    ax.set_ylabel("Mean ms per env step")
    ax.set_title("Scheduler loop: mean time per step (random policy)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.errorbar(
        xs,
        [r["mean_total_energy"] for r in rows],
        yerr=[r["std_total_energy"] for r in rows],
        fmt="^-",
        color="C2",
        capsize=3,
    )
    ax.set_xlabel(x_title)
    ax.set_ylabel("Sum of price (episode energy metric)")
    ax.set_title("Total energy metric vs scale (same random policy)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.errorbar(
        xs,
        [r["mean_init_s"] for r in rows],
        yerr=[r["std_init_s"] for r in rows],
        fmt="d-",
        color="C3",
        capsize=3,
        label="construct + reset",
    )
    ax.set_xlabel(x_title)
    ax.set_ylabel("Time (s)")
    ax.set_title("Env init: construct + reset (mean)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, rotation=25, ha="right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def print_amdahl_notes() -> None:
    print(
        "\n── Amdahl / 并行性（定性，用于报告）────────────────────────────────────\n"
        "• 本仿真中：时间轴事件推进与资源状态更新是串行离散事件模拟，单 episode 内难以多线程并行。\n"
        "• 强化学习训练时：各 worker 独立环境 rollout 彼此无依赖，属于易并行部分；梯度更新与\n"
        "  centralized critic 往往成为串行瓶颈，整体加速比受限于 Amdahl 定律。\n"
        "• 图中「每步耗时」反映观测维度与集群规模增长带来的软件路径开销，对应调度器实现\n"
        "  与体系结构建模的 cross-layer 成本。\n"
    )


# --- 多进程并行 rollout（仅测墙钟，用于说明采样可并行）---


def _chunk_seeds(seeds: list[int], num_chunks: int) -> list[list[int]]:
    """将 seeds 均分到 num_chunks 个子列表（每进程多 episode，摊销进程启动开销）。"""
    if num_chunks <= 0:
        return [seeds]
    n = len(seeds)
    base, rem = divmod(n, num_chunks)
    out: list[list[int]] = []
    i = 0
    for k in range(num_chunks):
        sz = base + (1 if k < rem else 0)
        out.append(seeds[i : i + sz])
        i += sz
    return [c for c in out if c]


def _worker_episode_chunk_seconds(
    payload: tuple[int, int, int, tuple[int, ...], int, float, str],
) -> float:
    """子进程内连续跑若干 episode，返回墙钟秒数（顶层函数，便于 Windows spawn pickle）。"""
    num_jobs, farms, total_servers, seeds_tuple, hw, hetero_w, prop_key = payload
    seeds = list(seeds_tuple)
    use_hetero = bool(hw)
    t0 = time.perf_counter()
    for seed in seeds:
        env = _make_env(
            num_jobs, farms, total_servers, seed,
            use_hetero, hetero_w, prop_key,
        )
        run_episode_profiled(env, seed)
    return time.perf_counter() - t0


def parse_int_list(s: str) -> list[int]:
    """解析逗号分隔正整数，如 '2,4,8' -> [2, 4, 8]。"""
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parallel_rollout_benchmark(
    num_jobs: int,
    farms: int,
    total_servers: int,
    base_seed: int,
    total_episodes: int,
    worker_counts: list[int],
    use_heterogeneity: bool,
    hetero_weight: float,
    proportion_key: str,
    results_dir: str,
) -> None:
    """固定规模，总 episode 数固定，改变进程数，记录完成全部 rollout 的墙钟时间。

    说明：Windows 下多进程有显著启动/导入开销；应使用较大的 num_jobs 与 total_episodes，
    使单进程内计算时间远大于进程开销，否则加速比可能长期小于 1（仍可作为讨论点写入报告）。
    """
    seeds = [base_seed + i for i in range(total_episodes)]
    rows = []

    # 串行基线：同一进程内连续跑完全部 episode
    t0 = time.perf_counter()
    for s in seeds:
        env = _make_env(
            num_jobs, farms, total_servers, s,
            use_heterogeneity, hetero_weight, proportion_key,
        )
        run_episode_profiled(env, s)
    serial_wall = time.perf_counter() - t0

    rows.append(
        {
            "workers": 1,
            "mode": "serial_for_loop",
            "wall_s": round(serial_wall, 4),
            "speedup_vs_serial": 1.0,
        }
    )

    hw_flag = 1 if use_heterogeneity else 0

    for w in worker_counts:
        if w <= 1:
            continue
        chunks = _chunk_seeds(seeds, w)
        payloads = [
            (
                num_jobs,
                farms,
                total_servers,
                tuple(chunk),
                hw_flag,
                hetero_weight,
                proportion_key,
            )
            for chunk in chunks
        ]
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=w) as ex:
            futures = [ex.submit(_worker_episode_chunk_seconds, p) for p in payloads]
            for fut in as_completed(futures):
                fut.result()
        wall = time.perf_counter() - t0
        speedup = serial_wall / wall if wall > 0 else float("inf")
        rows.append(
            {
                "workers": w,
                "mode": "process_pool_chunked",
                "wall_s": round(wall, 4),
                "speedup_vs_serial": round(speedup, 4),
            }
        )

    csv_path = os.path.join(results_dir, "exp3_parallel_rollout.csv")
    write_csv(csv_path, list(rows[0].keys()), rows)
    print(f"并行 rollout 结果已写入 {csv_path}")

    ws = [r["workers"] for r in rows]
    sp = [r["speedup_vs_serial"] for r in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(ws, sp, "o-", markersize=8)
    plt.xlabel("Parallel workers")
    plt.ylabel("Speedup vs serial for-loop")
    plt.title(
        f"Parallel rollouts ({total_episodes} episodes, "
        f"farms={farms}, total_servers={total_servers})"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(results_dir, "exp3_parallel_rollout.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"并行加速图已写入 {png_path}")
    if len(rows) > 1 and all(r["speedup_vs_serial"] <= 1.05 for r in rows[1:]):
        print(
            "\n提示：当前并行加速比仍接近或低于 1。"
            "可增大 --parallel-jobs / --parallel-episodes，或换用 Linux/WSL 复测，"
            "使单次 rollout 计算量显著大于进程启动成本。\n"
        )


def parallel_episodes_sweep(
    num_jobs: int,
    farms: int,
    total_servers: int,
    base_seed: int,
    episode_counts: list[int],
    parallel_workers: int,
    use_heterogeneity: bool,
    hetero_weight: float,
    proportion_key: str,
    results_dir: str,
) -> None:
    """固定并行 worker 数，扫描总 episode 数：看「工作量」增大时加速比是否更稳定。"""
    if parallel_workers < 2:
        raise ValueError("parallel_episodes_sweep 需要 parallel_workers >= 2")

    rows: list[dict[str, Any]] = []
    hw_flag = 1 if use_heterogeneity else 0

    for n_ep in episode_counts:
        seeds = [base_seed + i for i in range(n_ep)]

        t0 = time.perf_counter()
        for s in seeds:
            env = _make_env(
                num_jobs, farms, total_servers, s,
                use_heterogeneity, hetero_weight, proportion_key,
            )
            run_episode_profiled(env, s)
        serial_wall = time.perf_counter() - t0

        chunks = _chunk_seeds(seeds, parallel_workers)
        payloads = [
            (
                num_jobs,
                farms,
                total_servers,
                tuple(chunk),
                hw_flag,
                hetero_weight,
                proportion_key,
            )
            for chunk in chunks
        ]
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=parallel_workers) as ex:
            futures = [ex.submit(_worker_episode_chunk_seconds, p) for p in payloads]
            for fut in as_completed(futures):
                fut.result()
        par_wall = time.perf_counter() - t0

        sp = serial_wall / par_wall if par_wall > 0 else float("inf")
        rows.append(
            {
                "total_episodes": n_ep,
                "parallel_workers": parallel_workers,
                "serial_wall_s": round(serial_wall, 4),
                "parallel_wall_s": round(par_wall, 4),
                "speedup_vs_serial": round(sp, 4),
            }
        )
        print(
            f"  episodes={n_ep:3d} | serial {serial_wall:.3f}s | "
            f"{parallel_workers}w {par_wall:.3f}s | speedup {sp:.3f}x"
        )

    csv_path = os.path.join(results_dir, "exp3_parallel_vs_episodes.csv")
    write_csv(csv_path, list(rows[0].keys()), rows)
    print(f"已写入 {csv_path}")

    fig, ax1 = plt.subplots(figsize=(7, 4))
    xs = [r["total_episodes"] for r in rows]
    ax1.plot(xs, [r["speedup_vs_serial"] for r in rows], "o-", color="C0", label="Speedup")
    ax1.set_xlabel("Total episodes (fixed workload per episode)")
    ax1.set_ylabel("Speedup vs serial")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(xs, [r["serial_wall_s"] for r in rows], "s--", color="C1", alpha=0.7, label="Serial wall (s)")
    ax2.plot(xs, [r["parallel_wall_s"] for r in rows], "^--", color="C2", alpha=0.7, label=f"{parallel_workers}w wall (s)")
    ax2.set_ylabel("Wall time (s)")
    ax2.legend(loc="upper right")

    plt.title(
        f"Parallel benefit vs episode count (farms={farms}, total_servers={total_servers}, "
        f"num_jobs={num_jobs})"
    )
    plt.tight_layout()
    png_path = os.path.join(results_dir, "exp3_parallel_vs_episodes.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"已写入 {png_path}")
    if max(r["speedup_vs_serial"] for r in rows) < 1.05:
        print(
            "\n提示：episode 总时长仍偏短，进程启动/同步占主导，加速比 <1 属预期。"
            "请增大 --parallel-jobs 或 --parallel-sweep-episodes 中的最大值后重试。\n"
        )


def hier_marl_inference_scaling(
    n_values: list[int],
    m_per_farm: int,
    num_jobs: int,
    steps_per_point: int,
    results_dir: str,
) -> None:
    """Profile per-step inference time for HierMARL vs CommonActor at different N values.

    Uses randomly-initialized models — no trained weights required.
    Demonstrates that HierMARL actor inference scales linearly with N,
    while CommonActor actor scales similarly (both are O(N) for actors).
    The key architectural difference is the CRITIC dimension (see exp_critic_dim_analysis.py).
    """
    import time as _time
    import torch
    from env.cloud_scheduling_hier import CloudSchedulingEnvHier
    from schedulers.marl.hier_marl.HierMARL import HierMARL
    from schedulers.marl.common_actor.CommonActor import CommonActor

    rows: list[dict[str, Any]] = []

    for n in n_values:
        total_servers = n * m_per_farm
        env = CloudSchedulingEnvHier(
            num_jobs=num_jobs,
            num_server_farms=n,
            num_servers=total_servers,
        )
        obs, _ = env.reset(seed=42)

        dim_info: dict[str, Any] = {}
        for aid in env.agents:
            obs_space = env.observation_space(aid)
            dim_info[aid] = {
                "obs_shape": {k: s.shape for k, s in obs_space.spaces.items()},
                "action_dim": env.action_space(aid).n,
            }

        tmp_dir = os.path.join(results_dir, "_tmp_hier_scaling")
        os.makedirs(tmp_dir, exist_ok=True)

        hier = HierMARL(dim_info, capacity=1, batch_size=1,
                        actor_lr=3e-4, critic_lr=3e-4, res_dir=tmp_dir)
        ca   = CommonActor(dim_info, capacity=1, batch_size=1,
                           actor_lr=3e-4, critic_lr=3e-4, res_dir=tmp_dir)

        # Warm up
        for _ in range(5):
            hier.select_action(obs)
            ca.select_action(obs)

        # Time HierMARL inference
        t0 = _time.perf_counter()
        for _ in range(steps_per_point):
            hier.select_action(obs)
        hier_ms = (_time.perf_counter() - t0) / steps_per_point * 1000.0

        # Time CommonActor inference
        t0 = _time.perf_counter()
        for _ in range(steps_per_point):
            ca.select_action(obs)
        ca_ms = (_time.perf_counter() - t0) / steps_per_point * 1000.0

        rows.append({
            "N": n,
            "M": m_per_farm,
            "total_servers": total_servers,
            "hier_ms_per_step": round(hier_ms, 4),
            "ca_ms_per_step": round(ca_ms, 4),
            "hier_speedup": round(ca_ms / hier_ms, 4) if hier_ms > 0 else 0.0,
        })
        print(
            f"  N={n:3d} M={m_per_farm}  HierMARL={hier_ms:.3f}ms  "
            f"CommonActor={ca_ms:.3f}ms  speedup={ca_ms/hier_ms if hier_ms>0 else 0:.2f}x"
        )
        env.close()

    csv_path = os.path.join(results_dir, "exp3_hier_inference_scaling.csv")
    write_csv(csv_path, list(rows[0].keys()), rows)
    print(f"\nCSV 已写入: {csv_path}")

    ns   = [r["N"]              for r in rows]
    h_ms = [r["hier_ms_per_step"] for r in rows]
    c_ms = [r["ca_ms_per_step"]   for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(ns, h_ms, "o-",  color="C0", lw=2, label="HierMARL")
    ax.plot(ns, c_ms, "s--", color="C1", lw=2, label="CommonActor")
    ax.set_xlabel("Number of server farms (N)")
    ax.set_ylabel("Inference time per step (ms)")
    ax.set_title(f"Inference latency vs N  (M={m_per_farm} servers/farm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ns)

    ax = axes[1]
    sp = [r["hier_speedup"] for r in rows]
    ax.bar(ns, sp, color="C2", alpha=0.8, edgecolor="black")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of server farms (N)")
    ax.set_ylabel("Speedup (CommonActor / HierMARL)")
    ax.set_title("HierMARL inference speedup over CommonActor")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(ns)

    plt.tight_layout()
    png_path = os.path.join(results_dir, "exp3_hier_inference_scaling.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"图已写入: {png_path}")


def run_scalability_for_mode(
    scale_mode: str,
    args: argparse.Namespace,
    results_dir: str,
    hetero_weight: float,
    use_heterogeneity: bool,
) -> None:
    label, scale_grid = build_scale_grid(
        scale_mode,
        args.quick,
        args.fixed_farms,
        args.fixed_total_servers,
    )
    print(f"\n── 规模模式: {label} ──")
    print(f"  规模点 ({len(scale_grid)}): {scale_grid}\n")

    summary_rows: list[dict[str, Any]] = []
    for farms, total_servers in scale_grid:
        row = benchmark_scale(
            num_jobs=args.num_jobs,
            num_farms=farms,
            total_servers=total_servers,
            base_seed=args.seed,
            repeats=args.repeats,
            use_heterogeneity=use_heterogeneity,
            hetero_weight=hetero_weight,
            proportion_key=args.proportion,
        )
        summary_rows.append(row)
        print(
            f"farms={farms:2d} total_servers={total_servers:3d} | "
            f"episode {row['mean_episode_wall_s']:.4f}±{row['std_episode_wall_s']:.4f}s | "
            f"steps {row['mean_steps']:.1f}±{row['std_steps']:.1f} | "
            f"ms/step {row['ms_per_step_mean']:.3f}±{row.get('std_ms_per_step', 0):.3f}"
        )

    if label == "mixed":
        stem = os.path.join(results_dir, "exp3_scalability")
    else:
        stem = os.path.join(results_dir, f"exp3_scalability_{label}")
    csv_fields = list(summary_rows[0].keys())
    write_csv(f"{stem}.csv", csv_fields, summary_rows)
    print(f"\nCSV 已写入: {stem}.csv")
    plot_scalability(summary_rows, f"{stem}.png", scale_mode=label)
    print(f"图已写入: {stem}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="实验3：可扩展性测量")
    parser.add_argument("--quick", action="store_true", help="使用较小规模网格，便于快速试跑")
    parser.add_argument(
        "--scale-mode",
        type=str,
        default="mixed",
        choices=["mixed", "servers_only", "farms_only", "all"],
        help="mixed=默认组合网格; servers_only=固定 farm 扫总台数; farms_only=固定总台数扫 farm; all=三种各跑一遍",
    )
    parser.add_argument(
        "--fixed-farms",
        type=int,
        default=SERVERS_SWEEP_FIXED_FARMS,
        help="servers_only 模式下固定的 num_server_farms",
    )
    parser.add_argument(
        "--fixed-total-servers",
        type=int,
        default=FARMS_SWEEP_FIXED_TOTAL,
        help="farms_only 模式下固定的全集群总台数",
    )
    parser.add_argument(
        "--parallel-benchmark",
        action="store_true",
        help="多进程 rollout 加速比 vs 串行 for 循环",
    )
    parser.add_argument(
        "--parallel-workers",
        type=str,
        default="2,4,8",
        help="并行 worker 数列表（逗号分隔，不含串行基线 1），如 2,4,8",
    )
    parser.add_argument(
        "--parallel-episodes-sweep",
        action="store_true",
        help="扫描不同总 episode 数下固定 worker 数的加速比（见 --parallel-sweep-episodes）",
    )
    parser.add_argument(
        "--parallel-sweep-episodes",
        type=str,
        default="8,16,32,48,64",
        help="--parallel-episodes-sweep 时的 episode 数列表",
    )
    parser.add_argument(
        "--parallel-sweep-workers",
        type=int,
        default=4,
        help="--parallel-episodes-sweep 时使用的并行进程数",
    )
    parser.add_argument("--repeats", type=int, default=5, help="每个数据点重复次数（建议报告用 ≥5）")
    parser.add_argument("--seed", type=int, default=42, help="基准随机种子")
    parser.add_argument("--num-jobs", type=int, default=5, help="可扩展性实验的任务数")
    parser.add_argument(
        "--parallel-episodes",
        type=int,
        default=48,
        help="--parallel-benchmark 的总 episode 数",
    )
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=20,
        help="并行段专用 num_jobs（更长 episode 以摊销进程启动）",
    )
    parser.add_argument(
        "--parallel-farms",
        type=int,
        default=5,
        help="并行基准 / sweep 使用的 num_server_farms",
    )
    parser.add_argument(
        "--parallel-total-servers",
        type=int,
        default=50,
        help="并行基准 / sweep 使用的全集群总台数",
    )
    parser.add_argument(
        "--proportion",
        type=str,
        default="balanced",
        choices=list(PROPORTION_PRESETS.keys()),
        help="异构比例预设",
    )
    parser.add_argument(
        "--skip-scalability",
        action="store_true",
        help="跳过规模扫描，仅执行 --parallel-benchmark / --parallel-episodes-sweep / --hier-scaling",
    )
    parser.add_argument(
        "--hier-scaling",
        action="store_true",
        help="测量 HierMARL vs CommonActor 在不同 N 下的推理时延（随机初始化权重）",
    )
    parser.add_argument(
        "--hier-n-values",
        type=str,
        default="2,5,10,20,50",
        help="--hier-scaling 时扫描的 farm 数列表（逗号分隔）",
    )
    parser.add_argument(
        "--hier-m",
        type=int,
        default=6,
        help="--hier-scaling 时每个 farm 的服务器数（固定）",
    )
    parser.add_argument(
        "--hier-steps",
        type=int,
        default=200,
        help="--hier-scaling 每个数据点的推理次数（用于计时平均）",
    )
    args = parser.parse_args()

    if args.skip_scalability and not args.parallel_benchmark and not args.parallel_episodes_sweep and not args.hier_scaling:
        parser.error("--skip-scalability 需配合 --parallel-benchmark 或 --parallel-episodes-sweep")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    _ensure_results_dir(results_dir)

    hetero_weight = 0.3
    use_heterogeneity = True

    worker_list = [w for w in parse_int_list(args.parallel_workers) if w > 1]
    if not worker_list:
        worker_list = [2, 4]

    print("实验3：可扩展性（固定随机策略 + 异构预设）")
    print(f"  num_jobs={args.num_jobs}, repeats={args.repeats}, base_seed={args.seed}")
    print(f"  proportion={args.proportion}")

    modes_to_run: list[str]
    if args.scale_mode == "all":
        modes_to_run = ["mixed", "servers_only", "farms_only"]
    else:
        modes_to_run = [args.scale_mode]

    if not args.skip_scalability:
        for sm in modes_to_run:
            run_scalability_for_mode(sm, args, results_dir, hetero_weight, use_heterogeneity)
    else:
        print("已跳过规模扫描 (--skip-scalability)")

    print_amdahl_notes()

    if args.parallel_benchmark:
        print(
            f"\n并行 rollout 基准（episodes={args.parallel_episodes}, num_jobs={args.parallel_jobs}, "
            f"workers={worker_list}）…"
        )
        parallel_rollout_benchmark(
            num_jobs=args.parallel_jobs,
            farms=args.parallel_farms,
            total_servers=args.parallel_total_servers,
            base_seed=args.seed,
            total_episodes=args.parallel_episodes,
            worker_counts=worker_list,
            use_heterogeneity=use_heterogeneity,
            hetero_weight=hetero_weight,
            proportion_key=args.proportion,
            results_dir=results_dir,
        )

    if args.parallel_episodes_sweep:
        ep_list = parse_int_list(args.parallel_sweep_episodes)
        if len(ep_list) < 2:
            raise SystemExit("--parallel-sweep-episodes 至少需要 2 个值")
        print(
            f"\n并行 vs episode 数扫描（num_jobs={args.parallel_jobs}, workers="
            f"{args.parallel_sweep_workers}, episodes={ep_list}）…"
        )
        parallel_episodes_sweep(
            num_jobs=args.parallel_jobs,
            farms=args.parallel_farms,
            total_servers=args.parallel_total_servers,
            base_seed=args.seed,
            episode_counts=ep_list,
            parallel_workers=args.parallel_sweep_workers,
            use_heterogeneity=use_heterogeneity,
            hetero_weight=hetero_weight,
            proportion_key=args.proportion,
            results_dir=results_dir,
        )

    if args.hier_scaling:
        n_list = [int(x) for x in parse_int_list(args.hier_n_values) if x > 0]
        print(
            f"\nHierMARL vs CommonActor 推理时延扫描 "
            f"(N={n_list}, M={args.hier_m}, steps={args.hier_steps})…"
        )
        hier_marl_inference_scaling(
            n_values=n_list,
            m_per_farm=args.hier_m,
            num_jobs=args.num_jobs,
            steps_per_point=args.hier_steps,
            results_dir=results_dir,
        )


if __name__ == "__main__":
    main()
