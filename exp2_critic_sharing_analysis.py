"""
Exp 2 – Critic Neighborhood Sharing Effect Analysis

Section 3.1: Critic input dimension and inference latency analysis.

Part 1  (Table 4.2): Critic input dimension vs number of server farms N (M=5 fixed).
         Three strategies compared:
           - 集中式 critic (MADDPG/CommonActor): input = all agents' obs + acts
           - HierMARL global critic: input = global obs+act + one selected local obs+act
           - HierMARL local critic: input = local obs+act only (constant wrt N)

Part 2  (Table 4.3): Per-decision inference latency for HierMARL vs CommonActor
         at N ∈ {2, 4, 8, 16, 32, 64}, confirming that actor-only inference
         makes both algorithms' latency statistically indistinguishable.

Usage
-----
  python exp2_critic_sharing_analysis.py               # both parts
  python exp2_critic_sharing_analysis.py --dim-only    # Table 4.2 only (fast)
  python exp2_critic_sharing_analysis.py --lat-only    # Table 4.3 only
  python exp2_critic_sharing_analysis.py --trials 200  # fewer latency trials

Outputs
-------
  results/exp2_critic_dim.csv            — Table 4.2 data
  results/exp2_critic_dim.png            — dimension growth plot (linear + log scale)
  results/exp2_inference_latency.csv     — Table 4.3 data
  results/exp2_inference_latency.png     — latency comparison plot
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ── Dimension Formulas ─────────────────────────────────────────────────────────
# Derived from CloudSchedulingEnvHier observation spaces:
#   global obs = farm_avg_cpu_util(N) + farm_active_ratio(N) + farm_avg_efficiency(N)
#              + task_cpu(1) + task_ram(1) + task_deadline(1) + wall_time(1)  = 3N + 4
#   local  obs = cpus_utilization(M) + efficiency_tiers(M)
#              + task_cpu(1) + task_ram(1) + task_deadline(1) + wall_time(1) + is_selected(1)  = 2M + 5
#   global act = N   (which farm)
#   local  act = M   (which server within farm)

def central_critic_dim(n: int, m: int) -> int:
    """Centralized critic (MADDPG / CommonActor): concat all N+1 agents' obs + acts."""
    return (3 * n + 4) + n * (2 * m + 5) + n + n * m


def hier_global_critic_dim(n: int, m: int) -> int:
    """HierMARL global critic: global obs+act + ONE selected local obs+act."""
    return (3 * n + 4) + n + (2 * m + 5) + m


def hier_local_critic_dim(n: int, m: int) -> int:
    """HierMARL local critic: local obs+act only, completely independent of N."""
    return (2 * m + 5) + m


# ── Synthetic Dim-Info and Obs ─────────────────────────────────────────────────

def make_dim_info(n: int, m: int) -> dict:
    """Build dim_info matching CloudSchedulingEnvHier without instantiating the env."""
    dim_info: dict = {
        "global": {
            "obs_shape": {
                "farm_avg_cpu_util":   (n,),
                "farm_active_ratio":   (n,),
                "farm_avg_efficiency": (n,),
                "task_cpu":      (1,),
                "task_ram":      (1,),
                "task_deadline": (1,),
                "wall_time":     (1,),
            },
            "action_dim": n,
        }
    }
    for i in range(n):
        dim_info[f"local_{i}"] = {
            "obs_shape": {
                "cpus_utilization": (m,),
                "efficiency_tiers": (m,),
                "task_cpu":      (1,),
                "task_ram":      (1,),
                "task_deadline": (1,),
                "wall_time":     (1,),
                "is_selected":   (1,),
            },
            "action_dim": m,
        }
    return dim_info


def make_fake_obs(n: int, m: int) -> dict:
    """Random obs with the correct structure for timing purposes."""
    obs: dict = {
        "global": {
            "farm_avg_cpu_util":   np.random.rand(n).astype(np.float32),
            "farm_active_ratio":   np.random.rand(n).astype(np.float32),
            "farm_avg_efficiency": np.random.rand(n).astype(np.float32),
            "task_cpu":      np.random.rand(1).astype(np.float32),
            "task_ram":      np.random.rand(1).astype(np.float32),
            "task_deadline": np.random.rand(1).astype(np.float32),
            "wall_time":     np.random.rand(1).astype(np.float32),
        }
    }
    for i in range(n):
        obs[f"local_{i}"] = {
            "cpus_utilization": np.random.rand(m).astype(np.float32),
            "efficiency_tiers": np.random.rand(m).astype(np.float32),
            "task_cpu":      np.random.rand(1).astype(np.float32),
            "task_ram":      np.random.rand(1).astype(np.float32),
            "task_deadline": np.random.rand(1).astype(np.float32),
            "wall_time":     np.random.rand(1).astype(np.float32),
            "is_selected":   np.zeros(1, dtype=np.float32),
        }
    return obs


# ── Part 1: Critic Dimension Analysis ─────────────────────────────────────────

def run_dimension_analysis(
    n_values: list[int],
    m: int,
    results_dir: str,
) -> list[dict]:
    rows = []
    for n in n_values:
        row = {
            "N": n,
            "M": m,
            "central_critic":      central_critic_dim(n, m),
            "hier_global_critic":  hier_global_critic_dim(n, m),
            "hier_local_critic":   hier_local_critic_dim(n, m),
        }
        rows.append(row)

    csv_path = os.path.join(results_dir, "exp2_critic_dim.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Exp2-Part1] Saved: {csv_path}")

    print(f"\nTable 4.2 — Critic input dimension vs N  (M={m} servers/farm fixed)")
    print(f"{'N':>6}  {'集中式 critic':>16}  {'HierMARL global':>16}  {'HierMARL local':>16}")
    print("-" * 62)
    for r in rows:
        ratio = r["central_critic"] / r["hier_global_critic"]
        print(
            f"{r['N']:>6}  {r['central_critic']:>16}  "
            f"{r['hier_global_critic']:>16}  {r['hier_local_critic']:>16}"
            f"   (central/global = {ratio:.1f}×)"
        )

    _plot_dimension(rows, m, results_dir)
    return rows


def _plot_dimension(rows: list[dict], m: int, results_dir: str) -> None:
    ns = [r["N"] for r in rows]
    central = [r["central_critic"] for r in rows]
    hier_g  = [r["hier_global_critic"] for r in rows]
    hier_l  = [r["hier_local_critic"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, log in zip(axes, [False, True]):
        plot = ax.semilogy if log else ax.plot
        plot(ns, central, "o-",  color="C0", lw=2, label="Centralized critic (MADDPG/CommonActor)")
        plot(ns, hier_g,  "^--", color="C2", lw=2, label="HierMARL global critic")
        plot(ns, hier_l,  "s:",  color="C3", lw=2, label="HierMARL local critic")
        ax.set_xlabel("Number of server farms N")
        ax.set_ylabel("Critic input dimension" + (" (log)" if log else ""))
        ax.set_title(f"Critic input dim vs N  (M={m})" + (" — log scale" if log else ""))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both" if log else "major")
        ax.set_xticks(ns)

    local_dim = hier_local_critic_dim(1, m)
    axes[1].axhline(y=local_dim, color="C3", linestyle=":", alpha=0.5)
    axes[1].text(ns[0], local_dim * 1.2, f"local = {local_dim} (const)",
                 color="C3", fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(results_dir, "exp2_critic_dim.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[Exp2-Part1] Saved: {png_path}")


# ── Part 2: Inference Latency ──────────────────────────────────────────────────

def _init_hier_marl(dim_info: dict) -> object:
    from schedulers.marl.hier_marl.HierMARL import HierMARL
    return HierMARL(
        dim_info, capacity=1, batch_size=1,
        actor_lr=3e-4, critic_lr=3e-4,
        res_dir="/tmp",
        device=torch.device("cpu"),
    )


def _init_common_actor(dim_info: dict) -> object:
    import contextlib, io
    from schedulers.marl.common_actor.CommonActor import CommonActor
    # suppress the "[CommonActor] critic input dim = ..." print
    with contextlib.redirect_stdout(io.StringIO()):
        agent = CommonActor(
            dim_info, capacity=1, batch_size=1,
            actor_lr=3e-4, critic_lr=3e-4,
            res_dir="/tmp",
            device=torch.device("cpu"),
        )
    return agent


def measure_latency(algo: str, n: int, m: int, n_trials: int) -> tuple[float, float]:
    """Return (mean_ms, std_ms) for one scheduling decision with n_trials samples."""
    dim_info = make_dim_info(n, m)
    obs = make_fake_obs(n, m)

    if algo == "hier_marl":
        agent = _init_hier_marl(dim_info)
    else:
        agent = _init_common_actor(dim_info)

    # warm-up (JIT, cache)
    for _ in range(max(20, n_trials // 20)):
        agent.select_action(obs)

    times_ms: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        agent.select_action(obs)
        times_ms.append((time.perf_counter() - t0) * 1_000)

    return float(np.mean(times_ms)), float(np.std(times_ms))


def run_latency_analysis(
    n_values: list[int],
    m: int,
    n_trials: int,
    results_dir: str,
) -> list[dict]:
    rows = []
    for n in n_values:
        print(f"  N={n:3d}: measuring HierMARL latency ({n_trials} trials)...", end=" ", flush=True)
        h_mean, h_std = measure_latency("hier_marl", n, m, n_trials)
        print(f"{h_mean:.2f} ± {h_std:.2f} ms")

        print(f"         measuring CommonActor latency ({n_trials} trials)...", end=" ", flush=True)
        c_mean, c_std = measure_latency("common_actor", n, m, n_trials)
        print(f"{c_mean:.2f} ± {c_std:.2f} ms")

        rel_diff = (c_mean - h_mean) / c_mean * 100 if c_mean > 0 else 0.0
        rows.append({
            "N":            n,
            "M":            m,
            "hier_mean_ms": round(h_mean, 2),
            "hier_std_ms":  round(h_std, 2),
            "ca_mean_ms":   round(c_mean, 2),
            "ca_std_ms":    round(c_std, 2),
            "rel_diff_pct": round(rel_diff, 1),
        })

    csv_path = os.path.join(results_dir, "exp2_inference_latency.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Exp2-Part2] Saved: {csv_path}")

    print(f"\nTable 4.3 — Inference latency per scheduling decision  (M={m}, {n_trials} trials)")
    print(f"{'N':>5}  {'HierMARL (ms)':>18}  {'CommonActor (ms)':>18}  {'Rel diff (%)':>14}")
    print("-" * 62)
    for r in rows:
        print(
            f"{r['N']:>5}  "
            f"{r['hier_mean_ms']:>6.2f} ± {r['hier_std_ms']:>5.2f}    "
            f"{r['ca_mean_ms']:>6.2f} ± {r['ca_std_ms']:>5.2f}    "
            f"{r['rel_diff_pct']:>+.1f}%"
        )

    _plot_latency(rows, m, results_dir)
    return rows


def _plot_latency(rows: list[dict], m: int, results_dir: str) -> None:
    ns      = [r["N"] for r in rows]
    h_mean  = np.array([r["hier_mean_ms"] for r in rows])
    h_std   = np.array([r["hier_std_ms"]  for r in rows])
    c_mean  = np.array([r["ca_mean_ms"]   for r in rows])
    c_std   = np.array([r["ca_std_ms"]    for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: latency curves with error bars
    ax = axes[0]
    ax.errorbar(ns, h_mean, yerr=h_std, fmt="o-",  color="C0", lw=2,
                capsize=4, label="HierMARL")
    ax.errorbar(ns, c_mean, yerr=c_std, fmt="s--", color="C1", lw=2,
                capsize=4, label="CommonActor")
    ax.set_xlabel("Number of server farms N")
    ax.set_ylabel("Inference latency (ms)")
    ax.set_title(f"Per-decision inference latency (M={m})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ns)

    # Right: relative difference
    ax = axes[1]
    rel = [r["rel_diff_pct"] for r in rows]
    colors = ["C1" if v >= 0 else "C0" for v in rel]
    ax.bar(ns, rel, color=colors, alpha=0.7, edgecolor="black",
           width=[v * 0.4 for v in ns])
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(2.5,  color="gray", lw=1, linestyle="--", alpha=0.7)
    ax.axhline(-2.5, color="gray", lw=1, linestyle="--", alpha=0.7, label="±2.5% band")
    ax.set_xlabel("Number of server farms N")
    ax.set_ylabel("Relative latency diff: (CA − Hier) / CA  [%]")
    ax.set_title("Latency difference — CommonActor vs HierMARL")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(ns)

    plt.tight_layout()
    png_path = os.path.join(results_dir, "exp2_inference_latency.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[Exp2-Part2] Saved: {png_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp2: Critic neighborhood sharing effect (dimensions + latency)"
    )
    parser.add_argument(
        "--m", type=int, default=5,
        help="Servers per farm, fixed across all N (default: 5)",
    )
    parser.add_argument(
        "--n-values-dim", type=str, default="2,4,8,16,32,64,128",
        help="N values for dimension table (Table 4.2)",
    )
    parser.add_argument(
        "--n-values-lat", type=str, default="2,4,8,16,32,64",
        help="N values for latency measurement (Table 4.3)",
    )
    parser.add_argument(
        "--trials", type=int, default=1000,
        help="Number of timing trials per (N, algo) pair",
    )
    parser.add_argument(
        "--dim-only", action="store_true",
        help="Only run Part 1 (dimension analysis)",
    )
    parser.add_argument(
        "--lat-only", action="store_true",
        help="Only run Part 2 (latency measurement)",
    )
    args = parser.parse_args()

    m = args.m
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    run_dim = not args.lat_only
    run_lat = not args.dim_only

    if run_dim:
        n_dim = [int(x) for x in args.n_values_dim.split(",") if x.strip()]
        print(f"\n{'='*60}")
        print(f"Exp2 Part 1 — Critic Dimension Analysis  (M={m})")
        print(f"N values: {n_dim}")
        print(f"{'='*60}")
        run_dimension_analysis(n_dim, m, results_dir)

    if run_lat:
        n_lat = [int(x) for x in args.n_values_lat.split(",") if x.strip()]
        print(f"\n{'='*60}")
        print(f"Exp2 Part 2 — Inference Latency Measurement  (M={m}, {args.trials} trials each)")
        print(f"N values: {n_lat}")
        print(f"{'='*60}")
        run_latency_analysis(n_lat, m, args.trials, results_dir)

    print("\n[Exp2] Done.")


if __name__ == "__main__":
    main()
