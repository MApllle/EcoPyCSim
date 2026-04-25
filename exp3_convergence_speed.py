"""
Exp 3 – Convergence Speed Comparison: HierMARL vs CommonActor

Section 3.2: HierMARL's neighborhood critic reduces critic input dimension from
O(NM) to O(N+M), cutting gradient variance and accelerating convergence.

Both algorithms are trained for N_EPISODES episodes on the same environment
with N_SEEDS independent random seeds. Per-episode team average reward
(sum of all agents' reward / steps) is tracked. Final plot shows mean ± 1 std
across seeds, reproducing Fig 4.1 of the paper.

Two modes
---------
  (default)    Train both algorithms from scratch (N_SEEDS seeds each).
  --from-logs  Read existing reward.txt files instead of training.
               --hier-logs and --ca-logs accept comma-separated paths
               (one path per seed).

Usage
-----
  python exp3_convergence_speed.py                        # train from scratch
  python exp3_convergence_speed.py --episodes 500 --seeds 5
  python exp3_convergence_speed.py \\
      --from-logs \\
      --hier-logs results/hier_run1/reward.txt,results/hier_run2/reward.txt \\
      --ca-logs   results/ca_run1/reward.txt,results/ca_run2/reward.txt

Environment variables
---------------------
  NUM_FARMS   (default 4)   — number of server farms (N)
  NUM_SERVERS (default 20)  — total servers (must be divisible by NUM_FARMS; M = NUM_SERVERS/NUM_FARMS)
  NUM_JOBS    (default 100) — jobs per episode
  EPISODES    (default 500) — training episodes per seed
  N_SEEDS     (default 5)   — number of independent seeds
  WINDOW      (default 20)  — sliding-window size for smoothing
  CONV_THRESH (default 0.80)— fraction of peak reward for convergence metric

Outputs
-------
  results/exp3_convergence_speed.csv   — per-episode mean/std for both algorithms
  results/exp3_convergence_speed.png   — reward curves with mean ± 1 std shading
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Config (overridable via env vars / CLI) ────────────────────────────────────
NUM_FARMS    = int(os.getenv("NUM_FARMS",   "4"))
NUM_SERVERS  = int(os.getenv("NUM_SERVERS", "20"))   # M = NUM_SERVERS / NUM_FARMS = 5
NUM_JOBS     = int(os.getenv("NUM_JOBS",    "100"))
N_EPISODES   = int(os.getenv("EPISODES",   "500"))
N_SEEDS      = int(os.getenv("N_SEEDS",    "5"))
WINDOW       = int(os.getenv("WINDOW",     "20"))
CONV_THRESH  = float(os.getenv("CONV_THRESH", "0.80"))


# ── Utility Functions ──────────────────────────────────────────────────────────

def sliding_window(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = arr[start: i + 1].mean()
    return out


def episodes_to_convergence(arr: np.ndarray, threshold_frac: float = CONV_THRESH,
                            window: int = WINDOW) -> int:
    """First episode where smoothed reward reaches threshold_frac × peak."""
    smoothed = sliding_window(arr, window)
    peak = smoothed.max()
    idxs = np.where(smoothed >= threshold_frac * peak)[0]
    return int(idxs[0]) + 1 if len(idxs) > 0 else len(arr)


def parse_reward_log(path: str) -> np.ndarray:
    """Parse a reward.txt or console log file; return avg_reward_per_step per episode.

    Supports two line formats:
      reward.txt:  episode=N, ..., avg_reward_per_step=X, ...
      console log: [Algo] ep N/M  ... avg_step=X  ...
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # reward.txt format
            m = re.search(r"avg_reward_per_step=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", line)
            if m:
                records.append(float(m.group(1)))
                continue
            # console log format: avg_step=X
            m = re.search(r"\bavg_step=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", line)
            if m:
                records.append(float(m.group(1)))
    return np.array(records, dtype=np.float32)


# ── Training ───────────────────────────────────────────────────────────────────

def _build_dim_info(env) -> dict:
    dim_info = {}
    for aid in env.agents:
        obs_space = env.observation_space(aid)
        dim_info[aid] = {
            "obs_shape": {k: s.shape for k, s in obs_space.spaces.items()},
            "action_dim": env.action_space(aid).n,
        }
    return dim_info


def train_one_seed(algo: str, seed: int, n_episodes: int, res_dir: str) -> np.ndarray:
    """Train HierMARL or CommonActor for n_episodes with a given seed.

    Returns per-episode avg_reward_per_step array of shape (n_episodes,).
    """
    from env.cloud_scheduling_hier import CloudSchedulingEnvHier

    np.random.seed(seed)

    capacity        = 50_000
    batch_size      = 256
    actor_lr        = 3e-4
    critic_lr       = 3e-4
    gamma           = 0.9
    tau             = 0.02
    learn_interval  = 5
    eps_start       = 1.0
    eps_end         = 0.05
    random_steps    = max(int(NUM_JOBS * 2), int(NUM_JOBS * n_episodes * 0.1))
    eps_decay_steps = max(int(NUM_JOBS * n_episodes * 0.9), 1)

    env = CloudSchedulingEnvHier(
        num_jobs=NUM_JOBS,
        num_server_farms=NUM_FARMS,
        num_servers=NUM_SERVERS,
    )
    env.reset(seed=seed)
    dim_info = _build_dim_info(env)

    os.makedirs(res_dir, exist_ok=True)

    if algo == "hier_marl":
        from schedulers.marl.hier_marl.HierMARL import HierMARL
        agent = HierMARL(dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir)
    elif algo == "common_actor":
        import contextlib, io
        from schedulers.marl.common_actor.CommonActor import CommonActor
        with contextlib.redirect_stdout(io.StringIO()):
            agent = CommonActor(dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    reward_log_path = os.path.join(res_dir, "reward.txt")
    ep_rewards = np.zeros(n_episodes, dtype=np.float32)
    global_step = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        agent_reward = {aid: 0.0 for aid in env.agents}
        steps = 0

        while True:
            steps += 1
            global_step += 1

            if global_step <= random_steps:
                epsilon = 1.0
            else:
                progress = min(1.0, (global_step - random_steps) / eps_decay_steps)
                epsilon = eps_start - (eps_start - eps_end) * progress

            if np.random.rand() < epsilon:
                action = {aid: env.action_space(aid).sample() for aid in env.agents}
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = {a: terminated[a] or truncated[a] for a in env.agents}

            if algo == "hier_marl":
                sel_farm = info.get("global", {}).get("selected_farm_id", 0)
                agent.add(obs, action, reward, next_obs, done, sel_farm)
            else:
                agent.add(obs, action, reward, next_obs, done)

            for a, r in reward.items():
                agent_reward[a] += r
            obs = next_obs

            if global_step > random_steps and global_step % learn_interval == 0:
                agent.learn(gamma)
                agent.update_target(tau)

            if all(done.values()):
                break

        sum_r = sum(agent_reward.values())
        avg_r = sum_r / max(steps, 1)
        ep_rewards[ep] = avg_r

        g_info = info.get("global", {})
        rejected  = g_info.get("rejected_tasks_count", 0)
        completed = len(g_info.get("completed_job_ids", []))

        local_keys = sorted([k for k in agent_reward if k.startswith("local_")])
        local_str = " ".join(f"{k}={agent_reward[k]:.4f}" for k in local_keys)
        with open(reward_log_path, "a") as f:
            f.write(
                f"episode={ep + 1}, steps={steps}, epsilon={epsilon:.4f}, "
                f"global_reward={agent_reward.get('global', 0):.4f}, {local_str}, "
                f"sum_reward={sum_r:.4f}, avg_reward_per_step={avg_r:.4f}, "
                f"rejected_tasks={rejected}, completed_jobs={completed}\n"
            )

        if (ep + 1) % 50 == 0 or ep == 0:
            print(
                f"  [{algo}|seed={seed}] ep {ep+1:4d}/{n_episodes}  "
                f"eps={epsilon:.3f}  avg_r/step={avg_r:.4f}  "
                f"reject={rejected}  done={completed}"
            )

    agent.save(ep_rewards)
    env.close()
    return ep_rewards


def train_multi_seed(
    algo: str,
    n_episodes: int,
    n_seeds: int,
    results_base: str,
    timestamp: str,
) -> np.ndarray:
    """Train algo for n_seeds seeds; return reward matrix (n_seeds, n_episodes)."""
    all_rewards = []
    for seed in range(n_seeds):
        res_dir = os.path.join(results_base, f"exp3_{algo}_seed{seed}_{timestamp}")
        print(f"\n[Exp3] Training {algo} seed={seed} → {res_dir}")
        t0 = time.perf_counter()
        rewards = train_one_seed(algo, seed, n_episodes, res_dir)
        elapsed = time.perf_counter() - t0
        print(f"  Seed {seed} done in {elapsed:.1f}s")
        all_rewards.append(rewards)
    return np.stack(all_rewards, axis=0)  # (n_seeds, n_episodes)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_convergence_multiseed(
    hier_mat: np.ndarray,
    ca_mat: np.ndarray,
    out_png: str,
    n_episodes: int,
    window: int = WINDOW,
    conv_thresh: float = CONV_THRESH,
) -> None:
    """Plot mean ± 1 std reward curves and convergence bar chart.

    Args:
        hier_mat: (n_seeds, n_episodes) reward matrix for HierMARL
        ca_mat:   (n_seeds, n_episodes) reward matrix for CommonActor
    """
    def smooth_mat(mat):
        return np.stack([sliding_window(row, window) for row in mat], axis=0)

    hier_raw_mean  = hier_mat.mean(axis=0)
    hier_raw_std   = hier_mat.std(axis=0)
    hier_sm_mat    = smooth_mat(hier_mat)
    hier_sm_mean   = hier_sm_mat.mean(axis=0)
    hier_sm_std    = hier_sm_mat.std(axis=0)

    ca_raw_mean  = ca_mat.mean(axis=0)
    ca_raw_std   = ca_mat.std(axis=0)
    ca_sm_mat    = smooth_mat(ca_mat)
    ca_sm_mean   = ca_sm_mat.mean(axis=0)
    ca_sm_std    = ca_sm_mat.std(axis=0)

    xs = np.arange(1, n_episodes + 1)

    hier_conv = episodes_to_convergence(hier_sm_mean, conv_thresh, window=1)
    ca_conv   = episodes_to_convergence(ca_sm_mean,   conv_thresh, window=1)
    speedup   = ca_conv / hier_conv if hier_conv > 0 else float("inf")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: reward curves ────────────────────────────────────────────────
    ax = axes[0]
    # raw (faint)
    ax.plot(xs, hier_raw_mean, alpha=0.20, color="C0", lw=1)
    ax.plot(xs, ca_raw_mean,   alpha=0.20, color="C1", lw=1)
    # smoothed mean
    ax.plot(xs, hier_sm_mean, color="C0", lw=2,
            label=f"HierMARL (std={hier_sm_std.mean():.2f})")
    ax.plot(xs, ca_sm_mean,   color="C1", lw=2,
            label=f"CommonActor (std={ca_sm_std.mean():.2f})")
    # ±1 std band
    ax.fill_between(xs,
                    hier_sm_mean - hier_sm_std,
                    hier_sm_mean + hier_sm_std,
                    color="C0", alpha=0.15)
    ax.fill_between(xs,
                    ca_sm_mean - ca_sm_std,
                    ca_sm_mean + ca_sm_std,
                    color="C1", alpha=0.15)

    # convergence markers
    ax.axvline(x=hier_conv, color="C0", linestyle="--", lw=1, alpha=0.7)
    ax.axvline(x=ca_conv,   color="C1", linestyle="--", lw=1, alpha=0.7)
    ax.text(hier_conv + 3, ax.get_ylim()[0] + 0.02,
            f"ep{hier_conv}", color="C0", fontsize=7)
    ax.text(ca_conv + 3, ax.get_ylim()[0] + 0.02,
            f"ep{ca_conv}", color="C1", fontsize=7)

    m_per_farm = NUM_SERVERS // NUM_FARMS
    ax.set_xlabel("Episode")
    ax.set_ylabel("Team avg reward per step")
    ax.set_title(
        f"Training reward curves  (N={NUM_FARMS}, M={m_per_farm}, "
        f"{hier_mat.shape[0]} seeds, window={window})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Right: convergence bar chart ───────────────────────────────────────
    ax = axes[1]
    labels = ["HierMARL", "CommonActor"]
    vals   = [hier_conv, ca_conv]
    bars = ax.bar(labels, vals, color=["C0", "C1"], alpha=0.8, edgecolor="black")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Episodes to convergence")
    ax.set_title(
        f"Episodes to {int(conv_thresh * 100)}% peak reward\n"
        f"(HierMARL speedup: {speedup:.1f}×)"
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[Exp3] Saved plot: {out_png}")


# ── CSV Output ─────────────────────────────────────────────────────────────────

def save_csv(
    hier_mat: np.ndarray,
    ca_mat: np.ndarray,
    results_dir: str,
    window: int = WINDOW,
) -> None:
    n_ep = max(hier_mat.shape[1], ca_mat.shape[1])

    def _agg(mat, ep):
        if ep < mat.shape[1]:
            col = mat[:, ep]
            return round(float(col.mean()), 6), round(float(col.std()), 6)
        return "", ""

    hier_sm = np.stack([sliding_window(r, window) for r in hier_mat])
    ca_sm   = np.stack([sliding_window(r, window) for r in ca_mat])

    rows = []
    for i in range(n_ep):
        h_raw_mean, h_raw_std = _agg(hier_mat, i)
        c_raw_mean, c_raw_std = _agg(ca_mat, i)
        h_sm_mean,  h_sm_std  = _agg(hier_sm, i)
        c_sm_mean,  c_sm_std  = _agg(ca_sm, i)
        rows.append({
            "episode":         i + 1,
            "hier_raw_mean":   h_raw_mean,
            "hier_raw_std":    h_raw_std,
            "hier_smooth_mean": h_sm_mean,
            "hier_smooth_std":  h_sm_std,
            "ca_raw_mean":     c_raw_mean,
            "ca_raw_std":      c_raw_std,
            "ca_smooth_mean":  c_sm_mean,
            "ca_smooth_std":   c_sm_std,
        })

    csv_path = os.path.join(results_dir, "exp3_convergence_speed.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Exp3] Saved data: {csv_path}")


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(hier_mat: np.ndarray, ca_mat: np.ndarray, window: int = WINDOW) -> None:
    hier_sm  = np.stack([sliding_window(r, window) for r in hier_mat])
    ca_sm    = np.stack([sliding_window(r, window) for r in ca_mat])
    hier_mean = hier_sm.mean(axis=0)
    ca_mean   = ca_sm.mean(axis=0)

    hier_conv = episodes_to_convergence(hier_mean, CONV_THRESH, window=1)
    ca_conv   = episodes_to_convergence(ca_mean,   CONV_THRESH, window=1)
    speedup   = ca_conv / hier_conv if hier_conv > 0 else float("inf")

    hier_peak_mean = hier_sm.max(axis=1).mean()
    ca_peak_mean   = ca_sm.max(axis=1).mean()
    hier_final_std = hier_sm[:, -1].std()
    ca_final_std   = ca_sm[:, -1].std()

    print(f"\n{'─'*60}")
    print(f"{'Algorithm':20s}  {'Conv ep':>8}  {'Peak reward':>12}  {'Final std':>10}")
    print(f"{'─'*60}")
    print(f"{'HierMARL':20s}  {hier_conv:>8d}  {hier_peak_mean:>12.4f}  {hier_final_std:>10.4f}")
    print(f"{'CommonActor':20s}  {ca_conv:>8d}  {ca_peak_mean:>12.4f}  {ca_final_std:>10.4f}")
    print(f"\nHierMARL convergence speedup: {speedup:.2f}×  "
          f"(CommonActor takes {ca_conv} ep, HierMARL {hier_conv} ep)")
    print(f"{'─'*60}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp3: Convergence speed comparison — HierMARL vs CommonActor"
    )
    parser.add_argument(
        "--from-logs", action="store_true",
        help="Read existing reward.txt files instead of training from scratch",
    )
    parser.add_argument(
        "--hier-logs", type=str, default="",
        help="Comma-separated paths to HierMARL reward.txt files (one per seed)",
    )
    parser.add_argument(
        "--ca-logs", type=str, default="",
        help="Comma-separated paths to CommonActor reward.txt files (one per seed)",
    )
    parser.add_argument(
        "--episodes", type=int, default=N_EPISODES,
        help=f"Episodes per seed when training from scratch (default {N_EPISODES})",
    )
    parser.add_argument(
        "--seeds", type=int, default=N_SEEDS,
        help=f"Number of seeds when training from scratch (default {N_SEEDS})",
    )
    parser.add_argument(
        "--window", type=int, default=WINDOW,
        help=f"Sliding window for smoothing (default {WINDOW})",
    )
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    m_per_farm = NUM_SERVERS // NUM_FARMS
    print(f"\n{'='*60}")
    print(f"Exp3 — Convergence Speed Comparison")
    print(f"  N_FARMS={NUM_FARMS}, M={m_per_farm} servers/farm, "
          f"NUM_JOBS={NUM_JOBS}")
    print(f"{'='*60}")

    if args.from_logs:
        if not args.hier_logs or not args.ca_logs:
            parser.error(
                "--from-logs requires --hier-logs and --ca-logs "
                "(comma-separated paths to reward.txt files)"
            )
        hier_paths = [p.strip() for p in args.hier_logs.split(",") if p.strip()]
        ca_paths   = [p.strip() for p in args.ca_logs.split(",")   if p.strip()]

        print(f"Reading {len(hier_paths)} HierMARL log(s)...")
        hier_rewards = [parse_reward_log(p) for p in hier_paths]

        print(f"Reading {len(ca_paths)} CommonActor log(s)...")
        ca_rewards = [parse_reward_log(p) for p in ca_paths]

        # align both to the same length (pad shorter with last value)
        n_ep = max(
            max(len(r) for r in hier_rewards),
            max(len(r) for r in ca_rewards),
        )
        hier_mat = np.stack([
            np.pad(r, (0, n_ep - len(r)), constant_values=r[-1]) for r in hier_rewards
        ])
        ca_mat = np.stack([
            np.pad(r, (0, n_ep - len(r)), constant_values=r[-1]) for r in ca_rewards
        ])
        n_episodes_actual = n_ep

    else:
        n_episodes_actual = args.episodes
        n_seeds = args.seeds
        ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        print(f"\n=== Training HierMARL ({n_seeds} seeds × {n_episodes_actual} episodes) ===")
        t0 = time.perf_counter()
        hier_mat = train_multi_seed("hier_marl", n_episodes_actual, n_seeds, results_dir, ts)
        print(f"HierMARL total training time: {time.perf_counter() - t0:.1f}s")

        print(f"\n=== Training CommonActor ({n_seeds} seeds × {n_episodes_actual} episodes) ===")
        t0 = time.perf_counter()
        ca_mat = train_multi_seed("common_actor", n_episodes_actual, n_seeds, results_dir, ts)
        print(f"CommonActor total training time: {time.perf_counter() - t0:.1f}s")

    print_summary(hier_mat, ca_mat, window=args.window)
    save_csv(hier_mat, ca_mat, results_dir, window=args.window)

    png_path = os.path.join(results_dir, "exp3_convergence_speed.png")
    plot_convergence_multiseed(
        hier_mat, ca_mat,
        out_png=png_path,
        n_episodes=n_episodes_actual,
        window=args.window,
        conv_thresh=CONV_THRESH,
    )

    print("\n[Exp3] Done.")


if __name__ == "__main__":
    main()
