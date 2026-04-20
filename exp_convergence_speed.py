"""
Convergence speed comparison: HierMARL vs CommonActor (reproducing paper Fig.4).

Two modes:
  --from-logs   Read existing reward.txt files; paths set via HIER_LOG / CA_LOG env vars
  (default)     Train both algorithms from scratch for N episodes on the same small env
                and compare per-episode avg_reward_per_step.

Output
------
  results/exp_convergence_speed.png   — sliding-window reward curves
  results/exp_convergence_speed.csv   — per-episode data + convergence table
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

# ── config ─────────────────────────────────────────────────────────────────────
NUM_JOBS         = int(os.getenv("NUM_JOBS",   "30"))
NUM_FARMS        = int(os.getenv("NUM_FARMS",  "2"))
NUM_SERVERS      = int(os.getenv("NUM_SERVERS","6"))   # must be divisible by NUM_FARMS
N_EPISODES       = int(os.getenv("EPISODES",  "300"))
WINDOW           = int(os.getenv("WINDOW",    "20"))
CONV_THRESHOLD   = float(os.getenv("CONV_THRESH", "0.80"))  # fraction of peak reward

HIER_LOG_DEFAULT = os.getenv("HIER_LOG", "")
CA_LOG_DEFAULT   = os.getenv("CA_LOG",   "")

# ── reward.txt parser ──────────────────────────────────────────────────────────

def parse_reward_log(path: str) -> list[dict]:
    """Parse reward.txt lines into list of dicts with numeric fields."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec: dict[str, float | int | str] = {}
            for kv in line.split(","):
                kv = kv.strip()
                m = re.match(r"^(\w+)=(.+)$", kv)
                if not m:
                    continue
                key, val = m.group(1), m.group(2).strip()
                try:
                    rec[key] = float(val)
                except ValueError:
                    rec[key] = val
            if "episode" in rec and "avg_reward_per_step" in rec:
                records.append(rec)
    return records


def records_to_reward_array(records: list[dict]) -> np.ndarray:
    return np.array([r["avg_reward_per_step"] for r in records], dtype=np.float32)


# ── sliding window ─────────────────────────────────────────────────────────────

def sliding_window(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = arr[start : i + 1].mean()
    return out


# ── convergence metric ─────────────────────────────────────────────────────────

def episodes_to_convergence(arr: np.ndarray, threshold_frac: float = 0.80) -> int:
    """Return first episode where sliding-window reward exceeds threshold_frac × peak."""
    smoothed = sliding_window(arr, WINDOW)
    peak = smoothed.max()
    target = threshold_frac * peak
    idxs = np.where(smoothed >= target)[0]
    return int(idxs[0]) + 1 if len(idxs) > 0 else len(arr)


# ── training helper ────────────────────────────────────────────────────────────

def _build_dim_info(env):
    dim_info = {}
    for aid in env.agents:
        obs_space = env.observation_space(aid)
        dim_info[aid] = {
            "obs_shape": {k: s.shape for k, s in obs_space.spaces.items()},
            "action_dim": env.action_space(aid).n,
        }
    return dim_info


def train_algorithm(algo: str, n_episodes: int, res_dir: str) -> np.ndarray:
    """Train HierMARL or CommonActor for n_episodes; return per-episode avg_reward_per_step."""
    from env.cloud_scheduling_hier import CloudSchedulingEnvHier

    env = CloudSchedulingEnvHier(
        num_jobs=NUM_JOBS,
        num_server_farms=NUM_FARMS,
        num_servers=NUM_SERVERS,
    )
    env.reset()
    dim_info = _build_dim_info(env)

    capacity        = 10_000
    batch_size      = 128
    actor_lr        = 3e-4
    critic_lr       = 3e-4
    gamma           = 0.9
    tau             = 0.02
    learn_interval  = 5
    random_steps    = max(int(NUM_JOBS * 2), int(NUM_JOBS * n_episodes * 0.1))
    eps_decay_steps = max(int(NUM_JOBS * n_episodes * 0.8), 1)

    if algo == "hier_marl":
        from schedulers.marl.hier_marl.HierMARL import HierMARL
        agent = HierMARL(dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir)
    elif algo == "common_actor":
        from schedulers.marl.common_actor.CommonActor import CommonActor
        agent = CommonActor(dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    reward_log_path = os.path.join(res_dir, "reward.txt")
    ep_rewards = np.zeros(n_episodes, dtype=np.float32)
    global_step = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_agent_reward = {aid: 0.0 for aid in env.agents}
        steps = 0

        while True:
            steps += 1
            global_step += 1

            if global_step <= random_steps:
                epsilon = 1.0
            else:
                progress = min(1.0, (global_step - random_steps) / eps_decay_steps)
                epsilon = 1.0 - 0.95 * progress

            if np.random.rand() < epsilon:
                action = {aid: env.action_space(aid).sample() for aid in env.agents}
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = {a: terminated[a] or truncated[a] for a in env.agents}

            if algo == "hier_marl":
                g_info = info.get("global", {})
                sel_farm = g_info.get("selected_farm_id", 0)
                agent.add(obs, action, reward, next_obs, done, sel_farm)
            else:
                agent.add(obs, action, reward, next_obs, done)

            for a, r in reward.items():
                ep_agent_reward[a] += r
            obs = next_obs

            if global_step > random_steps and global_step % learn_interval == 0:
                agent.learn(gamma)
                agent.update_target(tau)

            if all(done.values()):
                break

        sum_r = sum(ep_agent_reward.values())
        avg_r = sum_r / max(steps, 1)
        ep_rewards[ep] = avg_r

        g_info = info.get("global", {})
        rejected = g_info.get("rejected_tasks_count", 0)
        completed = len(g_info.get("completed_job_ids", []))
        wall_time = g_info.get("wall_time", 0)

        local_keys = sorted([k for k in ep_agent_reward if k.startswith("local_")])
        local_str = " ".join(f"{k}={ep_agent_reward[k]:.4f}" for k in local_keys)
        with open(reward_log_path, "a") as f:
            f.write(
                f"episode={ep + 1}, steps={steps}, epsilon={epsilon:.4f}, "
                f"global_reward={ep_agent_reward.get('global', 0):.4f}, {local_str}, "
                f"sum_reward={sum_r:.4f}, avg_reward_per_step={avg_r:.4f}, "
                f"rejected_tasks={rejected}, completed_jobs={completed}, "
                f"wall_time={wall_time}\n"
            )

        if (ep + 1) % 50 == 0 or ep == 0:
            print(
                f"  [{algo}] ep {ep+1:4d}/{n_episodes}  "
                f"eps={epsilon:.3f}  avg_r/step={avg_r:.4f}  "
                f"reject={rejected}  done={completed}"
            )

    agent.save(ep_rewards)
    env.close()
    return ep_rewards


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_convergence(
    hier_rewards: np.ndarray,
    ca_rewards: np.ndarray,
    out_png: str,
    hier_label: str = "HierMARL",
    ca_label: str = "CommonActor",
) -> None:
    hier_smooth = sliding_window(hier_rewards, WINDOW)
    ca_smooth   = sliding_window(ca_rewards,   WINDOW)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: raw + smoothed
    ax = axes[0]
    xs_h = np.arange(1, len(hier_rewards) + 1)
    xs_c = np.arange(1, len(ca_rewards)   + 1)
    ax.plot(xs_h, hier_rewards, alpha=0.25, color="C0")
    ax.plot(xs_h, hier_smooth,  color="C0", lw=2, label=f"{hier_label} (smoothed)")
    ax.plot(xs_c, ca_rewards,   alpha=0.25, color="C1")
    ax.plot(xs_c, ca_smooth,    color="C1", lw=2, label=f"{ca_label} (smoothed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg reward per step")
    ax.set_title(f"Convergence curves (window={WINDOW})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: convergence speed bar chart
    ax = axes[1]
    hier_ep = episodes_to_convergence(hier_rewards, CONV_THRESHOLD)
    ca_ep   = episodes_to_convergence(ca_rewards,   CONV_THRESHOLD)
    bars = ax.bar([hier_label, ca_label], [hier_ep, ca_ep],
                  color=["C0", "C1"], alpha=0.8, edgecolor="black")
    for bar, val in zip(bars, [hier_ep, ca_ep]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontweight="bold")
    speedup = ca_ep / hier_ep if hier_ep > 0 else float("inf")
    ax.set_ylabel("Episodes to convergence")
    ax.set_title(
        f"Episodes to {int(CONV_THRESHOLD*100)}% peak reward\n"
        f"(speedup: {speedup:.1f}x)"
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved convergence plot: {out_png}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convergence speed: HierMARL vs CommonActor")
    parser.add_argument("--from-logs", action="store_true",
                        help="Read from existing reward.txt files (set HIER_LOG / CA_LOG env vars)")
    parser.add_argument("--hier-log", type=str, default=HIER_LOG_DEFAULT,
                        help="Path to HierMARL reward.txt")
    parser.add_argument("--ca-log", type=str, default=CA_LOG_DEFAULT,
                        help="Path to CommonActor reward.txt")
    parser.add_argument("--episodes", type=int, default=N_EPISODES,
                        help="Episodes to train if not using logs")
    parser.add_argument("--window", type=int, default=WINDOW,
                        help="Sliding window size for smoothing")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    if args.from_logs or (args.hier_log and args.ca_log):
        hier_log = args.hier_log or HIER_LOG_DEFAULT
        ca_log   = args.ca_log   or CA_LOG_DEFAULT
        if not hier_log or not ca_log:
            parser.error(
                "--from-logs requires --hier-log and --ca-log paths "
                "(or HIER_LOG/CA_LOG env vars)"
            )
        print(f"Reading HierMARL log: {hier_log}")
        print(f"Reading CommonActor log: {ca_log}")
        hier_records = parse_reward_log(hier_log)
        ca_records   = parse_reward_log(ca_log)
        hier_rewards = records_to_reward_array(hier_records)
        ca_rewards   = records_to_reward_array(ca_records)
        hier_label = "HierMARL (log)"
        ca_label   = "CommonActor (log)"
    else:
        n_ep = args.episodes
        print(
            f"Training from scratch: {n_ep} episodes each | "
            f"farms={NUM_FARMS}, servers={NUM_SERVERS}, jobs={NUM_JOBS}"
        )
        ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        hier_dir = os.path.join(results_dir, f"conv_hier_{ts}")
        ca_dir   = os.path.join(results_dir, f"conv_ca_{ts}")
        os.makedirs(hier_dir, exist_ok=True)
        os.makedirs(ca_dir,   exist_ok=True)

        print("\n=== Training HierMARL ===")
        t0 = time.perf_counter()
        hier_rewards = train_algorithm("hier_marl", n_ep, hier_dir)
        hier_time = time.perf_counter() - t0
        print(f"HierMARL training done in {hier_time:.1f}s")

        print("\n=== Training CommonActor ===")
        t0 = time.perf_counter()
        ca_rewards = train_algorithm("common_actor", n_ep, ca_dir)
        ca_time = time.perf_counter() - t0
        print(f"CommonActor training done in {ca_time:.1f}s")

        hier_label = f"HierMARL ({n_ep} ep)"
        ca_label   = f"CommonActor ({n_ep} ep)"

    # ── save CSV ───────────────────────────────────────────────────────────────
    hier_smooth = sliding_window(hier_rewards, args.window)
    ca_smooth   = sliding_window(ca_rewards,   args.window)
    hier_ep_conv = episodes_to_convergence(hier_rewards, CONV_THRESHOLD)
    ca_ep_conv   = episodes_to_convergence(ca_rewards,   CONV_THRESHOLD)
    speedup = ca_ep_conv / hier_ep_conv if hier_ep_conv > 0 else float("inf")

    n_max = max(len(hier_rewards), len(ca_rewards))
    csv_rows = []
    for i in range(n_max):
        row = {"episode": i + 1}
        if i < len(hier_rewards):
            row["hier_raw"]    = round(float(hier_rewards[i]), 6)
            row["hier_smooth"] = round(float(hier_smooth[i]),  6)
        else:
            row["hier_raw"] = row["hier_smooth"] = ""
        if i < len(ca_rewards):
            row["ca_raw"]    = round(float(ca_rewards[i]), 6)
            row["ca_smooth"] = round(float(ca_smooth[i]),  6)
        else:
            row["ca_raw"] = row["ca_smooth"] = ""
        csv_rows.append(row)

    csv_path = os.path.join(results_dir, "exp_convergence_speed.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "hier_raw", "hier_smooth",
                                               "ca_raw", "ca_smooth"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved data: {csv_path}")

    # ── print convergence summary ──────────────────────────────────────────────
    print(
        f"\n{'Algorithm':20s} {'Episodes to {:.0%} peak'.format(CONV_THRESHOLD):30s} {'Peak smooth reward':20s}"
    )
    print("-" * 72)
    print(f"{'HierMARL':20s} {hier_ep_conv:<30d} {hier_smooth.max():.4f}")
    print(f"{'CommonActor':20s} {ca_ep_conv:<30d} {ca_smooth.max():.4f}")
    print(f"\nSpeedup (CA/Hier): {speedup:.2f}x")

    # ── plot ───────────────────────────────────────────────────────────────────
    png_path = os.path.join(results_dir, "exp_convergence_speed.png")
    plot_convergence(hier_rewards, ca_rewards, png_path, hier_label, ca_label)


if __name__ == "__main__":
    main()
