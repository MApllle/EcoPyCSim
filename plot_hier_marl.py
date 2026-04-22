"""
python plot_hier_marl.py --reward-file results/hier_marl_2026_04_21_07_19_03/reward.txt --start-episode 0 --end-episode 900  --window 20
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def find_latest_reward(results_dir: Path, model_prefix: str = "hier_marl") -> Path:
    candidates = sorted(
        [p for p in results_dir.glob(f"{model_prefix}_*/reward.txt") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No {model_prefix} reward file found under: {results_dir}")
    return candidates[0]


def parse_metric(reward_file: Path, key: str) -> np.ndarray:
    pattern = re.compile(rf"{re.escape(key)}=(-?\d+(?:\.\d+)?)")
    values = []
    with reward_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                values.append(float(match.group(1)))
    if not values:
        raise ValueError(f"No metric `{key}` found in: {reward_file}")
    return np.array(values, dtype=np.float32)


def detect_local_keys(reward_file: Path) -> list[str]:
    # Supports entries like: local_0=..., local_1=..., ...
    local_pat = re.compile(r"(local_\d+)=(-?\d+(?:\.\d+)?)")
    keys: set[str] = set()
    with reward_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for match in local_pat.finditer(line):
                keys.add(match.group(1))
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def running_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = np.mean(arr[start : i + 1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward-file",
        type=str,
        default="results/hier_marl_2026_04_21_08_31_33/reward.txt",
        help="Path to hier_marl reward.txt",
    )
    parser.add_argument("--start-episode", type=int, default=0, help="Start episode index (inclusive)")
    parser.add_argument("--end-episode", type=int, default=300, help="End episode index (inclusive)")
    parser.add_argument("--window", type=int, default=20, help="Running mean window size")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "results"

    if args.reward_file:
        reward_file = (project_dir / args.reward_file).resolve() if not Path(args.reward_file).is_absolute() else Path(args.reward_file)
    else:
        reward_file = find_latest_reward(results_dir, "hier_marl")

    if not reward_file.exists():
        raise FileNotFoundError(f"Reward file not found: {reward_file}")

    global_rewards = parse_metric(reward_file, "global_reward")
    sum_rewards = parse_metric(reward_file, "sum_reward")
    local_keys = detect_local_keys(reward_file)
    local_rewards = {k: parse_metric(reward_file, k) for k in local_keys}

    n = len(global_rewards)
    start = max(0, args.start_episode)
    end = min(args.end_episode, n - 1)
    if start > end:
        raise ValueError(f"Invalid range: start={start}, end={end}, total={n}")

    x = np.arange(start, end + 1)
    g = global_rewards[start : end + 1]
    s = sum_rewards[start : end + 1]
    g_smooth = running_mean(g, window=max(1, args.window))
    s_smooth = running_mean(s, window=max(1, args.window))

    out_png_main = reward_file.parent / f"HIER_MARL_rewards_ep{start}_{end}.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, g, label="global_reward", alpha=0.8)
    ax.plot(x, g_smooth, linestyle="--", linewidth=2, label="global_reward (running avg)")
    ax.plot(x, s, label="sum_reward", alpha=0.8)
    ax.plot(x, s_smooth, linestyle="--", linewidth=2, label="sum_reward (running avg)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("HierMARL Training Rewards")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png_main, dpi=160)

    out_png_local = reward_file.parent / f"HIER_MARL_local_rewards_ep{start}_{end}.png"
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for key in local_keys:
        arr = local_rewards[key][start : end + 1]
        ax2.plot(x, arr, alpha=0.35, label=key)
        ax2.plot(x, running_mean(arr, window=max(1, args.window)), linestyle="--", linewidth=1.8, label=f"{key} (running avg)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.set_title("HierMARL Local-Agent Rewards")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(fontsize=8, ncol=2)
    fig2.tight_layout()
    fig2.savefig(out_png_local, dpi=160)

    print(f"reward_file={reward_file}")
    print(f"saved_plot_main={out_png_main}")
    print(f"saved_plot_local={out_png_local}")


if __name__ == "__main__":
    main()
