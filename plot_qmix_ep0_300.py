import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def find_latest_reward(results_dir: Path, model_prefix: str) -> Path:
    candidates = sorted(
        [p for p in results_dir.glob(f"{model_prefix}_*/reward.txt") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No {model_prefix} reward file found under: {results_dir}")
    return candidates[0]


def parse_agent_rewards(reward_file: Path) -> tuple[np.ndarray, np.ndarray]:
    sf_pattern = re.compile(r"server_farm_reward=(-?\d+(?:\.\d+)?)")
    s_pattern = re.compile(r"server_reward=(-?\d+(?:\.\d+)?)")
    server_farm_rewards = []
    server_rewards = []
    with reward_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            sf_match = sf_pattern.search(line)
            s_match = s_pattern.search(line)
            if sf_match and s_match:
                server_farm_rewards.append(float(sf_match.group(1)))
                server_rewards.append(float(s_match.group(1)))
    if not server_farm_rewards:
        raise ValueError(f"No server_farm_reward/server_reward found in: {reward_file}")
    return (
        np.array(server_farm_rewards, dtype=np.float32),
        np.array(server_rewards, dtype=np.float32),
    )


def parse_metric(reward_file: Path, key: str) -> np.ndarray:
    pattern = re.compile(rf"{key}=(-?\d+(?:\.\d+)?)")
    values = []
    with reward_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                values.append(float(match.group(1)))
    if not values:
        raise ValueError(f"No metric `{key}` found in: {reward_file}")
    return np.array(values, dtype=np.float32)


def running_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = np.mean(arr[start : i + 1])
    return out


def plot_five_models_first_n(
    results_dir: Path,
    end_episode: int,
    window: int,
    metric: str = "episode_total_reward",
) -> Path:
    models = ["idqn", "vdn", "qmix", "mappo", "maddpg"]
    start = 0
    end = max(0, end_episode)
    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"five_models_{metric}_ep{start}_{end}.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in models:
        reward_file = find_latest_reward(results_dir, model)
        values = parse_metric(reward_file, metric)
        cur_end = min(end, len(values) - 1)
        x = np.arange(start, cur_end + 1)
        sliced = values[start : cur_end + 1]
        smooth = running_mean(sliced, window=max(1, window))
        ax.plot(x, smooth, linewidth=2, label=f"{model} ({reward_file.parent.name})")

    ax.set_xlabel("Episode")
    ax.set_ylabel(metric)
    ax.set_title(f"Five-Model Comparison ({metric}, episode {start}-{end})")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    return out_png


def plot_per_model_agent_curves_first_n(
    results_dir: Path,
    end_episode: int,
    window: int,
) -> list[Path]:
    models = ["idqn", "vdn", "qmix", "mappo", "maddpg"]
    start = 0
    end = max(0, end_episode)
    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for model in models:
        reward_file = find_latest_reward(results_dir, model)
        sf_rewards, s_rewards = parse_agent_rewards(reward_file)
        cur_end = min(end, len(sf_rewards) - 1)
        if cur_end < start:
            continue

        x = np.arange(start, cur_end + 1)
        sf_sliced = sf_rewards[start : cur_end + 1]
        s_sliced = s_rewards[start : cur_end + 1]
        sf_smooth = running_mean(sf_sliced, window=max(1, window))
        s_smooth = running_mean(s_sliced, window=max(1, window))

        out_png = out_dir / f"{model}_training_performance_ep{start}_{cur_end}.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, sf_sliced, label="server_farm")
        ax.plot(x, sf_smooth, linestyle="--", label="server_farm (running avg)")
        ax.plot(x, s_sliced, label="server")
        ax.plot(x, s_smooth, linestyle="--", label="server (running avg)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title(f"{model.upper()} Training Performance")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        output_paths.append(out_png)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward-file",
        type=str,
        default=None,
        help="Optional explicit path to reward.txt",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Start episode index (inclusive)",
    )
    parser.add_argument(
        "--end-episode",
        type=int,
        default=300,
        help="End episode index (inclusive)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Running mean window size",
    )
    parser.add_argument(
        "--plot-five-models",
        action="store_true",
        help="Also plot five-model comparison for first N episodes",
    )
    parser.add_argument(
        "--five-end-episode",
        type=int,
        default=200,
        help="End episode (inclusive) for five-model comparison",
    )
    parser.add_argument(
        "--five-metric",
        type=str,
        default="episode_total_reward",
        help="Metric key for five-model comparison (default: episode_total_reward)",
    )
    parser.add_argument(
        "--plot-per-model-agent-curves",
        action="store_true",
        help="Plot 4-line training curves (server_farm/server + running avg) for each model",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "results"

    reward_file = Path(args.reward_file) if args.reward_file else find_latest_reward(results_dir, "qmix")
    server_farm_rewards, server_rewards = parse_agent_rewards(reward_file)

    start = max(0, args.start_episode)
    end = min(args.end_episode, len(server_farm_rewards) - 1)
    if start > end:
        raise ValueError(
            f"Invalid range: start={start}, end={end}, total={len(server_farm_rewards)}"
        )

    sf_sliced = server_farm_rewards[start : end + 1]
    s_sliced = server_rewards[start : end + 1]
    x = np.arange(start, end + 1)
    sf_smooth = running_mean(sf_sliced, window=max(1, args.window))
    s_smooth = running_mean(s_sliced, window=max(1, args.window))

    out_png = reward_file.parent / f"QMIX_performance_ep{start}_{end}.png"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, sf_sliced, label="server_farm", alpha=0.85)
    ax.plot(x, sf_smooth, linestyle="--", label="server_farm (running avg)", linewidth=2)
    ax.plot(x, s_sliced, label="server", alpha=0.85)
    ax.plot(x, s_smooth, linestyle="--", label="server (running avg)", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("QMIX Training Performance")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)

    print(f"reward_file={reward_file}")
    print(f"saved_plot={out_png}")

    if args.plot_five_models:
        five_out = plot_five_models_first_n(
            results_dir=results_dir,
            end_episode=args.five_end_episode,
            window=args.window,
            metric=args.five_metric,
        )
        print(f"saved_five_model_plot={five_out}")

    if args.plot_per_model_agent_curves:
        per_model_plots = plot_per_model_agent_curves_first_n(
            results_dir=results_dir,
            end_episode=args.five_end_episode,
            window=args.window,
        )
        for p in per_model_plots:
            print(f"saved_per_model_plot={p}")


if __name__ == "__main__":
    main()
