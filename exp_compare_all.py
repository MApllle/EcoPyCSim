"""
实验可视化：从 CSV 结果生成对比图表

用法：
  python exp_compare_all.py

输入（按需，均为可选）：
  results/exp1_baseline_comparison.csv      —— 全算法横向对比
  results/exp2_heterogeneity_ablation.csv   —— 异构消融实验

输出图表至 results/figures/
"""
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "DejaVu Sans"
FIGURES_DIR = os.path.join("results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Palette ─────────────────────────────────────────────────────────────────

STRATEGY_COLORS = {
    "random":       "#aaaaaa",
    "round_robin":  "#888888",
    "least_loaded": "#5b8dca",
    "best_fit":     "#78b7d0",
    "energy_greedy":"#4daf4a",
    "idqn":         "#ff7f00",
    "vdn":          "#e41a1c",
    "qmix":         "#984ea3",
    "mappo":        "#377eb8",
    "maddpg":       "#f781bf",
}

STRATEGY_LABELS = {
    "random":       "Random",
    "round_robin":  "Round-Robin",
    "least_loaded": "Least-Loaded",
    "best_fit":     "Best-Fit",
    "energy_greedy":"Energy-Greedy",
    "idqn":         "IDQN",
    "vdn":          "VDN",
    "qmix":         "QMIX",
    "mappo":        "MAPPO",
    "maddpg":       "MADDPG",
}


# ── Plot helpers ─────────────────────────────────────────────────────────────

def _bar_chart(df, mean_col, std_col, title, ylabel, filename,
               lower_is_better=True, highlight_marl=True):
    strategies = df["strategy"].tolist()
    means = df[mean_col].tolist()
    stds  = df[std_col].tolist() if std_col in df.columns else [0] * len(means)

    colors = [STRATEGY_COLORS.get(s, "#999999") for s in strategies]
    labels = [STRATEGY_LABELS.get(s, s) for s in strategies]

    fig, ax = plt.subplots(figsize=(max(8, len(strategies) * 0.9), 5))
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=4,
                  edgecolor="white", linewidth=0.6)

    # Highlight best bar
    best_idx = int(np.argmin(means) if lower_is_better else np.argmax(means))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")


def _grouped_bar(df, mean_col, std_col, group_col, hue_col,
                 title, ylabel, filename, lower_is_better=True):
    groups = df[group_col].unique()
    hues   = df[hue_col].unique()
    x      = np.arange(len(groups))
    width  = 0.8 / len(hues)

    fig, ax = plt.subplots(figsize=(max(9, len(groups) * 2), 5))
    for i, hue in enumerate(hues):
        sub   = df[df[hue_col] == hue].set_index(group_col)
        means = [sub.loc[g, mean_col] if g in sub.index else 0 for g in groups]
        stds  = ([sub.loc[g, std_col] if g in sub.index else 0 for g in groups]
                 if std_col in df.columns else [0] * len(groups))
        label  = STRATEGY_LABELS.get(str(hue), str(hue))
        color  = STRATEGY_COLORS.get(str(hue), "#999999")
        offset = (i - len(hues) / 2 + 0.5) * width
        ax.bar(x + offset, means, width * 0.9, yerr=stds, label=label,
               color=color, capsize=3, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")


# ── Exp1: Full algorithm comparison ─────────────────────────────────────────

def plot_exp1(csv_path):
    print("\n[实验一] 全算法横向对比图表")
    df = pd.read_csv(csv_path)

    metric_specs = [
        ("total_price_mean", "total_price_std",  "总能耗成本对比",         "Total Energy Cost",     "exp1_energy_cost.png",     True),
        ("eet_mean",          "eet_std",           "单任务能耗 (EET) 对比",  "Energy per Task (EET)", "exp1_eet.png",              True),
        ("rejected_tasks_mean","rejected_tasks_std","任务拒绝数对比",         "Rejected Tasks",        "exp1_rejected.png",         True),
        ("jains_fairness_mean","jains_fairness_std","负载均衡 Jain's J 对比","Jain's Fairness Index", "exp1_jains.png",            False),
        ("active_server_ratio_mean","active_server_ratio_std","活跃服务器比率 (ASR) 对比","Active Server Ratio","exp1_asr.png",True),
        ("her_mean",          "her_std",            "异构感知率 (HER) 对比",  "Heterogeneity Exploitation Rate","exp1_her.png",False),
    ]
    for mean_col, std_col, title, ylabel, filename, lib in metric_specs:
        if mean_col in df.columns:
            _bar_chart(df, mean_col, std_col, title, ylabel, filename, lower_is_better=lib)
        else:
            print(f"  跳过 {mean_col}（列不存在）")

    # Radar chart (normalised, lower-is-better metrics inverted)
    radar_metrics = {
        "能耗(↓)":     ("total_price_mean", True),
        "拒绝率(↓)":   ("rejected_tasks_mean", True),
        "Jain J(↑)":  ("jains_fairness_mean", False),
        "HER(↑)":     ("her_mean", False),
        "ASR(↓)":     ("active_server_ratio_mean", True),
    }
    _radar_chart(df, radar_metrics, "综合调度性能雷达图", "exp1_radar.png")


def _radar_chart(df, metric_specs, title, filename):
    """Draw a radar/spider chart for each strategy."""
    labels    = list(metric_specs.keys())
    n_labels  = len(labels)
    angles    = np.linspace(0, 2 * np.pi, n_labels, endpoint=False).tolist()
    angles   += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    # Normalise each metric to [0,1]
    normed = {}
    for lbl, (col, lower_is_better) in metric_specs.items():
        if col not in df.columns:
            normed[lbl] = np.zeros(len(df))
            continue
        vals = df[col].values.astype(float)
        v_min, v_max = vals.min(), vals.max()
        if v_max == v_min:
            normed[lbl] = np.ones(len(df)) * 0.5
        elif lower_is_better:
            normed[lbl] = 1 - (vals - v_min) / (v_max - v_min)
        else:
            normed[lbl] = (vals - v_min) / (v_max - v_min)

    for i, row in df.iterrows():
        strategy = row["strategy"]
        values   = [normed[lbl][i] for lbl in labels] + [normed[labels[0]][i]]
        color    = STRATEGY_COLORS.get(strategy, "#999999")
        label    = STRATEGY_LABELS.get(strategy, strategy)
        ax.plot(angles, values, color=color, linewidth=1.5, label=label)
        ax.fill(angles, values, color=color, alpha=0.06)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=8, ncol=2)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


# ── Exp2: Heterogeneity ablation ─────────────────────────────────────────────

def plot_exp2(csv_path):
    print("\n[实验二] 异构消融实验图表")
    df = pd.read_csv(csv_path)

    # Condition label
    df["condition"] = df["server_mix"] + "\n" + df["use_hetero"].map(
        {True: "hetero=on", False: "hetero=off", "True": "hetero=on", "False": "hetero=off"}
    )

    # 1. HER grouped by condition × strategy
    _grouped_bar(df, "her_mean", "her_std",
                 group_col="condition", hue_col="strategy",
                 title="各条件下异构感知率 (HER)",
                 ylabel="HER",
                 filename="exp2_her_grouped.png",
                 lower_is_better=False)

    # 2. Energy cost grouped
    _grouped_bar(df, "total_price_mean", "total_price_std",
                 group_col="condition", hue_col="strategy",
                 title="各条件下总能耗成本",
                 ylabel="Total Energy Cost",
                 filename="exp2_energy_grouped.png",
                 lower_is_better=True)

    # 3. Per server_mix: HER comparison hetero=on vs hetero=off (MARL only)
    marl_strategies = ["idqn", "vdn", "qmix", "mappo", "maddpg"]
    df_marl = df[df["strategy"].isin(marl_strategies)].copy()
    if not df_marl.empty:
        for mix in df["server_mix"].unique():
            sub = df_marl[df_marl["server_mix"] == mix]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            x      = np.arange(len(marl_strategies))
            width  = 0.35
            on_vals  = []
            off_vals = []
            on_stds  = []
            off_stds = []
            for s in marl_strategies:
                on_row  = sub[(sub["strategy"] == s) & (sub["use_hetero"].astype(str) == "True")]
                off_row = sub[(sub["strategy"] == s) & (sub["use_hetero"].astype(str) == "False")]
                on_vals.append(on_row["her_mean"].values[0] if not on_row.empty else 0)
                off_vals.append(off_row["her_mean"].values[0] if not off_row.empty else 0)
                on_stds.append(on_row["her_std"].values[0] if not on_row.empty else 0)
                off_stds.append(off_row["her_std"].values[0] if not off_row.empty else 0)

            ax.bar(x - width/2, on_vals,  width, yerr=on_stds,  label="hetero=on",  color="#377eb8", capsize=4)
            ax.bar(x + width/2, off_vals, width, yerr=off_stds, label="hetero=off", color="#aaaaaa", capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in marl_strategies])
            ax.set_title(f"异构奖励开关对 HER 的影响 ({mix})", fontsize=12)
            ax.set_ylabel("HER")
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            fig.tight_layout()
            path = os.path.join(FIGURES_DIR, f"exp2_her_ablation_{mix}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  保存: {path}")


# ── Learning curves (from training .npy reward logs) ────────────────────────

def _load_reward_series(path):
    if path.endswith(".npy"):
        arr = np.load(path)
        return np.asarray(arr, dtype=float).flatten()

    rewards = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = []
            for token in line.replace(",", " ").split():
                try:
                    nums.append(float(token))
                except ValueError:
                    continue
            if nums:
                rewards.append(sum(nums[-2:]) if len(nums) >= 2 else nums[-1])
    return np.asarray(rewards, dtype=float)


def plot_learning_curves(results_dir="results"):
    """
    Overlay learning curves for all algorithms that saved reward arrays as
    results/<algo>/<algo>_rewards.npy  or  results/<algo>/rewards.npy
    """
    algo_dirs = {
        "idqn":   "idqn",
        "vdn":    "vdn",
        "qmix":   "qmix",
        "mappo":  "mappo",
        "maddpg": "maddpg",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    found_any = False

    for algo, subdir in algo_dirs.items():
        candidates = [
            os.path.join(results_dir, subdir, f"{algo}_rewards.npy"),
            os.path.join(results_dir, subdir, "rewards.npy"),
            os.path.join(results_dir, subdir, "reward.txt"),
            os.path.join(results_dir, subdir, "rewards.txt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                rewards = _load_reward_series(path)
                if rewards.size == 0:
                    continue
                window  = min(10, len(rewards) // 5 + 1)
                smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
                color = STRATEGY_COLORS.get(algo, "#999999")
                label = STRATEGY_LABELS.get(algo, algo)
                ax.plot(smoothed, color=color, linewidth=1.8, label=label)
                ax.fill_between(range(len(smoothed)), smoothed, alpha=0.08, color=color)
                found_any = True
                break

    if not found_any:
        print("  未找到任何 rewards.npy 文件，跳过学习曲线图。")
        plt.close(fig)
        return

    ax.set_title("MARL 算法学习曲线对比", fontsize=13)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode Reward (smoothed)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "learning_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    exp1_csv = os.path.join("results", "exp1_baseline_comparison.csv")
    exp2_csv = os.path.join("results", "exp2_heterogeneity_ablation.csv")

    if os.path.exists(exp1_csv):
        plot_exp1(exp1_csv)
    else:
        print(f"[跳过] 未找到 {exp1_csv}，请先运行 python exp1_compare_baselines.py")

    if os.path.exists(exp2_csv):
        plot_exp2(exp2_csv)
    else:
        print(f"[跳过] 未找到 {exp2_csv}，请先运行 python exp_heterogeneity_ablation.py")

    print("\n生成学习曲线...")
    plot_learning_curves()

    print(f"\n所有图表已保存至 {FIGURES_DIR}/")
