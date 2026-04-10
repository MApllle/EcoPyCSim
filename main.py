"""
Heterogeneous-server ablation study for EcoPyCSim.

Controls
--------
CONFIG          — simulation scale (jobs / farms / servers)
PROPORTION_KEY  — which server-mix preset to run ("balanced" / "legacy" / "modern")
                  or None to run all three
USE_HETERO      — True/False, or None to run both
HETERO_WEIGHT   — strength of the heterogeneity-aware reward bonus
SEED            — reproducibility seed

Run all 6 combinations (3 proportions × hetero on/off):
    python main.py

Or edit CONFIG below to run only the combinations you care about.
"""

import pprint
from env.cloud_scheduling_v0 import CloudSchedulingEnv
from components.model_scripts.make_server_farms import PROPORTION_PRESETS

# ── Experiment config ────────────────────────────────────────────────────────
CONFIG = {
    "num_jobs":         5,
    "num_server_farms": 3,
    "num_servers":      9,   # 3 servers per farm
}

HETERO_WEIGHT = 0.3
SEED          = 42

# Set to a specific value to run only that combination; None = run all
PROPORTION_KEY = None   # "balanced" | "legacy" | "modern" | None
USE_HETERO     = None   # True | False | None


# ── Episode runner ───────────────────────────────────────────────────────────
def run_episode(env: CloudSchedulingEnv) -> dict:
    """Run one full episode with random actions and return summary metrics."""
    obs, infos = env.reset(seed=SEED)

    total_energy_cost   = 0.0
    total_hetero_reward = 0.0
    total_steps         = 0
    rejected_tasks      = 0

    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        sf_info = infos.get("server_farm", {})
        total_energy_cost   += sf_info.get("price", 0.0)
        total_hetero_reward += sf_info.get("hetero_reward_total", 0.0)
        rejected_tasks       = sf_info.get("rejected_tasks_count", 0)
        total_steps += 1

        if all(terminations.values()):
            env.close()
            break

    total_tasks = sum(job.num_tasks for job in env.jobs.values()) if env.jobs else 1
    rejection_rate = rejected_tasks / max(total_tasks, 1)

    return {
        "total_energy_cost":   round(total_energy_cost,   4),
        "rejection_rate":      round(rejection_rate,      4),
        "hetero_reward_total": round(total_hetero_reward, 4),
        "steps":               total_steps,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    proportion_keys = (
        [PROPORTION_KEY] if PROPORTION_KEY is not None
        else list(PROPORTION_PRESETS.keys())
    )
    hetero_flags = (
        [USE_HETERO] if USE_HETERO is not None
        else [True, False]
    )

    results = {}

    for prop_key in proportion_keys:
        proportions = PROPORTION_PRESETS[prop_key]
        for use_hetero in hetero_flags:
            label = f"{prop_key}_hetero={'on' if use_hetero else 'off'}"
            print(f"\n{'─'*60}")
            print(f"Config : {label}")
            print(f"  proportions : {proportions}")
            print(f"  use_hetero  : {use_hetero}   hetero_weight: {HETERO_WEIGHT}")
            print(f"{'─'*60}")

            env = CloudSchedulingEnv(
                num_jobs          = CONFIG["num_jobs"],
                num_server_farms  = CONFIG["num_server_farms"],
                num_servers       = CONFIG["num_servers"],
                use_heterogeneity = use_hetero,
                hetero_weight     = HETERO_WEIGHT,
                server_proportions= proportions,
            )

            metrics = run_episode(env)
            results[label] = metrics
            print(f"  → {metrics}")

    # ── Summary table ─────────────────────────────────────────────────────────
    col_w = 40
    print(f"\n\n{'='*80}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<{col_w}} {'Energy Cost':>12} {'Rej. Rate':>10} {'Hetero Rwrd':>12} {'Steps':>7}")
    print(f"{'-'*80}")
    for label, m in results.items():
        print(
            f"{label:<{col_w}}"
            f"{m['total_energy_cost']:>12.4f}"
            f"{m['rejection_rate']:>10.4f}"
            f"{m['hetero_reward_total']:>12.4f}"
            f"{m['steps']:>7}"
        )
    print(f"{'='*80}")

    # ── Quick sanity notes ────────────────────────────────────────────────────
    print("\nExpected patterns:")
    print("  • modern* energy cost  < legacy* energy cost  (newer servers are cheaper)")
    print("  • *hetero=on hetero_reward_total > 0          (bonus only when flag is on)")
    print("  • modern+hetero=on has highest hetero_reward  (most new servers to reward)")


if __name__ == "__main__":
    main()
