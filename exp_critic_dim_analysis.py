"""
Critic input dimension vs number of agents N — architectural scalability argument.

For a system with N server farms and M servers per farm:

  GlobalEnv obs dims (CloudSchedulingEnvHier):
    global agent obs  = 3N + 4   (per-farm stats × N + task info)
    local agent obs   = 2M + 5   (per-server stats × M + task info + is_selected)
    global act dim    = N
    local act dim     = M

  Algorithm critic input dims:
    MADDPG / CommonActor  = (3N+4 + N×(2M+5)) + (N + N×M)          — fully centralized
    HierMARL global       = (3N+4) + N + (2M+5) + M = 4N + 3M + 9  — one local farm
    HierMARL local        = (2M+5) + M = 3M + 5                      — constant wrt N

  Key insight: HierMARL local critic is O(M) — independent of N.
               HierMARL global critic is O(N + M) vs O(N×M) for centralized.

Output
------
  results/exp_critic_dim_analysis.png
  results/exp_critic_dim_analysis.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── dimension formulas ─────────────────────────────────────────────────────────

def common_actor_critic_dim(n: int, m: int) -> int:
    """CommonActor (and MADDPG) fully-centralized critic: concat all agents' obs + acts."""
    global_obs = 3 * n + 4
    local_obs_total = n * (2 * m + 5)
    global_act = n
    local_act_total = n * m
    return global_obs + local_obs_total + global_act + local_act_total


def maddpg_critic_dim(n: int, m: int) -> int:
    """MADDPG-style centralized critic (same formula as CommonActor for this env)."""
    return common_actor_critic_dim(n, m)


def hier_global_critic_dim(n: int, m: int) -> int:
    """HierMARL global critic: global obs+act + ONE selected local obs+act."""
    global_obs = 3 * n + 4
    global_act = n
    local_obs  = 2 * m + 5
    local_act  = m
    return global_obs + global_act + local_obs + local_act


def hier_local_critic_dim(n: int, m: int) -> int:
    """HierMARL local critic: only local farm obs + act (constant wrt N)."""
    return (2 * m + 5) + m


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Critic dimension vs N analysis")
    parser.add_argument(
        "--n-values", type=str, default="2,5,10,30,50,100",
        help="Comma-separated N values (num_server_farms)",
    )
    parser.add_argument(
        "--m", type=int, default=6,
        help="Servers per farm (fixed for the N sweep)",
    )
    parser.add_argument(
        "--m-values", type=str, default="",
        help="Optional: also do an M sweep at fixed N. Comma-separated M values.",
    )
    parser.add_argument(
        "--fixed-n", type=int, default=5,
        help="Fixed N for the M sweep",
    )
    args = parser.parse_args()

    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]
    m_fixed  = args.m

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    # ── compute dims over N sweep ──────────────────────────────────────────────
    rows = []
    for n in n_values:
        rows.append({
            "N": n,
            "M": m_fixed,
            "maddpg_critic": maddpg_critic_dim(n, m_fixed),
            "common_actor_critic": common_actor_critic_dim(n, m_fixed),
            "hier_global_critic": hier_global_critic_dim(n, m_fixed),
            "hier_local_critic": hier_local_critic_dim(n, m_fixed),
        })

    csv_path = os.path.join(results_dir, "exp_critic_dim_analysis.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")

    # ── print table ────────────────────────────────────────────────────────────
    print(f"\nCritic input dims vs N  (M={m_fixed} servers/farm fixed)")
    print(f"{'N':>5} {'MADDPG':>12} {'CommonActor':>12} {'Hier-Global':>12} {'Hier-Local':>12}")
    print("-" * 60)
    for r in rows:
        print(
            f"{r['N']:>5}  {r['maddpg_critic']:>12}  {r['common_actor_critic']:>12}  "
            f"{r['hier_global_critic']:>12}  {r['hier_local_critic']:>12}"
        )

    # ── plot N sweep ───────────────────────────────────────────────────────────
    ns = [r["N"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(ns, [r["maddpg_critic"]        for r in rows], "o-",  color="C0", lw=2, label="MADDPG (centralized)")
    ax.plot(ns, [r["common_actor_critic"]   for r in rows], "s--", color="C1", lw=2, label="CommonActor (centralized)")
    ax.plot(ns, [r["hier_global_critic"]    for r in rows], "^-",  color="C2", lw=2, label="HierMARL global critic")
    ax.plot(ns, [r["hier_local_critic"]     for r in rows], "d-",  color="C3", lw=2, label="HierMARL local critic")
    ax.set_xlabel("Number of server farms (N)")
    ax.set_ylabel("Critic input dimension")
    ax.set_title(f"Critic input dim vs N  (M={m_fixed} servers/farm)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ns)

    # Log-scale version for clarity
    ax = axes[1]
    ax.semilogy(ns, [r["maddpg_critic"]        for r in rows], "o-",  color="C0", lw=2, label="MADDPG (centralized)")
    ax.semilogy(ns, [r["common_actor_critic"]   for r in rows], "s--", color="C1", lw=2, label="CommonActor (centralized)")
    ax.semilogy(ns, [r["hier_global_critic"]    for r in rows], "^-",  color="C2", lw=2, label="HierMARL global critic")
    ax.semilogy(ns, [r["hier_local_critic"]     for r in rows], "d-",  color="C3", lw=2, label="HierMARL local critic")
    ax.set_xlabel("Number of server farms (N)")
    ax.set_ylabel("Critic input dimension (log scale)")
    ax.set_title(f"Critic input dim vs N — log scale  (M={m_fixed})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks(ns)

    # Annotate HierMARL local constant line
    local_dim = hier_local_critic_dim(1, m_fixed)
    axes[1].axhline(y=local_dim, color="C3", linestyle=":", alpha=0.5)
    axes[1].text(ns[0], local_dim * 1.15, f"Hier-Local = {local_dim} (const)",
                 color="C3", fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(results_dir, "exp_critic_dim_analysis.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved: {png_path}")

    # ── optional M sweep ───────────────────────────────────────────────────────
    if args.m_values:
        m_values = [int(x) for x in args.m_values.split(",") if x.strip()]
        n_fixed  = args.fixed_n

        m_rows = []
        for m in m_values:
            m_rows.append({
                "N": n_fixed,
                "M": m,
                "maddpg_critic": maddpg_critic_dim(n_fixed, m),
                "common_actor_critic": common_actor_critic_dim(n_fixed, m),
                "hier_global_critic": hier_global_critic_dim(n_fixed, m),
                "hier_local_critic": hier_local_critic_dim(n_fixed, m),
            })

        m_csv = os.path.join(results_dir, "exp_critic_dim_m_sweep.csv")
        with open(m_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(m_rows[0].keys()))
            writer.writeheader()
            writer.writerows(m_rows)

        print(f"\nM sweep (N={n_fixed} fixed)")
        print(f"{'M':>5} {'MADDPG':>12} {'CommonActor':>12} {'Hier-Global':>12} {'Hier-Local':>12}")
        print("-" * 60)
        for r in m_rows:
            print(
                f"{r['M']:>5}  {r['maddpg_critic']:>12}  {r['common_actor_critic']:>12}  "
                f"{r['hier_global_critic']:>12}  {r['hier_local_critic']:>12}"
            )
        print(f"Saved: {m_csv}")

    # ── Amdahl note ────────────────────────────────────────────────────────────
    max_n = max(n_values)
    ratio_hier = hier_global_critic_dim(max_n, m_fixed) / common_actor_critic_dim(max_n, m_fixed)
    print(
        f"\nAt N={max_n}, M={m_fixed}:"
        f"\n  MADDPG/CommonActor critic dim = {common_actor_critic_dim(max_n, m_fixed)}"
        f"\n  HierMARL global critic dim    = {hier_global_critic_dim(max_n, m_fixed)}"
        f"\n  HierMARL local  critic dim    = {hier_local_critic_dim(max_n, m_fixed)}"
        f"\n  Ratio (Hier-Global / Central) = {ratio_hier:.3f}"
        f"\n  HierMARL local is O(M) — completely independent of N."
    )


if __name__ == "__main__":
    main()
