"""Smoke-test: run one full episode of CloudSchedulingEnvHier with random actions."""
import random
from env.cloud_scheduling_hier import CloudSchedulingEnvHier

NUM_JOBS         = 20
NUM_SERVER_FARMS = 2
NUM_SERVERS      = 4   # must be divisible by NUM_SERVER_FARMS


def main():
    env = CloudSchedulingEnvHier(
        num_jobs=NUM_JOBS,
        num_server_farms=NUM_SERVER_FARMS,
        num_servers=NUM_SERVERS,
        use_heterogeneity=True,
    )

    obs, infos = env.reset(seed=42)
    print("=== Reset complete ===")
    print(f"Agents: {env.agents}")
    print(f"global obs space:  {env.observation_space('global')}")
    print(f"local_0 obs space: {env.observation_space('local_0')}")
    print(f"local_0 action space: {env.action_space('local_0')}")
    assert env.observation_space("local_0")["cpus_utilization"].shape == (env.num_servers_per_farm,), \
        "local obs dim mismatch"

    step_count = 0
    total_reward = {a: 0.0 for a in env.agents}

    while True:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rew, term, trunc, infos = env.step(actions)
        step_count += 1

        for a in env.agents:
            total_reward[a] += rew[a]

        if step_count % 50 == 0:
            global_info = infos["global"]
            print(
                f"step={step_count:4d}  wall_time={global_info['wall_time']:.2f}"
                f"  completed={len(global_info['completed_job_ids'])}"
                f"  rejected={len(global_info['rejected_job_ids'])}"
                f"  selected_farm={global_info['selected_farm_id']}"
            )

        if all(term.values()) or all(trunc.values()):
            break

    print("\n=== Episode complete ===")
    print(f"Total steps: {step_count}")
    print(f"Total reward: { {a: round(v, 2) for a, v in total_reward.items()} }")
    global_info = infos["global"]
    print(f"Completed jobs: {len(global_info['completed_job_ids'])}/{NUM_JOBS}")
    print(f"Rejected jobs:  {len(global_info['rejected_job_ids'])}")
    print(f"Jain's fairness: {global_info['jains_fairness']}")
    print(f"HER:             {global_info['her']}")
    print(f"selected_farm_id in last info: {global_info['selected_farm_id']}")


if __name__ == "__main__":
    main()
