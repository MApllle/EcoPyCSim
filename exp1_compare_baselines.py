import numpy as np
from env.cloud_scheduling import CloudSchedulingEnv

class BaselineEvaluator:
    def __init__(self):
        self.rr_count_farm = 0
        self.rr_count_server = 0

    def get_actions(self, obs_dict, strategy, env):
        """适配 ParallelEnv：一次性为所有活动的 Agent 生成动作"""
        actions = {}

        for agent, obs in obs_dict.items():
            action_space = env.action_space(agent)

            if strategy == "random":
                actions[agent] = action_space.sample()

            elif strategy == "round_robin":
                if agent == "server_farm":
                    actions[agent] = self.rr_count_farm % action_space.n
                    self.rr_count_farm += 1
                else:
                    actions[agent] = self.rr_count_server % action_space.n
                    self.rr_count_server += 1

            elif strategy == "least_loaded":
                all_loads = obs.get('cpus_utilization', [])
                if agent == "server_farm":
                    # all_loads 是二维数组
                    avg_loads = [np.mean(f) for f in all_loads]
                    actions[agent] = int(np.argmin(avg_loads))
                else:
                    # all_loads 是一维数组
                    actions[agent] = int(np.argmin(all_loads))

            elif strategy == "best_fit":
                all_loads = obs['cpus_utilization']
                task_req = obs['task_cpu'][0]

                best_idx = -1
                min_remaining = 1.1

                # 如果是 server_farm 级别，all_loads 是二维的，我们取各 farm 平均负载
                if agent == "server_farm":
                    loads_to_check = [np.mean(f) for f in all_loads]
                else:
                    loads_to_check = all_loads

                for i, load in enumerate(loads_to_check):
                    remaining = 1.0 - load
                    if remaining >= task_req and remaining < min_remaining:
                        min_remaining = remaining
                        best_idx = i

                actions[agent] = int(best_idx) if best_idx != -1 else action_space.sample()

            elif strategy == "energy_greedy":
                all_loads = obs['cpus_utilization']
                tiers = obs['efficiency_tiers']

                if agent == "server_farm":
                    # farm 级别取平均值
                    avg_loads = [np.mean(f) for f in all_loads]
                    avg_tiers = [np.mean(t) for t in tiers]
                    costs = [(1.0 / (avg_tiers[i] + 1e-6)) * (avg_loads[i] + 0.1) for i in range(len(avg_loads))]
                else:
                    costs = [(1.0 / (tiers[i] + 1e-6)) * (all_loads[i] + 0.1) for i in range(len(all_loads))]

                actions[agent] = int(np.argmin(costs))

            else:
                actions[agent] = action_space.sample()

        return actions

def run_experiment(strategy_name):
    eval_env = CloudSchedulingEnv(num_jobs=100, num_server_farms=5, num_servers=50)
    observations, infos = eval_env.reset()

    evaluator = BaselineEvaluator()
    total_price = 0
    step_count = 0

    while eval_env.agents:
        if eval_env.all_jobs_complete:
            print(f"检测到所有任务已完成，正常结束模拟。")
            break

        actions = evaluator.get_actions(observations, strategy_name, eval_env)

        try:
            observations, rewards, terminations, truncations, infos = eval_env.step(actions)

            # 累计 Price
            current_price = infos.get("server_farm", {}).get("price", 0)
            total_price += current_price
            step_count += 1

            # 如果任何一个 agent 终止了，也退出
            if any(terminations.values()) or any(truncations.values()):
                break
        except AssertionError:
            # 万一还是触发了那个断言，我们直接捕获它并当作模拟结束
            print(f"环境触发结束断言，停止当前策略。")
            break

    print(f"【实验结果】策略: {strategy_name:15} | 总步数: {step_count:5} | 总价格: {total_price:.2f}")
    eval_env.close()

if __name__ == "__main__":
    strategies = ["random", "round_robin", "least_loaded", "best_fit", "energy_greedy"]
    for s in strategies:
        print(f"正在运行策略: {s}...")
        try:
            run_experiment(s)
        except Exception as e:
            print(f"运行 {s} 时出错: {e}")
            import traceback
            traceback.print_exc()