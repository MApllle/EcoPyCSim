import numpy as np
from gym import spaces

def flatten_cloud_obs(obs):
    """将字典格式的 obs 展平成一维向量"""
    # 注意：这里需要确保提取的 key 在 obs_dict 中确实存在
    cpu_util = np.array(obs.get('cpus_utilization', [])).flatten()
    cpu_slack = np.array(obs.get('cpu_slack', [])).flatten()
    task_req = np.array(obs.get('task_cpu', [0])).flatten()
    tiers = np.array(obs.get('efficiency_tiers', [])).flatten()

    # 如果维度不对，可能会导致训练崩溃，建议在这里加一个 print 调试一次维度
    return np.concatenate([cpu_util, cpu_slack, task_req, tiers]).astype(np.float32)

class CloudEnvWrapper:
    def __init__(self, env):
        self.env = env
        # 保持与 ParallelEnv 中的 agents 顺序一致
        self.agents = ["server_farm", "server"]
        self.num_agents = len(self.agents)

        # 动态获取维度
        test_obs, _ = env.reset()
        obs_dim = flatten_cloud_obs(test_obs["server_farm"]).shape[0]

        # MAPPO 框架要求这些是列表形式的 Space 对象
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                                 for _ in range(self.num_agents)]

        self.action_space = [env.action_space(agent) for agent in self.agents]

        # 共享观测空间：所有 Agent 观测的拼接
        share_obs_dim = obs_dim * self.num_agents
        self.share_observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32)
                                       for _ in range(self.num_agents)]

    def seed(self, seed=None):
        """官方框架会调用此方法"""
        if seed is not None:
            np.random.seed(seed)

    def reset(self):
        obs_dict, infos = self.env.reset()

        local_obs = np.array([flatten_cloud_obs(obs_dict[agent]) for agent in self.agents])
        # global_obs 形状为 (share_obs_dim,)
        global_obs = local_obs.flatten()
        # share_obs 形状为 (num_agents, share_obs_dim)
        share_obs = np.array([global_obs for _ in range(self.num_agents)])

        return local_obs, share_obs, infos

    def step(self, actions):
        """
        actions 形状通常是 (num_agents, 1) 或 (num_agents,)
        """
        # 转换动作格式以适配你的 ParallelEnv
        # 确保 actions 是整数，因为环境通常需要离散索引
        action_dict = {
            self.agents[i]: int(np.squeeze(actions[i]))
            for i in range(self.num_agents)
        }

        obs_dict, rewards, terms, truncs, infos = self.env.step(action_dict)

        local_obs = np.array([flatten_cloud_obs(obs_dict[agent]) for agent in self.agents])
        global_obs = local_obs.flatten()
        share_obs = np.array([global_obs for _ in range(self.num_agents)])

        # 奖励形状：(num_agents, 1)
        reward_arr = np.array([[rewards[agent]] for agent in self.agents])

        # 结束标志：(num_agents,)
        dones = np.array([terms[agent] or truncs[agent] for agent in self.agents])

        # MAPPO 框架有时需要一个坏掉的标志位 (available_actions)，对离散动作通常设为全 1
        available_actions = None

        return local_obs, share_obs, reward_arr, dones, infos, available_actions

    def close(self):
        self.env.close()