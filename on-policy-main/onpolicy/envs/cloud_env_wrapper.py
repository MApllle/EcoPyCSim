import numpy as np
from gym import spaces
import torch

def flatten_cloud_obs(obs):
    """将字典格式的 obs 展平成一维向量"""
    # 提取并展平各个组件，确保即使为空也能返回 empty array
    cpu_util = np.array(obs.get('cpus_utilization', [])).flatten()
    task_req = np.array(obs.get('task_cpu', [0])).flatten()
    tiers = np.array(obs.get('efficiency_tiers', [])).flatten()

    # 拼接并确保类型
    return np.concatenate([cpu_util, task_req, tiers]).astype(np.float32)

class CloudEnvWrapper:
    def __init__(self, env):
        self.env = env
        # 保持与 ParallelEnv 中的 agents 顺序一致
        self.agents = ["server_farm", "server"]
        self.num_agents = len(self.agents)

        # 1. 动态获取所有 Agent 的最大维度以进行 Padding
        test_obs, _ = env.reset()
        dims = [flatten_cloud_obs(test_obs[agent]).shape[0] for agent in self.agents]
        self.max_obs_dim = max(dims)

        print(f"--- [Wrapper] Agent 原始维度: {dims}, 统一 Padding 维度: {self.max_obs_dim} ---")

        # 2. MAPPO 要求所有 Agent 的 Space 对象一致
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        # self.action_space = [env.action_space(agent) for agent in self.agents]
        self.action_space = []
        for agent in self.agents:
            # 直接从原始环境 env 中获取对应智能体的空间
            actual_space = self.env.action_space(agent)
            self.action_space.append(actual_space)

        print(f"--- [Wrapper] 智能体动作空间已对齐: {self.action_space} ---")
        # 在环境类中
        # num_servers = len(self.server_farms[self.server_farm_id].servers)
        # self.action_space = spaces.Discrete(num_servers)

        # 3. 共享观测空间：所有 Agent 补齐后的观测拼接
        self.share_obs_dim = self.max_obs_dim * self.num_agents
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        print(f"--- [Wrapper] 维度对齐完成 ---")
        print(f"--- [Wrapper] Obs Dim: {dims} -> Unified: {self.max_obs_dim}")
        print(f"--- [Wrapper] Action Space: {self.action_space}")

    def _get_padded_obs(self, obs_dict):
        """内部辅助函数：展平并补齐观测值"""
        padded_obs_list = []
        for agent in self.agents:
            obs = flatten_cloud_obs(obs_dict[agent])
            if len(obs) < self.max_obs_dim:
                # 后向补零对齐
                obs = np.concatenate([obs, np.zeros(self.max_obs_dim - len(obs), dtype=np.float32)])
            padded_obs_list.append(obs)

        local_obs = np.stack(padded_obs_list)
        global_obs = local_obs.flatten()
        share_obs = np.array([global_obs for _ in range(self.num_agents)])

        return local_obs, share_obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    # def reset(self):
    #     obs_dict, infos = self.env.reset()
    #     local_obs, share_obs = self._get_padded_obs(obs_dict)
    #
    #     # 很多版本的 on-policy 框架在 reset 时只接收 obs 和 share_obs
    #     # infos 往往是通过另一个管道发送的，或者根本不接收
    #     # 根据你 env_wrappers.py 第 272 行的报错，我们尝试返回一个统一的结构
    #     return local_obs, share_obs, infos
    # def reset(self):
    #     obs_dict, infos = self.env.reset()
    #     local_obs, share_obs = self._get_padded_obs(obs_dict)
    #     # 强制只返回两个核心 obs，infos 交给 step 处理或丢弃
    #     return local_obs, share_obs
    def reset(self):
        # 1. 正常重置
        obs_dict, infos = self.env.reset()

        # 2. 获取对齐后的观察
        local_obs, share_obs = self._get_padded_obs(obs_dict)

        # 3. 关键：在评估模式下，DummyVecEnv 往往只支持返回 obs
        # 我们返回 local_obs，确保它是一个纯粹的 numpy 数组
        return local_obs

    def step(self, actions):
        """
        actions 形状: (num_agents, action_dim)
        """
        # 构建环境需要的 Action 字典
        action_dict = {}
        for i, agent in enumerate(self.agents):
            act = actions[i]
            # 如果是离散动作且被包装成了 array，需要取标量
            if isinstance(act, (np.ndarray, list, torch.Tensor)):
                act = int(np.reshape(act, -1)[0])
            action_dict[agent] = act

        obs_dict, rewards, terms, truncs, infos = self.env.step(action_dict)

        # 获取对齐后的观测
        local_obs, share_obs = self._get_padded_obs(obs_dict)

        # 核心修改：将 share_obs 塞进 infos 字典里“偷渡”出去
        # 这样既符合 DummyVecEnv 的 4 值要求，Runner 又能拿到数据
        infos["share_obs"] = share_obs

        reward_arr = np.array([[rewards[agent]] for agent in self.agents], dtype=np.float32)
        dones = np.array([terms[agent] or truncs[agent] for agent in self.agents], dtype=bool)

        # 只返回 4 个值
        return local_obs, reward_arr, dones, infos

        # 奖励形状处理: (num_agents, 1)
        # reward_arr = np.array([[rewards[agent]] for agent in self.agents], dtype=np.float32)
        #
        # # 结束标志处理: (num_agents,)
        # dones = np.array([terms[agent] or truncs[agent] for agent in self.agents], dtype=bool)
        #
        # # 对于离散动作空间，通常设为 None，Runner 内部会自动处理为全 1
        # available_actions = None
        #
        #
        #
        # return local_obs, share_obs, reward_arr, dones, infos, available_actions
    def render(self, mode='human'):
        # 直接调用原始环境的 render
        return self.env.render()

    def close(self):
        self.env.close()