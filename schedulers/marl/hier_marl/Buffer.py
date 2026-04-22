import numpy as np
import torch


class LocalBuffer:
    """每个 local agent 独立的 replay buffer，仅在被选中那步入队。"""

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = capacity
        self.device = device

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

        self._index = 0
        self._size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self._index
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self._index = (self._index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, indices):
        obs = torch.from_numpy(self.obs[indices]).to(self.device)
        action = torch.from_numpy(self.action[indices]).to(self.device)
        reward = torch.from_numpy(self.reward[indices]).to(self.device)
        reward = reward / 2.0  # fixed scaling; batch normalisation is wrong for off-policy
        next_obs = torch.from_numpy(self.next_obs[indices]).to(self.device)
        done = torch.from_numpy(self.done[indices]).to(self.device)
        return obs, action, reward, next_obs, done

    def __len__(self):
        return self._size


class GlobalBuffer:
    """记录 global agent 的转移，并附带被选中 local 的邻域信息，用于邻域 critic 训练。

    每条记录包含：
        (g_obs, g_act, reward, g_next_obs, done,
         selected_id, l_obs, l_act, l_next_obs)
    Critic 输入维度与 local agent 总数 N 无关。
    """

    def __init__(self, capacity, global_obs_dim, global_act_dim,
                 local_obs_dim, local_act_dim, device):
        self.capacity = capacity
        self.device = device

        self.g_obs = np.zeros((capacity, global_obs_dim), dtype=np.float32)
        self.g_act = np.zeros((capacity, global_act_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.g_next_obs = np.zeros((capacity, global_obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

        self.selected_id = np.zeros(capacity, dtype=np.int64)
        self.l_obs = np.zeros((capacity, local_obs_dim), dtype=np.float32)
        self.l_act = np.zeros((capacity, local_act_dim), dtype=np.float32)
        self.l_next_obs = np.zeros((capacity, local_obs_dim), dtype=np.float32)

        self._index = 0
        self._size = 0

    def add(self, g_obs, g_act, reward, g_next_obs, done,
            selected_id, l_obs, l_act, l_next_obs):
        i = self._index
        self.g_obs[i] = g_obs
        self.g_act[i] = g_act
        self.reward[i] = reward
        self.g_next_obs[i] = g_next_obs
        self.done[i] = float(done)
        self.selected_id[i] = selected_id
        self.l_obs[i] = l_obs
        self.l_act[i] = l_act
        self.l_next_obs[i] = l_next_obs
        self._index = (self._index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self._size, batch_size, replace=False)
        to_t = lambda arr: torch.from_numpy(arr[indices]).to(self.device)

        reward = to_t(self.reward)
        reward = reward / 2.0  # fixed scaling; batch normalisation is wrong for off-policy

        return (
            to_t(self.g_obs),
            to_t(self.g_act),
            reward,
            to_t(self.g_next_obs),
            to_t(self.done),
            self.selected_id[indices],   # numpy int64 array，供索引 local buffer
            to_t(self.l_obs),
            to_t(self.l_act),
            to_t(self.l_next_obs),
        )

    def __len__(self):
        return self._size
