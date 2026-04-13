"""
MAPPO 轨迹缓冲区（On-Policy Rollout Buffer）。
简化自 on-policy-main/onpolicy/utils/shared_buffer.py：
  - 去除多线程支持（n_rollout_threads = 1）
  - 去除 gym.Space 依赖，直接接受整数维度
  - 保留 GAE 优势估计 / n-step 回报计算
  - 保留 feed_forward_generator 小批量生成器
  - 支持异构 agent（不同 obs_dim，用 dict 存储）
"""
import numpy as np
import torch


class RolloutBuffer:
    """
    存储单次 rollout（episode_length 步）的轨迹数据。

    数组形状约定（省略线程维度，n_agents = N）：
        obs         dict {agent_id: (T+1, obs_dim_i)}
        cent_obs    (T+1, N, cent_obs_dim)   集中观察（Critic 输入）
        actions     (T,   N, 1)              整数动作
        log_probs   (T,   N, 1)              log π(a|o)
        values      (T+1, N, 1)              Critic 预测值
        rewards     (T,   N, 1)
        masks       (T+1, N, 1)              1 = 存活, 0 = 终止
        returns     (T+1, N, 1)              GAE / n-step 回报
    """

    def __init__(self,
                 agent_ids: list,
                 obs_dims: dict,         # {agent_id: int}
                 cent_obs_dim: int,
                 episode_length: int,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 use_gae: bool = True):
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        self.T = episode_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_gae = use_gae

        N = self.num_agents
        # 异构 obs，每个 agent 单独存储
        self.obs = {
            aid: np.zeros((self.T + 1, obs_dims[aid]), dtype=np.float32)
            for aid in agent_ids
        }
        self.cent_obs = np.zeros((self.T + 1, N, cent_obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.T, N, 1), dtype=np.float32)
        self.log_probs = np.zeros((self.T, N, 1), dtype=np.float32)
        self.values = np.zeros((self.T + 1, N, 1), dtype=np.float32)
        self.rewards = np.zeros((self.T, N, 1), dtype=np.float32)
        self.masks = np.ones((self.T + 1, N, 1), dtype=np.float32)
        self.returns = np.zeros((self.T + 1, N, 1), dtype=np.float32)

        self.step = 0

    # ── 写入 ─────────────────────────────────────────────────────────────────

    def insert(self, step: int,
               obs_flat: dict,      # {agent_id: np.array (obs_dim_i,)}
               cent_obs: np.ndarray,  # (N, cent_obs_dim)
               actions: np.ndarray,   # (N, 1)
               log_probs: np.ndarray, # (N, 1)
               values: np.ndarray,    # (N, 1)
               rewards: np.ndarray,   # (N, 1)
               masks: np.ndarray):    # (N, 1)  已终止 = 0
        for aid in self.agent_ids:
            self.obs[aid][step + 1] = obs_flat[aid]
        self.cent_obs[step + 1] = cent_obs
        self.actions[step] = actions
        self.log_probs[step] = log_probs
        self.values[step] = values
        self.rewards[step] = rewards
        self.masks[step + 1] = masks

    def set_initial_obs(self, obs_flat: dict, cent_obs: np.ndarray):
        """在 episode 开始时初始化第 0 帧。"""
        for aid in self.agent_ids:
            self.obs[aid][0] = obs_flat[aid]
        self.cent_obs[0] = cent_obs

    # ── GAE 回报计算 ─────────────────────────────────────────────────────────

    def compute_returns(self, next_value: np.ndarray,
                        value_normalizer=None):
        """
        计算每一步的 GAE 回报。

        Args:
            next_value:       (N, 1) 最后一步之后的 bootstrap 值
            value_normalizer: ValueNorm 实例（可选）
        """
        self.values[-1] = next_value

        if self.use_gae:
            gae = np.zeros((self.num_agents, 1), dtype=np.float32)
            for t in reversed(range(self.T)):
                if value_normalizer is not None:
                    v_t = value_normalizer.denormalize(self.values[t])
                    v_tp1 = value_normalizer.denormalize(self.values[t + 1])
                else:
                    v_t = self.values[t]
                    v_tp1 = self.values[t + 1]

                delta = self.rewards[t] + self.gamma * v_tp1 * self.masks[t + 1] - v_t
                gae = delta + self.gamma * self.gae_lambda * self.masks[t + 1] * gae

                if value_normalizer is not None:
                    self.returns[t] = gae + v_t
                else:
                    self.returns[t] = gae + self.values[t]
        else:
            self.returns[-1] = next_value
            for t in reversed(range(self.T)):
                self.returns[t] = (self.returns[t + 1] * self.gamma
                                   * self.masks[t + 1] + self.rewards[t])

    # ── 帧推进 ───────────────────────────────────────────────────────────────

    def after_update(self):
        """episode 结束后将末帧数据复制到初始帧，供下一 episode 使用。"""
        for aid in self.agent_ids:
            self.obs[aid][0] = self.obs[aid][-1].copy()
        self.cent_obs[0] = self.cent_obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()

    # ── 小批量生成器 ──────────────────────────────────────────────────────────

    def feed_forward_generator(self, advantages: np.ndarray, num_mini_batch: int):
        """
        生成无 RNN 的 MLP 小批量数据。

        以**时间步**为采样单位（batch_size = T // num_mini_batch 个时间步），
        每个样本包含所有 N 个 agent 在该时间步的数据。
        这样可以自然支持异构 obs_dim（不同 agent 的 obs 分开存储）。

        Args:
            advantages:    (T, N, 1) 优势估计（已归一化）
            num_mini_batch: 小批量数量

        Yields:
            obs_batch:    {agent_id: np.array (B, obs_dim_i)}  各 agent 局部观察
            cent_obs_b:   np.array (B, N, cent_obs_dim)         集中观察
            actions_b:    np.array (B, N, 1)                    动作
            log_probs_b:  np.array (B, N, 1)                    旧 log π
            values_b:     np.array (B, N, 1)                    旧值预测
            returns_b:    np.array (B, N, 1)                    GAE 回报
            masks_b:      np.array (B, N, 1)                    存活掩码
            adv_b:        np.array (B, N, 1)                    归一化优势
        """
        T = self.T
        assert T >= num_mini_batch, (
            f"episode_length {T} < num_mini_batch {num_mini_batch}"
        )
        mini_batch_size = T // num_mini_batch

        rand = torch.randperm(T).numpy()
        for i in range(num_mini_batch):
            idx = rand[i * mini_batch_size: (i + 1) * mini_batch_size]
            obs_batch = {aid: self.obs[aid][idx] for aid in self.agent_ids}
            yield (
                obs_batch,
                self.cent_obs[idx],           # (B, N, cent_obs_dim)
                self.actions[idx],             # (B, N, 1)
                self.log_probs[idx],           # (B, N, 1)
                self.values[idx],              # (B, N, 1)
                self.returns[idx],             # (B, N, 1)
                self.masks[idx],               # (B, N, 1)
                advantages[idx],               # (B, N, 1)
            )
