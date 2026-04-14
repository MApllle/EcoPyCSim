"""
MAPPO（Multi-Agent Proximal Policy Optimization）原生实现。

核心设计（对齐 schedulers/marl/ 中其他算法的接口风格）：
  - 每个 agent 持有独立 Actor（支持异构观察维度）
  - 所有 agent 共享一个集中式 Critic（输入为所有 agent 观察拼接）
  - On-Policy：每个 episode 收集完整轨迹 → GAE → 多轮 PPO 更新
  - 自包含实现，不依赖外部仓库

参考文献：
  Lowe et al., 2017 (MADDPG)  — 集中式 Critic 思想
  Yu et al., 2021 (MAPPO)     — PPO 在 MARL 中的应用
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn

from schedulers.marl.mappo.buffer import RolloutBuffer
from schedulers.marl.mappo.networks import Actor, Critic
from schedulers.marl.mappo.utils import (
    ValueNorm, check, get_gard_norm, huber_loss, mse_loss,
    update_linear_schedule,
)


def _setup_logger(path: str) -> logging.Logger:
    logger = logging.getLogger(f"mappo_{path}")
    logger.setLevel(logging.INFO)
    h = logging.FileHandler(path, mode='w')
    h.setFormatter(logging.Formatter(
        '%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(h)
    return logger


class MAPPO:
    """
    MAPPO 协调器：集中式 Critic + 分散式 Actor，on-policy PPO 更新。

    Args:
        dim_info:       {agent_id: {'obs_shape': {key: shape,...}, 'action_dim': int}}
                        与 MADDPG/QMIX/VDN 完全相同的格式。
        episode_length: 每个 episode 的固定步数（= num_jobs）。
        num_mini_batch: PPO 更新时的小批量数量。
        lr:             Actor 与 Critic 共用学习率。
        res_dir:        结果保存目录。
        hidden_size:    隐层宽度。
        num_layers:     MLP 隐层数（不含输入层）。
        ppo_epoch:      每次 rollout 后 PPO 更新轮数。
        clip_param:     PPO clip 范围 ε。
        value_loss_coef: 值函数损失权重。
        entropy_coef:   熵正则化系数。
        max_grad_norm:  梯度裁剪阈值。
        use_huber_loss: 值函数损失使用 Huber 还是 MSE。
        huber_delta:    Huber loss δ。
        use_valuenorm:  是否用 ValueNorm 归一化值函数目标。
        gamma:          折扣因子。
        gae_lambda:     GAE λ。
        use_linear_lr_decay: 是否线性衰减学习率。
        device:         计算设备。
    """

    def __init__(self,
                 dim_info: dict,
                 episode_length: int,
                 num_mini_batch: int = 1,
                 lr: float = 5e-4,
                 res_dir: str = '.',
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 ppo_epoch: int = 10,
                 clip_param: float = 0.2,
                 value_loss_coef: float = 1.0,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 10.0,
                 use_huber_loss: bool = True,
                 huber_delta: float = 10.0,
                 use_valuenorm: bool = True,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 use_linear_lr_decay: bool = False,
                 device=None):

        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.tpdv = dict(dtype=torch.float32, device=self.device)

        # ── 维度计算 ──────────────────────────────────────────────────────────
        self.agent_ids = list(dim_info.keys())
        self.num_agents = len(self.agent_ids)

        self.obs_dims = {
            aid: int(sum(np.prod(s) for s in info['obs_shape'].values()))
            for aid, info in dim_info.items()
        }
        self.act_dims = {aid: info['action_dim'] for aid, info in dim_info.items()}
        self.cent_obs_dim = sum(self.obs_dims.values())

        # ── 超参数 ────────────────────────────────────────────────────────────
        self.episode_length = episode_length
        self.num_mini_batch = num_mini_batch
        self.lr = lr
        self.ppo_epoch = ppo_epoch
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        self.use_valuenorm = use_valuenorm
        self.use_linear_lr_decay = use_linear_lr_decay

        # ── 网络 ──────────────────────────────────────────────────────────────
        self.actors = nn.ModuleDict({
            aid: Actor(self.obs_dims[aid], self.act_dims[aid],
                       hidden_size, num_layers, device=self.device)
            for aid in self.agent_ids
        })
        self.critic = Critic(self.cent_obs_dim, hidden_size,
                             num_layers, device=self.device)

        # ── 优化器 ────────────────────────────────────────────────────────────
        self.actor_optimizers = {
            aid: torch.optim.Adam(self.actors[aid].parameters(), lr=lr, eps=1e-5)
            for aid in self.agent_ids
        }
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr, eps=1e-5)

        # ── 值函数归一化 ──────────────────────────────────────────────────────
        self.value_normalizer = ValueNorm(1, device=self.device) if use_valuenorm else None

        # ── 缓冲区 ────────────────────────────────────────────────────────────
        self.buffer = RolloutBuffer(
            agent_ids=self.agent_ids,
            obs_dims=self.obs_dims,
            cent_obs_dim=self.cent_obs_dim,
            episode_length=episode_length,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # ── 日志 ──────────────────────────────────────────────────────────────
        self.res_dir = res_dir
        os.makedirs(res_dir, exist_ok=True)
        self.logger = _setup_logger(os.path.join(res_dir, 'mappo.log'))

    # ── 辅助：obs 处理 ────────────────────────────────────────────────────────

    def flatten_obs(self, obs_dict: dict) -> np.ndarray:
        """将单个 agent 的 obs dict 展平为 1-D numpy array。"""
        parts = []
        for key in sorted(obs_dict.keys()):
            v = obs_dict[key]
            parts.extend(v.flatten() if isinstance(v, np.ndarray) else [v])
        return np.array(parts, dtype=np.float32)

    def _flatten_all(self, obs: dict) -> dict:
        """对所有 agent 的 obs 执行展平，返回 {agent_id: np.array}。"""
        return {aid: self.flatten_obs(obs[aid]) for aid in self.agent_ids if aid in obs}

    def _cent_obs(self, flat_obs: dict) -> np.ndarray:
        """拼接所有 agent 展平后的 obs，得到集中观察 (N, cent_obs_dim)。"""
        single = np.concatenate([flat_obs[aid] for aid in self.agent_ids])
        return np.tile(single, (self.num_agents, 1))  # (N, cent_obs_dim)

    # ── 推理：选取动作 ────────────────────────────────────────────────────────

    @torch.no_grad()
    def collect(self, obs: dict, deterministic: bool = False):
        """
        根据当前观察采样动作、log prob 及 Critic 值。

        Args:
            obs:          {agent_id: obs_dict}
            deterministic: 是否取贪心动作

        Returns:
            actions:   {agent_id: int}
            log_probs: np.array (N, 1)
            values:    np.array (N, 1)
            flat_obs:  {agent_id: np.array}  展平后的局部观察
            cent_obs:  np.array (N, cent_obs_dim)
        """
        flat_obs = self._flatten_all(obs)
        cent = self._cent_obs(flat_obs)

        actions = {}
        log_probs_list = []

        for i, aid in enumerate(self.agent_ids):
            self.actors[aid].eval()
            o = torch.tensor(flat_obs[aid], dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            act, lp, _ = self.actors[aid](o, deterministic=deterministic)
            actions[aid] = act.item()
            log_probs_list.append(lp.cpu().numpy().reshape(1, 1))

        # Critic 评估（以任意一个 agent 的集中观察为输入，此处取 agent 0）
        self.critic.eval()
        cent_t = torch.tensor(cent, dtype=torch.float32, device=self.device)  # (N, D)
        values, _ = self.critic(cent_t)
        values_np = values.cpu().numpy()  # (N, 1)

        log_probs = np.concatenate(log_probs_list, axis=0)  # (N, 1)
        return actions, log_probs, values_np, flat_obs, cent

    # ── 缓冲区写入 ────────────────────────────────────────────────────────────

    def insert(self, step: int,
               flat_obs: dict, cent_obs: np.ndarray,
               actions: dict, log_probs: np.ndarray,
               values: np.ndarray, rewards: dict, dones: dict):
        """
        将当前步的数据存入 buffer。

        注意：obs / cent_obs 存入的是"next obs"（下一步的初始状态）。
        """
        # 把 dict 形式的 actions/rewards/dones 转为 (N, 1) 数组
        act_arr = np.array([[actions[aid]] for aid in self.agent_ids], dtype=np.float32)
        rew_arr = np.array([[rewards[aid]] for aid in self.agent_ids], dtype=np.float32)
        mask_arr = np.array(
            [[0.0] if dones.get(aid, False) else [1.0] for aid in self.agent_ids],
            dtype=np.float32)

        self.buffer.insert(step, flat_obs, cent_obs, act_arr,
                           log_probs, values, rew_arr, mask_arr)

    def set_initial_obs(self, obs: dict):
        """episode 开始时设置初始帧（step=0）的 obs。"""
        flat_obs = self._flatten_all(obs)
        cent = self._cent_obs(flat_obs)
        self.buffer.set_initial_obs(flat_obs, cent)

    # ── 回报计算 ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_returns(self, last_obs: dict, last_dones: dict):
        """
        用 bootstrap 值计算 GAE 回报。

        Args:
            last_obs:   episode 结束后的最后观察
            last_dones: {agent_id: bool}
        """
        flat_obs = self._flatten_all(last_obs)
        cent = self._cent_obs(flat_obs)
        cent_t = torch.tensor(cent, dtype=torch.float32, device=self.device)

        self.critic.eval()
        next_values, _ = self.critic(cent_t)
        next_values_np = next_values.cpu().numpy()  # (N, 1)

        # 若已终止则 bootstrap = 0
        for i, aid in enumerate(self.agent_ids):
            if last_dones.get(aid, False):
                next_values_np[i] = 0.0

        self.buffer.compute_returns(next_values_np, self.value_normalizer)

    # ── PPO 更新 ──────────────────────────────────────────────────────────────

    def learn(self) -> dict:
        """
        在当前 rollout 数据上执行 ppo_epoch 轮 PPO 更新。

        Returns:
            train_info: {'value_loss', 'policy_loss', 'dist_entropy',
                         'actor_grad_norm', 'critic_grad_norm', 'ratio'}
        """
        # 计算优势估计
        if self.value_normalizer is not None:
            advantages = (self.buffer.returns[:-1]
                          - self.value_normalizer.denormalize(self.buffer.values[:-1]))
        else:
            advantages = self.buffer.returns[:-1] - self.buffer.values[:-1]

        # 归一化优势（忽略已终止 agent）
        active = self.buffer.masks[:-1]  # (T, N, 1)
        adv_copy = advantages.copy()
        adv_copy[active == 0.0] = np.nan
        mean_adv = np.nanmean(adv_copy)
        std_adv = np.nanstd(adv_copy)
        advantages = (advantages - mean_adv) / (std_adv + 1e-5)

        train_info = {k: 0.0 for k in
                      ['value_loss', 'policy_loss', 'dist_entropy',
                       'actor_grad_norm', 'critic_grad_norm', 'ratio']}
        num_updates = 0

        for _ in range(self.ppo_epoch):
            # batch 形状：obs_b {aid: (B, obs_dim)}, cent_obs_b (B, N, D),
            #              actions_b/log_probs_b/values_b/returns_b/masks_b/adv_b: (B, N, 1)
            for batch in self.buffer.feed_forward_generator(advantages, self.num_mini_batch):
                (obs_b, cent_obs_b, actions_b, old_log_probs_b,
                 values_b, returns_b, masks_b, adv_b) = batch

                # ── Actor 更新（每个 agent 独立） ─────────────────────────────
                policy_loss_total = torch.tensor(0.0, device=self.device)
                entropy_total = torch.tensor(0.0, device=self.device)
                actor_grad = 0.0

                for i, aid in enumerate(self.agent_ids):
                    # 取 agent i 在每个时间步的数据：索引第 i 列
                    obs_t = check(obs_b[aid]).to(**self.tpdv)          # (B, obs_dim_i)
                    act_t = check(actions_b[:, i, :]).to(**self.tpdv)  # (B, 1)
                    old_lp_i = check(old_log_probs_b[:, i, :]).to(**self.tpdv)  # (B, 1)
                    adv_i = check(adv_b[:, i, :]).to(**self.tpdv)      # (B, 1)

                    self.actors[aid].train()
                    log_probs_new, entropy = self.actors[aid].evaluate_actions(obs_t, act_t)

                    imp_w = torch.exp(log_probs_new - old_lp_i)
                    surr1 = imp_w * adv_i
                    surr2 = torch.clamp(imp_w, 1 - self.clip_param,
                                        1 + self.clip_param) * adv_i
                    actor_loss = -torch.min(surr1, surr2).mean()

                    self.actor_optimizers[aid].zero_grad()
                    (actor_loss - entropy * self.entropy_coef).backward()
                    grad_n = nn.utils.clip_grad_norm_(
                        self.actors[aid].parameters(), self.max_grad_norm)
                    self.actor_optimizers[aid].step()

                    policy_loss_total += actor_loss.detach()
                    entropy_total += entropy.detach()
                    actor_grad += float(grad_n)
                    train_info['ratio'] += imp_w.mean().item()

                # ── Critic 更新 ───────────────────────────────────────────────
                # 将 (B, N, D) 展平为 (B*N, D) 统一处理
                B = cent_obs_b.shape[0]
                N = self.num_agents
                cent_flat = check(cent_obs_b.reshape(B * N, -1)).to(**self.tpdv)
                values_flat = check(values_b.reshape(B * N, 1)).to(**self.tpdv)
                returns_flat = check(returns_b.reshape(B * N, 1)).to(**self.tpdv)

                self.critic.train()
                new_values, _ = self.critic(cent_flat)  # (B*N, 1)

                if self.value_normalizer is not None:
                    self.value_normalizer.update(returns_flat)
                    norm_ret = self.value_normalizer.normalize(returns_flat)
                    error_clip = norm_ret - (values_flat
                                            + (new_values - values_flat)
                                            .clamp(-self.clip_param, self.clip_param))
                    error_orig = norm_ret - new_values
                else:
                    error_clip = returns_flat - (values_flat
                                                 + (new_values - values_flat)
                                                 .clamp(-self.clip_param, self.clip_param))
                    error_orig = returns_flat - new_values

                if self.use_huber_loss:
                    vl_clip = huber_loss(error_clip, self.huber_delta)
                    vl_orig = huber_loss(error_orig, self.huber_delta)
                else:
                    vl_clip = mse_loss(error_clip)
                    vl_orig = mse_loss(error_orig)

                value_loss = torch.max(vl_orig, vl_clip).mean()

                self.critic_optimizer.zero_grad()
                (value_loss * self.value_loss_coef).backward()
                critic_grad = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += (policy_loss_total / self.num_agents).item()
                train_info['dist_entropy'] += (entropy_total / self.num_agents).item()
                train_info['actor_grad_norm'] += actor_grad / self.num_agents
                train_info['critic_grad_norm'] += float(critic_grad)
                num_updates += 1

        for k in train_info:
            train_info[k] /= max(num_updates, 1)

        return train_info

    # ── 学习率调度 ────────────────────────────────────────────────────────────

    def lr_decay(self, episode: int, total_episodes: int):
        if self.use_linear_lr_decay:
            for aid in self.agent_ids:
                update_linear_schedule(
                    self.actor_optimizers[aid], episode, total_episodes, self.lr)
            update_linear_schedule(
                self.critic_optimizer, episode, total_episodes, self.lr)

    # ── 存储 / 加载 ───────────────────────────────────────────────────────────

    def save(self, reward=None):
        """保存所有 Actor 与 Critic 的参数。"""
        state = {
            'actors': {aid: self.actors[aid].state_dict() for aid in self.agent_ids},
            'critic': self.critic.state_dict(),
        }
        torch.save(state, os.path.join(self.res_dir, 'model.pt'))

    @classmethod
    def load(cls, dim_info: dict, file: str, episode_length: int,
             num_mini_batch: int = 1, lr: float = 5e-4,
             hidden_size: int = 64, device=None, **kwargs):
        """从保存文件还原 MAPPO 实例（用于推理）。"""
        instance = cls(dim_info, episode_length, num_mini_batch, lr,
                       res_dir=os.path.dirname(file),
                       hidden_size=hidden_size, device=device, **kwargs)
        state = torch.load(file, map_location=instance.device, weights_only=False)
        for aid in instance.agent_ids:
            instance.actors[aid].load_state_dict(state['actors'][aid])
        instance.critic.load_state_dict(state['critic'])
        return instance
