import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner

class CloudRunner(Runner):
    """
    云调度任务专用 Runner
    """
    def __init__(self, config):
        super(CloudRunner, self).__init__(config)
        # 初始化一些统计变量
        self.total_price = 0
        self.total_steps = 0

    def run(self):
        self.all_eval_prices = [] # 用来存每一轮的价格
        self.total_cost_accumulator = 0.0

        print("--- [Runner] 开始运行评估/训练 ---")

        # 1. 重置环境并处理观察空间维度
        results = self.envs.reset()

        # 统一 obs 维度为 (n_threads, n_agents, obs_shape)
        # 针对 DummyVecEnv 返回的 (1, 2, 101) 或 (2, 101) 进行对齐
        obs = np.array(results)
        if len(obs.shape) == 4: # 针对某些 Wrapper 多包了一层的情况
            obs = np.squeeze(obs, axis=0)
        elif len(obs.shape) == 2: # 只有 (n_agents, dim) 时补上 thread 维度
            obs = np.expand_dims(obs, axis=0)

        # 2. 手动构造共享观察空间 (share_obs)
        # 将所有智能体的 obs 拼接：(n_threads, n_agents * obs_shape)
        # 然后复制给每个智能体，形成 (n_threads, n_agents, share_obs_shape)
        share_obs = obs.reshape(self.n_rollout_threads, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)

        # 3. 初始化 Buffer
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

        # 4. 开始主循环
        # 计算总的迭代次数
        episodes = int(self.all_args.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay and not self.all_args.use_eval:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # 采样动作：利用当前策略网络进行推理
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # 与环境交互
                # 注意：这里需要确保返回的 obs/share_obs 维度正确
                obs, rewards, dones, infos = self.envs.step(actions)

                temp_env = self.envs.envs[0]
                while hasattr(temp_env, "env"):
                    temp_env = temp_env.env

                # 获取当前步的瞬时总开销并累加
                current_step_price = sum(sf.get_price for sf in temp_env.server_farms.values())
                self.total_cost_accumulator += current_step_price

                share_obs = np.array([info.get("share_obs") for info in infos])
                # # 渲染：如果是 eval 模式，这一步会打印出价格和调度细节
                if self.all_args.use_render:
                    self.envs.render()

                # 整理数据并存入 Buffer
                data = (obs, share_obs, rewards, dones, infos,
                        values, actions, action_log_probs,
                        rnn_states, rnn_states_critic)
                self.insert(data)

            # 5. 训练/评估后的逻辑
            if not self.all_args.use_eval:
                # 仅在非评估模式下更新网络
                self.compute()
                train_infos = self.train()

                # 记录日志
                total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
                if episode % self.all_args.log_interval == 0:
                    print(f"Steps: {total_num_steps}/{self.all_args.num_env_steps}, "
                          f"Reward: {np.mean(self.buffer.rewards):.4f}")
                    self.log_train(train_infos, total_num_steps)

                # 定期保存模型
                if episode % self.all_args.save_interval == 0 or episode == episodes - 1:
                    self.save()
                    print(f"--- [INFO] 模型已保存至 Episode {episode} ---")
            else:
                # 评估模式下，每跑完一个 episode 打印简单的提示
                print(f"--- [Eval] Episode {episode} 完成 ---")

                # 这里的 total_cost_accumulator 已经存了该 Episode 几百步下来的累加总和
                final_accumulated_price = self.total_cost_accumulator

                print(f"--- [Episode {episode}] 运行步数: {self.episode_length} | 累加总价格: {final_accumulated_price:.2f} ---")

                temp_env = self.envs.envs[0]
                while hasattr(temp_env, "env"):
                    temp_env = temp_env.env

                # # 实时计算当前环境的总开销
                # final_price = sum(sf.get_price for sf in temp_env.server_farms.values())
                # print(f"--- [Episode {episode}] 结算价格: {final_price:.2f} ---")

                # 存入列表，而不是覆盖单个变量
                self.all_eval_prices.append(final_accumulated_price)

                # 为了兼容外部读取，计算目前的平均值
                self.final_eval_price = np.mean(self.all_eval_prices)
                self.total_cost_accumulator = 0.0

        print(f"--- [Runner] 运行结束 | {episodes}轮评估平均总价格: {self.final_eval_price:.2f} ---")


    # def run(self):
    #     # 1. 重置环境
    #     # obs: (n_rollout_threads, num_agents, obs_shape)
    #     # obs, share_obs, _ = self.envs.reset()
    #     obs = self.envs.reset()
    #     # 因为是共享策略模式，share_obs 通常是 obs 展平后的拼接
    #     # 我们在 Runner 里手动构造它，或者直接让它等于 obs (取决于你 policy 的输入要求)
    #     if len(obs.shape) == 4: # 说明返回的是 [threads, agents, dim] 之外还有一层
    #         obs = np.squeeze(obs, axis=0)
    #
    #     # 构造 share_obs: 将所有 agent 的 obs 拼在一起
    #     # 形状从 (n_threads, n_agents, max_obs_dim) -> (n_threads, n_agents, share_obs_dim)
    #     share_obs = obs.reshape(self.n_rollout_threads, -1)
    #     share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
    #
    #     # 2. 将初始观察存入 Buffer
    #     self.buffer.share_obs[0] = share_obs.copy()
    #     self.buffer.obs[0] = obs.copy()
    #
    #     # 开始训练循环
    #     for episode in range(self.all_args.num_env_steps // self.episode_length // self.n_rollout_threads):
    #         if self.use_linear_lr_decay:
    #             self.trainer.policy.lr_decay(episode, self.num_env_steps)
    #
    #         for step in range(self.episode_length):
    #             # Sample actions
    #             # values, actions, action_log_probs, rnn_states, rnn_states_critic
    #             values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
    #
    #             # Obser reward and next obs
    #             obs, share_obs, rewards, dones, infos, _ = self.envs.step(actions)
    #
    #             data = (obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic)
    #
    #             # insert data into buffer
    #             self.insert(data)
    #
    #         # 训练阶段
    #         self.compute()
    #         train_infos = self.train()
    #
    #         # 记录日志
    #         total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
    #         if episode % self.all_args.log_interval == 0:
    #             print(f"Steps: {total_num_steps}, Episode: {episode}, Reward: {np.mean(self.buffer.rewards)}")
    #             self.log_train(train_infos, total_num_steps)
    #
    #         # 3. 【新增保存逻辑】
    #         if (episode % self.all_args.save_interval == 0 or episode == self.all_args.num_env_steps // self.episode_length // self.n_rollout_threads - 1):
    #             self.save() # 调用父类的 save 方法
    #             print(f"--- [INFO] 模型已保存至 episode {episode} ---")

    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))

        # [threads, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        # 1. 生成 masks (如果 dones 是 True，mask 就是 0)
        # dones 维度通常是 [n_threads, n_agents]
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # 2. 处理 RNN 状态：如果环境结束了，下一轮的 RNN 初始状态要清零
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)

        # 3. 存入 buffer
        # 注意顺序：share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                          actions, action_log_probs, values, rewards, masks)

def _t2n(x):
    return x.detach().cpu().numpy()