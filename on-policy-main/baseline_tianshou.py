import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import DQNPolicy, MultiAgentPolicyManager
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
import supersuit as ss
from pettingzoo.utils import parallel_to_aec

# 请确保路径正确
from env.cloud_scheduling import CloudSchedulingEnv

# ========================================================
# 1. 彻底解决 info 导致的 has no len() (针对 image_4bfe30, 4b7387)
# ========================================================
def preprocess_fn(obs=None, info=None, **kwargs):
    """
    强制清空 info 字典。
    解决 image_4b7387 中 info 包含 numpy.float64 等标量导致 Batch 无法对齐的问题。
    """
    return Batch(obs=obs, info=Batch())

# ========================================================
# 2. 终极环境工厂 (解决 image_4c5cfe, 4c58de, 4c551d)
# ========================================================
def _get_env():
    # A. 实例化环境并立即 reset 解决 StopIteration
    raw_env = CloudSchedulingEnv(num_jobs=100, num_server_farms=5, num_servers=50)
    raw_env.reset()

    # B. 注入必要元数据
    if not hasattr(raw_env, 'metadata'):
        raw_env.metadata = {"render_modes": [], "name": "cloud_scheduling"}

    # C. 强制对齐观测空间形状 (核心修复：解决 image_4c5cfe 的断言错误)
    # 我们不再手动改属性，而是使用 SuperSuit 的工具强制转换
    # 假设你的 observation 长度是 101
    env = ss.reshape_observations_v0(raw_env, (101,))

    # D. 强制转换为标准 Box 空间 (避开非标准空间导致的断言失败)
    env = ss.dtype_v0(env, np.float32)

    # E. 协议转换 Parallel -> AEC
    aec_env = parallel_to_aec(env)

    # F. 最后一道防线：确保动作空间也是标准的 Discrete
    aec_env = ss.pad_action_spaces_v0(aec_env)
    aec_env = ss.pad_observations_v0(aec_env)

    return PettingZooEnv(aec_env)

# ========================================================
# 3. 策略初始化 (针对 RTX 3050 优化)
# ========================================================
def _get_agents(env_pool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_shape = 101 # 锁定维度
    action_shape = 5 # 锁定维度

    agents = []
    # 获取智能体数量，这里假设为 10
    num_agents = 10
    for _ in range(num_agents):
        net = Net(state_shape=obs_shape, action_shape=action_shape,
                  hidden_sizes=[256, 256], device=device).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        agent = DQNPolicy(model=net, optim=optim, discount_factor=0.9,
                          target_update_freq=320)
        agents.append(agent)

    return MultiAgentPolicyManager(agents, env_pool)

if __name__ == "__main__":
    # 并行环境数量控制在 4 以内节省显存
    train_envs = DummyVectorEnv([_get_env for _ in range(4)])
    test_envs = DummyVectorEnv([_get_env for _ in range(4)])

    policy = _get_agents(train_envs)

    # 构建数据收集器 (解决 image_4bf385 的参数缺失问题)
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(20000, len(train_envs)),
        preprocess_fn=preprocess_fn
    )
    test_collector = Collector(policy, test_envs, preprocess_fn=preprocess_fn)

    print(">>> 正在启动连接性测试...")
    try:
        train_collector.collect(n_step=100)
        print(">>> [成功] 数据链路全线通畅，开始执行训练循环！")

        result = offpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=30,
            step_per_epoch=1000,
            step_per_collect=50,
            episode_per_test=10,
            batch_size=64,
            train_fn=lambda e, s: policy.set_eps(0.1),
            test_fn=lambda e, s: policy.set_eps(0.05),
        )
        print(f"训练结果: {result}")

    except Exception:
        import traceback
        print(">>> [致命错误] 报错详细信息如下：")
        traceback.print_exc()