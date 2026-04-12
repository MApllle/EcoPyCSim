import sys
import os
import types
import numpy as np
import torch
from pathlib import Path

# --- 1. Windows 兼容性与路径补丁 (必须置顶) ---
if sys.platform == "win32":
    for module_name in ["grp", "pwd"]:
        if module_name not in sys.modules:
            sys.modules[module_name] = types.ModuleType(module_name)

# 确保项目根目录在 sys.path 中
root_dir = str(Path(__file__).resolve().parents[3])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- 2. 导入框架与环境组件 ---
import wandb
from onpolicy.config import get_config
from onpolicy.envs.cloud_env_wrapper import CloudEnvWrapper
from env.cloud_scheduling import CloudSchedulingEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from onpolicy.runner.shared.cloud_runner import CloudRunner as Runner

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            base_env = CloudSchedulingEnv(
                num_jobs=all_args.num_jobs,
                num_server_farms=5,
                num_servers=50
            )
            return CloudEnvWrapper(base_env)
        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def main(args):
    parser = get_config()
    # 添加自定义参数
    parser.add_argument('--num_jobs', type=int, default=100)
    all_args = parser.parse_known_args(args)[0]

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() and all_args.cuda else "cpu")

    # 运行目录设置
    run_dir = Path(root_dir) / "onpolicy" / "scripts" / "results" / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # 初始化环境
    envs = make_train_env(all_args)
    eval_envs = None

    # 智能体数量识别
    try:
        if isinstance(envs.observation_space, list):
            num_agents = len(envs.observation_space)
        else:
            num_agents = envs.num_agents
    except AttributeError:
        num_agents = 2  # 兜底：server_farm 和 server

    print(f"--- [INFO] 环境启动成功，检测到智能体数量: {num_agents} ---")

    # --- 3. 处理 Wandb 离线模式 ---
    # os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_MODE"] = "offline"

    if wandb.run is None:
        wandb.init(
            project=all_args.env_name,
            name=all_args.experiment_name,
            config=all_args,
            dir=str(run_dir),
            # mode="disabled"
            mode="offline"
        )

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # --- 4. 启动训练 ---
    runner = Runner(config)
    runner.run()

    # 训练结束关闭环境
    envs.close()
    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    main(sys.argv[1:])

# import sys
# if sys.platform == "win32":
#     import types
#     m = types.ModuleType("grp")
#     sys.modules["grp"] = m
#     sys.modules["pwd"] = m
#
# import os
# from pathlib import Path
# import wandb
#
# # 这一步是关键：手动把项目根目录添加到 Python 的搜索路径中
# # __file__ 是当前文件，parents[3] 刚好指向 on-policy-main
# root_dir = str(Path(__file__).resolve().parents[3])
# if root_dir not in sys.path:
#     sys.path.insert(0, root_dir)
#
# # 确保根目录下的 env 文件夹也能被找到
# if os.getcwd() not in sys.path:
#     sys.path.insert(0, os.getcwd())
#
#
# from onpolicy.runner.shared.cloud_runner import CloudRunner as Runner
# from onpolicy.config import get_config
#
# from onpolicy.runner.shared.cloud_runner import CloudRunner as Runner
# import sys
# import os
# import numpy as np
# from pathlib import Path
# import torch
#
# # 核心：把根目录加入路径，否则找不到 env 文件夹
# sys.path.append(str(Path(__file__).resolve().parents[3]))
#
# from onpolicy.config import get_config
# from onpolicy.envs.cloud_env_wrapper import CloudEnvWrapper
# from env.cloud_scheduling import CloudSchedulingEnv
# from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
# from onpolicy.runner.shared.cloud_runner import CloudRunner as Runner # 稍后创建 Runner
#
# def make_train_env(all_args):
#     def get_env_fn(rank):
#         def init_env():
#             base_env = CloudSchedulingEnv(num_jobs=all_args.num_jobs,
#                                           num_server_farms=10,
#                                           num_servers=50)
#             return CloudEnvWrapper(base_env)
#         return init_env
#     if all_args.n_rollout_threads == 1:
#         return DummyVecEnv([get_env_fn(0)])
#     else:
#         return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
#
# def main(args):
#     parser = get_config()
#     # 添加你自定义的参数，比如 num_jobs
#     parser.add_argument('--num_jobs', type=int, default=100)
#     all_args = parser.parse_known_args(args)[0]
#
#     # 设置设备
#     device = torch.device("cuda:0" if torch.cuda.is_available() and all_args.cuda else "cpu")
#
#     # 运行目录
#     run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name
#     if not run_dir.exists():
#         os.makedirs(str(run_dir))
#
#     # 初始化环境
#     envs = make_train_env(all_args)
#     eval_envs = None # 暂时不需要单独的 eval 环境
#
#     # 绕过 get_attr，直接通过第一个 remote 获取属性
#     # SubprocVecEnv 的 remotes 是一组管道，发送一个 'get_num_agents' 指令（如果没定义，会报错）
#     # 或者我们采用最保险的“硬核”办法：
#     try:
#         # 尝试通过标准接口
#         num_agents = envs.num_agents
#     except AttributeError:
#         # 如果是 SubprocVecEnv 且没有 num_agents 属性
#         # 我们直接去第一个进程里拿。注意：通常 envs.observation_space 是个列表，长度就是 agent 数量
#         if isinstance(envs.observation_space, list):
#             num_agents = len(envs.observation_space)
#         else:
#             # 万能兜底：既然你知道是两个（server_farm 和 server），这里直接写 2 也可以
#             num_agents = 2
#
#     print(f"检测到智能体数量: {num_agents}")
#
#     config = {
#         "all_args": all_args,
#         "envs": envs,
#         "eval_envs": eval_envs,
#         "num_agents": num_agents,
#         "device": device,
#         "run_dir": run_dir
#     }
#
#     # 1. 禁用 wandb 联网同步
#     os.environ["WANDB_MODE"] = "disabled"
#
#     # 2. 如果你的 base_runner 还是强行要访问 wandb.run.dir
#     # 我们就手动初始化一个本地模式的 wandb
#     if wandb.run is None:
#         wandb.init(
#             project=all_args.env_name,
#             name=all_args.experiment_name,
#             config=all_args,
#             dir=str(run_dir), # 指定本地保存路径
#             job_type="training",
#             mode="disabled"   # 再次确保不联网
#         )
#
#     runner = Runner(config)
#     runner.run()
#
#     envs.close()
#
# if __name__ == "__main__":
#     main(sys.argv[1:])