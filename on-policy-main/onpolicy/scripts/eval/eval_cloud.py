import sys
import os

# 将项目根目录直接加入系统路径
sys.path.insert(0, "E:/on-policy-main")

import csv
from datetime import datetime
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.cloud_env_wrapper import CloudEnvWrapper
from env.cloud_scheduling import CloudSchedulingEnv
from onpolicy.envs.env_wrappers import DummyVecEnv
from onpolicy.runner.shared.cloud_runner import CloudRunner as Runner

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            base_env = CloudSchedulingEnv(num_jobs=all_args.num_jobs,
                                          num_server_farms=5,
                                          num_servers=50)
            return CloudEnvWrapper(base_env)
        return init_env
    return DummyVecEnv([get_env_fn(0)])

def main(args):
    parser = get_config()
    parser.add_argument('--num_jobs', type=int, default=100)
    all_args = parser.parse_known_args(args)[0]

    # 强制设置为测试模式
    all_args.use_wandb = False
    all_args.use_eval = True
    all_args.n_rollout_threads = 1 # 测试只开一个线程

    # 检查是否指定了模型路径
    if all_args.model_dir == None or all_args.model_dir == "":
        print("错误：请通过 --model_dir 指定 actor.pt 所在的文件夹！")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() and all_args.cuda else "cpu")

    # --- 关键修改：手动创建一个 run_dir，防止 base_runner 报错 ---
    run_dir = Path(all_args.model_dir)
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # --- 核心修复：mock 一个 wandb.run 对象或者绕过它 ---
    # 在有些版本中，最快的方法是直接在初始化 Runner 前加上这行：
    import wandb
    if wandb.run is None:
        # 如果你不想初始化 wandb，可以手动给 runner 传一个路径
        # 或者在这里执行一个极简的 init
        wandb.init(project="eval", mode="disabled")

    # 环境初始化
    envs = make_eval_env(all_args)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": envs,
        "num_agents": 2, # 你的环境是 2 个智能体
        "device": device,
        "run_dir": Path(all_args.model_dir)
    }

    runner = Runner(config)

    print("--- 开始测试 MAPPO 模型性能 ---")
    # ... 在 main 函数里 ...
    print("--- 正在重置环境以获取初始观察 ---")

    # 同步改为解包 3 个值
    obs = envs.reset()

    print(f"Agent obs 加载成功，维度: {np.array(obs).shape}")

    # 这样进入 runner.run() 后，里面的 reset 也能正常解包了
    runner.run()
    # 改为从 runner 身上取刚才截胡到的数据
    # 拿到 8 次测试的平均价格
    final_avg_price = getattr(runner, "final_eval_price", 0.0)
    print(f"--- [DONE] 8次评估平均总开销: {final_avg_price:.2f} ---")

    print("--- 正在提取并保存实验数据 ---")

    # 1. 确定保存路径
    save_path = Path(all_args.model_dir) / "eval_results.csv"
    file_exists = save_path.exists()

    # 2. 深度提取原始环境对象
    # 循环剥开 Wrapper 直到找到真正的 CloudSchedulingEnv
    temp_env = envs.envs[0]
    while hasattr(temp_env, "env"):
        temp_env = temp_env.env
    inner_env = temp_env

    # 3. 构造数据字典 (请检查 CloudSchedulingEnv.py 里的实际变量名)
    # 如果你的变量名是 self.total_cost，请把下面的 "total_price" 改成 "total_cost"
    # 在 stats 字典中修改对应项
    # 3. 构造数据字典
    stats = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_jobs": all_args.num_jobs,
        "total_price": getattr(runner, "final_eval_price", 0.0),
        "rejected_jobs": len(inner_env.rejected_job_ids),
        "completed_jobs": len(inner_env.completed_job_ids),
        "success_rate": round(len(inner_env.completed_job_ids) / all_args.num_jobs, 4) if all_args.num_jobs > 0 else 0
    }
    # 4. 写入 CSV
    with open(save_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        if not file_exists:
            writer.writeheader()  # 如果文件是新创建的，写表头
        writer.writerow(stats)

    print(f"--- [SUCCESS] 实验数据已存入: {save_path} ---")
    print(f"--- 最终总开销: {stats['total_price']:.2f} ---")

    envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])