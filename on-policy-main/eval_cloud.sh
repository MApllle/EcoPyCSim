#!/bin/bash

# --- 基础配置 ---
env="CloudScheduling"
algo="mappo"
exp="eval_test"
num_jobs=100

# --- 模型路径 ---
model_path="E:/on-policy-main/onpolicy/scripts/results/CloudScheduling/mappo/baseline_compare/wandb/offline-run-20260412_155826-vrdl3i7g/files"

echo "开始评估 MAPPO 模型: ${env} ..."

# --- 核心：使用绝对路径确保 Python 能找到项目根目录 ---
# 这里直接指定你项目所在的 E 盘路径
export PYTHONPATH=$PYTHONPATH:/e/on-policy-main

# --- 执行测试 ---
python onpolicy/scripts/eval/eval_cloud.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --num_jobs ${num_jobs} \
    --model_dir "${model_path}" \
    --n_rollout_threads 1 \
    --num_env_steps 1600 \
    --use_render \
    --cuda