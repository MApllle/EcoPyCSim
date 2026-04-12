#!/bin/bash

env="CloudScheduling"
algo="mappo"
exp="baseline_compare"
num_jobs=100

echo "开始训练 ${env} 任务，算法: ${algo}..."

python onpolicy/scripts/train/train_cloud.py \
    --use_valan \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --num_jobs ${num_jobs} \
    --seed 1 \
    --n_training_threads 1 \
    --n_rollout_threads 8 \
    --num_mini_batch 1 \
    --episode_length 200 \
    --num_env_steps 1000000 \
    --ppo_epoch 10 \
    --use_relu \
    --lr 3e-4 \
    --critic_lr 3e-4 \
    --user_name "84838"\
    --use_tensorboard\
    --save_interval 10 \
    --log_interval 1\
    --use_proper_time_limits