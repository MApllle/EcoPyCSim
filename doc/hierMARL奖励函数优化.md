两个改动方案：
  方案 A（去掉恒正偏置）                                                                                                                                                                                   
  - _take_action()：成功调度时设 self.task_just_scheduled = True，拒绝/开始时重置为 False                                                                                                                  
  - step()：获取奖励后重置 self.task_just_scheduled = False                                                                                                                                                
  - reset()：初始化 self.task_just_scheduled = False       
  - _get_rewards()：r_task ∈ {-1, 0, +1}（已存在但缺少 task_just_scheduled 变量导致崩溃，现已修复）                                                                                                        
                                                                                                                                                                                                           
  方案 B（相对基线的瞬时功率节约）                                                                                                                                                                         
  - reset()：计算 baseline_power_per_step = Σ(0.035 + opt_util × alpha)，即各服务器在最优利用率下的功率估计（old≈0.485、mid≈0.385、new≈0.275/台）                                                          
  - _get_rewards()：r_energy = clip((baseline - curr_power) / baseline, -1, 1)，在整个 episode 中尺度一致，去掉了旧的 prev_total_energy 状态                                                               
                                                                                                                                                                                  
  另外 r_completion = 0.2 × cmp_delta（原计划写的 2.0，但考虑到 credit assignment 问题改为小权重 0.2），已加入 r_shared。

对两个问题的改动：
  问题 1：credit assignment（cloud_scheduling_hier.py）
  - 从 _get_rewards() 彻底删除 cmp_delta 和 r_completion，每步不再给完成信号
  - 在 step() 的 all_jobs_complete 分支加 terminal reward：completion_rate = num_completed_jobs / num_jobs ∈ [0, 1]，一次性给所有 agent
  - 删除 prev_rejected_tasks_count、prev_completed_jobs_count 两个无用状态

  问题 2：Buffer 归一化（两个 Buffer.py）
  - hier_marl/Buffer.py：LocalBuffer.sample 和 GlobalBuffer.sample 各一行 batch 归一化 → reward / 2.0
  - maddpg/Buffer.py：同上
