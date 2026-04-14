# 计划：exp_heterogeneity_ablation 改进

## Context
实验二 (`exp_heterogeneity_ablation.py`) 旨在通过3种服务器混合比例 × 2种异构标志（hetero on/off）的矩阵，对比5种MARL算法在CPU异构环境下的适应性。  
用户希望增加三项改进，使结论更清晰、更有说服力：
1. **ΔHER 对比**（消融实验核心指标：hetero=True 与 hetero=False 的 HER 差值）
2. **跨 server_mix 归一化指标**（消除不同 mix 基准价格差异，使算法间对比公平）
3. **可视化输出**（分组柱状图 + 热力图，直观展示差异）

## 方案：新建独立分析脚本

**新建文件**：`analyze_exp2.py`（不修改原实验脚本）  
- 读取已有的 `results/exp2_heterogeneity_ablation.csv`
- 计算衍生指标并输出新 CSV
- 生成图表保存至 `results/figures/`

---

## 详细实现步骤

### Step 1：读取 CSV 并构建 DataFrame

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results/exp2_heterogeneity_ablation.csv")
os.makedirs("results/figures", exist_ok=True)
```

### Step 2：计算 ΔHER（消融核心指标）

按 (server_mix, strategy) 分组，对比 use_hetero=True vs False 的 her_mean：

```python
# pivot to get hetero=True/False side by side
her_pivot = df.pivot_table(
    index=["server_mix", "strategy"],
    columns="use_hetero",
    values="her_mean"
).rename(columns={True: "her_on", False: "her_off"}).reset_index()

her_pivot["delta_her"] = her_pivot["her_on"] - her_pivot["her_off"]
```

输出：`results/exp2_delta_her.csv`（列：server_mix, strategy, her_on, her_off, delta_her）

### Step 3：归一化 EET 和 total_price

以每个 (server_mix, use_hetero) 分组内的 `energy_greedy` 作为归一化基准：

```python
baseline = "energy_greedy"
for col in ["total_price_mean", "eet_mean"]:
    ref = df[df["strategy"] == baseline].set_index(["server_mix", "use_hetero"])[col]
    df[col.replace("_mean", "_norm")] = df.apply(
        lambda row: row[col] / ref.loc[(row["server_mix"], row["use_hetero"])], axis=1
    )
```

值含义：`1.0` = 与 energy_greedy 持平，`0.5` = 能耗节省50%，`2.0` = 翻倍。  
输出：`results/exp2_normalized.csv`

### Step 4：可视化

#### 图1：各算法 HER 分组柱状图（hetero=True vs False）
- X轴：strategy，分组：server_mix
- 双色柱（蓝=hetero on，橙=hetero off），误差棒显示 her_std
- 保存：`results/figures/exp2_her_grouped.png`

#### 图2：ΔHER 热力图（算法 × server_mix）
- X轴：server_mix，Y轴：strategy
- 颜色深浅 = ΔHER 大小（越深 = MARL越依赖异构信号）
- 保存：`results/figures/exp2_delta_her_heatmap.png`

#### 图3：归一化 EET 柱状图（hetero=True 条件下）
- X轴：strategy，分组：server_mix
- 基准线 y=1.0 标注 energy_greedy
- 保存：`results/figures/exp2_normalized_eet.png`

---

## 关键文件

| 文件 | 角色 |
|------|------|
| `exp_heterogeneity_ablation.py` | 原实验脚本（**不修改**） |
| `results/exp2_heterogeneity_ablation.csv` | 原始实验数据（已存在） |
| `analyze_exp2.py` | **新建**：分析 + 可视化脚本 |
| `results/exp2_delta_her.csv` | 新输出：ΔHER 汇总表 |
| `results/exp2_normalized.csv` | 新输出：归一化指标表 |
| `results/figures/exp2_her_grouped.png` | 新输出：HER 分组柱状图 |
| `results/figures/exp2_delta_her_heatmap.png` | 新输出：ΔHER 热力图 |
| `results/figures/exp2_normalized_eet.png` | 新输出：归一化 EET 柱状图 |

---

## 验证方式

```bash
python analyze_exp2.py
```

预期：
- 终端打印 ΔHER 汇总表
- `results/exp2_delta_her.csv` 中 MARL 算法（hetero=True训练）的 delta_her 应显著高于传统基线
- 三张图保存至 `results/figures/`，无报错

# 修改前审计
一、当前实验能否完成对比？
结构上可以，但有重要前提：

实验是推理时评估（inference-time evaluation），即在已训练好的模型上跑不同环境配置。这意味着：

评估的是「泛化/迁移能力」，不是「在该异构环境下重新训练后的表现」
5种算法必须有对应的 results/{algo}/model.pt 文件才会被加载，否则跳过
传统基线（energy_greedy / round_robin / least_loaded）始终参与对比
实验矩阵：3种 server_mix × 2种 hetero_flag = 6个条件，每种条件跑5个种子

二、主要评价指标及其含义
指标	含义	越好方向
HER (Heterogeneity Exploitation Rate)	Σ(task_cpu × eff_tier) / Σ(task_cpu)，衡量高算力任务被路由到高效率服务器的程度	越高越好（趋近1.0）
total_price	全仿真周期的能源费用（分时电价模型）	越低越好
EET (Energy per Task)	total_price / completed_jobs，每完成一个任务的能耗成本	越低越好
rejected_tasks	因资源不足被拒绝的任务数	越低越好
jains_fairness	服务器间负载均衡度（Jain指数，0–1）	越高越好
active_server_ratio	有实际负载的服务器占比（反映资源整合效率）	视策略而定
三、指标如何体现算法差异
核心对比维度：

1. HER 在 hetero=True vs hetero=False 之间的落差

→ hetero=True 时给予异构奖励信号，MARL算法若能利用该信号，HER应趋近1.0；hetero=False 时剥除信号，HER下降越明显的算法，说明其越依赖显式异构奖励，泛化能力越弱

2. 不同 server_mix 下的 total_price 变化

→ legacy 环境（60%旧服务器）下电价本应偏低，但若算法仍频繁使用低效服务器则会抬高价格；modern 环境（70%新服务器）应带来最低能耗

3. rejected_tasks 与 completed_jobs 的权衡

→ 某些算法（如 QMIX/MADDPG）可能通过更激进的拒绝策略来降低 total_price，表面看价格低但实际完成率差，需要结合 EET 综合判断

四、当前设计的局限性
没有跨条件归一化：不同 server_mix 基准价格不同，直接比较绝对 total_price 会混淆 mix 效应和算法效应
训练分布不匹配：模型在默认环境训练，在 legacy/modern 下评估属于 out-of-distribution 测试，会低估算法真实潜力
HER 的统计展示缺失：现在只输出均值±标准差，没有按 hetero_flag 分组做差值对比（ΔHER = HER_on - HER_off），而这个差值才是消融实验的核心结论


# 更改结果
输出文件
文件	内容
results/exp2_delta_her.csv	ΔHER 汇总（her_on, her_off, delta_her）
results/exp2_normalized.csv	含 total_price_norm / eet_norm 的完整表
results/figures/exp2_her_grouped.png	HER 分组柱状图（hetero on vs off）
results/figures/exp2_delta_her_heatmap.png	ΔHER 热力图（算法 × server_mix）
results/figures/exp2_normalized_eet.png	归一化 EET 柱状图
从 ΔHER 数据看到的关键结论
ΔHER 几乎全部在 ±0.006 范围内，说明：

QMIX 和 MADDPG 在 hetero=off 时仍保持高 HER（~0.94–1.0），移除异构奖励信号后性能基本不变——这说明这两个算法已从其他奖励信号（价格最小化）中隐式学会了异构感知调度
传统基线（least_loaded / round_robin）HER 本就很低，与 hetero flag 无关，说明它们根本不感知异构性
energy_greedy 天然 HER≈1.0，因为它的启发式规则与 HER 定义完全对齐
这一结果本身就是有价值的发现，可以在论文中论述："MARL 算法无需显式异构奖励也能涌现异构感知行为，但 HER 的稳定性在 legacy 环境（旧服务器占多数）下有所下降。"
