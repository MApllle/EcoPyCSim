"""
工具函数与 ValueNorm 归一化类。
从 on-policy-main 移植，去除外部依赖。
"""
import math

import numpy as np
import torch
import torch.nn as nn


# ── 张量转换 ──────────────────────────────────────────────────────────────────

def check(x):
    """numpy array -> torch tensor，已是 tensor 则直接返回。"""
    return torch.from_numpy(x) if isinstance(x, np.ndarray) else x


# ── 网络初始化 ────────────────────────────────────────────────────────────────

def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


# ── 损失函数 ──────────────────────────────────────────────────────────────────

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e ** 2 / 2


# ── 梯度范数 ──────────────────────────────────────────────────────────────────

def get_gard_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.norm() ** 2
    return math.sqrt(total)


# ── 学习率线性衰减 ────────────────────────────────────────────────────────────

def update_linear_schedule(optimizer, epoch, total_epochs, initial_lr):
    lr = initial_lr * (1.0 - epoch / float(total_epochs))
    for g in optimizer.param_groups:
        g['lr'] = lr


# ── 值函数归一化（running-mean/var） ─────────────────────────────────────────

class ValueNorm(nn.Module):
    """
    对值函数目标进行运行均值/方差归一化（去偏指数移动平均）。
    移植自 on-policy-main/onpolicy/utils/valuenorm.py，接口保持一致。
    """

    def __init__(self, input_shape, beta=0.99999, epsilon=1e-5,
                 device=torch.device("cpu")):
        super().__init__()
        self.epsilon = epsilon
        self.beta = beta
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.running_mean = nn.Parameter(
            torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(
            torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(
            torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

    def _running_mean_var(self):
        mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        var = (mean_sq - mean ** 2).clamp(min=1e-2)
        return mean, var

    @torch.no_grad()
    def update(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(**self.tpdv)
        batch_mean = x.mean()
        batch_sq_mean = (x ** 2).mean()
        w = self.beta
        self.running_mean.mul_(w).add_(batch_mean * (1.0 - w))
        self.running_mean_sq.mul_(w).add_(batch_sq_mean * (1.0 - w))
        self.debiasing_term.mul_(w).add_(1.0 * (1.0 - w))

    def normalize(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(**self.tpdv)
        mean, var = self._running_mean_var()
        return (x - mean) / torch.sqrt(var)

    def denormalize(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(**self.tpdv)
        mean, var = self._running_mean_var()
        return (x * torch.sqrt(var) + mean).cpu().numpy()
