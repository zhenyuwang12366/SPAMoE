# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

import torch
import math
from bisect import bisect_right

# Scheduler adopted from the original repo
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not milestones == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr *
            warmup_factor *
            self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup + Cosine Annealing LR Scheduler.

    Args:
        optimizer (Optimizer): 优化器
        T_max (int): 从 warmup 结束到训练结束的“有效”步数（通常=总步数 - warmup_iters）
                     若你按“迭代”step()，这里就传“迭代数”
        eta_min (float): 余弦阶段的最小学习率下界，默认 0.0
        warmup_factor (float): warmup 初始 lr 相对 base_lr 的比例，如 0.1 或 1e-3
        warmup_iters (int): warmup 的步数（迭代或 epoch，取决于你在哪里 step）
        warmup_method (str): "linear" 或 "constant"
        last_epoch (int): 上次调用 step() 的索引，默认 -1（表示从头开始）

    注意：
        - 若你在“迭代”处调用 scheduler.step()，请把 T_max 与 warmup_iters 都用“迭代数”。
        - get_lr() 的返回值是 *每个 param_group* 的当前 lr。
    """
    def __init__(
        self,
        optimizer,
        T_max,
        eta_min=0.0,
        warmup_factor=1.0/3,
        warmup_iters=0,
        warmup_method="linear",
        last_epoch: int = -1,
    ):
        if warmup_method not in ("constant", "linear"):
            raise ValueError("Only 'constant' or 'linear' warmup_method accepted, got {}".format(warmup_method))

        if T_max < 1:
            raise ValueError("T_max must be >= 1 after warmup. Got T_max={}".format(T_max))

        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        self.warmup_factor = float(warmup_factor)
        self.warmup_iters = int(warmup_iters)
        self.warmup_method = warmup_method

        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def _get_warmup_factor(self):
        # 计算当前步的 warmup 缩放因子
        if self.warmup_iters <= 0:
            return 1.0

        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                return self.warmup_factor
            else:  # linear
                # 从 warmup_factor 线性增至 1
                alpha = float(self.last_epoch) / float(max(1, self.warmup_iters))
                return self.warmup_factor * (1.0 - alpha) + alpha
        return 1.0

    def _get_cosine_factor(self, base_lr):
        """
        余弦阶段的绝对 lr：
            lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * t)) / 2
        其中 t ∈ [0, 1] 表示从 warmup 结束到 T_max 的归一化进度
        """
        # 在 warmup 期间不使用 cosine
        if self.last_epoch < self.warmup_iters:
            return base_lr  # 实际会再乘 warmup_factor

        # 已进入余弦阶段的“局部步数”
        t = self.last_epoch - self.warmup_iters
        t = min(max(t, 0), self.T_max)  # clamp 到 [0, T_max]

        cos_term = (1 + math.cos(math.pi * (t / float(max(1, self.T_max))))) / 2.0
        return self.eta_min + (base_lr - self.eta_min) * cos_term

    def get_lr(self):
        # 在 PyTorch 的 _LRScheduler 里，self.base_lrs 是各 param_group 的初始 base_lr
        warm = self._get_warmup_factor()
        lrs = []
        for base_lr in self.base_lrs:
            if self.last_epoch < self.warmup_iters:
                # warmup 阶段：base_lr * warm_factor
                lr = base_lr * warm
            else:
                # 余弦阶段：先计算“未乘 warmup 的余弦 lr”，再乘 warmup_factor(此时=1)
                lr_cos = self._get_cosine_factor(base_lr)
                # warm==1.0（超过 warmup_iters），这里乘不乘等效；保持形式一致
                lr = lr_cos * warm
            lrs.append(lr)
        return lrs