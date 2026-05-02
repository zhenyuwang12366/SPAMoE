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
from typing import Tuple

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
        optimizer (Optimizer): wrapped optimizer
        T_max (int): effective steps from end of warmup to end of training
                     (often total_steps - warmup_iters). If you call step() per iteration, pass iterations.
        eta_min (float): minimum LR in cosine phase, default 0.0
        warmup_factor (float): initial LR scale vs base_lr during warmup, e.g. 0.1 or 1e-3
        warmup_iters (int): warmup length in steps (iterations or epochs depending where you step)
        warmup_method (str): "linear" or "constant"
        last_epoch (int): last step index, default -1 (start fresh)

    Notes:
        - If you call scheduler.step() per iteration, use iteration counts for T_max and warmup_iters.
        - get_lr() returns the current LR for each param group.
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
        # Current-step warmup scale factor
        if self.warmup_iters <= 0:
            return 1.0

        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                return self.warmup_factor
            else:  # linear
                # Linear ramp from warmup_factor to 1
                alpha = float(self.last_epoch) / float(max(1, self.warmup_iters))
                return self.warmup_factor * (1.0 - alpha) + alpha
        return 1.0

    def _get_cosine_factor(self, base_lr):
        """
        Absolute LR in cosine phase:
            lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * t)) / 2
        where t in [0, 1] is normalized progress from end of warmup to T_max.
        """
        # No cosine during warmup
        if self.last_epoch < self.warmup_iters:
            return base_lr  # Actually multiplied again by warmup_factor

        # Local step count inside cosine phase
        t = self.last_epoch - self.warmup_iters
        t = min(max(t, 0), self.T_max)  # clamp to [0, T_max]

        cos_term = (1 + math.cos(math.pi * (t / float(max(1, self.T_max))))) / 2.0
        return self.eta_min + (base_lr - self.eta_min) * cos_term

    def get_lr(self):
        # In PyTorch _LRScheduler, self.base_lrs holds initial base_lr per param group
        warm = self._get_warmup_factor()
        lrs = []
        for base_lr in self.base_lrs:
            if self.last_epoch < self.warmup_iters:
                # Warmup: base_lr * warm_factor
                lr = base_lr * warm
            else:
                # Cosine phase: lr_cos already accounts for base_lr; warm is 1 after warmup
                lr_cos = self._get_cosine_factor(base_lr)
                lr = lr_cos * warm
            lrs.append(lr)
        return lrs

class WarmupCosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing warm restarts with an optional warmup stage."""

    def __init__(
        self,
        optimizer,
        T_0,
        T_mult: int = 1,
        eta_min: float = 0.0,
        warmup_factor: float = 1.0 / 3,
        warmup_iters: int = 0,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        if T_0 <= 0:
            raise ValueError(f"Expected T_0 > 0, got {T_0}.")
        if T_mult < 1:
            raise ValueError(f"Expected T_mult >= 1, got {T_mult}.")
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted, got {}".format(
                    warmup_method
                )
            )

        self.T_0 = int(T_0)
        self.T_mult = int(T_mult)
        self.eta_min = float(eta_min)
        self.warmup_factor = float(warmup_factor)
        self.warmup_iters = int(warmup_iters)
        self.warmup_method = warmup_method

        super().__init__(optimizer, last_epoch)

    def _get_warmup_factor(self) -> float:
        if self.warmup_iters <= 0:
            return 1.0
        if self.last_epoch >= self.warmup_iters:
            return 1.0
        if self.warmup_method == "constant":
            return self.warmup_factor

        # Linear warmup from warmup_factor -> 1.0
        alpha = float(self.last_epoch) / float(max(1, self.warmup_iters))
        return self.warmup_factor * (1.0 - alpha) + alpha

    def _resolve_cycle(self, effective_epoch: float) -> Tuple[float, float]:
        """Return (position_in_cycle, current_cycle_length)."""
        if self.T_mult == 1:
            cycle_length = float(self.T_0)
            position = float(effective_epoch % cycle_length)
            return position, cycle_length

        cycle_length = float(self.T_0)
        position = float(effective_epoch)
        while position >= cycle_length:
            position -= cycle_length
            cycle_length *= self.T_mult
        return position, cycle_length

    def _cosine_lr(self, base_lr: float) -> float:
        effective_epoch = float(self.last_epoch - self.warmup_iters)
        if effective_epoch <= 0.0:
            # First cosine step or warmup-only configuration
            return base_lr

        position, cycle_length = self._resolve_cycle(effective_epoch)
        cos_inner = math.pi * position / float(max(1.0, cycle_length))
        cosine = (1.0 + math.cos(cos_inner)) / 2.0
        return self.eta_min + (base_lr - self.eta_min) * cosine

    def get_lr(self):
        if self.last_epoch == -1:
            return list(self.base_lrs)

        if self.last_epoch < self.warmup_iters:
            warmup_scale = self._get_warmup_factor()
            return [base_lr * warmup_scale for base_lr in self.base_lrs]

        return [self._cosine_lr(base_lr) for base_lr in self.base_lrs]
