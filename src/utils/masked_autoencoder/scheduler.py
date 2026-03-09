import math
import torch


class WarmupCosineLR:
    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, warmup_epochs: int,
                 max_epochs: int, eta_min: float = 0.0,) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min

    def step(self, epoch_progress: float) -> float:
        lr = self._get_lr(epoch_progress)
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
        return lr

    def _get_lr(self, epoch_progress: float) -> float:
        if self.warmup_epochs > 0 and epoch_progress < self.warmup_epochs:
            return self.base_lr * epoch_progress / self.warmup_epochs
        if self.max_epochs <= self.warmup_epochs:
            return self.eta_min

        progress = (epoch_progress - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
