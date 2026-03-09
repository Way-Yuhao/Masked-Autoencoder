from typing import Any, Optional, Tuple

import torch
from timm.optim import param_groups_weight_decay


def build_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    return param_groups_weight_decay(model, weight_decay=weight_decay)


def resolve_learning_rate(lr: Optional[float], blr: float, batch_size_per_device: int, world_size: int,
                          accum_iter: int) -> Tuple[float, int]:
    effective_batch_size = batch_size_per_device * max(world_size, 1) * max(accum_iter, 1)
    actual_lr = float(lr) if lr is not None else blr * effective_batch_size / 256.0
    return actual_lr, effective_batch_size
