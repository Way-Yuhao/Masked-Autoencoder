from typing import Any, Optional, Tuple

import torch

try:
    from timm.optim import optim_factory
except Exception:  # pragma: no cover - API location changed across timm versions
    try:
        import timm.optim.optim_factory as optim_factory  # type: ignore[no-redef]
    except Exception:  # pragma: no cover - fallback below handles no helper API
        optim_factory = None


def build_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    if optim_factory is not None and hasattr(optim_factory, "add_weight_decay"):
        return optim_factory.add_weight_decay(model, weight_decay)
    if optim_factory is not None and hasattr(optim_factory, "param_groups_weight_decay"):
        return optim_factory.param_groups_weight_decay(model, weight_decay=weight_decay)

    decay_params, no_decay_params = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": decay_params, "weight_decay": weight_decay},
    ]


def resolve_learning_rate(lr: Optional[float], blr: float, batch_size_per_device: int, world_size: int,
                          accum_iter: int) -> Tuple[float, int]:
    effective_batch_size = batch_size_per_device * max(world_size, 1) * max(accum_iter, 1)
    actual_lr = float(lr) if lr is not None else blr * effective_batch_size / 256.0
    return actual_lr, effective_batch_size
