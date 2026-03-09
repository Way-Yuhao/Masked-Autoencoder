import math

import pytest

torch = pytest.importorskip("torch")

from src.utils.masked_autoencoder.optim import build_param_groups, resolve_learning_rate
from src.utils.masked_autoencoder.scheduler import WarmupCosineLR


class TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)


def test_build_param_groups_excludes_bias_and_norm_from_weight_decay() -> None:
    net = TinyNet()
    param_groups = build_param_groups(net, weight_decay=0.05)

    no_decay_group = next(group for group in param_groups if group["weight_decay"] == 0.0)
    decay_group = next(group for group in param_groups if group["weight_decay"] == 0.05)

    no_decay_ids = {id(param) for param in no_decay_group["params"]}
    decay_ids = {id(param) for param in decay_group["params"]}

    assert id(net.linear.bias) in no_decay_ids
    assert id(net.norm.weight) in no_decay_ids
    assert id(net.norm.bias) in no_decay_ids
    assert id(net.linear.weight) in decay_ids


def test_resolve_learning_rate_matches_linear_scaling_rule() -> None:
    actual_lr, effective_batch_size = resolve_learning_rate(
        lr=None, blr=1.5e-4, batch_size_per_device=64, world_size=8, accum_iter=8
    )

    assert effective_batch_size == 4096
    assert math.isclose(actual_lr, 1.5e-4 * 4096 / 256.0)


def test_warmup_cosine_lr_matches_mae_schedule() -> None:
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1.0e-3)
    scheduler = WarmupCosineLR(
        optimizer=optimizer,
        base_lr=1.0e-3,
        warmup_epochs=40,
        max_epochs=400,
        eta_min=0.0,
    )

    assert math.isclose(scheduler.step(0.0), 0.0)
    assert math.isclose(optimizer.param_groups[0]["lr"], 0.0)

    warmup_mid = scheduler.step(20.0)
    assert math.isclose(warmup_mid, 5.0e-4)

    cosine_mid = scheduler.step(220.0)
    expected_mid = 5.0e-4
    assert math.isclose(cosine_mid, expected_mid, rel_tol=1e-6)

    final_lr = scheduler.step(400.0)
    assert math.isclose(final_lr, 0.0, abs_tol=1e-12)
