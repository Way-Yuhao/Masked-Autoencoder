from collections import OrderedDict
from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import TQDMProgressBar


class NoVersionTQDMProgressBar(TQDMProgressBar):
    """Default Lightning TQDM progress bar without the logger version (`v_num`)."""

    def get_metrics(self, trainer: Trainer, pl_module: LightningModule) -> dict[str, Any]:
        metrics = super().get_metrics(trainer, pl_module)
        if isinstance(metrics, OrderedDict):
            metrics = dict(metrics)
        metrics.pop("v_num", None)
        return metrics
