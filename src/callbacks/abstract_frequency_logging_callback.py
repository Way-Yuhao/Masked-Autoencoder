from abc import ABC, abstractmethod
from typing import Any, Mapping

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT


class AbstractFrequencyLoggingCallback(Callback, ABC):
    """Base callback for frequency-gated batch-end media logging."""

    def __init__(self, stage_log_freqs: Mapping[str, int], check_freq_via: str = "epoch",
                 log_test_once: bool = False, skip_sanity: bool = True) -> None:
        super().__init__()
        self.stage_log_freqs = dict(stage_log_freqs)
        self.check_freq_via = check_freq_via
        self.log_test_once = log_test_once
        self.skip_sanity = skip_sanity
        self.next_log_idx = {stage: 0 for stage in self.stage_log_freqs}

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        del batch_idx
        self._handle_stage_batch_end(
            trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage="train")

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                                outputs: STEP_OUTPUT, batch: Any, batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        del batch_idx, dataloader_idx
        self._handle_stage_batch_end(
            trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage="val")

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                          outputs: STEP_OUTPUT, batch: Any, batch_idx: int,
                          dataloader_idx: int = 0) -> None:
        del dataloader_idx
        if not self.log_test_once:
            return

        self.handle_batch_end(
            trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage="test")
        if batch_idx == 0:
            self.log_scheduled_batch(
                trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage="test")

    @abstractmethod
    def log_scheduled_batch(self, trainer: Trainer, pl_module: LightningModule,
                            outputs: STEP_OUTPUT, batch: Any, stage: str) -> None:
        """Log scheduled media for a stage."""

    def handle_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT,
                         batch: Any, stage: str) -> None:
        del trainer, pl_module, outputs, batch, stage

    def _handle_stage_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                                outputs: STEP_OUTPUT, batch: Any, stage: str) -> None:
        self.handle_batch_end(
            trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage=stage)
        if self._should_log_stage(trainer=trainer, stage=stage):
            self.log_scheduled_batch(trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage=stage)

    def _should_log_stage(self, trainer: Trainer, stage: str, update: bool = True) -> bool:
        freq = self.stage_log_freqs.get(stage, -1)
        if freq == -1:
            return False
        if self.skip_sanity and trainer.current_epoch == 0:
            return False

        check_idx = self._get_check_idx(trainer)
        next_log_idx = self.next_log_idx.setdefault(stage, 0)
        if check_idx >= next_log_idx:
            if update:
                self.next_log_idx[stage] = check_idx + freq
            return True
        return False

    def _get_check_idx(self, trainer: Trainer) -> int:
        if self.check_freq_via == "global_step":
            return int(trainer.global_step)
        if self.check_freq_via == "epoch":
            return int(trainer.current_epoch)
        raise ValueError(
            f"Invalid check frequency method: {self.check_freq_via}. "
            "Expected 'global_step' or 'epoch'."
        )
