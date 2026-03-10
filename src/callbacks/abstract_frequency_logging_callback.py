from abc import ABC, abstractmethod
from typing import Any, Mapping

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT


class AbstractFrequencyLoggingCallback(Callback, ABC):
    """Base callback for stage-aware, frequency-gated batch-end logging.

    This class centralizes train/val/test hook wiring so child callbacks only need
    to define *what* gets logged, not *when* hooks fire.

    How to use this class:
    1. Subclass it and implement :meth:`log_scheduled_batch`.
    2. Optionally override :meth:`custom_handle_batch_end` for logic that should run
       on every batch (for example, counters or side-effect metrics).
    3. Configure ``stage_log_freqs`` with per-stage frequencies:
       - ``{"train": 1, "val": 1}`` logs every epoch/step for both stages.
       - ``-1`` disables scheduled logging for a stage.
    4. Choose ``check_freq_via="epoch"`` or ``"global_step"`` based on how you want
       schedule boundaries to be interpreted.

    Notes:
    - Scheduled logging can be skipped during sanity checks by setting
      ``skip_sanity=True`` (default behavior for epoch-based schedules at epoch 0).
    - Test logging is opt-in through ``log_test_once=True`` and logs only the first
      test batch.
    """

    def __init__(self, stage_log_freqs: Mapping[str, int], check_freq_via: str = "epoch",
                 log_test_once: bool = False, skip_sanity: bool = True) -> None:
        """Initialize stage frequencies and scheduling behavior.

        :param stage_log_freqs: Mapping from stage name to logging frequency.
            Frequency ``-1`` disables scheduled logging for that stage.
        :param check_freq_via: Schedule unit, either ``"epoch"`` or ``"global_step"``.
        :param log_test_once: Whether to run scheduled logging for test stage once
            (on ``batch_idx == 0``).
        :param skip_sanity: Whether to skip scheduled logging during sanity stage
            (epoch 0 when using epoch-based schedules).
        """
        super().__init__()
        self.stage_log_freqs = dict(stage_log_freqs)
        self.check_freq_via = check_freq_via
        self.log_test_once = log_test_once
        self.skip_sanity = skip_sanity
        self.next_log_idx = {stage: 0 for stage in self.stage_log_freqs}

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """Handle train-batch end and run scheduled train logging if due."""
        self._handle_stage_batch_end(trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage="train")

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                                outputs: STEP_OUTPUT, batch: Any, batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        """Handle validation-batch end and run scheduled val logging if due."""
        self._handle_stage_batch_end(trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage="val")

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                          outputs: STEP_OUTPUT, batch: Any, batch_idx: int,
                          dataloader_idx: int = 0) -> None:
        """Optionally run test logging once, on the first test batch."""
        if not self.log_test_once:
            return

        self.custom_handle_batch_end(
            trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage="test")
        if batch_idx == 0:
            self.log_scheduled_batch(
                trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage="test")

    @abstractmethod
    def log_scheduled_batch(self, trainer: Trainer, pl_module: LightningModule,
                            outputs: STEP_OUTPUT, batch: Any, stage: str) -> None:
        """Log media or artifacts for a stage when the schedule gate passes."""

    def custom_handle_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                                outputs: STEP_OUTPUT, batch: Any, stage: str) -> None:
        """Optional always-run hook executed on every batch, before schedule checks."""
        del trainer, pl_module, outputs, batch, stage

    def _handle_stage_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                                outputs: STEP_OUTPUT, batch: Any, stage: str) -> None:
        """Run per-batch hook first, then scheduled logging for the given stage."""
        self.custom_handle_batch_end(
            trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage=stage)
        if self._should_log_stage(trainer=trainer, stage=stage):
            self.log_scheduled_batch(
                trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage=stage)

    def _should_log_stage(self, trainer: Trainer, stage: str, update: bool = True) -> bool:
        """Return whether scheduled logging should run for ``stage`` at this point."""
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
        """Resolve the current schedule index from trainer state."""
        if self.check_freq_via == "global_step":
            return int(trainer.global_step)
        if self.check_freq_via == "epoch":
            return int(trainer.current_epoch)
        raise ValueError(
            f"Invalid check frequency method: {self.check_freq_via}. "
            "Expected 'global_step' or 'epoch'."
        )
