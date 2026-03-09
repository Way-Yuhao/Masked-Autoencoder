"""
Lightning callhook that enables offline & real-time sync on slurm, where compute nodes have no internet access,
and therefore relies on login nodes to indirectly log to weights& biases.
Modified from https://github.com/klieret/wandb-offline-sync-hook/blob/main/src/wandb_osh/lightning_hooks.py
"""
from __future__ import annotations
from os import PathLike
import wandb_osh
from wandb_osh.hooks import TriggerWandbSyncHook, _comm_default_dir
from wandb_osh.util.log import logger
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
from src.utils import RankedLogger

wandb_osh.set_log_level("ERROR")
logger = RankedLogger(name=__name__, rank_zero_only=True)



class TriggerWandbSyncLightningCallback(pl.Callback):
    def __init__(self, communication_dir: PathLike = _comm_default_dir, enabled: bool = False):
        """Hook to be used when interfacing wandb with Lightning.
        Args:
            communication_dir: Directory used for communication with wandb-osh.
            enabled: Whether this callback should be enabled.
        """
        super().__init__()
        self.enabled = enabled
        if self.enabled:
            logger.info("TriggerWandbSyncLightningCallback is enabled.")
        else:
            logger.info("TriggerWandbSyncLightningCallback is disabled.")
        self._hook = (
            TriggerWandbSyncHook(communication_dir=communication_dir)
            if self.enabled
            else None
        )

    @rank_zero_only
    def _call_hook(self, trainer: pl.Trainer) -> None:
        if trainer.sanity_checking or not self.enabled:
            return
        self._hook()

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._call_hook(trainer)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
       self._call_hook(trainer)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._call_hook(trainer)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._call_hook(trainer)

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self._call_hook(trainer)