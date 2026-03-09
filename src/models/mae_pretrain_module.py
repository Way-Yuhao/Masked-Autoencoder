from typing import Any, Dict, Optional, Tuple
import torch
from lightning import LightningModule
from lightning_utilities.core.rank_zero import rank_zero_info
from src.utils.masked_autoencoder.optim import build_param_groups, resolve_learning_rate
from src.utils.masked_autoencoder.scheduler import WarmupCosineLR
from src.utils import RankedLogger

__author__ = "yuhao liu"
log = RankedLogger(name=__name__, rank_zero_only=True)


def _to_base_optimizer(optimizer: Any) -> torch.optim.Optimizer:
    return optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer


def _is_optimizer_step(batch_idx: int, total_steps: int, accum_iter: int) -> bool:
    if accum_iter <= 1:
        return True
    return ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == total_steps)


class MAEPretrainLitModule(LightningModule):
    def __init__(self, net: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Any] = None,
                 mask_ratio: float = 0.75, weight_decay: float = 0.05, lr: Optional[float] = None, blr: float = 1e-3,
                 accum_iter: int = 1, compile: bool = False) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.actual_lr: Optional[float] = None
        self.effective_batch_size: Optional[int] = None
        self.current_lr: float = 0.0
        self.mae_scheduler: Optional[WarmupCosineLR] = None

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.net(images, mask_ratio=self.hparams.mask_ratio)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def on_fit_start(self) -> None:
        optimizer = _to_base_optimizer(self.optimizers())
        batch_size = self._resolve_per_device_batch_size()
        world_size = max(1, int(self.trainer.world_size))
        accum_iter = self._resolve_accum_iter()
        self.actual_lr, self.effective_batch_size = resolve_learning_rate(
            lr=self.hparams.lr,
            blr=float(self.hparams.blr),
            batch_size_per_device=batch_size,
            world_size=world_size,
            accum_iter=accum_iter,
        )
        self.current_lr = self.actual_lr

        self._set_lr(optimizer, self.actual_lr)
        if self.hparams.scheduler is not None:
            self.mae_scheduler = self.hparams.scheduler(optimizer=optimizer, base_lr=self.actual_lr)
        base_lr = self.actual_lr * 256.0 / self.effective_batch_size

        rank_zero_info(f"base lr: {base_lr:.2e}")
        rank_zero_info(f"actual lr: {self.actual_lr:.2e}")
        rank_zero_info(f"accumulate grad iterations: {accum_iter}")
        rank_zero_info(f"effective batch size: {self.effective_batch_size}")

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        del batch
        accum_iter = self._resolve_accum_iter()
        if self.actual_lr is None or self.mae_scheduler is None or batch_idx % accum_iter != 0:
            return

        total_steps = int(self.trainer.num_training_batches)
        epoch_progress = batch_idx / max(total_steps, 1) + float(self.current_epoch)
        # MAE updates LR before the forward pass of each accumulation window.
        self.current_lr = self.mae_scheduler.step(epoch_progress)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, pred, mask, samples = self._shared_step(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=samples.shape[0])
        total_steps = int(self.trainer.num_training_batches)
        if _is_optimizer_step(batch_idx, total_steps, self._resolve_accum_iter()):
            self.log("train/lr", self.current_lr, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss, "pred": pred.detach(), "mae_mask": mask.detach()}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        del batch_idx
        loss, pred, mask, samples = self._shared_step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=samples.shape[0])
        return {"loss": loss, "pred": pred.detach(), "mae_mask": mask.detach()}

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        samples, _ = batch
        loss, pred, mask = self.forward(samples)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss is {loss.item()}, stopping training.")
        return loss, pred, mask, samples

    def configure_optimizers(self) -> Dict[str, Any]:
        param_groups = build_param_groups(self.net, float(self.hparams.weight_decay))
        initial_lr = (float(self.hparams.lr) if self.hparams.lr is not None else float(self.hparams.blr))
        optimizer = self.hparams.optimizer(params=param_groups, lr=initial_lr)
        return {"optimizer": optimizer}

    def _resolve_per_device_batch_size(self) -> int:
        datamodule = self.trainer.datamodule
        if datamodule is not None and hasattr(datamodule, "batch_size_per_device"):
            return int(datamodule.batch_size_per_device)
        raise ValueError(
            "Could not infer per-device batch size from datamodule. "
            "Please expose `batch_size_per_device` on the datamodule."
        )

    def _resolve_accum_iter(self) -> int:
        accum_iter = getattr(self.trainer, "accumulate_grad_batches", None)
        if isinstance(accum_iter, int) and accum_iter > 0:
            return accum_iter
        return max(1, int(self.hparams.accum_iter))

    @staticmethod
    def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
        for group in optimizer.param_groups:
            if "lr_scale" in group:
                group["lr"] = lr * group["lr_scale"]
            else:
                group["lr"] = lr
