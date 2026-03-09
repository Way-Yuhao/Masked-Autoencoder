import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning_utilities.core.rank_zero import rank_zero_info
from PIL import Image, ImageDraw, ImageFont

try:
    from timm.optim import optim_factory
except Exception:  # pragma: no cover - API location changed across timm versions
    try:
        import timm.optim.optim_factory as optim_factory  # type: ignore[no-redef]
    except Exception:  # pragma: no cover - fallback below handles no helper API
        optim_factory = None


def _build_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    if optim_factory is not None and hasattr(optim_factory, "add_weight_decay"):
        return optim_factory.add_weight_decay(model, weight_decay)
    if optim_factory is not None and hasattr(optim_factory, "param_groups_weight_decay"):
        return optim_factory.param_groups_weight_decay(model, weight_decay=weight_decay)

    decay_params, no_decay_params = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return [
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": decay_params, "weight_decay": weight_decay},
    ]


def _to_base_optimizer(optimizer: Any) -> torch.optim.Optimizer:
    return optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer


def _is_optimizer_step(batch_idx: int, total_steps: int, accum_iter: int) -> bool:
    if accum_iter <= 1:
        return True
    return ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == total_steps)


class MAEPretrainLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        mask_ratio: float = 0.75,
        weight_decay: float = 0.05,
        lr: Optional[float] = None,
        blr: float = 1e-3,
        min_lr: float = 0.0,
        warmup_epochs: int = 40,
        accum_iter: int = 1,
        vis_log_every_n_steps: int = 500,
        vis_num_images: int = 1,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.actual_lr: Optional[float] = None
        self.effective_batch_size: Optional[int] = None
        self.current_lr: float = 0.0

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.net(images, mask_ratio=self.hparams.mask_ratio)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def on_fit_start(self) -> None:
        optimizer = _to_base_optimizer(self.optimizers())
        batch_size = self._resolve_per_device_batch_size()
        world_size = max(1, int(self.trainer.world_size))
        self.effective_batch_size = batch_size * self.hparams.accum_iter * world_size

        if self.hparams.lr is None:
            self.actual_lr = self.hparams.blr * self.effective_batch_size / 256.0
        else:
            self.actual_lr = float(self.hparams.lr)
        self.current_lr = self.actual_lr

        self._set_lr(optimizer, self.actual_lr)
        base_lr = self.actual_lr * 256.0 / self.effective_batch_size

        rank_zero_info(f"base lr: {base_lr:.2e}")
        rank_zero_info(f"actual lr: {self.actual_lr:.2e}")
        rank_zero_info(f"accumulate grad iterations: {self.hparams.accum_iter}")
        rank_zero_info(f"effective batch size: {self.effective_batch_size}")

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        del batch
        if self.actual_lr is None or batch_idx % self.hparams.accum_iter != 0:
            return

        total_steps = int(self.trainer.num_training_batches)
        epoch_progress = batch_idx / max(total_steps, 1) + float(self.current_epoch)
        optimizer = _to_base_optimizer(self.optimizers())
        self.current_lr = self._adjust_learning_rate(optimizer, epoch_progress)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        samples, _ = batch
        loss, pred, mask = self.forward(samples)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss is {loss.item()}, stopping training.")

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=samples.shape[0],
        )

        total_steps = int(self.trainer.num_training_batches)
        if _is_optimizer_step(batch_idx, total_steps, int(self.hparams.accum_iter)):
            self.log("train/lr", self.current_lr, on_step=True, on_epoch=False, prog_bar=True)

        self._maybe_log_visualization(samples, pred, mask, batch_idx)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        param_groups = _build_param_groups(self.net, float(self.hparams.weight_decay))
        initial_lr = (
            float(self.hparams.lr) if self.hparams.lr is not None else float(self.hparams.blr)
        )
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

    def _adjust_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
        epoch_progress: float,
    ) -> float:
        if self.actual_lr is None:
            raise RuntimeError("actual_lr is not initialized; `on_fit_start` must run first.")

        warmup_epochs = int(self.hparams.warmup_epochs)
        max_epochs = int(self.trainer.max_epochs)
        min_lr = float(self.hparams.min_lr)

        if warmup_epochs > 0 and epoch_progress < warmup_epochs:
            lr = self.actual_lr * epoch_progress / warmup_epochs
        else:
            if max_epochs <= warmup_epochs:
                lr = min_lr
            else:
                progress = (epoch_progress - warmup_epochs) / (max_epochs - warmup_epochs)
                progress = min(max(progress, 0.0), 1.0)
                lr = (
                    min_lr
                    + (self.actual_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
                )

        self._set_lr(optimizer, lr)
        return lr

    @staticmethod
    def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
        for group in optimizer.param_groups:
            if "lr_scale" in group:
                group["lr"] = lr * group["lr_scale"]
            else:
                group["lr"] = lr

    def _maybe_log_visualization(
        self,
        samples: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if not self.trainer.is_global_zero:
            return
        if self.hparams.vis_log_every_n_steps <= 0 or self.hparams.vis_num_images <= 0:
            return
        if batch_idx % self.hparams.vis_log_every_n_steps != 0:
            return

        image = self._build_vis_4panel(samples=samples, pred=pred, mask=mask)
        if image is None:
            return

        total_steps = int(self.trainer.num_training_batches)
        vis_step = int(self.current_epoch * total_steps + batch_idx)
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image("pretrain/vis_4panel", image, vis_step)

    def _build_vis_4panel(
        self,
        samples: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        vis_num = min(int(self.hparams.vis_num_images), samples.shape[0])
        if vis_num <= 0:
            return None

        original = samples[:vis_num].float()
        pred = pred[:vis_num].float()
        mask = mask[:vis_num].float()

        if getattr(self.net, "norm_pix_loss", False):
            target = self.net.patchify(original)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            pred = pred * (var + 1.0e-6).sqrt() + mean

        reconstruction = self.net.unpatchify(pred)

        patch_size = self.net.patch_embed.patch_size[0]
        mask = mask.unsqueeze(-1).repeat(1, 1, patch_size**2 * 3)
        mask = self.net.unpatchify(mask)

        masked = original * (1.0 - mask)
        reconstruction_with_visible = masked + reconstruction * mask

        mean = torch.tensor([0.485, 0.456, 0.406], device=original.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=original.device).view(1, 3, 1, 1)
        panels = [
            (original * std + mean).clamp(0, 1),
            (masked * std + mean).clamp(0, 1),
            (reconstruction * std + mean).clamp(0, 1),
            (reconstruction_with_visible * std + mean).clamp(0, 1),
        ]

        border, caption_h = 4, 26
        rows = []
        for i in range(vis_num):
            single = [panels[0][i]]
            sep = torch.ones(3, panels[0].shape[-2], border, device=original.device)
            for panel_idx in range(1, 4):
                single.extend([sep, panels[panel_idx][i]])
            rows.append(torch.cat(single, dim=2))

        if len(rows) == 1:
            grid = rows[0].detach().cpu()
        else:
            stacked = [rows[0]]
            row_sep = torch.ones(3, border, rows[0].shape[2], device=original.device)
            for row in rows[1:]:
                stacked.extend([row_sep, row])
            grid = torch.cat(stacked, dim=1).detach().cpu()

        grid_uint8 = (grid.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
        grid_uint8 = np.transpose(grid_uint8, (1, 2, 0))
        canvas = np.full(
            (caption_h + grid_uint8.shape[0], grid_uint8.shape[1], 3),
            255,
            dtype=np.uint8,
        )
        canvas[caption_h:, :, :] = grid_uint8

        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        titles = ["original", "masked", "reconstruction", "reconstruction + visible"]
        panel_w = int(panels[0].shape[-1])
        starts = [0, panel_w + border, 2 * panel_w + 2 * border, 3 * panel_w + 3 * border]
        for x_pos, title in zip(starts, titles):
            draw.text((x_pos + 6, 6), title, fill=(0, 0, 0), font=font)

        output = np.transpose(np.asarray(image), (2, 0, 1))
        return torch.from_numpy(output).float() / 255.0
