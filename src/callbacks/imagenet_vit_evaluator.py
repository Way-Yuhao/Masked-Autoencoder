from typing import Any, Optional, Sequence

import numpy as np
import torch
import wandb
from PIL import Image, ImageDraw, ImageFont
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT

from src.utils import RankedLogger

__author__ = "yuhao liu"
log = RankedLogger(name=__name__, rank_zero_only=False)


class ImagenetViTEvaluator(Callback):
    """RGB ViT MAE reconstruction visualizer and baseline reporter."""

    def __init__(self, train_log_img_freq: int = 10, val_log_img_freq: int = 10,
                 check_freq_via: str = "epoch", disable_image_logging: bool = False,
                 log_test_on_wandb: bool = False, report_zero_baseline: bool = True,
                 vis_num_images: int = 1, mean: Sequence[float] = (0.485, 0.456, 0.406),
                 std: Sequence[float] = (0.229, 0.224, 0.225)) -> None:
        super().__init__()
        self.train_log_img_freq = train_log_img_freq
        self.val_log_img_freq = val_log_img_freq
        self.check_freq_via = check_freq_via
        self.disable_image_logging = disable_image_logging
        self.log_test_on_wandb = log_test_on_wandb
        self.report_zero_baseline = report_zero_baseline
        self.vis_num_images = vis_num_images
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)

        self.freqs = {"train_img": train_log_img_freq, "val_img": val_log_img_freq}
        self.next_log_idx = {"train_img": 0, "val_img": 0}
        self._warned_missing_keys = False
        log.info("Imagenet ViT Evaluator callback initialized.")

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        del batch_idx
        if self._check_frequency(trainer, "train_img", update=True, skip_sanity=True):
            self._log_reconstruction(trainer=trainer, pl_module=pl_module, outputs=outputs,
                                     batch=batch, stage="train")
        self.report_baseline_metric(trainer=trainer, pl_module=pl_module, outputs=outputs,
                                    batch=batch, stage="train")

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                                outputs: STEP_OUTPUT, batch: Any, batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        del batch_idx, dataloader_idx
        if self._check_frequency(trainer, "val_img", update=True, skip_sanity=True):
            self._log_reconstruction(trainer=trainer, pl_module=pl_module, outputs=outputs,
                                     batch=batch, stage="val")
        self.report_baseline_metric(trainer=trainer, pl_module=pl_module, outputs=outputs,
                                    batch=batch, stage="val")

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                          outputs: STEP_OUTPUT, batch: Any, batch_idx: int,
                          dataloader_idx: int = 0) -> None:
        del dataloader_idx
        if self.log_test_on_wandb and batch_idx == 0:
            self._log_reconstruction(trainer=trainer, pl_module=pl_module, outputs=outputs,
                                     batch=batch, stage="test")

    def _log_reconstruction(self, trainer: Trainer, pl_module: LightningModule,
                            outputs: STEP_OUTPUT, batch: Any, stage: str) -> None:
        if self.disable_image_logging or not trainer.is_global_zero:
            return
        if not isinstance(outputs, dict):
            return

        pred = outputs.get("pred")
        mask = outputs.get("mae_mask")
        samples = self._extract_images(batch)
        if pred is None or mask is None or samples is None:
            if not self._warned_missing_keys:
                log.warning("ImagenetViTEvaluator expects 'pred', 'mae_mask', and image batch data.")
                self._warned_missing_keys = True
            return

        mae_model = self._get_mae_model(pl_module)
        if mae_model is None:
            log.warning("Unable to find MAE model with patchify/unpatchify for visualization.")
            return

        with torch.no_grad():
            vis_image = self._build_vis_4panel(mae_model=mae_model, samples=samples.detach(),
                                               pred=pred.detach(), mask=mask.detach())

        if vis_image is None:
            return

        loss = outputs.get("loss")
        loss_val = (
            float(loss.detach().float().item())
            if isinstance(loss, torch.Tensor)
            else float("nan")
        )
        mask_ratio = float(mask.detach().float().mean().item())
        caption = (
            f"stage={stage}, epoch={trainer.current_epoch}, step={trainer.global_step}, "
            f"loss={loss_val:.6f}, mask_ratio={mask_ratio:.3f}"
        )

        self._log_image_to_loggers(trainer=trainer, vis_image=vis_image, stage=stage,
                                   caption=caption)

    def _log_image_to_loggers(self, trainer: Trainer, vis_image: torch.Tensor, stage: str,
                              caption: str) -> None:
        step = int(trainer.global_step)
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(f"{stage}/vis_4panel", vis_image, step)
                continue
            if not isinstance(logger, WandbLogger):
                continue

            image_uint8 = (
                (vis_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            )
            logger.experiment.log(
                {f"{stage}_vis/mae_rgb_reconstruction": wandb.Image(image_uint8, caption=caption)},
                step=step,
            )

    def report_baseline_metric(self, trainer: Trainer, pl_module: LightningModule,
                               outputs: STEP_OUTPUT, batch: Any, stage: str) -> None:
        del trainer
        if not self.report_zero_baseline or not isinstance(outputs, dict):
            return

        loss = outputs.get("loss")
        if not isinstance(loss, torch.Tensor):
            return

        zero_baseline = self._masked_zero_baseline(
            pl_module=pl_module, outputs=outputs, batch=batch)
        if not np.isfinite(zero_baseline):
            return

        model_loss = float(loss.detach().float().item())
        ratio = model_loss / zero_baseline if zero_baseline > 0 else float("nan")
        batch_size = self._extract_batch_size(batch)
        pl_module.log_dict(
            {
                f"{stage}/model_loss": model_loss,
                f"{stage}/zero_baseline": zero_baseline,
                f"{stage}/loss_over_zero_baseline": ratio,
            },
            on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)

    def _masked_zero_baseline(self, pl_module: LightningModule, outputs: STEP_OUTPUT,
                              batch: Any) -> float:
        if not isinstance(outputs, dict):
            return float("nan")

        mask = outputs.get("mae_mask")
        samples = self._extract_images(batch)
        if not isinstance(mask, torch.Tensor) or samples is None:
            return float("nan")

        mae_model = self._get_mae_model(pl_module)
        if mae_model is None:
            return float("nan")

        with torch.no_grad():
            target = mae_model.patchify(samples.detach())
            if getattr(mae_model, "norm_pix_loss", False):
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6).sqrt()
            baseline = (target ** 2).mean(dim=-1)
            denom = mask.detach().sum().clamp_min(1.0e-6)
            baseline = (baseline * mask.detach()).sum() / denom
        return float(baseline.detach().float().item())

    @staticmethod
    def _extract_images(batch: Any) -> Optional[torch.Tensor]:
        if (
            isinstance(batch, (tuple, list))
            and len(batch) > 0
            and isinstance(batch[0], torch.Tensor)
        ):
            return batch[0]
        if isinstance(batch, dict):
            for key in ("images", "image", "x", "env_features"):
                if key in batch and isinstance(batch[key], torch.Tensor):
                    return batch[key]
        return None

    @staticmethod
    def _extract_batch_size(batch: Any) -> int:
        images = ImagenetViTEvaluator._extract_images(batch)
        return int(images.shape[0]) if images is not None else 1

    @staticmethod
    def _get_mae_model(pl_module: LightningModule) -> Optional[torch.nn.Module]:
        model = getattr(pl_module, "net", None)
        if model is not None and hasattr(model, "patchify") and hasattr(model, "unpatchify"):
            return model
        return None

    def _build_vis_4panel(self, mae_model: torch.nn.Module, samples: torch.Tensor,
                          pred: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
        vis_num = min(int(self.vis_num_images), samples.shape[0])
        if vis_num <= 0:
            return None

        original = samples[:vis_num].float()
        pred = pred[:vis_num].float()
        mask = mask[:vis_num].float()

        if getattr(mae_model, "norm_pix_loss", False):
            target = mae_model.patchify(original)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            pred = pred * (var + 1.0e-6).sqrt() + mean

        reconstruction = mae_model.unpatchify(pred)
        patch_size = mae_model.patch_embed.patch_size[0]
        mask = mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
        mask = mae_model.unpatchify(mask)

        masked = original * (1.0 - mask)
        reconstruction_with_visible = masked + reconstruction * mask
        panels = [
            self._to_rgb_tensor(original),
            self._to_rgb_tensor(masked),
            self._to_rgb_tensor(reconstruction),
            self._to_rgb_tensor(reconstruction_with_visible),
        ]

        border, caption_h = 4, 26
        rows = []
        for i in range(vis_num):
            single = [panels[0][i]]
            sep = torch.ones(3, panels[0].shape[-2], border, device=panels[0].device)
            for panel_idx in range(1, 4):
                single.extend([sep, panels[panel_idx][i]])
            rows.append(torch.cat(single, dim=2))

        if len(rows) == 1:
            grid = rows[0].detach().cpu()
        else:
            stacked = [rows[0]]
            row_sep = torch.ones(3, border, rows[0].shape[2], device=rows[0].device)
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

        out = np.transpose(np.asarray(image), (2, 0, 1))
        return torch.from_numpy(out).float() / 255.0

    def _to_rgb_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.new_tensor(self.mean).view(1, 3, 1, 1)
        std = x.new_tensor(self.std).view(1, 3, 1, 1)
        return (x * std + mean).clamp(0, 1)

    def _check_frequency(self, trainer: Trainer, key: str, update: bool = True,
                         skip_sanity: bool = True) -> bool:
        if self.freqs[key] == -1:
            return False
        if skip_sanity and trainer.current_epoch == 0:
            return False

        if self.check_freq_via == "global_step":
            check_idx = trainer.global_step
        elif self.check_freq_via == "epoch":
            check_idx = trainer.current_epoch
        else:
            raise ValueError(
                f"Invalid check frequency method: {self.check_freq_via}. "
                "Expected 'global_step' or 'epoch'."
            )

        if check_idx >= self.next_log_idx[key]:
            if update:
                self.next_log_idx[key] = check_idx + self.freqs[key]
            return True
        return False
