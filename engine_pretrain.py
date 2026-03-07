# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import util.misc as misc
import util.lr_sched as lr_sched


def _build_vis_4panel(model, samples, pred, mask, vis_num_images, norm_pix_loss):
    model_without_ddp = model.module if hasattr(model, 'module') else model
    vis_num = min(vis_num_images, samples.shape[0])
    if vis_num <= 0:
        return None

    original = samples[:vis_num].float()
    pred = pred[:vis_num].float()
    mask = mask[:vis_num].float()

    if norm_pix_loss:
        target = model_without_ddp.patchify(original)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        pred = pred * (var + 1.e-6).sqrt() + mean

    reconstruction = model_without_ddp.unpatchify(pred)

    patch_size = model_without_ddp.patch_embed.patch_size[0]
    mask = mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
    mask = model_without_ddp.unpatchify(mask)

    masked = original * (1 - mask)
    reconstruction_with_visible = masked + reconstruction * mask

    mean = torch.tensor([0.485, 0.456, 0.406], device=original.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=original.device).view(1, 3, 1, 1)

    original = (original * std + mean).clamp(0, 1)
    masked = (masked * std + mean).clamp(0, 1)
    reconstruction = (reconstruction * std + mean).clamp(0, 1)
    reconstruction_with_visible = (reconstruction_with_visible * std + mean).clamp(0, 1)

    border = 4
    caption_h = 26
    panel_titles = ['original', 'masked', 'reconstruction', 'reconstruction + visible']

    rows = []
    for i in range(vis_num):
        panels = [original[i], masked[i], reconstruction[i], reconstruction_with_visible[i]]
        h = panels[0].shape[1]
        white_sep = torch.ones(3, h, border, device=panels[0].device)
        row = [panels[0]]
        for p in panels[1:]:
            row.extend([white_sep, p])
        rows.append(torch.cat(row, dim=2))

    white_h_sep = torch.ones(3, border, rows[0].shape[2], device=rows[0].device)
    stacked = [rows[0]]
    for r in rows[1:]:
        stacked.extend([white_h_sep, r])
    grid = torch.cat(stacked, dim=1).detach().cpu()

    grid_uint8 = (grid.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    grid_uint8 = np.transpose(grid_uint8, (1, 2, 0))

    canvas_h = caption_h + grid_uint8.shape[0]
    canvas_w = grid_uint8.shape[1]
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    canvas[caption_h:, :, :] = grid_uint8
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    panel_w = original.shape[-1]
    starts = [0, panel_w + border, 2 * panel_w + 2 * border, 3 * panel_w + 3 * border]
    for x0, title in zip(starts, panel_titles):
        draw.text((x0 + 6, 6), title, fill=(0, 0, 0), font=font)

    result = np.transpose(np.asarray(img), (2, 0, 1))
    return torch.from_numpy(result).float() / 255.0


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        should_log_vis = (
            log_writer is not None
            and args.vis_log_every_n_steps > 0
            and args.vis_num_images > 0
            and data_iter_step % args.vis_log_every_n_steps == 0
        )
        if should_log_vis:
            with torch.no_grad():
                vis_image = _build_vis_4panel(
                    model=model,
                    samples=samples,
                    pred=pred,
                    mask=mask,
                    vis_num_images=args.vis_num_images,
                    norm_pix_loss=args.norm_pix_loss,
                )
            if vis_image is not None:
                vis_step = epoch * len(data_loader) + data_iter_step
                log_writer.add_image('pretrain/vis_4panel', vis_image, vis_step)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
