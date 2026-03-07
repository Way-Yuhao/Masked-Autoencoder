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

    rows = []
    for i in range(vis_num):
        row = torch.cat([original[i], masked[i], reconstruction[i], reconstruction_with_visible[i]], dim=2)
        rows.append(row)
    return torch.cat(rows, dim=1).detach().cpu()


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

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

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
