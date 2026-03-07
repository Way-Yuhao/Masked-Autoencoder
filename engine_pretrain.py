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
import time
import datetime
from typing import Iterable

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import util.misc as misc
import util.lr_sched as lr_sched


def _select_progress_backend():
    if not sys.stdout.isatty() or not misc.is_main_process():
        return None

    try:
        from rich.progress import Progress
        from rich.progress import TextColumn
        from rich.progress import BarColumn
        from rich.progress import TaskProgressColumn
        from rich.progress import TimeRemainingColumn

        return (
            'rich',
            {
                'Progress': Progress,
                'TextColumn': TextColumn,
                'BarColumn': BarColumn,
                'TaskProgressColumn': TaskProgressColumn,
                'TimeRemainingColumn': TimeRemainingColumn,
            },
        )
    except Exception:
        pass

    try:
        from tqdm import tqdm

        return 'tqdm', {'tqdm': tqdm}
    except Exception:
        return None


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

    total_steps = len(data_loader)
    progress_backend = _select_progress_backend()

    def _run_step(data_iter_step, samples):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / total_steps + epoch, args
            )

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss, optimizer, parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0
        )
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
                vis_step = epoch * total_steps + data_iter_step
                log_writer.add_image('pretrain/vis_4panel', vis_image, vis_step)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / total_steps + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        return lr

    if progress_backend is None:
        for data_iter_step, (samples, _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            _run_step(data_iter_step, samples)
    else:
        start_time = time.time()
        end = time.time()
        iter_time = misc.SmoothedValue(fmt='{avg:.4f}')
        data_time = misc.SmoothedValue(fmt='{avg:.4f}')
        mb = 1024.0 * 1024.0
        has_cuda = torch.cuda.is_available()

        if progress_backend[0] == 'rich':
            classes = progress_backend[1]
            progress = classes['Progress'](
                classes['TextColumn'](header + ' {task.completed}/{task.total}'),
                classes['BarColumn'](),
                classes['TaskProgressColumn'](),
                classes['TimeRemainingColumn'](),
                classes['TextColumn'](
                    'lr:{task.fields[lr]} loss:{task.fields[loss]} time:{task.fields[time]} '
                    'data:{task.fields[data]}{task.fields[mem]}'
                ),
                transient=False,
            )
            with progress:
                task_id = progress.add_task(
                    'train', total=total_steps, lr='0.000000', loss='0.0000',
                    time='0.0000', data='0.0000', mem=''
                )
                for data_iter_step, (samples, _) in enumerate(data_loader):
                    data_time.update(time.time() - end)
                    lr = _run_step(data_iter_step, samples)
                    iter_time.update(time.time() - end)
                    mem = ''
                    if has_cuda:
                        mem = ' max mem:{:.0f}'.format(torch.cuda.max_memory_allocated() / mb)
                    progress.update(
                        task_id,
                        advance=1,
                        lr='{:.6f}'.format(lr),
                        loss='{:.4f}'.format(metric_logger.loss.global_avg),
                        time='{:.4f}'.format(iter_time.avg),
                        data='{:.4f}'.format(data_time.avg),
                        mem=mem,
                    )
                    end = time.time()
        else:
            tqdm = progress_backend[1]['tqdm']
            with tqdm(total=total_steps, desc=header, dynamic_ncols=True) as pbar:
                for data_iter_step, (samples, _) in enumerate(data_loader):
                    data_time.update(time.time() - end)
                    lr = _run_step(data_iter_step, samples)
                    iter_time.update(time.time() - end)
                    postfix = {
                        'lr': '{:.6f}'.format(lr),
                        'loss': '{:.4f}'.format(metric_logger.loss.global_avg),
                        'time': '{:.4f}'.format(iter_time.avg),
                        'data': '{:.4f}'.format(data_time.avg),
                    }
                    if has_cuda:
                        max_mem = torch.cuda.max_memory_allocated() / mb
                        postfix['max mem'] = '{:.0f}'.format(max_mem)
                    pbar.set_postfix(postfix)
                    pbar.update(1)
                    end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / total_steps))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
