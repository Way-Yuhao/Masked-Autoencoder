# Masked Autoencoder (MAE) - Lightning + Hydra

This repository is a PyTorch Lightning + Hydra adaptation of the original Masked Autoencoder (MAE)
implementation from Facebook Research:
[facebookresearch/mae](https://github.com/facebookresearch/mae).

## What This Repo Contains

- MAE pretraining with `src/train.py` + Hydra configs
- ImageNet-style datamodule for MAE pretraining
- Multi-GPU training support via Lightning DDP
- SLURM launching via Hydra Submitit (`configs/hydra/slurm.yaml`)

## Installation

### Option 1: pip

```bash
pip install -r requirements.txt
```

### Option 2: conda

```bash
conda env create -f environment.yaml -n health-env
conda activate health-env
```

## Data Setup (ImageNet-Style)

The MAE pretrain datamodule expects this folder layout:

```text
/path/to/imagenet
├── train/
│   ├── class_000/
│   │   ├── img1.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class_000/
    │   ├── img1.jpg
    │   └── ...
    └── ...
```

Set the dataset path through a local config file (ignored by git):

`configs/local/default.yaml`

```yaml
image_net_dir: /path/to/imagenet
```

`experiment=mae_pretrain` reads `data.data_dir` from `${local.image_net_dir}`.

## Training

### MAE pretraining (single GPU)

```bash
python src/train.py experiment=mae_pretrain trainer=gpu trainer.devices=1
```

### MAE pretraining (multi-GPU DDP on one node)

```bash
python src/train.py experiment=mae_pretrain trainer=ddp trainer.devices=6
```

### MAE pretraining on SLURM (Submitit)

```bash
python src/train.py experiment=mae_pretrain hydra=slurm trainer=ddp trainer.devices=6
```

### Resume from checkpoint

```bash
python src/train.py experiment=mae_pretrain ckpt_path=/path/to/last.ckpt
```

## Common Overrides

- Per-device batch size:

```bash
python src/train.py experiment=mae_pretrain data.batch_size=256
```

- Mixed precision mode:

```bash
python src/train.py experiment=mae_pretrain trainer.precision=bf16-mixed
```

- Disable validation set for pretraining:

```bash
python src/train.py experiment=mae_pretrain data.val_subdir=null
```

## Logs and Checkpoints

- Single runs: `logs/<YYYY-MM-DD_HH-MM-SS>/`
- Multiruns: `logs/<task_name>/multiruns/<YYYY-MM-DD_HH-MM-SS>/<job_id>/`
- Checkpoints (for pretrain callbacks): `<output_dir>/checkpoints/`

## Evaluation

`src/eval.py` is available for models/datamodules with a `test_dataloader`. For MAE pretraining,
monitor validation metrics from training runs.

## Tests

```bash
make test
make test-full
```

## Reference

- Original MAE repository: [facebookresearch/mae](https://github.com/facebookresearch/mae)
- Lightning-Hydra-Template: [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

## License

MIT (see `LICENSE`).
