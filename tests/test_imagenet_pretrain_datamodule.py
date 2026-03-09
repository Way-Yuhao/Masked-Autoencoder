from pathlib import Path

import pytest
from PIL import Image

from src.data.imagenet_pretrain_datamodule import ImageNetPretrainDataModule


def _write_rgb_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), color=(64, 128, 192)).save(path)


def test_imagenet_pretrain_datamodule_builds_train_and_val_loaders(tmp_path: Path) -> None:
    _write_rgb_image(tmp_path / "train" / "class0" / "sample.jpg")
    _write_rgb_image(tmp_path / "val" / "class0" / "sample.jpg")

    datamodule = ImageNetPretrainDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
    )
    datamodule.setup(stage="fit")

    assert datamodule.data_train is not None
    assert datamodule.data_val is not None
    assert datamodule.val_dataloader() is not None

    train_images, train_targets = next(iter(datamodule.train_dataloader()))
    val_images, val_targets = next(iter(datamodule.val_dataloader()))

    assert tuple(train_images.shape) == (1, 3, 224, 224)
    assert tuple(val_images.shape) == (1, 3, 224, 224)
    assert int(train_targets[0]) == 0
    assert int(val_targets[0]) == 0


def test_imagenet_pretrain_datamodule_requires_val_dir_when_enabled(tmp_path: Path) -> None:
    _write_rgb_image(tmp_path / "train" / "class0" / "sample.jpg")

    datamodule = ImageNetPretrainDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
    )

    with pytest.raises(FileNotFoundError, match="Validation directory does not exist"):
        datamodule.setup(stage="fit")
