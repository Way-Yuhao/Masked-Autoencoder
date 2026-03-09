from pathlib import Path
from typing import Any, Dict, Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import InterpolationMode, transforms


class ImageNetPretrainDataModule(LightningDataModule):
    """LightningDataModule for MAE pretraining on ImageNet-style folder data."""

    def __init__(self, data_dir: str, train_subdir: str = "train", val_subdir: Optional[str] = "val",
                 input_size: int = 224, min_scale: float = 0.2, max_scale: float = 1.0, interpolation: str = "bicubic",
                 batch_size: int = 64, num_workers: int = 10, pin_memory: bool = True, drop_last: bool = True,
                 persistent_workers: bool = True) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.batch_size_per_device = batch_size
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        interpolation_mode = self._resolve_interpolation(interpolation)
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    input_size,
                    scale=(min_scale, max_scale),
                    interpolation=interpolation_mode,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(
                    self._resolve_eval_resize_size(input_size),
                    interpolation=interpolation_mode,
                ),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    @staticmethod
    def _resolve_interpolation(name: str) -> InterpolationMode:
        resolved = str(name).strip().upper()
        if not hasattr(InterpolationMode, resolved):
            raise ValueError(f"Unsupported interpolation mode: {name}")
        return getattr(InterpolationMode, resolved)

    @staticmethod
    def _resolve_eval_resize_size(input_size: int) -> int:
        crop_pct = 224 / 256 if input_size <= 224 else 1.0
        return int(input_size / crop_pct)

    def setup(self, stage: Optional[str] = None) -> None:
        load_train = stage in (None, "fit")
        load_val = stage in (None, "fit", "validate")

        if load_train and self.data_train is None:
            train_root = Path(self.hparams.data_dir) / self.hparams.train_subdir
            self.data_train = datasets.ImageFolder(str(train_root), transform=self.train_transform)
        if load_val and self.data_val is None and self.hparams.val_subdir is not None:
            val_root = Path(self.hparams.data_dir) / self.hparams.val_subdir
            if not val_root.is_dir():
                raise FileNotFoundError(
                    f"Validation directory does not exist: {val_root}. "
                    "Set `val_subdir=null` to disable validation."
                )
            self.data_val = datasets.ImageFolder(str(val_root), transform=self.val_transform)

    def train_dataloader(self) -> DataLoader[Any]:
        persistent_workers = self.hparams.persistent_workers and self.hparams.num_workers > 0
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            persistent_workers=persistent_workers,
        )

    def val_dataloader(self) -> Optional[DataLoader[Any]]:
        if self.data_val is None:
            return None

        persistent_workers = self.hparams.persistent_workers and self.hparams.num_workers > 0
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )

    def test_dataloader(self) -> None:
        return None

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        del state_dict
