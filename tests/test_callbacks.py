from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import hydra
import pytest
import torch
from omegaconf import OmegaConf

import src.callbacks.imagenet_vit_evaluator as imagenet_vit_module
from src.callbacks.abstract_frequency_logging_callback import AbstractFrequencyLoggingCallback
from src.callbacks.imagenet_vit_evaluator import ImagenetViTEvaluator


class RecordingFrequencyCallback(AbstractFrequencyLoggingCallback):
    def __init__(self, stage_log_freqs: dict[str, int], check_freq_via: str = "epoch",
                 log_test_once: bool = False, skip_sanity: bool = True) -> None:
        super().__init__(
            stage_log_freqs=stage_log_freqs, check_freq_via=check_freq_via,
            log_test_once=log_test_once, skip_sanity=skip_sanity)
        self.handled: list[tuple[str, int, int]] = []
        self.logged: list[tuple[str, int, int]] = []

    def handle_batch_end(self, trainer, pl_module, outputs, batch, stage: str) -> None:
        del pl_module, outputs, batch
        self.handled.append((stage, int(trainer.current_epoch), int(trainer.global_step)))

    def log_scheduled_batch(self, trainer, pl_module, outputs, batch, stage: str) -> None:
        del pl_module, outputs, batch
        self.logged.append((stage, int(trainer.current_epoch), int(trainer.global_step)))


class DummyLightningModule:
    def __init__(self) -> None:
        self.logged = []

    def log_dict(self, metrics, **kwargs) -> None:
        self.logged.append((metrics, kwargs))


def make_trainer(epoch: int = 1, step: int = 0, is_global_zero: bool = True):
    return SimpleNamespace(
        current_epoch=epoch, global_step=step, is_global_zero=is_global_zero, loggers=[])


def test_frequency_callback_tracks_train_and_val_schedules_independently() -> None:
    callback = RecordingFrequencyCallback(
        stage_log_freqs={"train": 2, "val": 3}, check_freq_via="epoch", skip_sanity=False)
    trainer = make_trainer(epoch=0)

    callback.on_train_batch_end(trainer, object(), {}, object(), batch_idx=0)
    callback.on_validation_batch_end(trainer, object(), {}, object(), batch_idx=0)
    trainer.current_epoch = 1
    callback.on_train_batch_end(trainer, object(), {}, object(), batch_idx=0)
    callback.on_validation_batch_end(trainer, object(), {}, object(), batch_idx=0)
    trainer.current_epoch = 2
    callback.on_train_batch_end(trainer, object(), {}, object(), batch_idx=0)
    callback.on_validation_batch_end(trainer, object(), {}, object(), batch_idx=0)
    trainer.current_epoch = 3
    callback.on_validation_batch_end(trainer, object(), {}, object(), batch_idx=0)

    assert callback.logged == [
        ("train", 0, 0),
        ("val", 0, 0),
        ("train", 2, 0),
        ("val", 3, 0),
    ]
    assert len(callback.handled) == 7


def test_frequency_callback_uses_global_step_when_requested() -> None:
    callback = RecordingFrequencyCallback(
        stage_log_freqs={"train": 2}, check_freq_via="global_step", skip_sanity=False)
    trainer = make_trainer(epoch=1, step=0)

    for step in range(4):
        trainer.global_step = step
        callback.on_train_batch_end(trainer, object(), {}, object(), batch_idx=step)

    assert callback.logged == [("train", 1, 0), ("train", 1, 2)]


def test_frequency_callback_disables_scheduled_logging_with_minus_one() -> None:
    callback = RecordingFrequencyCallback(stage_log_freqs={"train": -1}, skip_sanity=False)
    trainer = make_trainer(epoch=1)

    callback.on_train_batch_end(trainer, object(), {}, object(), batch_idx=0)

    assert callback.handled == [("train", 1, 0)]
    assert callback.logged == []


def test_frequency_callback_skips_epoch_zero_when_skip_sanity_enabled() -> None:
    callback = RecordingFrequencyCallback(stage_log_freqs={"train": 1}, skip_sanity=True)
    trainer = make_trainer(epoch=0)

    callback.on_train_batch_end(trainer, object(), {}, object(), batch_idx=0)
    trainer.current_epoch = 1
    callback.on_train_batch_end(trainer, object(), {}, object(), batch_idx=1)

    assert callback.handled == [("train", 0, 0), ("train", 1, 0)]
    assert callback.logged == [("train", 1, 0)]


def test_frequency_callback_handles_every_test_batch_and_logs_once() -> None:
    callback = RecordingFrequencyCallback(stage_log_freqs={}, log_test_once=True)
    trainer = make_trainer(epoch=1, step=0)

    callback.on_test_batch_end(trainer, object(), {}, object(), batch_idx=0)
    trainer.global_step = 1
    callback.on_test_batch_end(trainer, object(), {}, object(), batch_idx=1)

    assert callback.handled == [("test", 1, 0), ("test", 1, 1)]
    assert callback.logged == [("test", 1, 0)]


def test_frequency_callback_skips_test_when_disabled() -> None:
    callback = RecordingFrequencyCallback(stage_log_freqs={}, log_test_once=False)
    trainer = make_trainer(epoch=1)

    callback.on_test_batch_end(trainer, object(), {}, object(), batch_idx=0)

    assert callback.handled == []
    assert callback.logged == []


def test_imagenet_vit_evaluator_config_instantiates() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "callbacks"
    cfg = OmegaConf.load(config_path / "imagenet_vit_evaluator.yaml")

    callback = hydra.utils.instantiate(cfg.imagenet_vit_evaluator)

    assert isinstance(callback, ImagenetViTEvaluator)


def test_imagenet_vit_evaluator_reports_baseline_when_image_logging_skips(monkeypatch) -> None:
    callback = ImagenetViTEvaluator(train_log_img_freq=5, val_log_img_freq=5)
    trainer = make_trainer(epoch=0)
    baseline_stages = []
    image_stages = []
    monkeypatch.setattr(
        callback, "report_baseline_metric",
        lambda trainer, pl_module, outputs, batch, stage: baseline_stages.append(stage))
    monkeypatch.setattr(
        callback, "_log_reconstruction",
        lambda trainer, pl_module, outputs, batch, stage: image_stages.append(stage))

    callback.on_train_batch_end(trainer, object(), {}, object(), batch_idx=0)
    callback.on_validation_batch_end(trainer, object(), {}, object(), batch_idx=0)

    assert baseline_stages == ["train", "val"]
    assert image_stages == []


def test_imagenet_vit_evaluator_scheduled_logging_calls_reconstruction(monkeypatch) -> None:
    callback = ImagenetViTEvaluator(train_log_img_freq=1, val_log_img_freq=1)
    trainer = make_trainer(epoch=1)
    image_stages = []
    monkeypatch.setattr(
        callback, "_log_reconstruction",
        lambda trainer, pl_module, outputs, batch, stage: image_stages.append(stage))
    monkeypatch.setattr(callback, "report_baseline_metric", lambda *args, **kwargs: None)

    callback.on_train_batch_end(trainer, object(), {}, object(), batch_idx=0)
    callback.on_validation_batch_end(trainer, object(), {}, object(), batch_idx=0)

    assert image_stages == ["train", "val"]


@pytest.mark.parametrize(("enabled", "expected_count"), [(False, 0), (True, 1)])
def test_imagenet_vit_evaluator_test_logging_respects_flag(
    monkeypatch, enabled: bool, expected_count: int
) -> None:
    callback = ImagenetViTEvaluator(log_test_on_wandb=enabled)
    trainer = make_trainer(epoch=1)
    image_stages = []
    monkeypatch.setattr(
        callback, "_log_reconstruction",
        lambda trainer, pl_module, outputs, batch, stage: image_stages.append(stage))

    callback.on_test_batch_end(trainer, object(), {}, object(), batch_idx=0)
    callback.on_test_batch_end(trainer, object(), {}, object(), batch_idx=1)

    assert image_stages == ["test"] * expected_count


def test_imagenet_vit_evaluator_missing_keys_warns_once(monkeypatch) -> None:
    callback = ImagenetViTEvaluator()
    trainer = make_trainer(epoch=1)
    warning = Mock()
    monkeypatch.setattr(imagenet_vit_module.log, "warning", warning)
    outputs = {"loss": torch.tensor(1.0)}
    batch = {"image": torch.zeros(1, 3, 4, 4)}

    callback._log_reconstruction(trainer=trainer, pl_module=DummyLightningModule(),
                                 outputs=outputs, batch=batch, stage="train")
    callback._log_reconstruction(trainer=trainer, pl_module=DummyLightningModule(),
                                 outputs=outputs, batch=batch, stage="train")

    warning.assert_called_once()
    assert callback._warned_missing_keys is True
