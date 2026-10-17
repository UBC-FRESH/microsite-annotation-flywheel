"""Unit tests for the model helpers and the training loop."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from fakes import make_config
from model import FocalLoss, build_model, dice_loss, load_model, prob_map
from train import SegDataset, _split_counts, binary_iou, run_train


def _write_split(root: Path, split: str, count: int = 1) -> None:
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(directory / f"img{index}.jpg"), image)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        cv2.imwrite(str(directory / f"img{index}_mask.png"), mask)


def test_binary_iou() -> None:
    a = np.array([[1, 1], [0, 0]], dtype=np.int64)
    assert binary_iou(a, a) == pytest.approx(1.0)
    b = np.zeros((2, 2), dtype=np.int64)
    assert binary_iou(a, b) < 0.01


def test_seg_dataset_excludes_masks_and_handles_missing(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", 2)
    dataset = SegDataset(tmp_path, "train", 32)
    assert len(dataset) == 2

    image, mask = dataset[0]
    assert image.shape == (3, 32, 32)
    assert mask.shape == (32, 32)

    (tmp_path / "train" / "img0_mask.png").unlink()
    _, zero_mask = SegDataset(tmp_path, "train", 32)[0]
    assert int(zero_mask.sum()) == 0


def test_seg_dataset_empty_directory(tmp_path: Path) -> None:
    assert len(SegDataset(tmp_path, "valid", 32)) == 0


def test_split_counts(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", 1)
    _write_split(tmp_path, "test", 2)
    assert _split_counts(tmp_path) == {"train": 1, "valid": 0, "test": 2}


def test_focal_and_dice_losses() -> None:
    pred = torch.randn(2, 2, 8, 8)
    target = torch.randint(0, 2, (2, 8, 8))
    assert float(FocalLoss()(pred, target)) > 0.0
    assert float(dice_loss()(pred, target)) >= 0.0


def test_build_model_and_prob_map_with_tta() -> None:
    model = build_model("resnet34", encoder_weights=None)
    image = np.random.rand(32, 32, 3).astype(np.float32)
    assert prob_map(model, image, torch.device("cpu")).shape == (32, 32)
    assert prob_map(model, image, torch.device("cpu"), tta=True).shape == (32, 32)


def test_load_model_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save(build_model("resnet34", encoder_weights=None).state_dict(), path)
    assert load_model(path, "resnet34", device=torch.device("cpu")) is not None


def test_load_model_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "absent.pt")


def test_run_train_tiny(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    for split in ("train", "valid", "test"):
        _write_split(config.dataset_dir, split)
    result = run_train(config, encoder_weights=None, device=torch.device("cpu"))

    assert result.epochs_run == 1
    assert result.checkpoint_path.is_file()
    assert result.metrics_path.is_file()
    assert result.per_split["train"] == 1


def test_run_train_requires_training_images(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    (config.dataset_dir / "train").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no training images"):
        run_train(config, encoder_weights=None, device=torch.device("cpu"))


def test_run_train_without_validation_keeps_latest(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    _write_split(config.dataset_dir, "train", 2)  # no valid, no test
    result = run_train(config, encoder_weights=None, device=torch.device("cpu"))

    assert result.epochs_run == config.model.epochs
    assert result.best_epoch == config.model.epochs
    assert result.checkpoint_path.is_file()
