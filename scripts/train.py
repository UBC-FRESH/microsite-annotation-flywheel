"""Train the pre-labeller from a synced dataset.

This reproduces the proven recipe: U-Net ResNet34, ``0.5 * focal + 0.5 * dice``, AdamW with a
cosine schedule, dihedral augmentation, and early stopping on validation IoU. It writes a best
checkpoint and a metrics-and-provenance record.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, Subset

from config import Config
from model import FocalLoss, build_model, dice_loss

Logger = Callable[[str], None]


@dataclass(frozen=True)
class TrainingResult:
    """The outcome of one ``train`` run.

    Attributes
    ----------
    checkpoint_path : Path
        The best checkpoint, by validation IoU.
    metrics_path : Path
        The metrics-and-provenance record.
    best_val_iou : float
        Best validation IoU reached.
    best_epoch : int
        Epoch of the best checkpoint.
    test_iou : float
        Test IoU of the best checkpoint.
    epochs_run : int
        Number of epochs actually run.
    per_split : dict of str to int
        Number of images per split used for training.
    holdout : int
        Number of training images held out for validation because the project had no
        validation split.
    """

    checkpoint_path: Path
    metrics_path: Path
    best_val_iou: float
    best_epoch: int
    test_iou: float
    epochs_run: int
    per_split: dict[str, int] = field(default_factory=dict)
    holdout: int = 0


class SegDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A train/valid/test segmentation dataset of ``.jpg`` images and ``_mask.png`` masks.

    Parameters
    ----------
    root : Path
        Dataset root, containing ``train``, ``valid``, and ``test`` subdirectories.
    split : str
        Which split to read.
    size : int
        Square size images and masks are resized to.
    augment : bool, optional
        Whether to apply random dihedral augmentation.
    """

    def __init__(self, root: Path, split: str, size: int, *, augment: bool = False) -> None:
        self.directory = root / split
        self.size = size
        self.augment = augment
        self.images: list[Path] = (
            sorted(
                p
                for p in self.directory.glob("*")
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                and not p.name.endswith("_mask.png")
            )
            if self.directory.is_dir()
            else []
        )

    def __len__(self) -> int:
        """Return the number of images in the split.

        Returns
        -------
        int
            The image count.
        """
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one image and its mask.

        Parameters
        ----------
        index : int
            Index into the split.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            The normalised image of shape ``(3, size, size)`` and the mask of shape
            ``(size, size)``.
        """
        path = self.images[index]
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise FileNotFoundError(f"could not read image: {path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.imread(str(self.directory / f"{path.stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((self.size, self.size), dtype=np.uint8)
        else:
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        image_f = image.astype(np.float32) / 255.0
        mask_i = (mask > 0).astype(np.int64)
        if self.augment:
            if np.random.random() > 0.5:
                image_f = np.fliplr(image_f).copy()
                mask_i = np.fliplr(mask_i).copy()
            if np.random.random() > 0.5:
                image_f = np.flipud(image_f).copy()
                mask_i = np.flipud(mask_i).copy()
            k = int(np.random.randint(0, 4))
            if k:
                image_f = np.rot90(image_f, k).copy()
                mask_i = np.rot90(mask_i, k).copy()
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_f = (image_f - mean) / std
        return (
            torch.from_numpy(image_f.transpose(2, 0, 1).astype(np.float32)),
            torch.from_numpy(mask_i),
        )


def binary_iou(pred: NDArray[np.int64], target: NDArray[np.int64], eps: float = 1e-7) -> float:
    """Return the intersection-over-union of the positive class.

    Parameters
    ----------
    pred : numpy.ndarray
        Predicted integer labels.
    target : numpy.ndarray
        Target integer labels.
    eps : float, optional
        Small constant that avoids division by zero.

    Returns
    -------
    float
        The IoU of class 1.
    """
    pred_positive = pred == 1
    target_positive = target == 1
    intersection = int(np.logical_and(pred_positive, target_positive).sum())
    union = int(np.logical_or(pred_positive, target_positive).sum())
    return (intersection + eps) / (union + eps)


def _split_counts(dataset_dir: Path) -> dict[str, int]:
    """Count the ``.jpg`` images in each split directory.

    Parameters
    ----------
    dataset_dir : Path
        Dataset root.

    Returns
    -------
    dict of str to int
        The count per split, always with ``train``, ``valid``, and ``test`` keys.
    """
    counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        directory = dataset_dir / split
        counts[split] = (
            len([p for p in directory.glob("*") if p.suffix.lower() in (".jpg", ".jpeg")])
            if directory.is_dir()
            else 0
        )
    return counts


def run_train(
    config: Config,
    *,
    encoder_weights: str | None = "imagenet",
    device: torch.device | None = None,
    logger: Logger = print,
    seed: int = 0,
) -> TrainingResult:
    """Train the pre-labeller and write a checkpoint plus a metrics record.

    Parameters
    ----------
    config : Config
        The flywheel configuration.
    encoder_weights : str or None, optional
        Encoder pretraining; ``None`` for random initialisation (used in tests).
    device : torch.device or None, optional
        Device to train on. Defaults to CUDA when available.
    logger : callable, optional
        Progress sink; defaults to ``print``.
    seed : int, optional
        Random seed.

    Returns
    -------
    TrainingResult
        Paths and metrics.

    Raises
    ------
    FileNotFoundError
        When the dataset directory has no training images.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    target = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = config.model.image_size
    counts = _split_counts(config.dataset_dir)
    if counts["train"] == 0:
        raise FileNotFoundError(f"no training images in {config.dataset_dir / 'train'}")
    train_base = SegDataset(config.dataset_dir, "train", size, augment=True)
    train_count = len(train_base)
    valid_base = SegDataset(config.dataset_dir, "valid", size, augment=False)
    valid_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] | None = (
        valid_base if len(valid_base) > 0 else None
    )
    test_dataset = SegDataset(config.dataset_dir, "test", size, augment=False)

    holdout = 0
    if valid_dataset is None and train_count >= 5:
        # The project has no validation split (all Dataset images are 'train'). Hold out a
        # deterministic slice so the best epoch can be chosen instead of blindly keeping the
        # last one, which can collapse to all-background.
        holdout = max(1, round(train_count * config.model.valid_fraction))
        indices = list(range(train_count))
        step = max(1, train_count // holdout)
        valid_indices = indices[::step][:holdout]
        valid_set = set(valid_indices)
        train_indices = [i for i in indices if i not in valid_set]
        valid_view = SegDataset(config.dataset_dir, "train", size, augment=False)
        valid_dataset = Subset(valid_view, valid_indices)
        train_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] = Subset(
            train_base, train_indices
        )
    else:
        train_dataset = train_base

    loaders: dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]] = {
        "train": DataLoader(train_dataset, batch_size=config.model.batch_size, shuffle=True)
    }
    if valid_dataset is not None:
        loaders["valid"] = DataLoader(valid_dataset, batch_size=config.model.batch_size)
    if len(test_dataset) > 0:
        loaders["test"] = DataLoader(test_dataset, batch_size=config.model.batch_size)
    logger(f"train: {counts} on {target}; held out {holdout} for validation")
    has_valid = "valid" in loaders
    if not has_valid:
        logger("train: too few images to hold out; keeping the latest epoch's weights")

    model = build_model(config.model.encoder, encoder_weights=encoder_weights).to(target)
    focal = FocalLoss()
    dice = dice_loss()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.model.learning_rate, weight_decay=1e-4
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(config.model.epochs, 1), eta_min=1e-6
    )

    def loss_fn(pred: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        total: torch.Tensor = 0.5 * focal(pred, target_mask) + 0.5 * dice(pred, target_mask)
        return total

    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.checkpoints_dir / "best.pt"
    best_iou, best_epoch, stale = 0.0, 0, 0
    history: list[dict[str, float | int]] = []
    epochs_run = 0

    for epoch in range(1, config.model.epochs + 1):
        epochs_run = epoch
        model.train()
        train_loss = 0.0
        for images, masks in loaders["train"]:
            images, masks = images.to(target), masks.to(target)
            optimiser.zero_grad()
            loss: torch.Tensor = loss_fn(model(images), masks)
            # torch's Tensor.backward is unannotated in the shipped stubs.
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
            train_loss += float(loss.item())
        schedule.step()

        model.eval()
        ious: list[float] = []
        valid_loader = loaders.get("valid")
        if valid_loader is not None:
            with torch.no_grad():
                for images, masks in valid_loader:
                    images = images.to(target)
                    predictions = model(images).argmax(1).cpu().numpy()
                    for pred, mask in zip(predictions, masks.numpy(), strict=True):
                        ious.append(binary_iou(pred.astype(np.int64), mask.astype(np.int64)))
        val_iou = float(np.mean(ious)) if ious else 0.0
        train_batches = max(len(loaders["train"]), 1)
        history.append(
            {"epoch": epoch, "train_loss": train_loss / train_batches, "val_iou": val_iou}
        )
        improved = (not has_valid) or val_iou > best_iou or best_epoch == 0
        if improved:
            best_iou, best_epoch, stale = val_iou, epoch, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale += 1
        logger(
            f"train: epoch {epoch} train_loss={train_loss / train_batches:.4f} "
            f"val_iou={val_iou:.4f}{' *' if improved else ''}"
        )
        if stale >= config.model.patience:
            logger(f"train: early stop at epoch {epoch} (best {best_epoch}: {best_iou:.4f})")
            break

    if checkpoint_path.is_file():
        model.load_state_dict(torch.load(checkpoint_path, map_location=target))
    model.eval()
    test_ious: list[float] = []
    test_loader = loaders.get("test")
    if test_loader is not None:
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(target)
                predictions = model(images).argmax(1).cpu().numpy()
                for pred, mask in zip(predictions, masks.numpy(), strict=True):
                    test_ious.append(binary_iou(pred.astype(np.int64), mask.astype(np.int64)))
    test_iou = float(np.mean(test_ious)) if test_ious else 0.0
    logger(f"train: test IoU {test_iou:.4f} (best val {best_iou:.4f} @ epoch {best_epoch})")

    record: dict[str, Any] = {
        "project": config.train_project,
        "round": config.round_number,
        "timestamp": dt.datetime.now(dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "recipe": {
            "arch": "unet",
            "encoder": config.model.encoder,
            "encoder_weights": encoder_weights,
            "loss": "0.5*focal(alpha=0.25,gamma=2)+0.5*dice",
            "optimiser": "AdamW",
            "learning_rate": config.model.learning_rate,
            "batch_size": config.model.batch_size,
            "image_size": size,
            "patience": config.model.patience,
            "augmentation": "dihedral",
        },
        "dataset": {"per_split": counts},
        "metrics": {
            "best_val_iou": best_iou,
            "best_epoch": best_epoch,
            "test_iou": test_iou,
            "epochs_run": epochs_run,
        },
        "checkpoint": checkpoint_path.name,
        "history": history,
    }
    metrics_path = config.checkpoints_dir / "metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        best_val_iou=best_iou,
        best_epoch=best_epoch,
        test_iou=test_iou,
        epochs_run=epochs_run,
        per_split=counts,
        holdout=holdout,
    )
