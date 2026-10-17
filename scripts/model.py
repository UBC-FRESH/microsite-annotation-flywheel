"""The pre-labeller model: architecture, focal loss, and TTA inference.

The architecture and loss are the proven recipe (U-Net ResNet34, ``0.5 * focal + 0.5 * dice``)
documented in the source project. This module is the single definition of the network, shared
by training and pre-labelling. The heavy imports live here, so modules that do not need torch
stay importable without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from numpy.typing import NDArray

# ImageNet normalisation, matching the encoder's pretraining.
MEAN: NDArray[np.float32] = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD: NDArray[np.float32] = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FocalLoss(nn.Module):
    """Binary focal loss over two classes.

    Parameters
    ----------
    alpha : float, optional
        Class-balancing weight on the focal term.
    gamma : float, optional
        Focusing parameter; higher values down-weight easy examples more.

    Attributes
    ----------
    alpha : float
        The stored alpha.
    gamma : float
        The stored gamma.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the mean focal loss.

        Parameters
        ----------
        pred : torch.Tensor
            Logits of shape ``(N, C, H, W)``.
        target : torch.Tensor
            Integer labels of shape ``(N, H, W)``.

        Returns
        -------
        torch.Tensor
            The scalar loss.
        """
        ce = nn.functional.cross_entropy(pred, target, reduction="none")
        p_t = torch.softmax(pred, 1).gather(1, target.unsqueeze(1)).squeeze(1)
        return (self.alpha * (1 - p_t) ** self.gamma * ce).mean()


def build_model(
    encoder: str = "resnet34",
    *,
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 2,
) -> nn.Module:
    """Build the U-Net segmentation model.

    Parameters
    ----------
    encoder : str, optional
        Encoder backbone name.
    encoder_weights : str or None, optional
        Pretrained weights, or ``None`` for random initialisation.
    in_channels : int, optional
        Number of input channels.
    classes : int, optional
        Number of output classes.

    Returns
    -------
    torch.nn.Module
        The model, on the CPU and in training mode.
    """
    model: nn.Module = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    return model


def dice_loss() -> nn.Module:
    """Return the soft Dice loss used alongside the focal loss.

    Returns
    -------
    torch.nn.Module
        A multiclass Dice loss.
    """
    loss: nn.Module = smp.losses.DiceLoss(mode="multiclass")
    return loss


def load_model(
    checkpoint: Path,
    encoder: str = "resnet34",
    *,
    device: torch.device | None = None,
) -> nn.Module:
    """Load a trained checkpoint for inference.

    Parameters
    ----------
    checkpoint : Path
        Path to a saved ``state_dict``.
    encoder : str, optional
        Encoder backbone name the checkpoint was trained with.
    device : torch.device or None, optional
        Device to place the model on. Defaults to CUDA when available.

    Returns
    -------
    torch.nn.Module
        The model in evaluation mode.

    Raises
    ------
    FileNotFoundError
        When the checkpoint does not exist.
    """
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    target = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(encoder, encoder_weights=None, in_channels=3, classes=2)
    state: dict[str, Any] = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    return model.to(target).eval()


def _dihedral(array: NDArray[np.float32], k: int, flip: bool) -> NDArray[np.float32]:
    """Apply one dihedral transform to a batched image.

    Parameters
    ----------
    array : numpy.ndarray
        Array of shape ``(N, C, H, W)``.
    k : int
        Number of counter-clockwise 90-degree rotations.
    flip : bool
        Whether to flip horizontally before rotating.

    Returns
    -------
    numpy.ndarray
        The transformed array.
    """
    if flip:
        array = np.flip(array, axis=3)
    return np.rot90(array, k, axes=(2, 3))


def _dihedral_inv(prob: NDArray[np.float32], k: int, flip: bool) -> NDArray[np.float32]:
    """Invert one dihedral transform on a single probability map.

    Parameters
    ----------
    prob : numpy.ndarray
        Array of shape ``(H, W)``.
    k : int
        The rotation used forward.
    flip : bool
        The flip used forward.

    Returns
    -------
    numpy.ndarray
        The inverse-transformed array.
    """
    prob = np.rot90(prob, -k, axes=(0, 1))
    if flip:
        prob = np.flip(prob, axis=1)
    return prob


def prob_map(
    model: nn.Module,
    image_rgb: NDArray[np.float32],
    device: torch.device,
    *,
    tta: bool = False,
) -> NDArray[np.float32]:
    """Return the per-pixel microsite probability for one image.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded model in evaluation mode.
    image_rgb : numpy.ndarray
        RGB image in ``float32`` ``[0, 1]``, shape ``(H, W, 3)``.
    device : torch.device
        Device to run on.
    tta : bool, optional
        When ``True``, average the eight dihedral views.

    Returns
    -------
    numpy.ndarray
        Probability of the positive class, shape ``(H, W)``.
    """
    model.eval()
    normalised = (image_rgb - MEAN) / STD
    batch = normalised.transpose(2, 0, 1)[None]
    operations = [(0, False)] if not tta else [(k, f) for k in range(4) for f in (False, True)]
    probs: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for k, flip in operations:
            view = _dihedral(batch, k, flip)
            tensor = torch.from_numpy(np.ascontiguousarray(view).astype(np.float32)).to(device)
            logits = model(tensor)
            prob = torch.softmax(logits, 1)[0, 1].cpu().numpy().astype(np.float32)
            probs.append(_dihedral_inv(prob, k, flip))
    return np.mean(probs, axis=0).astype(np.float32)
