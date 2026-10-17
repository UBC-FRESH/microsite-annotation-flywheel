"""Pre-label a project's annotation-free images and upload the predictions.

The model predicts a binary mask; the mask becomes polygons; the polygons are uploaded as a
COCO document labelled with the project's microsite class. Every target image is re-checked
against the API immediately before upload, and any image that already carries an annotation is
skipped. This is the hard rule: a pre-label never overwrites a human annotation.
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

from config import Config
from geometry import mask_to_polygons, polygons_to_coco
from model import load_model, prob_map
from roboflow_api import RoboflowClient

Logger = Callable[[str], None]


@dataclass(frozen=True)
class PrelabelResult:
    """The outcome of one ``prelabel`` run.

    Attributes
    ----------
    project : str
        Project that was pre-labelled.
    round_number : int
        Round identifier.
    checkpoint : Path
        The checkpoint used.
    uploaded : int
        Number of images that received a pre-label.
    skipped : int
        Number of images skipped because they were already annotated.
    empty : int
        Number of target images skipped because the model predicted no objects.
    uploaded_images : list of str
        Image ids that received a pre-label.
    skipped_images : list of str
        Image ids that were skipped.
    uploads_path : Path
        Where the upload record was written, for later convergence.
    """

    project: str
    round_number: int
    checkpoint: Path
    uploaded: int = 0
    skipped: int = 0
    empty: int = 0
    uploaded_images: list[str] = field(default_factory=list)
    skipped_images: list[str] = field(default_factory=list)
    uploads_path: Path = Path()


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    *,
    threshold: float,
    tta: bool,
    size: int,
) -> tuple[np.ndarray, int, int]:
    """Predict the positive-class mask for one image file.

    The image is resized to the square training size before inference, then the probability
    map is resized back to the native resolution, so training and inference see the same
    geometry.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded model.
    image_path : Path
        Image file to read.
    device : torch.device
        Device to run on.
    threshold : float
        Probability threshold for the positive class.
    tta : bool
        Whether to average the dihedral views.
    size : int
        Square size the model was trained at.

    Returns
    -------
    tuple of (numpy.ndarray, int, int)
        The ``uint8`` mask at native resolution, the width, and the height.

    Raises
    ------
    FileNotFoundError
        When the image cannot be read.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resized: NDArray[np.float32] = cv2.resize(
        rgb, (size, size), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    probability = prob_map(model, resized, device, tta=tta)
    restored: NDArray[np.float32] = cv2.resize(
        probability, (width, height), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    return (restored > threshold).astype(np.uint8), width, height


def run_prelabel(
    config: Config,
    client: RoboflowClient,
    checkpoint: Path,
    *,
    limit: int = 0,
    device: torch.device | None = None,
    logger: Logger = print,
) -> PrelabelResult:
    """Predict on a project's unannotated images and upload the pre-labels.

    Parameters
    ----------
    config : Config
        The flywheel configuration.
    client : RoboflowClient
        The Roboflow client.
    checkpoint : Path
        Checkpoint to load.
    limit : int, optional
        Maximum number of images to pre-label; ``0`` means all unannotated images.
    device : torch.device or None, optional
        Device to run on. Defaults to CUDA when available.
    logger : callable, optional
        Progress sink; defaults to ``print``.

    Returns
    -------
    PrelabelResult
        Counts and the record path.
    """
    project = config.prelabel_project
    target = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.predictions_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = config.predictions_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint, config.model.encoder, device=target)
    logger(f"prelabel: loaded {checkpoint.name} on {target}")

    candidates = [s for s in client.iter_images(project, in_dataset=False) if not s.is_annotated]
    chosen = candidates[:limit] if limit else candidates
    logger(
        "prelabel: guard ON - only images with zero annotations are targeted; "
        "existing annotations are never overwritten and images are never added to the Dataset"
    )
    logger(f"prelabel: {len(candidates)} annotation-free images, processing {len(chosen)}")

    uploaded_ids: list[str] = []
    skipped_ids: list[str] = []
    empty_count = 0
    records: list[dict[str, Any]] = []
    for index, summary in enumerate(chosen, start=1):
        # Guard: re-read the image and refuse to write over any existing annotation.
        detail = client.image_detail(project, summary.id)
        if detail.objects:
            skipped_ids.append(summary.id)
            logger(f"prelabel: skip {summary.name} (already annotated)")
            continue
        image_path = config.predictions_dir / "images" / summary.name
        client.download(detail.image_url, image_path)
        mask, width, height = predict_image(
            model,
            image_path,
            target,
            threshold=config.prelabel.threshold,
            tta=config.prelabel.tta,
            size=config.model.image_size,
        )
        cv2.imwrite(str(masks_dir / f"{Path(summary.name).stem}_mask.png"), mask * 255)
        polygons = mask_to_polygons(mask, min_area=config.prelabel.min_polygon_area)
        if not polygons:
            empty_count += 1
            logger(f"prelabel: [{index}/{len(chosen)}] {summary.name} (no objects; skipped)")
            continue
        document = polygons_to_coco(
            image_name=summary.name,
            width=width,
            height=height,
            polygons=polygons,
            category_name=config.prelabel_class,
        )
        client.upload_annotation(
            project,
            summary.id,
            document,
            name="annotation.coco.json",
            labelmap={"1": config.prelabel_class},
        )
        uploaded_ids.append(summary.id)
        records.append(
            {
                "image_id": summary.id,
                "name": summary.name,
                "width": width,
                "height": height,
                "polygons": [[[float(x), float(y)] for x, y in polygon] for polygon in polygons],
            }
        )
        logger(f"prelabel: [{index}/{len(chosen)}] {summary.name} ({len(polygons)} polygons)")

    uploads_path = config.predictions_dir / f"uploads-round-{config.round_number}.json"
    uploads_path.write_text(
        json.dumps(
            {
                "project": project,
                "round": config.round_number,
                "timestamp": _now(),
                "checkpoint": str(checkpoint),
                "prelabel_class": config.prelabel_class,
                "uploaded": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger(
        f"prelabel: uploaded {len(uploaded_ids)}, skipped {len(skipped_ids)}, empty {empty_count}"
    )
    return PrelabelResult(
        project=project,
        round_number=config.round_number,
        checkpoint=checkpoint,
        uploaded=len(uploaded_ids),
        skipped=len(skipped_ids),
        empty=empty_count,
        uploaded_images=uploaded_ids,
        skipped_images=skipped_ids,
        uploads_path=uploads_path,
    )
