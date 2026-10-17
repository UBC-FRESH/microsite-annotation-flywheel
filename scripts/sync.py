"""Sync a Roboflow project's annotations into a local dataset with provenance.

``sync`` lists the project's images, downloads the annotated ones, rasterises their polygons
into masks, and records which images are still annotation-free. The result is a training
dataset in the layout the trainer expects, plus a provenance record that names the project,
round, timestamp, and counts.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from config import Config
from geometry import rasterize
from roboflow_api import RoboflowClient, RoboflowError

Logger = Callable[[str], None]
SPLITS: tuple[str, ...] = ("train", "valid", "test")


@dataclass(frozen=True)
class SyncResult:
    """The outcome of one ``sync`` run.

    Attributes
    ----------
    project : str
        Project slug that was synced.
    round_number : int
        Round identifier.
    dataset_dir : Path
        Where the training dataset was written.
    per_split : dict of str to int
        Number of annotated images written per split.
    unannotated : int
        Number of annotation-free images recorded for pre-labelling.
    provenance_path : Path
        Where the provenance record was written.
    unannotated_path : Path
        Where the annotation-free image list was written.
    """

    project: str
    round_number: int
    dataset_dir: Path
    per_split: dict[str, int] = field(default_factory=dict)
    unannotated: int = 0
    provenance_path: Path = Path()
    unannotated_path: Path = Path()

    @property
    def annotated(self) -> int:
        """Return the total number of annotated images written."""
        return sum(self.per_split.values())


def _timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _split_for(summary_split: str) -> str:
    """Map a Roboflow split to a local dataset split.

    Parameters
    ----------
    summary_split : str
        The split reported by Roboflow, possibly empty.

    Returns
    -------
    str
        One of ``train``, ``valid``, or ``test``; defaults to ``train``.
    """
    return summary_split if summary_split in SPLITS else "train"


def run_sync(
    config: Config,
    client: RoboflowClient,
    *,
    limit: int = 0,
    logger: Logger = print,
) -> SyncResult:
    """Download a project's annotations and image list into a local dataset.

    Parameters
    ----------
    config : Config
        The flywheel configuration.
    client : RoboflowClient
        The Roboflow client.
    limit : int, optional
        Maximum number of annotated images to download; ``0`` means all.
    logger : callable, optional
        Progress sink; defaults to ``print``.

    Returns
    -------
    SyncResult
        Counts, paths, and the split breakdown.

    Raises
    ------
    RoboflowError
        When a listing, detail, or download call fails.
    """
    project = config.train_project
    logger(f"sync: listing images in '{project}'")
    summaries = list(client.iter_images(project))
    annotated = [s for s in summaries if s.is_annotated]
    unannotated = [s for s in summaries if not s.is_annotated]
    logger(f"sync: {len(annotated)} annotated, {len(unannotated)} unannotated")

    chosen = annotated[:limit] if limit else annotated
    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (config.dataset_dir / split).mkdir(parents=True, exist_ok=True)

    per_split: dict[str, int] = dict.fromkeys(SPLITS, 0)
    synced_ids: list[str] = []
    for index, summary in enumerate(chosen, start=1):
        detail = client.image_detail(project, summary.id)
        if detail.width <= 0 or detail.height <= 0:
            logger(f"sync: skip {summary.name} (no geometry)")
            continue
        split = _split_for(summary.split)
        image_path = config.dataset_dir / split / summary.name
        client.download(detail.image_url, image_path)
        mask = rasterize(detail.objects, detail.width, detail.height)
        mask_path = config.dataset_dir / split / f"{Path(summary.name).stem}_mask.png"
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        per_split[split] += 1
        synced_ids.append(summary.id)
        logger(
            f"sync: [{index}/{len(chosen)}] {split}/{summary.name} ({len(detail.objects)} objects)"
        )

    provenance_dir = config.round_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = provenance_dir / "sync.json"
    provenance = {
        "project": project,
        "round": config.round_number,
        "timestamp": _timestamp(),
        "workspace": config.workspace,
        "annotated_total": len(annotated),
        "unannotated_total": len(unannotated),
        "synced": len(synced_ids),
        "per_split": per_split,
        "synced_image_ids": synced_ids,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    unannotated_path = config.round_dir / "unannotated.json"
    unannotated_path.write_text(
        json.dumps(
            {
                "project": config.prelabel_project,
                "round": config.round_number,
                "timestamp": _timestamp(),
                "images": [{"id": s.id, "name": s.name} for s in unannotated],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = SyncResult(
        project=project,
        round_number=config.round_number,
        dataset_dir=config.dataset_dir,
        per_split=per_split,
        unannotated=len(unannotated),
        provenance_path=provenance_path,
        unannotated_path=unannotated_path,
    )
    logger(f"sync: wrote {result.annotated} images to {config.dataset_dir}")
    logger(f"sync: {result.unannotated} unannotated images recorded for pre-labelling")
    return result


def load_provenance(path: Path) -> dict[str, object]:
    """Read a provenance record.

    Parameters
    ----------
    path : Path
        Path to a ``sync.json`` record.

    Returns
    -------
    dict of str to object
        The decoded record.

    Raises
    ------
    RoboflowError
        When the file is missing or is not a JSON object.
    """
    if not path.is_file():
        raise RoboflowError(f"provenance file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RoboflowError(f"provenance file is not a JSON object: {path}")
    return data


def result_to_dict(result: SyncResult) -> dict[str, object]:
    """Return a JSON-serialisable view of a :class:`SyncResult`.

    Parameters
    ----------
    result : SyncResult
        The result to serialise.

    Returns
    -------
    dict of str to object
        The result with its ``Path`` fields rendered as strings.
    """
    data = asdict(result)
    data["dataset_dir"] = str(result.dataset_dir)
    data["provenance_path"] = str(result.provenance_path)
    data["unannotated_path"] = str(result.unannotated_path)
    data["annotated"] = result.annotated
    return data
