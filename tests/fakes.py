"""Test doubles and helpers shared by the flywheel test suite.

The doubles are deliberately small: a fake HTTP session, a fake SDK project, and a fake
:class:`RoboflowClient` that serves canned summaries and details. No test touches the network.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import Config, parse_config
from roboflow_api import ImageDetail, ImageSummary, RoboflowClient, RoboflowError


class FakeResponse:
    """A minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code: int, payload: Any = None, content: bytes = b"") -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = content

    def json(self) -> Any:
        """Return the canned JSON payload."""
        return self._payload


class FakeSession:
    """A fake HTTP session that returns queued responses in order."""

    def __init__(self, responses: list[FakeResponse] | None = None) -> None:
        self.responses = responses if responses is not None else []
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> FakeResponse:
        """Return the next queued response, recording the URL."""
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


class FakeProject:
    """A fake Roboflow SDK project object."""

    def __init__(self, pages: list[list[dict[str, Any]]] | None = None) -> None:
        self.pages = pages if pages is not None else []
        self.uploaded: list[dict[str, Any]] = []
        self.raise_on_search = False
        self.raise_on_save = False

    def search(self, limit: int, offset: int, fields: list[str]) -> Any:
        """Return the page for the requested offset."""
        if self.raise_on_search:
            raise RuntimeError("search failed")
        index = offset // 100
        return self.pages[index] if index < len(self.pages) else []

    def save_annotation(self, **kwargs: Any) -> Any:
        """Record an uploaded annotation."""
        if self.raise_on_save:
            raise RuntimeError("save failed")
        self.uploaded.append(kwargs)
        return {"ok": True}


def summary(image_id: str, name: str, count: int, split: str = "train") -> ImageSummary:
    """Build an :class:`ImageSummary` for tests."""
    return ImageSummary(id=image_id, name=name, split=split, annotation_count=count)


def detail(
    image_id: str,
    name: str,
    *,
    width: int = 32,
    height: int = 32,
    objects: list[Any] | None = None,
    url: str = "https://example.test/image.jpg",
) -> ImageDetail:
    """Build an :class:`ImageDetail` for tests."""
    return ImageDetail(
        id=image_id,
        name=name,
        width=width,
        height=height,
        objects=objects if objects is not None else [],
        image_url=url,
    )


class FakeClient(RoboflowClient):
    """A :class:`RoboflowClient` that never touches the network."""

    def __init__(self, *, summaries: list[ImageSummary] | None = None) -> None:
        # Deliberately do not call super().__init__: there is no network and no key.
        self.summaries = summaries if summaries is not None else []
        self.details: dict[str, ImageDetail] = {}
        self.uploads: list[tuple[str, Mapping[str, Any]]] = []
        self.uploads_error = False
        self.image = np.zeros((32, 32, 3), dtype=np.uint8)

    def iter_images(self, slug: str) -> Iterator[ImageSummary]:
        """Yield the canned summaries."""
        return iter(self.summaries)

    def image_detail(self, slug: str, image_id: str) -> ImageDetail:
        """Return the canned detail for an image id."""
        return self.details[image_id]

    def download(self, url: str, destination: Path) -> int:
        """Write a small image to the destination."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), self.image)
        return int(self.image.nbytes)

    def upload_annotation(
        self,
        slug: str,
        image_id: str,
        document: Mapping[str, Any],
        *,
        name: str = "annotation.coco.json",
        labelmap: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        """Record an upload, or fail when configured to."""
        if self.uploads_error:
            raise RoboflowError("upload failed")
        self.uploads.append((image_id, document))
        return {"ok": True}


def make_config(tmp_path: Path, **prelabel: Any) -> Config:
    """Build a small valid :class:`Config` in a temporary directory.

    Parameters
    ----------
    tmp_path : Path
        Directory the round writes to.
    **prelabel : Any
        Overrides merged into the ``prelabel`` section.

    Returns
    -------
    Config
        The configuration.
    """
    document: dict[str, Any] = {
        "round": 1,
        "workspace": "ws",
        "train_project": "train-project",
        "prelabel_project": "prelabel-project",
        "prelabel_class": "microsite",
        "data_root": str(tmp_path / "data"),
        "state_path": str(tmp_path / "loop_state.json"),
        "model": {
            "encoder": "resnet34",
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
            "patience": 1,
            "image_size": 64,
        },
        "prelabel": {
            "threshold": 0.5,
            "min_polygon_area": 1,
            "tta": False,
            "limit": 0,
            "split": "train",
        },
    }
    document["prelabel"].update(prelabel)
    return parse_config(document, base_dir=tmp_path)


def square_object(label: str = "microsite") -> Any:
    """Return a square :class:`DetectedObject` for tests."""
    from geometry import DetectedObject

    return DetectedObject(
        label=label,
        points=((4.0, 4.0), (20.0, 4.0), (20.0, 20.0), (4.0, 20.0)),
    )
