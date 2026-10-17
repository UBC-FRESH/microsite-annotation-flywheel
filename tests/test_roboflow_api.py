"""Unit tests for the Roboflow client and its parsing helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import roboflow

import roboflow_api
from fakes import FakeProject, FakeResponse, FakeSession
from roboflow_api import (
    HttpSession,
    RoboflowClient,
    RoboflowError,
    _default_project_loader,
    _summary_from,
)


def test_summary_from_reads_counts() -> None:
    assert _summary_from({"annotations": {"count": 3}}).annotation_count == 3
    assert _summary_from({"annotations": [1, 2]}).annotation_count == 2
    assert _summary_from({}).annotation_count == 0
    assert _summary_from({"annotations": {"count": 0}}).is_annotated is False


def test_default_project_loader_builds_a_project(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeWorkspace:
        def project(self, slug: str) -> object:
            return {"slug": slug}

    class FakeRoboflow:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def workspace(self, name: str) -> FakeWorkspace:
            return FakeWorkspace()

    monkeypatch.setattr(roboflow, "Roboflow", FakeRoboflow)
    assert _default_project_loader("key", "ws", "proj") == {"slug": "proj"}


def test_default_project_loader_wraps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom:
        def __init__(self, api_key: str) -> None:
            raise RuntimeError("nope")

    monkeypatch.setattr(roboflow, "Roboflow", Boom)
    with pytest.raises(RoboflowError, match="could not load project"):
        _default_project_loader("key", "ws", "proj")


def _client(
    session: HttpSession | None = None,
    project: object | None = None,
) -> RoboflowClient:
    loader = (lambda k, w, s: project) if project is not None else None
    return RoboflowClient("key", "ws", project_loader=loader, session=session)


def test_project_info_ok() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200, {"project": {"name": "P", "type": "instance-segmentation", "annotation": "g"}}
            )
        ]
    )
    info = _client(session).project_info("proj")
    assert info.name == "P"
    assert info.project_type == "instance-segmentation"
    assert session.calls == [f"{roboflow_api.API_BASE}/ws/proj"]


def test_project_info_missing_object() -> None:
    session = FakeSession([FakeResponse(200, {})])
    with pytest.raises(RoboflowError, match="no project object"):
        _client(session).project_info("proj")


def test_get_json_errors() -> None:
    with pytest.raises(RoboflowError, match="HTTP 500"):
        _client(FakeSession([FakeResponse(500, {})])).project_info("p")
    with pytest.raises(RoboflowError, match="non-object"):
        _client(FakeSession([FakeResponse(200, [])])).project_info("p")


def test_iter_images_pages_through() -> None:
    page1 = [
        {"id": f"i{n}", "name": f"f{n}.jpg", "annotations": [], "split": "train"}
        for n in range(100)
    ]
    page2 = [{"id": "last", "name": "last.jpg", "annotations": {"count": 1}, "split": "valid"}]
    images = list(_client(project=FakeProject([page1, page2])).iter_images("p"))
    assert len(images) == 101
    assert images[-1].is_annotated


def test_iter_images_wraps_search_errors() -> None:
    project = FakeProject()
    project.raise_on_search = True
    with pytest.raises(RoboflowError, match="failed"):
        list(_client(project=project).iter_images("p"))


def test_iter_images_rejects_non_list() -> None:
    class NonList:
        def search(
            self, limit: int, offset: int, fields: list[str], in_dataset: object = None
        ) -> object:
            return {"nope": True}

    with pytest.raises(RoboflowError, match="was not a list"):
        list(_client(project=NonList()).iter_images("p"))


def test_image_detail_ok() -> None:
    payload = {
        "image": {
            "id": "a",
            "name": "a.jpg",
            "annotation": {
                "width": 32,
                "height": 16,
                "boxes": [{"label": "microsite", "points": [[0, 0], [5, 0], [5, 5]]}],
            },
            "urls": {"original": "https://x/a.jpg"},
        }
    }
    result = _client(FakeSession([FakeResponse(200, payload)])).image_detail("p", "a")
    assert (result.width, result.height) == (32, 16)
    assert result.objects[0].label == "microsite"


def test_image_detail_missing_object() -> None:
    with pytest.raises(RoboflowError, match="no image object"):
        _client(FakeSession([FakeResponse(200, {})])).image_detail("p", "a")


def test_download_writes_file(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(200, content=b"bytes")])
    destination = tmp_path / "sub" / "image.jpg"
    written = _client(session).download("https://x/a.jpg", destination)
    assert written == 5
    assert destination.read_bytes() == b"bytes"


def test_download_errors() -> None:
    with pytest.raises(RoboflowError, match="empty URL"):
        _client().download("", Path("x"))
    session = FakeSession([FakeResponse(404)])
    with pytest.raises(RoboflowError, match="HTTP 404"):
        _client(session).download("https://x/a.jpg", Path("x"))


def test_upload_annotation_ok() -> None:
    project = FakeProject()
    result = _client(project=project).upload_annotation(
        "p", "a", {"images": []}, labelmap={"1": "m"}
    )
    assert result == {"response": {"ok": True}}
    sent = project.uploaded[0]
    assert sent["image_id"] == "a"
    assert sent["is_prediction"] is True
    assert sent["annotation_overwrite"] is False
    assert sent["add_to_dataset"] is False
    assert sent["annotation_path"]["name"] == "annotation.coco.json"


def test_upload_annotation_tuple_and_error() -> None:
    class TupleProject:
        def save_annotation(self, **kwargs: object) -> object:
            return ("payload", 0.1, 0)

    assert _client(project=TupleProject()).upload_annotation("p", "a", {}) == {
        "response": "payload"
    }

    project = FakeProject()
    project.raise_on_save = True
    with pytest.raises(RoboflowError, match="uploading annotation"):
        _client(project=project).upload_annotation("p", "a", {})


def test_http_session_protocol_is_importable() -> None:
    assert roboflow_api.HttpSession is HttpSession
