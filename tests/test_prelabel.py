"""Unit tests for the pre-labelling step and inference helper."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from config import Config
from fakes import FakeClient, detail, make_config, square_object, summary
from model import build_model
from prelabel import predict_image, run_prelabel
from roboflow_api import RoboflowError


def _checkpoint(config: Config) -> Path:
    path = config.checkpoints_dir / "best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model("resnet34", encoder_weights=None)
    torch.save(model.state_dict(), path)
    return path


def test_run_prelabel_uploads_only_unannotated(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=0.0)
    # u2 is listed as annotation-free but is annotated on re-read: the guard must skip it.
    client = FakeClient(summaries=[summary("u1", "u.jpg", 0, "train"), summary("u2", "v.jpg", 0)])
    client.details["u1"] = detail("u1", "u.jpg", objects=[])
    client.details["u2"] = detail("u2", "v.jpg", objects=[square_object()])
    checkpoint = _checkpoint(config)

    result = run_prelabel(config, client, checkpoint, device=torch.device("cpu"))

    assert result.uploaded == 1
    assert result.skipped == 1
    assert result.uploaded_images == ["u1"]
    assert result.skipped_images == ["u2"]
    assert client.uploads[0][0] == "u1"
    assert result.uploads_path.is_file()


def test_run_prelabel_honours_limit(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=0.0)
    client = FakeClient(summaries=[summary("u1", "u.jpg", 0), summary("u2", "v.jpg", 0)])
    client.details["u1"] = detail("u1", "u.jpg", objects=[])
    checkpoint = _checkpoint(config)

    result = run_prelabel(config, client, checkpoint, limit=1, device=torch.device("cpu"))

    assert result.uploaded == 1


def test_run_prelabel_skips_images_without_objects(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=1.5)  # no pixel can exceed 1.0, so no objects
    client = FakeClient(summaries=[summary("u1", "u.jpg", 0)])
    client.details["u1"] = detail("u1", "u.jpg", objects=[])
    checkpoint = _checkpoint(config)

    result = run_prelabel(config, client, checkpoint, device=torch.device("cpu"))

    assert result.uploaded == 0
    assert result.empty == 1
    assert result.empty_images == ["u1"]
    assert client.uploads == []


def test_run_prelabel_propagates_upload_error(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=0.0)
    client = FakeClient(summaries=[summary("u1", "u.jpg", 0)])
    client.details["u1"] = detail("u1", "u.jpg", objects=[])
    client.uploads_error = True
    checkpoint = _checkpoint(config)

    with pytest.raises(RoboflowError, match="upload failed"):
        run_prelabel(config, client, checkpoint, device=torch.device("cpu"))


def test_predict_image_missing_file(tmp_path: Path) -> None:
    model = build_model("resnet34", encoder_weights=None)
    with pytest.raises(FileNotFoundError, match="could not read image"):
        predict_image(
            model, tmp_path / "absent.jpg", torch.device("cpu"), threshold=0.5, tta=False, size=64
        )
