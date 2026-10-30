"""Unit tests for the pre-labelling step and inference helper."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from config import Config
from fakes import FakeClient, detail, make_config, square_object, summary
from model import build_model
from prelabel import load_empty_history, predict_image, run_prelabel, save_empty_history
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


def test_run_prelabel_never_targets_dataset_images(tmp_path: Path) -> None:
    # A Dataset image with zero annotations must still be excluded: Dataset membership, not
    # annotation presence alone, is what protects it.
    config = make_config(tmp_path, threshold=0.0)
    client = FakeClient(
        summaries=[summary("d1", "in_dataset.jpg", 0), summary("u1", "free.jpg", 0)],
        dataset_ids={"d1"},
    )
    client.details["d1"] = detail("d1", "in_dataset.jpg", objects=[])
    client.details["u1"] = detail("u1", "free.jpg", objects=[])
    checkpoint = _checkpoint(config)

    result = run_prelabel(config, client, checkpoint, device=torch.device("cpu"))

    assert "d1" not in result.uploaded_images
    assert [i for i, _ in client.uploads] == ["u1"]


def test_load_and_save_empty_history(tmp_path: Path) -> None:
    path = tmp_path / "empty_history.json"
    assert load_empty_history(path) == {}
    save_empty_history(path, {"a": 3, "b": 5})
    assert load_empty_history(path) == {"a": 3, "b": 5}
    path.write_text("not json", encoding="utf-8")
    assert load_empty_history(path) == {}
    path.write_text("[1, 2]", encoding="utf-8")
    assert load_empty_history(path) == {}


def test_run_prelabel_skips_recently_empty_frames(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=0.0)  # round 1, cooldown 3
    save_empty_history(tmp_path / "empty_history.json", {"u1": 1})
    client = FakeClient(summaries=[summary("u1", "old.jpg", 0), summary("u2", "new.jpg", 0)])
    client.details["u1"] = detail("u1", "old.jpg", objects=[])
    client.details["u2"] = detail("u2", "new.jpg", objects=[])
    checkpoint = _checkpoint(config)

    result = run_prelabel(config, client, checkpoint, device=torch.device("cpu"))

    assert "u1" not in result.uploaded_images
    assert "u2" in result.uploaded_images


def test_predict_image_missing_file(tmp_path: Path) -> None:
    model = build_model("resnet34", encoder_weights=None)
    with pytest.raises(FileNotFoundError, match="could not read image"):
        predict_image(
            model, tmp_path / "absent.jpg", torch.device("cpu"), threshold=0.5, tta=False, size=64
        )
