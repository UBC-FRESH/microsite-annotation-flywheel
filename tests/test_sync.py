"""Unit tests for the sync step."""

from __future__ import annotations

from pathlib import Path

import pytest

from fakes import FakeClient, detail, make_config, square_object, summary
from roboflow_api import RoboflowError
from sync import load_provenance, result_to_dict, run_sync


def _client() -> FakeClient:
    client = FakeClient(
        summaries=[summary("a1", "annotated.jpg", 2, "train"), summary("u1", "unannotated.jpg", 0)]
    )
    client.details["a1"] = detail("a1", "annotated.jpg", objects=[square_object()])
    return client


def test_run_sync_writes_dataset_and_provenance(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result = run_sync(config, _client())

    assert result.annotated == 1
    assert result.per_split == {"train": 1, "valid": 0, "test": 0}
    assert result.unannotated == 1
    assert (config.dataset_dir / "train" / "annotated.jpg").is_file()
    assert (config.dataset_dir / "train" / "annotated_mask.png").is_file()
    assert result.provenance_path.is_file()
    assert result.unannotated_path.is_file()


def test_run_sync_honours_limit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result = run_sync(config, _client(), limit=1)
    assert result.annotated == 1


def test_run_sync_skips_images_without_geometry(tmp_path: Path) -> None:
    client = FakeClient(summaries=[summary("a1", "annotated.jpg", 2, "train")])
    client.details["a1"] = detail("a1", "annotated.jpg", width=0, height=0)
    result = run_sync(make_config(tmp_path), client)
    assert result.annotated == 0


def test_run_sync_trains_only_on_dataset_images(tmp_path: Path) -> None:
    # A prediction-only image carries an annotation count but is not in the Dataset, and must
    # never be used as training ground truth.
    client = FakeClient(
        summaries=[
            summary("h1", "human.jpg", 3, "train"),
            summary("p1", "prediction.jpg", 1, "train"),
        ],
        dataset_ids={"h1"},
    )
    client.details["h1"] = detail("h1", "human.jpg", objects=[square_object()])
    config = make_config(tmp_path)
    result = run_sync(config, client)
    assert result.annotated == 1
    assert (config.dataset_dir / "train" / "human.jpg").is_file()
    assert not (config.dataset_dir / "train" / "prediction.jpg").exists()


def test_load_provenance_round_trip(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result = run_sync(config, _client())
    record = load_provenance(result.provenance_path)
    assert record["project"] == "train-project"
    assert record["synced"] == 1


def test_load_provenance_missing(tmp_path: Path) -> None:
    with pytest.raises(RoboflowError, match="not found"):
        load_provenance(tmp_path / "absent.json")


def test_result_to_dict_is_serialisable(tmp_path: Path) -> None:
    result = run_sync(make_config(tmp_path), _client())
    data = result_to_dict(result)
    assert data["dataset_dir"] == str(result.dataset_dir)
    assert data["annotated"] == 1
