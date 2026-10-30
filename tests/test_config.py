"""Unit tests for the configuration loader."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from config import Config, ConfigError, load_config, parse_config

VALID = {
    "round": 2,
    "workspace": "ws",
    "train_project": "train-proj",
    "prelabel_project": "prelabel-proj",
    "prelabel_class": "microsite",
    "data_root": "_data",
    "state_path": "loop_state.json",
    "model": {
        "encoder": "resnet34",
        "epochs": 3,
        "batch_size": 2,
        "learning_rate": 0.001,
        "patience": 2,
        "image_size": 64,
    },
    "prelabel": {
        "threshold": 0.4,
        "min_polygon_area": 10,
        "tta": False,
        "limit": 5,
        "split": "test",
    },
}


def test_parse_config_reads_every_field(tmp_path: Path) -> None:
    config = parse_config(VALID, base_dir=tmp_path)

    assert isinstance(config, Config)
    assert config.round_number == 2
    assert config.prelabel_class == "microsite"
    assert config.model.encoder == "resnet34"
    assert config.model.image_size == 64
    assert config.prelabel.split == "test"
    assert config.prelabel.tta is False
    assert config.data_root == (tmp_path / "_data").resolve()
    assert config.dataset_dir == (tmp_path / "_data" / "round-2" / "dataset").resolve()
    assert config.checkpoints_dir.name == "checkpoints"


def test_parse_config_applies_defaults() -> None:
    minimal = {k: v for k, v in VALID.items() if k not in ("model", "prelabel")}
    minimal["model"] = {
        "encoder": "resnet34",
        "epochs": 1,
        "batch_size": 1,
        "patience": 1,
        "image_size": 32,
    }
    config = parse_config(minimal, base_dir=Path("."))

    assert config.prelabel.threshold == 0.5
    assert config.prelabel.min_polygon_area == 64
    assert config.prelabel.limit == 0
    assert config.prelabel.tta is True
    assert config.model.learning_rate == pytest.approx(3e-4)
    assert config.model.valid_fraction == pytest.approx(0.15)
    assert config.prelabel.empty_cooldown == 3


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.pop("round"), "round"),
        (lambda d: d.update(round=0), "at least"),
        (lambda d: d.update(round=True), "integer"),
        (lambda d: d.pop("workspace"), "workspace"),
        (lambda d: d.update(workspace="   "), "workspace"),
        (lambda d: d.update(prelabel={"split": "nope"}), "split"),
        (lambda d: d.update(model="x"), "model"),
    ],
)
def test_parse_config_rejects_bad_values(
    mutate: Callable[[dict[str, object]], object], message: str
) -> None:
    document: dict[str, object] = {
        k: (dict(v) if isinstance(v, dict) else v) for k, v in VALID.items()
    }
    mutate(document)
    with pytest.raises(ConfigError) as excinfo:
        parse_config(document, base_dir=Path("."))
    assert message in str(excinfo.value)


def test_parse_config_rejects_non_mapping_document() -> None:
    with pytest.raises(ConfigError, match="<root>"):
        parse_config([], base_dir=Path("."))  # type: ignore[arg-type]  # deliberately wrong type


def test_load_config_reads_a_file(tmp_path: Path) -> None:
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(VALID), encoding="utf-8")
    config = load_config(path)
    assert config.train_project == "train-proj"
    assert config.state_path == (tmp_path / "loop_state.json").resolve()


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_load_config_rejects_empty(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(path)


def test_load_config_rejects_broken_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("round: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="valid YAML"):
        load_config(path)
