"""Tests for the operator command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

import loop
from fakes import FakeClient, detail, make_config, square_object, summary
from model import build_model
from state import LoopState, RoundRecord, load_state, save_state
from train import TrainingResult


def test_read_dotenv(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text('# comment\nA=1\nB = "two"\n\n', encoding="utf-8")
    assert loop.read_dotenv(path) == {"A": "1", "B": "two"}
    assert loop.read_dotenv(tmp_path / "absent") == {}


def test_resolve_api_key(tmp_path: Path) -> None:
    assert loop.resolve_api_key(tmp_path, {"ROBOFLOW_API_KEY": "env-key"}) == "env-key"
    (tmp_path / ".env").write_text("ROBOFLOW_API_KEY=file-key\n", encoding="utf-8")
    assert loop.resolve_api_key(tmp_path, {}) == "file-key"
    with pytest.raises(SystemExit):
        loop.resolve_api_key(tmp_path / "empty", {})


def test_build_parser_has_every_command() -> None:
    parser = loop.build_parser()
    for command in ("sync", "train", "prelabel", "compare", "status"):
        assert parser.parse_args([command]).command == command


def test_cmd_status_prints_the_trend(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = make_config(tmp_path)
    state = LoopState()
    record = RoundRecord(1, "t", "p", "c")
    record.convergence = {"round": 1, "images": 2}
    state.upsert(record)
    save_state(config.state_path, state)

    args = loop.build_parser().parse_args(["status"])
    assert loop.cmd_status(config, args) == 0
    assert "rounds recorded: 1" in capsys.readouterr().out


def test_cmd_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_config(tmp_path)
    client = FakeClient(summaries=[summary("a1", "a.jpg", 1, "train")])
    client.details["a1"] = detail("a1", "a.jpg", objects=[square_object()])
    monkeypatch.setattr(loop, "build_client", lambda config, root: client)

    assert loop.cmd_sync(config, loop.build_parser().parse_args(["sync"])) == 0
    assert (config.dataset_dir / "train" / "a.jpg").is_file()


def test_cmd_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_config(tmp_path)
    result = TrainingResult(
        checkpoint_path=tmp_path / "best.pt",
        metrics_path=tmp_path / "metrics.json",
        best_val_iou=0.5,
        best_epoch=1,
        test_iou=0.4,
        epochs_run=1,
    )
    monkeypatch.setattr(loop, "run_train", lambda config, encoder_weights="imagenet": result)

    assert loop.cmd_train(config, loop.build_parser().parse_args(["train"])) == 0
    assert load_state(config.state_path).for_round(1) is not None


def test_cmd_prelabel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_config(tmp_path)
    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        build_model("resnet34", encoder_weights=None).state_dict(),
        config.checkpoints_dir / "best.pt",
    )
    client = FakeClient(summaries=[summary("u1", "u.jpg", 0, "train")])
    client.details["u1"] = detail("u1", "u.jpg", objects=[])
    monkeypatch.setattr(loop, "build_client", lambda config, root: client)

    assert loop.cmd_prelabel(config, loop.build_parser().parse_args(["prelabel"])) == 0
    assert client.uploads[0][0] == "u1"


def test_cmd_compare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_config(tmp_path)
    config.predictions_dir.mkdir(parents=True, exist_ok=True)
    uploads = {
        "round": 1,
        "uploaded": [
            {
                "image_id": "u1",
                "name": "u.jpg",
                "width": 32,
                "height": 32,
                "polygons": [[[4, 4], [20, 4], [20, 20], [4, 20]]],
            }
        ],
    }
    (config.predictions_dir / "uploads-round-1.json").write_text(
        json.dumps(uploads), encoding="utf-8"
    )
    client = FakeClient()
    client.details["u1"] = detail("u1", "u.jpg", objects=[square_object()])
    monkeypatch.setattr(loop, "build_client", lambda config, root: client)

    assert loop.cmd_compare(config, loop.build_parser().parse_args(["compare"])) == 0
    record = load_state(config.state_path).for_round(1)
    assert record is not None
    assert record.convergence is not None


def test_cmd_compare_without_uploads(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert loop.cmd_compare(config, loop.build_parser().parse_args(["compare"])) == 1


def test_main_reports_a_bad_config(tmp_path: Path) -> None:
    assert loop.main(["--config", str(tmp_path / "absent.yaml"), "status"]) == 2


def test_main_runs_status(tmp_path: Path) -> None:
    document = {
        "round": 1,
        "workspace": "ws",
        "train_project": "t",
        "prelabel_project": "p",
        "prelabel_class": "microsite",
        "data_root": "data",
        "state_path": "loop_state.json",
        "model": {
            "encoder": "resnet34",
            "epochs": 1,
            "batch_size": 1,
            "patience": 1,
            "image_size": 32,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    assert loop.main(["--config", str(path), "status"]) == 0
