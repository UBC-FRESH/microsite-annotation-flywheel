"""Unit tests for the loop state store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from state import LoopState, RoundRecord, load_state, save_state


def _record(number: int, **kwargs: object) -> RoundRecord:
    base: dict[str, object] = {
        "round_number": number,
        "train_project": "train",
        "prelabel_project": "prelabel",
        "prelabel_class": "microsite",
    }
    base.update(kwargs)
    return RoundRecord(**base)  # type: ignore[arg-type]  # kwargs is dynamic in tests


def test_load_state_missing_file_is_empty(tmp_path: Path) -> None:
    state = load_state(tmp_path / "absent.json")
    assert state.rounds == []


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "loop_state.json"
    state = LoopState()
    state.upsert(_record(1, uploaded=3, skipped=1, uploaded_images=["a", "b", "c"]))
    save_state(path, state)

    reloaded = load_state(path)
    assert len(reloaded.rounds) == 1
    assert reloaded.rounds[0].uploaded == 3
    assert reloaded.rounds[0].uploaded_images == ["a", "b", "c"]
    assert reloaded.rounds[0].timestamp != ""


def test_upsert_replaces_and_preserves_convergence() -> None:
    state = LoopState()
    original = _record(1)
    original.convergence = {"mean_pixel_iou": 0.5}
    state.upsert(original)

    updated = _record(1, uploaded=5)
    state.upsert(updated)

    assert len(state.rounds) == 1
    assert state.rounds[0].uploaded == 5
    assert state.rounds[0].convergence == {"mean_pixel_iou": 0.5}


def test_upsert_keeps_rounds_sorted() -> None:
    state = LoopState()
    state.upsert(_record(2))
    state.upsert(_record(1))
    assert [r.round_number for r in state.rounds] == [1, 2]


def test_for_round_returns_record_or_none() -> None:
    state = LoopState()
    state.upsert(_record(3))
    assert state.for_round(3) is not None
    assert state.for_round(9) is None


def test_load_state_rejects_malformed(tmp_path: Path) -> None:
    path = tmp_path / "loop_state.json"
    path.write_text(json.dumps({"rounds": "not-a-list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_state(path)


def test_load_state_rejects_non_object_round(tmp_path: Path) -> None:
    path = tmp_path / "loop_state.json"
    path.write_text(json.dumps({"rounds": [42]}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-object"):
        load_state(path)


def test_to_dict_is_json_serialisable() -> None:
    state = LoopState()
    state.upsert(_record(1))
    json.dumps(state.to_dict())
