"""Unit tests for the convergence metrics."""

from __future__ import annotations

import numpy as np
import pytest

from convergence import (
    aggregate,
    image_convergence,
    object_diff,
    pixel_diff,
    polygon_iou,
    trend_table,
)
from geometry import DetectedObject

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def _obj(points: list[tuple[float, float]], label: str = "microsite") -> DetectedObject:
    return DetectedObject(label=label, points=tuple(points))


def test_polygon_iou_identical_and_disjoint() -> None:
    assert polygon_iou(SQUARE, SQUARE) == pytest.approx(1.0)
    far = [(100.0, 100.0), (110.0, 100.0), (110.0, 110.0), (100.0, 110.0)]
    assert polygon_iou(SQUARE, far) == 0.0
    assert polygon_iou([(0.0, 0.0)], SQUARE) == 0.0


def test_object_diff_identical_has_no_change() -> None:
    diff = object_diff([_obj(SQUARE)], [_obj(SQUARE)])
    assert (diff.matched, diff.added, diff.deleted, diff.edited) == (1, 0, 0, 0)
    assert diff.change_rate == 0.0


def test_object_diff_detects_added_and_deleted() -> None:
    other = [(20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0)]
    added = object_diff([_obj(SQUARE)], [_obj(SQUARE), _obj(other)])
    assert added.added == 1
    assert added.deleted == 0
    deleted = object_diff([_obj(SQUARE), _obj(other)], [_obj(SQUARE)])
    assert deleted.deleted == 1
    assert deleted.added == 0


def test_object_diff_marks_a_shifted_object_edited() -> None:
    shifted = [(2.0, 0.0), (12.0, 0.0), (12.0, 10.0), (2.0, 10.0)]
    diff = object_diff([_obj(SQUARE)], [_obj(shifted)])
    assert diff.matched == 1
    assert diff.edited == 1
    assert diff.change_rate == pytest.approx(1.0)


def test_object_diff_change_rate_zero_when_nothing_uploaded() -> None:
    diff = object_diff([], [_obj(SQUARE)])
    assert diff.change_rate == 0.0
    assert diff.added == 1


def test_pixel_diff_identical_and_changed() -> None:
    first = np.zeros((4, 4), dtype=np.uint8)
    first[0:2, 0:2] = 1
    assert pixel_diff(first, first).iou == pytest.approx(1.0)

    grown = first.copy()
    grown[2, 2] = 1
    diff = pixel_diff(first, grown)
    assert 0.0 < diff.iou < 1.0
    assert diff.added_fraction > 0.0
    assert diff.removed_fraction == 0.0


def test_pixel_diff_empty_masks() -> None:
    empty = np.zeros((2, 2), dtype=np.uint8)
    diff = pixel_diff(empty, empty)
    assert diff.iou == 1.0
    assert diff.added_fraction == 0.0


def test_image_convergence_reports_both_signals() -> None:
    record = image_convergence([_obj(SQUARE)], [_obj(SQUARE)], 16, 16, label="microsite")
    assert record["matched"] == 1
    assert record["pixel_iou"] == pytest.approx(1.0)
    assert "pixel_added_fraction" in record


def test_aggregate_empty_and_non_empty() -> None:
    empty = aggregate([])
    assert empty["images"] == 0
    assert empty["mean_pixel_iou"] == 0.0

    one = image_convergence([_obj(SQUARE)], [_obj(SQUARE)], 16, 16, label="microsite")
    summary = aggregate([one, one])
    assert summary["images"] == 2
    assert summary["matched"] == 2
    assert summary["mean_pixel_iou"] == pytest.approx(1.0)


def test_trend_table_empty_and_rows() -> None:
    assert "no completed rounds" in trend_table([])
    table = trend_table([{"round": 1, "images": 3, "added": 1, "deleted": 0, "edited": 2}])
    assert "ROUND" in table
    assert "1" in table
