"""Unit tests for the polygon and mask geometry helpers."""

from __future__ import annotations

import numpy as np

from geometry import (
    DetectedObject,
    mask_to_polygons,
    objects_from_annotation,
    polygons_to_coco,
    rasterize,
)


def test_objects_from_annotation_reads_polygons() -> None:
    annotation = {
        "boxes": [
            {"label": "microsite", "points": [[0, 0], [10, 0], [10, 10]]},
            {"label": "ignore", "points": [[1, 1]]},  # too few points, skipped
            {"label": "no-points"},
        ]
    }
    objects = objects_from_annotation(annotation)

    assert len(objects) == 1
    assert objects[0].label == "microsite"
    assert objects[0].points == ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))


def test_rasterize_fills_polygon_and_filters_by_label() -> None:
    square = DetectedObject(
        label="microsite", points=((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0))
    )

    mask = rasterize([square], 8, 8)
    assert mask.shape == (8, 8)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0
    assert rasterize([square], 8, 8, label="other").sum() == 0


def test_mask_to_polygons_drops_small_components() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:12, 2:12] = 1  # large square
    mask[18, 18] = 1  # single pixel, below the area floor

    polygons = mask_to_polygons(mask, min_area=4)

    assert len(polygons) == 1
    assert len(polygons[0]) >= 3


def test_polygons_to_coco_structure() -> None:
    square = [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)]
    document = polygons_to_coco(
        image_name="frame.jpg",
        width=32,
        height=32,
        polygons=[square],
        category_name="a58-sandbox-microsite",
    )

    assert document["images"][0]["file_name"] == "frame.jpg"
    assert document["categories"][0]["name"] == "a58-sandbox-microsite"
    assert len(document["annotations"]) == 1
    annotation = document["annotations"][0]
    assert annotation["segmentation"][0][:2] == [0.0, 0.0]
    assert annotation["area"] > 0
    assert annotation["bbox"] == [0.0, 0.0, 6.0, 6.0]


def test_objects_from_annotation_handles_no_boxes() -> None:
    assert objects_from_annotation({}) == []
