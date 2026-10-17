"""Polygon and mask geometry shared by pre-labelling and convergence.

Roboflow stores instance-segmentation annotations as polygons, while the model predicts a
binary mask. This module converts between the two representations, and to COCO JSON for
upload. It uses OpenCV for rasterisation and contour extraction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

# A polygon is a sequence of (x, y) points. Coordinates are floats in image pixels.
Point = tuple[float, float]
Polygon = list[Point]


@dataclass(frozen=True)
class DetectedObject:
    """One annotated or predicted object.

    Attributes
    ----------
    label : str
        Class name, such as ``a58-sandbox-microsite``.
    points : tuple of tuple of float
        The polygon outline, as ``(x, y)`` pixels.
    """

    label: str
    points: tuple[Point, ...]


def objects_from_annotation(annotation: Mapping[str, Any]) -> list[DetectedObject]:
    """Read the objects out of a Roboflow image ``annotation`` payload.

    Parameters
    ----------
    annotation : mapping of str to Any
        The ``annotation`` field of a Roboflow image detail, with a ``boxes`` list. Each box
        may carry ``points`` (a polygon) or a bounding ``x``/``y``/``width``/``height``.

    Returns
    -------
    list of DetectedObject
        One object per non-empty box, in payload order. Boxes without points are skipped.
    """
    boxes = annotation.get("boxes") or []
    objects: list[DetectedObject] = []
    for box in boxes:
        points = box.get("points") or []
        if len(points) < 3:
            continue
        polygon: list[Point] = [(float(p[0]), float(p[1])) for p in points]
        objects.append(DetectedObject(label=str(box.get("label", "")), points=tuple(polygon)))
    return objects


def rasterize(
    objects: list[DetectedObject],
    width: int,
    height: int,
    *,
    label: str | None = None,
) -> NDArray[np.uint8]:
    """Draw objects into a binary mask.

    Parameters
    ----------
    objects : list of DetectedObject
        Objects to draw.
    width : int
        Mask width in pixels.
    height : int
        Mask height in pixels.
    label : str or None, optional
        When given, draw only objects with this label.

    Returns
    -------
    numpy.ndarray
        A ``uint8`` array of shape ``(height, width)`` with ``1`` inside any polygon and
        ``0`` elsewhere.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for obj in objects:
        if label is not None and obj.label != label:
            continue
        if len(obj.points) < 3:
            continue
        contour = np.round(np.asarray(obj.points, dtype=np.float64)).astype(np.int32)
        cv2.fillPoly(mask, [contour.reshape(-1, 1, 2)], 1)
    return mask


def mask_to_polygons(
    mask: NDArray[np.uint8],
    *,
    min_area: int = 64,
    epsilon_ratio: float = 0.01,
) -> list[Polygon]:
    """Extract simplified polygons from a binary mask.

    Parameters
    ----------
    mask : numpy.ndarray
        A ``uint8`` mask where non-zero pixels are the object of interest.
    min_area : int, optional
        Connected components with a contour area below this many pixels are dropped.
    epsilon_ratio : float, optional
        The Douglas-Peucker tolerance as a fraction of the contour perimeter. Larger values
        simplify more aggressively.

    Returns
    -------
    list of Polygon
        One simplified polygon per kept component, each a list of ``(x, y)`` integer points.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[Polygon] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        polygons.append([(float(pt[0][0]), float(pt[0][1])) for pt in approx])
    return polygons


def _polygon_area(points: Polygon) -> float:
    """Return the absolute area of a polygon by the shoelace formula.

    Parameters
    ----------
    points : Polygon
        The polygon outline.

    Returns
    -------
    float
        The polygon area in square pixels.
    """
    area = 0.0
    count = len(points)
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygons_to_coco(
    *,
    image_name: str,
    width: int,
    height: int,
    polygons: list[Polygon],
    category_name: str,
    category_id: int = 1,
) -> dict[str, Any]:
    """Build a single-image COCO document for polymorphic polygons.

    Parameters
    ----------
    image_name : str
        File name recorded for the image.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    polygons : list of Polygon
        The predicted polygons to upload.
    category_name : str
        Class name for every polygon, such as ``a58-sandbox-microsite``.
    category_id : int, optional
        COCO category id; Roboflow resolves the class by name.

    Returns
    -------
    dict of str to Any
        A COCO document with ``images``, ``categories``, and ``annotations`` lists.
    """
    annotations: list[dict[str, Any]] = []
    for index, polygon in enumerate(polygons):
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]
        flat = [coord for pt in polygon for coord in (pt[0], pt[1])]
        annotations.append(
            {
                "id": index + 1,
                "image_id": 1,
                "category_id": category_id,
                "segmentation": [flat],
                "area": _polygon_area(polygon),
                "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                "iscrowd": 0,
            }
        )
    return {
        "images": [{"id": 1, "file_name": image_name, "width": width, "height": height}],
        "categories": [{"id": category_id, "name": category_name}],
        "annotations": annotations,
    }
