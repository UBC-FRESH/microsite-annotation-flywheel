"""Convergence: how much the human edited the pre-labels.

After a round of correction, the operator wants to know whether the loop is still improving.
Two signals are computed and shown side by side:

- Object level: pre-label and corrected objects are matched at IoU >= 0.5, and the counts of
  added, deleted, and edited objects are reported.
- Pixel level: the fraction of positive pixels added and removed, and the IoU between the
  uploaded and corrected masks.

Both are per round; :func:`trend_table` prints the history so the operator can stop when it
flattens.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from geometry import DetectedObject, Polygon, rasterize

MATCH_IOU: float = 0.5
EDIT_IOU: float = 0.9


@dataclass(frozen=True)
class ObjectDiff:
    """Object-level agreement between pre-labels and corrections.

    Attributes
    ----------
    matched : int
        Objects present in both, at IoU >= :data:`MATCH_IOU`.
    edited : int
        Matched objects whose shapes changed materially (IoU < :data:`EDIT_IOU`).
    added : int
        Objects the human added.
    deleted : int
        Pre-label objects the human removed.
    uploaded : int
        Total pre-label objects.
    corrected : int
        Total corrected objects.
    """

    matched: int
    edited: int
    added: int
    deleted: int
    uploaded: int
    corrected: int

    @property
    def change_rate(self) -> float:
        """Return the fraction of uploaded objects that were not left untouched."""
        if self.uploaded == 0:
            return 0.0
        untouched = self.matched - self.edited
        return max(0.0, (self.uploaded - untouched) / self.uploaded)


@dataclass(frozen=True)
class PixelDiff:
    """Pixel-level agreement between the uploaded and corrected masks.

    Attributes
    ----------
    iou : float
        IoU of the positive class between the two masks.
    added_fraction : float
        Positive pixels in the correction that were not positive in the pre-label, as a
        fraction of the union.
    removed_fraction : float
        Positive pixels in the pre-label that the correction removed, as a fraction of the
        union.
    """

    iou: float
    added_fraction: float
    removed_fraction: float


def polygon_iou(first: Polygon, second: Polygon) -> float:
    """Return the IoU of two polygons.

    The polygons are rasterised into a canvas covering their combined bounding box, which is
    exact enough for object matching at IoU 0.5.

    Parameters
    ----------
    first : Polygon
        First polygon.
    second : Polygon
        Second polygon.

    Returns
    -------
    float
        The IoU, in ``[0, 1]``. ``0.0`` when either polygon is degenerate.
    """
    if len(first) < 3 or len(second) < 3:
        return 0.0
    xs = [pt[0] for pt in first] + [pt[0] for pt in second]
    ys = [pt[1] for pt in first] + [pt[1] for pt in second]
    x0, x1 = int(np.floor(min(xs))), int(np.ceil(max(xs))) + 1
    y0, y1 = int(np.floor(min(ys))), int(np.ceil(max(ys))) + 1
    width, height = max(x1 - x0, 1), max(y1 - y0, 1)

    def canvas(polygon: Polygon) -> NDArray[np.uint8]:
        mask = np.zeros((height, width), dtype=np.uint8)
        shifted = np.round(np.asarray(polygon) - np.array([x0, y0])).astype(np.int32)
        cv2.fillPoly(mask, [shifted.reshape(-1, 1, 2)], 1)
        return mask

    first_mask, second_mask = canvas(first), canvas(second)
    intersection = int(np.logical_and(first_mask, second_mask).sum())
    union = int(np.logical_or(first_mask, second_mask).sum())
    return intersection / union if union else 0.0


def object_diff(
    uploaded: list[DetectedObject],
    corrected: list[DetectedObject],
    *,
    iou_threshold: float = MATCH_IOU,
    edit_threshold: float = EDIT_IOU,
) -> ObjectDiff:
    """Match pre-label objects to corrected objects and count the differences.

    Greedy matching: each uploaded object is paired with the best corrected object that is
    still free, when their IoU is at least ``iou_threshold``.

    Parameters
    ----------
    uploaded : list of DetectedObject
        Objects that were uploaded as pre-labels.
    corrected : list of DetectedObject
        Objects present after human correction.
    iou_threshold : float, optional
        IoU at or above which two objects count as the same object.
    edit_threshold : float, optional
        IoU below which a matched pair counts as edited.

    Returns
    -------
    ObjectDiff
        The match counts.
    """
    uploaded_polys = [list(obj.points) for obj in uploaded]
    corrected_polys = [list(obj.points) for obj in corrected]
    used: set[int] = set()
    matched = 0
    edited = 0
    for poly in uploaded_polys:
        best_index, best_iou = -1, iou_threshold
        for index, candidate in enumerate(corrected_polys):
            if index in used:
                continue
            score = polygon_iou(poly, candidate)
            if score > best_iou:
                best_index, best_iou = index, score
        if best_index >= 0:
            used.add(best_index)
            matched += 1
            if best_iou < edit_threshold:
                edited += 1
    return ObjectDiff(
        matched=matched,
        edited=edited,
        added=len(corrected_polys) - matched,
        deleted=len(uploaded_polys) - matched,
        uploaded=len(uploaded_polys),
        corrected=len(corrected_polys),
    )


def pixel_diff(
    uploaded_mask: NDArray[np.uint8],
    corrected_mask: NDArray[np.uint8],
) -> PixelDiff:
    """Compare the uploaded and corrected masks at the pixel level.

    Parameters
    ----------
    uploaded_mask : numpy.ndarray
        The mask that was uploaded as a pre-label.
    corrected_mask : numpy.ndarray
        The mask after human correction.

    Returns
    -------
    PixelDiff
        IoU and the added and removed fractions.
    """
    uploaded_positive = np.asarray(uploaded_mask) > 0
    corrected_positive = np.asarray(corrected_mask) > 0
    intersection = int(np.logical_and(uploaded_positive, corrected_positive).sum())
    union = int(np.logical_or(uploaded_positive, corrected_positive).sum())
    added = int(np.logical_and(corrected_positive, ~uploaded_positive).sum())
    removed = int(np.logical_and(uploaded_positive, ~corrected_positive).sum())
    denominator = union if union else 1
    return PixelDiff(
        iou=(intersection / union) if union else 1.0,
        added_fraction=added / denominator,
        removed_fraction=removed / denominator,
    )


def image_convergence(
    uploaded: list[DetectedObject],
    corrected: list[DetectedObject],
    width: int,
    height: int,
    *,
    label: str | None = None,
) -> dict[str, float | int]:
    """Compute both convergence signals for one image.

    Parameters
    ----------
    uploaded : list of DetectedObject
        Pre-label objects.
    corrected : list of DetectedObject
        Corrected objects.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    label : str or None, optional
        Restrict the pixel comparison to one class label.

    Returns
    -------
    dict of str to (float or int)
        Object counts plus pixel IoU and added and removed fractions.
    """
    objects = object_diff(uploaded, corrected)
    pixels = pixel_diff(
        rasterize(uploaded, width, height, label=label),
        rasterize(corrected, width, height, label=label),
    )
    return {
        "uploaded_objects": objects.uploaded,
        "corrected_objects": objects.corrected,
        "matched": objects.matched,
        "edited": objects.edited,
        "added": objects.added,
        "deleted": objects.deleted,
        "object_change_rate": round(objects.change_rate, 4),
        "pixel_iou": round(pixels.iou, 4),
        "pixel_added_fraction": round(pixels.added_fraction, 4),
        "pixel_removed_fraction": round(pixels.removed_fraction, 4),
    }


def aggregate(per_image: list[dict[str, float | int]]) -> dict[str, float | int]:
    """Average per-image convergence records into one round record.

    Parameters
    ----------
    per_image : list of dict of str to (float or int)
        One record per image, from :func:`image_convergence`.

    Returns
    -------
    dict of str to (float or int)
        The summed object counts, the mean object-change rate, and the mean pixel metrics.
        Returns zeros for an empty input.
    """
    if not per_image:
        return {
            "images": 0,
            "uploaded_objects": 0,
            "corrected_objects": 0,
            "matched": 0,
            "edited": 0,
            "added": 0,
            "deleted": 0,
            "mean_object_change_rate": 0.0,
            "mean_pixel_iou": 0.0,
            "mean_pixel_added_fraction": 0.0,
            "mean_pixel_removed_fraction": 0.0,
        }
    count = len(per_image)

    def total(key: str) -> int:
        return int(sum(int(record[key]) for record in per_image))

    def mean(key: str) -> float:
        return round(sum(float(record[key]) for record in per_image) / count, 4)

    return {
        "images": count,
        "uploaded_objects": total("uploaded_objects"),
        "corrected_objects": total("corrected_objects"),
        "matched": total("matched"),
        "edited": total("edited"),
        "added": total("added"),
        "deleted": total("deleted"),
        "mean_object_change_rate": mean("object_change_rate"),
        "mean_pixel_iou": mean("pixel_iou"),
        "mean_pixel_added_fraction": mean("pixel_added_fraction"),
        "mean_pixel_removed_fraction": mean("pixel_removed_fraction"),
    }


def trend_table(rounds: list[dict[str, float | int]]) -> str:
    """Render the per-round convergence history as a text table.

    Parameters
    ----------
    rounds : list of dict of str to (float or int)
        One aggregated convergence record per round, in round order. Each record must carry a
        ``round`` key.

    Returns
    -------
    str
        A fixed-width table, or a one-line notice when there is no history.
    """
    if not rounds:
        return "no completed rounds yet"
    header = (
        f"{'ROUND':>5}  {'IMAGES':>6}  {'ADDED':>6}  {'DELETED':>7}  "
        f"{'EDITED':>6}  {'OBJ_CHG':>7}  {'PIX_IOU':>7}  {'PIX_ADD':>7}  {'PIX_DEL':>7}"
    )
    lines = [header, "-" * len(header)]
    for record in rounds:
        lines.append(
            f"{int(record.get('round', 0)):>5}  {int(record.get('images', 0)):>6}  "
            f"{int(record.get('added', 0)):>6}  {int(record.get('deleted', 0)):>7}  "
            f"{int(record.get('edited', 0)):>6}  "
            f"{float(record.get('mean_object_change_rate', 0.0)):>7.3f}  "
            f"{float(record.get('mean_pixel_iou', 0.0)):>7.3f}  "
            f"{float(record.get('mean_pixel_added_fraction', 0.0)):>7.3f}  "
            f"{float(record.get('mean_pixel_removed_fraction', 0.0)):>7.3f}"
        )
    return "\n".join(lines)
