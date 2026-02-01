from __future__ import annotations

import math
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_AREA_THRESHOLD = 0.0025
DEFAULT_PERSISTENCE_CAP = 10.0
DEFAULT_DISPERSION_CAP = 0.25
DEFAULT_BURSTINESS_CAP = 2.0


def compute_snow_metrics(
    meta: Dict[str, Any],
    area_threshold: float = DEFAULT_AREA_THRESHOLD,
) -> Dict[str, float]:
    """Compute snow/precip heuristics from a meta.json payload.

    Threshold notes:
    - area_threshold: bbox area <= this is treated as small (default 0.0025).
    - hit_persistence is normalized against DEFAULT_PERSISTENCE_CAP (10 frames).
    - spatial_dispersion is normalized against DEFAULT_DISPERSION_CAP (0.25).
    - burstiness is normalized against DEFAULT_BURSTINESS_CAP (2.0).

    Metrics are normalized to the [0, 1] domain where possible.
    """
    hits = _iter_model_hits(meta)
    bboxes = _extract_bboxes(hits)

    areas = [_bbox_area(bbox) for bbox in bboxes]
    valid_areas = [area for area in areas if area is not None]

    if valid_areas:
        small_box_count = sum(1 for area in valid_areas if area <= area_threshold)
        small_box_ratio = small_box_count / len(valid_areas)
    else:
        small_box_ratio = 0.0

    centers = _extract_centers(bboxes)
    spatial_dispersion = _center_dispersion(centers)

    frames = _extract_frames(hits)
    hit_persistence = _median_run_length(frames)

    burstiness = _burstiness(frames)

    return {
        "small_box_ratio": float(small_box_ratio),
        "hit_persistence": float(hit_persistence),
        "spatial_dispersion": float(spatial_dispersion),
        "burstiness": float(burstiness),
    }


def compute_snow_score(
    meta: Dict[str, Any],
    area_threshold: float = DEFAULT_AREA_THRESHOLD,
) -> Tuple[float, Dict[str, float], str]:
    metrics = compute_snow_metrics(meta, area_threshold=area_threshold)

    persistence_norm = min(1.0, metrics["hit_persistence"] / DEFAULT_PERSISTENCE_CAP)
    dispersion_norm = min(1.0, metrics["spatial_dispersion"] / DEFAULT_DISPERSION_CAP)
    burstiness_norm = min(1.0, metrics["burstiness"] / DEFAULT_BURSTINESS_CAP)

    score = (
        0.45 * metrics["small_box_ratio"]
        + 0.25 * (1.0 - persistence_norm)
        + 0.2 * dispersion_norm
        + 0.1 * burstiness_norm
    )
    score = max(0.0, min(1.0, score))
    reason = summarize_snow_reason(metrics)
    return score, metrics, reason


def summarize_snow_reason(metrics: Dict[str, float]) -> str:
    reasons: List[str] = []

    if metrics["small_box_ratio"] >= 0.6:
        reasons.append("high small-box ratio")
    elif metrics["small_box_ratio"] <= 0.2:
        reasons.append("low small-box ratio")

    if metrics["hit_persistence"] <= 2:
        reasons.append("low hit persistence")
    elif metrics["hit_persistence"] >= 6:
        reasons.append("persistent bbox track")

    if metrics["spatial_dispersion"] >= 0.12:
        reasons.append("high spatial dispersion")
    elif metrics["spatial_dispersion"] <= 0.05:
        reasons.append("tight spatial clustering")

    if metrics["burstiness"] >= 1.5:
        reasons.append("bursty hit cadence")

    if not reasons:
        reasons.append("mixed signal")

    return "; ".join(reasons)


def _iter_model_hits(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    models = meta.get("models")
    if isinstance(models, dict):
        model_iter: Iterable[Any] = models.values()
    elif isinstance(models, list):
        model_iter = models
    else:
        model_iter = []

    hits: List[Dict[str, Any]] = []
    for model in model_iter:
        if not isinstance(model, dict):
            continue
        model_hits = model.get("hits")
        if not isinstance(model_hits, list):
            continue
        for hit in model_hits:
            if isinstance(hit, dict):
                hits.append(hit)
    return hits


def _extract_bboxes(hits: List[Dict[str, Any]]) -> List[List[float]]:
    bboxes: List[List[float]] = []
    for hit in hits:
        bbox = hit.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        try:
            bboxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
        except (TypeError, ValueError):
            continue
    return bboxes


def _bbox_area(bbox: List[float]) -> Optional[float]:
    try:
        width = float(bbox[2])
        height = float(bbox[3])
    except (TypeError, ValueError, IndexError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width * height


def _extract_centers(bboxes: List[List[float]]) -> List[Tuple[float, float]]:
    centers: List[Tuple[float, float]] = []
    for bbox in bboxes:
        if len(bbox) < 4:
            continue
        x, y, w, h = bbox[:4]
        centers.append((x + w / 2.0, y + h / 2.0))
    return centers


def _center_dispersion(centers: List[Tuple[float, float]]) -> float:
    if not centers:
        return 0.0
    xs = [pt[0] for pt in centers]
    ys = [pt[1] for pt in centers]
    x_std = _stddev(xs)
    y_std = _stddev(ys)
    return math.sqrt(x_std ** 2 + y_std ** 2)


def _extract_frames(hits: List[Dict[str, Any]]) -> List[int]:
    frames: List[int] = []
    for hit in hits:
        frame_value = hit.get("frame")
        frame_num = _parse_frame_number(frame_value)
        if frame_num is not None:
            frames.append(frame_num)
    return frames


def _parse_frame_number(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        raw = value
        if raw.startswith("frame_"):
            raw = raw[6:]
        if raw.isdigit():
            return int(raw)
        try:
            return int(float(raw))
        except ValueError:
            return None
    return None


def _median_run_length(frames: List[int]) -> float:
    if not frames:
        return 0.0
    sorted_frames = sorted(frames)
    runs: List[int] = []
    current_run = 1
    for prev, current in zip(sorted_frames, sorted_frames[1:]):
        if current - prev <= 1:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    return float(median(runs)) if runs else 0.0


def _burstiness(frames: List[int]) -> float:
    if not frames:
        return 0.0
    counts: Dict[int, int] = {}
    for frame in frames:
        counts[frame] = counts.get(frame, 0) + 1
    values = list(counts.values())
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    return _stddev(values) / mean


def _stddev(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)
