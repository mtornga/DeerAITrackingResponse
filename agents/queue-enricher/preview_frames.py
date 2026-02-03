"""Queue Enricher preview frame helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2

DEFAULT_FRAME_COUNT = 5
DEFAULT_RATIOS = (0.1, 0.5)
DEFAULT_MAX_DIM = 640
DEFAULT_JPEG_QUALITY = 75
DEFAULT_CROP_PADDING = 0.1
DEFAULT_CENTER_CROP_RATIO = 0.5
DEFAULT_BUCKET_MOTION_WEIGHT = 0.6
DEFAULT_DIVERSITY_RADIUS = 0.12
DEFAULT_MOTION_SAMPLE_COUNT = 30
DEFAULT_MOTION_MAX_DIM = 160
DEFAULT_MOTION_FRAME_GAP_RATIO = 0.06
DEFAULT_HIT_FRAME_WINDOW = 24
DEFAULT_DETECT_SAMPLE_COUNT = 12
DEFAULT_DETECT_FRAME_GAP_RATIO = 0.08


@dataclass
class PreviewSelection:
    frame_indices: List[int]
    peak_frame: Optional[int]
    total_frames: int


@dataclass
class PreviewResult:
    output_paths: List[Path]
    frame_indices: List[int]
    errors: List[str]


def load_meta_json(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def extract_peak_frame(meta: Dict[str, Any]) -> Optional[int]:
    timeline = meta.get("detection_timeline") or meta.get("timeline") or {}
    peak_frame = timeline.get("peak_frame") or meta.get("peak_frame")
    try:
        return int(peak_frame) if peak_frame is not None else None
    except (TypeError, ValueError):
        return None


def extract_hit_frames(
    meta: Dict[str, Any],
    preferred_category: Optional[str] = None,
    max_frames: int = DEFAULT_FRAME_COUNT,
) -> List[int]:
    models = meta.get("models")
    hits: List[Dict[str, Any]] = []
    if isinstance(models, dict):
        model_iter: Iterable[Any] = models.values()
    elif isinstance(models, list):
        model_iter = models
    else:
        model_iter = []

    for model in model_iter:
        if not isinstance(model, dict):
            continue
        model_hits = model.get("hits")
        if not isinstance(model_hits, list):
            continue
        for hit in model_hits:
            if isinstance(hit, dict):
                hits.append(hit)

    if not hits:
        return []

    preferred_hits = hits
    if preferred_category:
        filtered = [hit for hit in hits if hit.get("category") == preferred_category]
        if filtered:
            preferred_hits = filtered

    def _confidence(hit: Dict[str, Any]) -> float:
        value = hit.get("confidence")
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _frame(hit: Dict[str, Any]) -> Optional[int]:
        frame_value = hit.get("frame")
        if frame_value is None:
            return None
        if isinstance(frame_value, str):
            raw = frame_value
            if raw.startswith("frame_"):
                raw = raw[6:]
            if raw.isdigit():
                return int(raw)
            try:
                return int(float(raw))
            except ValueError:
                return None
        try:
            return int(frame_value)
        except (TypeError, ValueError):
            return None

    sorted_hits = sorted(preferred_hits, key=_confidence, reverse=True)
    frames: List[int] = []
    seen: set[int] = set()
    for hit in sorted_hits:
        frame = _frame(hit)
        if frame is None or frame in seen:
            continue
        frames.append(frame)
        seen.add(frame)
        if len(frames) >= max_frames:
            break
    return frames


def extract_hit_entries(
    meta: Dict[str, Any],
    preferred_category: Optional[str] = None,
    max_hits: int = DEFAULT_FRAME_COUNT,
) -> List[Dict[str, Any]]:
    models = meta.get("models")
    hits: List[Dict[str, Any]] = []
    if isinstance(models, dict):
        model_iter: Iterable[Any] = models.values()
    elif isinstance(models, list):
        model_iter = models
    else:
        model_iter = []

    for model in model_iter:
        if not isinstance(model, dict):
            continue
        model_hits = model.get("hits")
        if not isinstance(model_hits, list):
            continue
        for hit in model_hits:
            if isinstance(hit, dict):
                hits.append(hit)

    if not hits:
        return []

    preferred_hits = hits
    if preferred_category:
        filtered = [hit for hit in hits if hit.get("category") == preferred_category]
        if filtered:
            preferred_hits = filtered

    def _confidence(hit: Dict[str, Any]) -> float:
        value = hit.get("confidence")
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    sorted_hits = sorted(preferred_hits, key=_confidence, reverse=True)
    output: List[Dict[str, Any]] = []
    for hit in sorted_hits:
        output.append(hit)
        if len(output) >= max_hits:
            break
    if not output:
        return []
    while len(output) < max_hits:
        output.append(output[-1])
    return output


def _parse_hit_frame(hit: Dict[str, Any]) -> Optional[int]:
    frame_value = hit.get("frame")
    if frame_value is None:
        return None
    if isinstance(frame_value, str):
        raw = frame_value
        if raw.startswith("frame_"):
            raw = raw[6:]
        if raw.isdigit():
            return int(raw)
        try:
            return int(float(raw))
        except ValueError:
            return None
    try:
        return int(frame_value)
    except (TypeError, ValueError):
        return None


def _hit_center(hit: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    bbox = hit.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    normalized = max(x, y, w, h) <= 1.5
    if not normalized:
        return None
    return x + w * 0.5, y + h * 0.5


def _hit_confidence(hit: Dict[str, Any]) -> float:
    value = hit.get("confidence")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _motion_scores(sorted_hits: List[Dict[str, Any]]) -> List[float]:
    frames = [_parse_hit_frame(hit) for hit in sorted_hits]
    centers = [_hit_center(hit) for hit in sorted_hits]
    scores: List[float] = []
    for idx, hit in enumerate(sorted_hits):
        center = centers[idx]
        frame = frames[idx]
        if center is None or frame is None:
            scores.append(0.0)
            continue
        best = 0.0
        for neighbor in (idx - 1, idx + 1):
            if neighbor < 0 or neighbor >= len(sorted_hits):
                continue
            n_center = centers[neighbor]
            n_frame = frames[neighbor]
            if n_center is None or n_frame is None:
                continue
            frame_delta = max(abs(frame - n_frame), 1)
            dx = center[0] - n_center[0]
            dy = center[1] - n_center[1]
            dist = (dx * dx + dy * dy) ** 0.5
            speed = dist / frame_delta
            if speed > best:
                best = speed
        scores.append(best)
    return scores


def _center_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def select_hit_entries_for_buckets(
    meta: Dict[str, Any],
    preferred_category: Optional[str],
    bucket_count: int,
    total_frames: int,
    motion_weight: float = DEFAULT_BUCKET_MOTION_WEIGHT,
    diversity_radius: float = DEFAULT_DIVERSITY_RADIUS,
) -> List[Dict[str, Any]]:
    models = meta.get("models")
    hits: List[Dict[str, Any]] = []
    if isinstance(models, dict):
        model_iter: Iterable[Any] = models.values()
    elif isinstance(models, list):
        model_iter = models
    else:
        model_iter = []

    for model in model_iter:
        if not isinstance(model, dict):
            continue
        model_hits = model.get("hits")
        if not isinstance(model_hits, list):
            continue
        for hit in model_hits:
            if isinstance(hit, dict):
                hits.append(hit)

    if not hits:
        return []

    preferred_hits = hits
    if preferred_category:
        filtered = [hit for hit in hits if hit.get("category") == preferred_category]
        if filtered:
            preferred_hits = filtered

    timeline = meta.get("detection_timeline") or {}
    first_frame = timeline.get("first_frame")
    last_frame = timeline.get("last_frame")
    try:
        first_frame = int(first_frame) if first_frame is not None else None
    except (TypeError, ValueError):
        first_frame = None
    try:
        last_frame = int(last_frame) if last_frame is not None else None
    except (TypeError, ValueError):
        last_frame = None

    hit_frames = [_parse_hit_frame(hit) for hit in preferred_hits]
    hit_frames_clean = [frame for frame in hit_frames if frame is not None]
    if hit_frames_clean:
        first_frame = min(hit_frames_clean) if first_frame is None else first_frame
        last_frame = max(hit_frames_clean) if last_frame is None else last_frame
    if first_frame is None:
        first_frame = 0
    if last_frame is None:
        last_frame = max(total_frames - 1, first_frame)

    span = max(last_frame - first_frame + 1, 1)
    bucket_size = span / float(max(bucket_count, 1))

    def _bucket_index(frame: int) -> int:
        if bucket_size <= 0:
            return 0
        idx = int((frame - first_frame) / bucket_size)
        return max(0, min(bucket_count - 1, idx))

    sorted_hits = sorted(preferred_hits, key=lambda h: (_parse_hit_frame(h) or 0))
    motion_scores = _motion_scores(sorted_hits)
    scored_hits: List[Tuple[int, Dict[str, Any], float, Optional[Tuple[float, float]]]] = []
    for hit, motion in zip(sorted_hits, motion_scores):
        frame = _parse_hit_frame(hit)
        if frame is None:
            continue
        confidence = _hit_confidence(hit)
        score = confidence + motion_weight * motion
        scored_hits.append((frame, hit, score, _hit_center(hit)))

    buckets: List[List[Tuple[int, Dict[str, Any], float, Optional[Tuple[float, float]]]]] = [
        [] for _ in range(bucket_count)
    ]
    for frame, hit, score, center in scored_hits:
        buckets[_bucket_index(frame)].append((frame, hit, score, center))

    selected: List[Dict[str, Any]] = []
    used_frames: set[int] = set()
    used_centers: List[Tuple[float, float]] = []
    for bucket in buckets:
        if not bucket:
            continue
        sorted_bucket = sorted(bucket, key=lambda item: item[2], reverse=True)
        best = None
        for candidate in sorted_bucket:
            center = candidate[3]
            if center is None:
                best = candidate
                break
            if not used_centers:
                best = candidate
                break
            if all(_center_distance(center, prev) >= diversity_radius for prev in used_centers):
                best = candidate
                break
        if best is None:
            best = sorted_bucket[0]
        frame = best[0]
        if frame in used_frames:
            continue
        selected.append(best[1])
        used_frames.add(frame)
        if best[3] is not None:
            used_centers.append(best[3])
        if len(selected) >= bucket_count:
            break

    if len(selected) < bucket_count:
        remaining = [item for item in scored_hits if item[0] not in used_frames]
        remaining_sorted = sorted(remaining, key=lambda item: item[2], reverse=True)
        for frame, hit, _score, center in remaining_sorted:
            if frame in used_frames:
                continue
            if center is not None and used_centers:
                if any(_center_distance(center, prev) < diversity_radius for prev in used_centers):
                    continue
            selected.append(hit)
            used_frames.add(frame)
            if center is not None:
                used_centers.append(center)
            if len(selected) >= bucket_count:
                break

    return selected


def select_preview_frame_indices(
    total_frames: int,
    peak_frame: Optional[int],
    count: int = DEFAULT_FRAME_COUNT,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> List[int]:
    indices: List[int] = []
    if peak_frame is not None:
        indices.append(int(peak_frame))

    if total_frames > 0:
        max_index = max(total_frames - 1, 0)
        for ratio in ratios:
            indices.append(int(round(max_index * ratio)))
    else:
        indices.extend([0, 1])

    if total_frames <= 0:
        base = [idx for idx in indices if idx >= 0]
        if not base:
            base = [0, 1, 2]
        return _pad_indices(base, count)

    clamped = [max(0, min(total_frames - 1, idx)) for idx in indices]
    return _pad_indices(_unique_preserve(clamped), count, total_frames=total_frames)


def _unique_preserve(items: Iterable[int]) -> List[int]:
    seen: set[int] = set()
    output: List[int] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _pad_indices(indices: List[int], count: int, total_frames: int = 0) -> List[int]:
    output = list(indices)
    if total_frames > 0:
        for idx in range(total_frames):
            if len(output) >= count:
                break
            if idx not in output:
                output.append(idx)
    while len(output) < count:
        output.append(output[-1] if output else 0)
    return output[:count]


def find_queue_video(queue_dir: Path) -> Optional[Path]:
    candidates = sorted(queue_dir.glob("segment_*.mkv"))
    if candidates:
        return candidates[0]
    candidates = sorted(queue_dir.glob("segment_*.mp4"))
    return candidates[0] if candidates else None


def get_video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return frame_count


def resize_frame(frame: Any, max_dim: int = DEFAULT_MAX_DIM) -> Any:
    if max_dim <= 0:
        return frame
    height, width = frame.shape[:2]
    if max(height, width) <= max_dim:
        return frame
    scale = max_dim / float(max(height, width))
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def extract_frame(video_path: Path, frame_index: int) -> Tuple[Optional[Any], Optional[str]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "failed to open video"
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, f"failed to read frame {frame_index}"
    return frame, None


def _bbox_to_pixels(
    bbox: Sequence[float],
    frame_width: int,
    frame_height: int,
    padding: float = DEFAULT_CROP_PADDING,
) -> Optional[Tuple[int, int, int, int]]:
    if len(bbox) != 4:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    normalized = max(x, y, w, h) <= 1.5
    if normalized:
        x *= frame_width
        y *= frame_height
        w *= frame_width
        h *= frame_height
    if w <= 0 or h <= 0:
        return None
    pad_w = w * padding
    pad_h = h * padding
    x1 = int(round(max(0.0, x - pad_w)))
    y1 = int(round(max(0.0, y - pad_h)))
    x2 = int(round(min(frame_width, x + w + pad_w)))
    y2 = int(round(min(frame_height, y + h + pad_h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _center_crop(
    frame: Any,
    ratio: float = DEFAULT_CENTER_CROP_RATIO,
) -> Any:
    height, width = frame.shape[:2]
    crop_w = int(round(width * ratio))
    crop_h = int(round(height * ratio))
    x1 = max(0, (width - crop_w) // 2)
    y1 = max(0, (height - crop_h) // 2)
    x2 = min(width, x1 + crop_w)
    y2 = min(height, y1 + crop_h)
    return frame[y1:y2, x1:x2]


def _resize_for_motion(frame: Any, max_dim: int = DEFAULT_MOTION_MAX_DIM) -> Any:
    height, width = frame.shape[:2]
    if max(height, width) <= max_dim:
        return frame
    scale = max_dim / float(max(height, width))
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def select_motion_frames(
    video_path: Path,
    total_frames: int,
    select_count: int,
    sample_count: int = DEFAULT_MOTION_SAMPLE_COUNT,
    min_gap_ratio: float = DEFAULT_MOTION_FRAME_GAP_RATIO,
) -> List[int]:
    if total_frames <= 0:
        return []
    if select_count <= 0:
        return []

    sample_count = min(sample_count, total_frames)
    if sample_count <= 1:
        return [0]
    indices = [int(round(i * (total_frames - 1) / (sample_count - 1))) for i in range(sample_count)]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    prev_gray = None
    scores: List[Tuple[int, float]] = []
    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = _resize_for_motion(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            score = 0.0
        else:
            diff = cv2.absdiff(gray, prev_gray)
            score = float(diff.mean())
        scores.append((frame_index, score))
        prev_gray = gray
    cap.release()

    scores_sorted = sorted(scores, key=lambda item: item[1], reverse=True)
    min_gap = max(int(round(total_frames * min_gap_ratio)), 1)
    selected: List[int] = []
    for frame_index, _score in scores_sorted:
        if any(abs(frame_index - other) < min_gap for other in selected):
            continue
        selected.append(frame_index)
        if len(selected) >= select_count:
            break
    if len(selected) < select_count:
        for frame_index, _score in scores_sorted:
            if frame_index in selected:
                continue
            selected.append(frame_index)
            if len(selected) >= select_count:
                break
    return selected


def select_person_hits_with_local_model(
    video_path: Path,
    first_frame: int,
    last_frame: int,
    select_count: int,
    sample_count: int = DEFAULT_DETECT_SAMPLE_COUNT,
    min_gap_ratio: float = DEFAULT_DETECT_FRAME_GAP_RATIO,
) -> List[Dict[str, Any]]:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        return []

    if select_count <= 0:
        return []
    if last_frame < first_frame:
        return []

    total_span = max(last_frame - first_frame + 1, 1)
    sample_count = min(sample_count, total_span)
    if sample_count <= 0:
        return []
    if sample_count == 1:
        indices = [first_frame]
    else:
        indices = [
            int(round(first_frame + i * (total_span - 1) / (sample_count - 1)))
            for i in range(sample_count)
        ]

    model_path = Path(__file__).resolve().parents[2] / "yolov8n.pt"
    try:
        model = YOLO(str(model_path))
    except Exception:
        return []

    detections: List[Dict[str, Any]] = []
    for frame_index in indices:
        frame, err = extract_frame(video_path, frame_index)
        if err or frame is None:
            continue
        height, width = frame.shape[:2]
        results = model(frame, verbose=False)
        if not results:
            continue
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or boxes.conf is None or boxes.cls is None:
            continue
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clses = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            continue
        best = None
        for idx, cls_id in enumerate(clses):
            if int(cls_id) != 0:
                continue
            conf = float(confs[idx])
            x1, y1, x2, y2 = [float(v) for v in xyxy[idx]]
            if x2 <= x1 or y2 <= y1:
                continue
            bbox = [
                x1 / width,
                y1 / height,
                (x2 - x1) / width,
                (y2 - y1) / height,
            ]
            if best is None or conf > best["confidence"]:
                best = {
                    "frame": frame_index,
                    "bbox": bbox,
                    "confidence": conf,
                    "category": "person",
                    "source": "local_person",
                }
        if best is not None:
            detections.append(best)

    if not detections:
        return []

    min_gap = max(int(round(total_span * min_gap_ratio)), 1)
    bucket_count = max(select_count, 1)
    bucket_size = total_span / float(bucket_count)

    def _bucket_index(frame: int) -> int:
        if bucket_size <= 0:
            return 0
        idx = int((frame - first_frame) / bucket_size)
        return max(0, min(bucket_count - 1, idx))

    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(bucket_count)]
    for det in detections:
        buckets[_bucket_index(int(det["frame"]))].append(det)

    selected: List[Dict[str, Any]] = []
    used_frames: List[int] = []
    for bucket in buckets:
        if not bucket:
            continue
        best = max(bucket, key=lambda d: float(d.get("confidence", 0.0)))
        frame = int(best["frame"])
        if any(abs(frame - other) < min_gap for other in used_frames):
            continue
        selected.append(best)
        used_frames.append(frame)
        if len(selected) >= select_count:
            break

    if len(selected) < select_count:
        remaining = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
        for det in remaining:
            frame = int(det["frame"])
            if any(abs(frame - other) < min_gap for other in used_frames):
                continue
            selected.append(det)
            used_frames.append(frame)
            if len(selected) >= select_count:
                break

    return selected


def write_preview_frames(
    video_path: Path,
    output_dir: Path,
    frame_indices: Sequence[int],
    max_dim: int = DEFAULT_MAX_DIM,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> PreviewResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []
    errors: List[str] = []

    for idx, frame_index in enumerate(frame_indices, start=1):
        frame, err = extract_frame(video_path, frame_index)
        if err:
            errors.append(err)
            continue
        frame = resize_frame(frame, max_dim=max_dim)
        output_path = output_dir / f"preview_{idx:03d}.jpg"
        ok = cv2.imwrite(
            str(output_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            errors.append(f"failed to write {output_path}")
            continue
        output_paths.append(output_path)

    return PreviewResult(
        output_paths=output_paths,
        frame_indices=list(frame_indices),
        errors=errors,
    )


def write_preview_bundle(
    meta_path: Path,
    video_path: Path,
    output_dir: Path,
    full_count: int = 1,
    crop_count: int = DEFAULT_FRAME_COUNT,
    preferred_category: Optional[str] = None,
    max_dim: int = DEFAULT_MAX_DIM,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> PreviewResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []
    errors: List[str] = []
    frame_indices: List[int] = []

    meta = load_meta_json(meta_path)
    total_frames = get_video_frame_count(video_path)
    timeline = meta.get("detection_timeline") or {}
    peak_frame = extract_peak_frame(meta)
    first_frame = timeline.get("first_frame")
    last_frame = timeline.get("last_frame")
    try:
        first_frame = int(first_frame) if first_frame is not None else None
    except (TypeError, ValueError):
        first_frame = None
    try:
        last_frame = int(last_frame) if last_frame is not None else None
    except (TypeError, ValueError):
        last_frame = None
    if first_frame is None:
        first_frame = 0
    if last_frame is None:
        last_frame = max(total_frames - 1, 0)

    full_indices: List[int] = []
    if peak_frame is not None:
        full_indices.append(peak_frame)
    if last_frame is not None and last_frame not in full_indices:
        full_indices.append(last_frame)
    if not full_indices:
        full_indices = select_preview_frame_indices(
            total_frames=total_frames,
            peak_frame=peak_frame,
            count=max(full_count, 1),
            ratios=DEFAULT_RATIOS,
        )
    if len(full_indices) > max(full_count, 1):
        full_indices = full_indices[:max(full_count, 1)]

    frame_cache: Dict[int, Any] = {}
    for idx, frame_index in enumerate(full_indices, start=1):
        frame, err = extract_frame(video_path, frame_index)
        if err:
            errors.append(err)
            continue
        frame_cache[frame_index] = frame
        frame = resize_frame(frame, max_dim=max_dim)
        output_path = output_dir / f"preview_full_{idx:03d}.jpg"
        ok = cv2.imwrite(
            str(output_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            errors.append(f"failed to write {output_path}")
            continue
        output_paths.append(output_path)
        frame_indices.append(frame_index)

    motion_frames = select_motion_frames(
        video_path=video_path,
        total_frames=total_frames,
        select_count=crop_count,
    )
    local_person_hits: List[Dict[str, Any]] = []
    if preferred_category == "person":
        local_person_hits = select_person_hits_with_local_model(
            video_path=video_path,
            first_frame=first_frame,
            last_frame=last_frame,
            select_count=crop_count,
        )
    hit_entries = select_hit_entries_for_buckets(
        meta,
        preferred_category=preferred_category,
        bucket_count=max(crop_count + 3, crop_count),
        total_frames=total_frames,
    )
    if not hit_entries:
        hit_entries = extract_hit_entries(meta, preferred_category=preferred_category, max_hits=crop_count)
    hit_entries_by_frame: Dict[int, Dict[str, Any]] = {}
    for hit in hit_entries:
        frame = _parse_hit_frame(hit)
        if frame is None:
            continue
        if frame not in hit_entries_by_frame:
            hit_entries_by_frame[frame] = hit

    for hit in local_person_hits:
        frame = _parse_hit_frame(hit)
        if frame is None:
            continue
        hit_entries_by_frame[frame] = hit

    if not motion_frames and local_person_hits:
        motion_frames = [int(hit["frame"]) for hit in local_person_hits if "frame" in hit]
    if not motion_frames:
        motion_frames = [frame for frame in hit_entries_by_frame.keys()]
    if not motion_frames:
        motion_frames = select_preview_frame_indices(
            total_frames=total_frames,
            peak_frame=peak_frame,
            count=crop_count,
            ratios=DEFAULT_RATIOS,
        )

    for idx, target_frame in enumerate(motion_frames, start=1):
        if idx > crop_count:
            break
        frame_index = target_frame
        hit = None
        if hit_entries_by_frame:
            nearest = min(hit_entries_by_frame.keys(), key=lambda f: abs(f - target_frame))
            if abs(nearest - target_frame) <= DEFAULT_HIT_FRAME_WINDOW:
                hit = hit_entries_by_frame[nearest]

        if frame_index is None:
            errors.append("missing frame index for crop hit")
            continue

        frame = frame_cache.get(frame_index)
        if frame is None:
            frame, err = extract_frame(video_path, frame_index)
            if err:
                errors.append(err)
                continue
            frame_cache[frame_index] = frame

        height, width = frame.shape[:2]
        bbox = hit.get("bbox") if isinstance(hit, dict) else None
        crop_box = None
        if isinstance(bbox, list):
            crop_box = _bbox_to_pixels(bbox, width, height)
        if crop_box is None:
            crop = _center_crop(frame)
        else:
            x1, y1, x2, y2 = crop_box
            crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            errors.append(f"empty crop for frame {frame_index}")
            continue
        crop = resize_frame(crop, max_dim=max_dim)
        output_path = output_dir / f"preview_crop_{idx:03d}.jpg"
        ok = cv2.imwrite(
            str(output_path),
            crop,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            errors.append(f"failed to write {output_path}")
            continue
        output_paths.append(output_path)
        frame_indices.append(frame_index)

    return PreviewResult(
        output_paths=output_paths,
        frame_indices=frame_indices,
        errors=errors,
    )


def select_preview_frames_from_meta(
    meta_path: Path,
    video_path: Path,
    count: int = DEFAULT_FRAME_COUNT,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    preferred_category: Optional[str] = None,
) -> PreviewSelection:
    meta = load_meta_json(meta_path)
    hit_frames = extract_hit_frames(meta, preferred_category=preferred_category, max_frames=count)
    peak_frame = extract_peak_frame(meta)
    total_frames = get_video_frame_count(video_path)
    if hit_frames:
        indices = _pad_indices(_unique_preserve(hit_frames), count, total_frames=total_frames)
    else:
        indices = select_preview_frame_indices(
            total_frames=total_frames,
            peak_frame=peak_frame,
            count=count,
            ratios=ratios,
        )
    return PreviewSelection(
        frame_indices=indices,
        peak_frame=peak_frame,
        total_frames=total_frames,
    )
