"""Queue Enricher preview frame helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2

DEFAULT_FRAME_COUNT = 3
DEFAULT_RATIOS = (0.1, 0.5)
DEFAULT_MAX_DIM = 640
DEFAULT_JPEG_QUALITY = 75


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


def select_preview_frames_from_meta(
    meta_path: Path,
    video_path: Path,
    count: int = DEFAULT_FRAME_COUNT,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> PreviewSelection:
    meta = load_meta_json(meta_path)
    peak_frame = extract_peak_frame(meta)
    total_frames = get_video_frame_count(video_path)
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
