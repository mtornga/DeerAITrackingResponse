"""Shared helpers for training loop agents."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

VIDEO_EXTS = {".mkv", ".mp4"}


def resolve_share_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()

    candidates: List[Path] = []
    env_server = os.environ.get("DEER_SHARE_SERVER_PATH")
    env_local = os.environ.get("DEER_SHARE_LOCAL_MOUNT")

    if env_server:
        candidates.append(Path(env_server).expanduser())
    if env_local:
        candidates.append(Path(env_local).expanduser())

    candidates.append(Path("/srv/deer-share"))
    candidates.append(Path.home() / "DeerShare")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise SystemExit(
        f"Unable to locate shared storage root; checked: {', '.join(str(c) for c in candidates)}"
    )


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def find_video_file(clip_dir: Path) -> Optional[Path]:
    for ext in VIDEO_EXTS:
        matches = sorted(clip_dir.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def extract_best_bbox(meta: Dict[str, Any], min_conf: float = 0.3) -> Optional[Tuple[float, float, float, float]]:
    best_conf = -1.0
    best_bbox: Optional[Tuple[float, float, float, float]] = None

    models = meta.get("models")
    if isinstance(models, dict):
        for model_data in models.values():
            hits = model_data.get("hits") if isinstance(model_data, dict) else None
            if not isinstance(hits, list):
                continue
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                conf = hit.get("confidence") or hit.get("conf")
                try:
                    conf_val = float(conf)
                except (TypeError, ValueError):
                    conf_val = 0.0
                if conf_val < min_conf:
                    continue
                bbox = hit.get("bbox") or hit.get("box")
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    if conf_val > best_conf:
                        best_conf = conf_val
                        best_bbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    if best_bbox:
        return best_bbox

    detections = meta.get("detections")
    if isinstance(detections, list):
        for det in detections:
            if not isinstance(det, dict):
                continue
            conf = det.get("confidence") or det.get("conf")
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                conf_val = 0.0
            if conf_val < min_conf:
                continue
            bbox = det.get("bbox") or det.get("box")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                if conf_val > best_conf:
                    best_conf = conf_val
                    best_bbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    return best_bbox


def extract_frames(
    video_path: Path,
    output_dir: Path,
    prefix: str,
    frame_interval: int,
    max_frames: int,
) -> List[Path]:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependency
        raise SystemExit(f"OpenCV not available: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    extracted: List[Path] = []
    frame_idx = 0
    saved = 0

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            output_path = output_dir / f"{prefix}_{saved:04d}.jpg"
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted.append(output_path)
            saved += 1
        frame_idx += 1

    cap.release()
    return extracted


def write_label_files(
    frames: Iterable[Path],
    labels_dir: Path,
    class_id: int,
    bbox: Sequence[float],
) -> int:
    labels_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for frame_path in frames:
        label_path = labels_dir / f"{frame_path.stem}.txt"
        label_line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
        label_path.write_text(label_line)
        count += 1
    return count


def choose_split(clip_id: str, val_split: float) -> str:
    if val_split <= 0:
        return "train"
    if val_split >= 1:
        return "val"
    digest = hashlib.md5(clip_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_split else "train"
