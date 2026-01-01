#!/usr/bin/env python3
"""
Multi-model detector for A/B testing different detection models.

Runs multiple models (e.g., YOLOv8n and RT-DETR) on each segment,
stores results from all models, and promotes events when any model
exceeds the threshold.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch

# Import routing module (if available)
try:
    from detection_router import (
        RouteDecision,
        RoutingConfig,
        RoutingResult,
        PseudoLabelStore,
        route_detection,
    )
    ROUTING_AVAILABLE = True
except ImportError:
    ROUTING_AVAILABLE = False

# Import notification module (if available)
try:
    from notify import notify_wildlife_detection, PushoverConfig
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False


def _ensure_repo_root_on_path() -> Path:
    script_path = Path(__file__).resolve()
    for parent in (script_path.parent, *script_path.parents):
        candidate = parent / ".env"
        if candidate.exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    root = script_path.parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_path()

from env_loader import load_env_file


def default_live_path(relative: str) -> Path:
    overrides = []
    shared_root = os.environ.get("DEER_SHARE_ROOT")
    if shared_root:
        overrides.append(Path(shared_root).expanduser())
    overrides.append(Path("/srv/deer-share"))
    for root in overrides:
        if root.exists():
            return root / relative
    return Path(relative)


# Defaults
DEFAULT_SEGMENTS_DIR = default_live_path("runs/live/analysis")
DEFAULT_DETECTIONS_DIR = default_live_path("runs/live/detections")
DEFAULT_EVENTS_DIR = default_live_path("runs/live/events")
DEFAULT_LOG_PATH = Path("logs/live_detector_multimodel.log")
# Lighting defaults for day/night selection
DEFAULT_LIGHTING_MODE = "auto"  # auto|day|night
DEFAULT_FRAME_RATIO = 0.5

# Category mapping (MegaDetector style)
CATEGORY_MAP = {"animal": "1", "person": "2", "vehicle": "3"}
CATEGORY_NAMES = {"1": "animal", "2": "person", "3": "vehicle"}
INTERESTING_CATEGORIES = {"1", "2"}  # animal, person

# COCO classes that map to our categories
COCO_ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "deer", "rabbit", "squirrel", "raccoon", "fox"
}
COCO_PERSON_CLASSES = {"person"}
COCO_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "airplane", "boat", "train"}

# Classes to ignore (not interesting for wildlife monitoring)
IGNORED_CLASSES = {
    "apriltag",
    "unknown_animal",  # Ambiguous class from training
    "april_tag",
    "tag",
}


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    path: Path
    model: Any = None  # Loaded model instance


@dataclass
class Detection:
    """A single detection from a model."""
    frame_idx: int
    category: str  # "1", "2", or "3"
    label: str     # human-readable label
    confidence: float
    bbox: List[float]  # [x, y, w, h] normalized

    def to_dict(self) -> Dict:
        return {
            "frame": f"frame_{self.frame_idx:06d}",
            "category": CATEGORY_NAMES.get(self.category, self.category),
            "confidence": self.confidence,
            "bbox": self.bbox,
        }


@dataclass
class ModelResult:
    """Results from running a single model on a segment."""
    model_name: str
    detections: List[Detection] = field(default_factory=list)
    max_confidence: float = 0.0
    inference_time_ms: float = 0.0
    frame_count: int = 0

    def to_dict(self) -> Dict:
        counts: Dict[str, int] = {}
        for det in self.detections:
            label = CATEGORY_NAMES.get(det.category, det.category)
            counts[label] = counts.get(label, 0) + 1

        return {
            "max_confidence": self.max_confidence,
            "inference_ms": self.inference_time_ms,
            "frame_count": self.frame_count,
            "counts": counts,
            "hits": [d.to_dict() for d in self.detections[:30]],  # Limit to 30
        }


def utc_now() -> datetime:
    return datetime.now(UTC)


def format_utc(ts: datetime) -> str:
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_frame_number(frame_str: str) -> int:
    """Extract frame number from 'frame_000123' format."""
    if frame_str.startswith("frame_"):
        return int(frame_str[6:])
    return int(frame_str)


def find_detection_clusters(
    hits: List[Dict],
    max_frame_gap: int = 30,
) -> List[List[Dict]]:
    """
    Group detections into temporal clusters.

    Args:
        hits: List of detection dicts with 'frame' field
        max_frame_gap: Max frames between detections to be in same cluster

    Returns:
        List of clusters, each cluster is a list of hits
    """
    if not hits:
        return []

    # Sort by frame number
    sorted_hits = sorted(hits, key=lambda h: parse_frame_number(h.get("frame", "frame_0")))

    clusters = []
    current_cluster = [sorted_hits[0]]

    for hit in sorted_hits[1:]:
        current_frame = parse_frame_number(hit.get("frame", "frame_0"))
        last_frame = parse_frame_number(current_cluster[-1].get("frame", "frame_0"))

        if current_frame - last_frame <= max_frame_gap:
            current_cluster.append(hit)
        else:
            clusters.append(current_cluster)
            current_cluster = [hit]

    clusters.append(current_cluster)
    return clusters


def has_temporal_consistency(
    model_results: Dict[str, "ModelResult"],
    min_detections: int = 3,
    max_frame_gap: int = 30,
    min_confidence: float = 0.25,
) -> bool:
    """
    Check if detections cluster together (real animal walking through).

    Args:
        model_results: Dict of model name -> ModelResult
        min_detections: Minimum detections in a cluster
        max_frame_gap: Max frames between detections to be in same cluster
        min_confidence: Minimum confidence to count detection

    Returns:
        True if there's a cluster with min_detections
    """
    # Collect all interesting hits from all models
    all_hits = []
    for result in model_results.values():
        for det in result.detections:
            if det.category in INTERESTING_CATEGORIES and det.confidence >= min_confidence:
                all_hits.append(det.to_dict())

    if len(all_hits) < min_detections:
        return False

    clusters = find_detection_clusters(all_hits, max_frame_gap)

    # Check if any cluster has enough detections
    for cluster in clusters:
        if len(cluster) >= min_detections:
            return True

    return False


def is_static_detection(
    model_results: Dict[str, "ModelResult"],
    min_confidence: float = 0.25,
    max_position_variance: float = 0.02,
    min_samples: int = 5,
) -> bool:
    """
    Check if detections are static (same position across frames = likely FP).

    Static detections are typically fixed scene features (posts, shadows,
    dark patches) that the model consistently misclassifies.

    Args:
        model_results: Dict of model name -> ModelResult
        min_confidence: Minimum confidence to include detection
        max_position_variance: Max x-position variance for "static" (0.02 = 2% of frame)
        min_samples: Minimum samples needed to determine if static

    Returns:
        True if detection appears to be static (likely FP)
    """
    all_x_positions = []

    for result in model_results.values():
        for det in result.detections:
            if det.category in INTERESTING_CATEGORIES and det.confidence >= min_confidence:
                # bbox is [x, y, w, h] normalized
                x_center = det.bbox[0] + det.bbox[2] / 2
                all_x_positions.append(x_center)

    if len(all_x_positions) < min_samples:
        return False

    x_variance = max(all_x_positions) - min(all_x_positions)
    return x_variance < max_position_variance


def check_bbox_size(
    model_results: Dict[str, "ModelResult"],
    min_confidence: float = 0.25,
    min_width: float = 0.04,
    min_height: float = 0.08,
) -> bool:
    """
    Check if any detection has a reasonably sized bounding box.

    Very small bounding boxes (< 4% width or < 8% height) are often
    false positives from texture patterns, not real animals.

    Args:
        model_results: Dict of model name -> ModelResult
        min_confidence: Minimum confidence to include detection
        min_width: Minimum bbox width as fraction of frame (0.04 = 4%)
        min_height: Minimum bbox height as fraction of frame (0.08 = 8%)

    Returns:
        True if at least one detection passes size threshold
    """
    for result in model_results.values():
        for det in result.detections:
            if det.category in INTERESTING_CATEGORIES and det.confidence >= min_confidence:
                # bbox is [x, y, w, h] normalized
                width = det.bbox[2]
                height = det.bbox[3]
                if width >= min_width and height >= min_height:
                    return True

    return False


def check_model_agreement(
    model_results: Dict[str, "ModelResult"],
    min_confidence: float = 0.25,
    min_models: int = 2,
) -> bool:
    """
    Check if multiple models agree on detection (reduces single-model FPs).

    Args:
        model_results: Dict of model name -> ModelResult
        min_confidence: Minimum confidence to count as detection
        min_models: Minimum number of models that must detect something

    Returns:
        True if at least min_models detected something
    """
    detecting_models = 0

    for result in model_results.values():
        has_detection = any(
            det.category in INTERESTING_CATEGORIES and det.confidence >= min_confidence
            for det in result.detections
        )
        if has_detection:
            detecting_models += 1

    return detecting_models >= min_models


def check_edge_exclusion(
    model_results: Dict[str, "ModelResult"],
    min_confidence: float = 0.25,
    edge_margin: float = 0.05,
) -> bool:
    """
    Check if any detection is NOT in the edge exclusion zone.

    Detections at frame edges (especially corners) are often camera overlays,
    timestamps, or IR artifacts rather than real animals.

    Args:
        model_results: Dict of model name -> ModelResult
        min_confidence: Minimum confidence to include detection
        edge_margin: Margin from edge to exclude (0.05 = 5% of frame)

    Returns:
        True if at least one detection is outside the edge exclusion zone
    """
    for result in model_results.values():
        for det in result.detections:
            if det.category in INTERESTING_CATEGORIES and det.confidence >= min_confidence:
                # bbox is [x, y, w, h] normalized
                x, y = det.bbox[0], det.bbox[1]
                # Detection is valid if it's not touching the edge
                if x >= edge_margin and y >= edge_margin:
                    return True

    return False


def calculate_detection_timeline(
    model_results: Dict[str, "ModelResult"],
    fps: float = 15.0,
    min_confidence: float = 0.25,
) -> Optional[Dict]:
    """
    Calculate timeline info for detections.

    Returns dict with first_frame, last_frame, first_second, etc.
    """
    all_hits = []
    for result in model_results.values():
        for det in result.detections:
            if det.category in INTERESTING_CATEGORIES and det.confidence >= min_confidence:
                all_hits.append({
                    "frame": det.frame_idx,
                    "confidence": det.confidence,
                    "category": det.category,
                    "bbox": det.bbox,
                })

    if not all_hits:
        return None

    frames = [h["frame"] for h in all_hits]
    confidences = [h["confidence"] for h in all_hits]

    first_frame = min(frames)
    last_frame = max(frames)
    peak_idx = confidences.index(max(confidences))
    peak_hit = all_hits[peak_idx]

    return {
        "first_frame": first_frame,
        "last_frame": last_frame,
        "first_second": round(first_frame / fps, 1),
        "last_second": round(last_frame / fps, 1),
        "duration": round((last_frame - first_frame) / fps, 1),
        "peak_frame": peak_hit["frame"],
        "peak_confidence": round(peak_hit["confidence"], 3),
        "peak_bbox": peak_hit["bbox"],
        "total_detections": len(all_hits),
    }


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple detection models on segments for A/B comparison."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="yolov8n.pt,rtdetr-l.pt",
        help="Comma-separated list of model paths (default: yolov8n.pt,rtdetr-l.pt)",
    )
    parser.add_argument(
        "--models-day",
        type=str,
        default="",
        help="Comma-separated list of model paths for daytime lighting (optional).",
    )
    parser.add_argument(
        "--models-night",
        type=str,
        default="",
        help="Comma-separated list of model paths for nighttime/IR lighting (optional).",
    )
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--detections-dir", type=Path, default=DEFAULT_DETECTIONS_DIR)
    parser.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR)
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold to promote an event (default: 0.25)",
    )
    parser.add_argument(
        "--lighting-mode",
        type=str,
        choices=["auto", "day", "night"],
        default=DEFAULT_LIGHTING_MODE,
        help="Lighting selection mode: auto (choose day/night models), or force day/night.",
    )
    parser.add_argument(
        "--lighting-frame-ratio",
        type=float,
        default=DEFAULT_FRAME_RATIO,
        help="Frame ratio to sample for lighting detection (default: 0.5 for middle).",
    )
    parser.add_argument(
        "--frame-sample",
        type=int,
        default=15,
        help="Process every Nth frame (default: 15)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Seconds between polls for new segments (default: 10)",
    )
    parser.add_argument(
        "--min-segment-age",
        type=float,
        default=30.0,
        help="Minimum file age in seconds before processing (default: 30)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Log file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to run inference on (default: auto)",
    )
    parser.add_argument(
        "--enable-routing",
        action="store_true",
        help="Enable confidence-based routing for autonomous improvement",
    )
    parser.add_argument(
        "--pseudo-label-dir",
        type=Path,
        default=None,
        help="Directory for pseudo-label storage (default: <events_dir>/../pseudo_labels)",
    )
    parser.add_argument(
        "--auto-accept-threshold",
        type=float,
        default=0.85,
        help="Minimum confidence for auto-accept (default: 0.85)",
    )
    parser.add_argument(
        "--min-cluster-detections",
        type=int,
        default=3,
        help="Minimum detections in a temporal cluster to promote (default: 3)",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=30,
        help="Maximum frames between detections to be in same cluster (default: 30)",
    )
    parser.add_argument(
        "--disable-temporal-filter",
        action="store_true",
        help="Disable temporal consistency filter (promote on single detection)",
    )
    # FP reduction filters
    parser.add_argument(
        "--min-bbox-width",
        type=float,
        default=0.04,
        help="Minimum bbox width as fraction of frame (default: 0.04 = 4%%)",
    )
    parser.add_argument(
        "--min-bbox-height",
        type=float,
        default=0.08,
        help="Minimum bbox height as fraction of frame (default: 0.08 = 8%%)",
    )
    parser.add_argument(
        "--static-variance-threshold",
        type=float,
        default=0.02,
        help="Max position variance before detection is rejected as static (default: 0.02)",
    )
    parser.add_argument(
        "--require-model-agreement",
        action="store_true",
        help="Require multiple models to agree before promotion (reduces single-model FPs)",
    )
    parser.add_argument(
        "--min-models-agree",
        type=int,
        default=2,
        help="Minimum number of models that must agree (default: 2)",
    )
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=0.05,
        help="Edge exclusion margin as fraction of frame (default: 0.05 = 5%%)",
    )
    return parser.parse_args()


def load_models(model_specs: str, device: str) -> List[ModelConfig]:
    """Load all specified models."""
    from ultralytics import YOLO, RTDETR

    models = []
    for spec in model_specs.split(","):
        spec = spec.strip()
        if not spec:
            continue

        path = Path(spec)
        if not path.is_absolute():
            path = REPO_ROOT / "models" / spec

        name = path.stem  # e.g., "yolov8n" or "rtdetr-l"

        logging.info(f"Loading model: {name} from {path}")

        try:
            if "rtdetr" in name.lower():
                model = RTDETR(str(path))
            else:
                model = YOLO(str(path))

            if device:
                model.to(device)

            models.append(ModelConfig(name=name, path=path, model=model))
            logging.info(f"  Loaded {name} successfully")
        except Exception as e:
            logging.error(f"  Failed to load {name}: {e}")

    return models


def detect_lighting(
    video_path: Path,
    frame_ratio: float = DEFAULT_FRAME_RATIO,
) -> Tuple[str, float, Dict[str, float]]:
    """Classify segment lighting using a single sampled frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "day", 0.0, {}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ratios = []
    for r in (frame_ratio, 0.5, 0.25, 0.75, 0.1):
        if r not in ratios:
            ratios.append(r)

    frame = None
    for r in ratios:
        target = int(frame_count * r)
        if target > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, candidate = cap.read()
        if ok:
            frame = candidate
            break

    cap.release()
    if frame is None:
        return "day", 0.0, {}

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_hsv = hsv.reshape(-1, 3).mean(axis=0)
    mean_sat = float(mean_hsv[1])
    mean_val = float(mean_hsv[2])

    sat_score = float(np.clip((mean_sat - 15.0) / 70.0, 0.0, 1.0))
    val_score = float(np.clip((mean_val - 60.0) / 140.0, 0.0, 1.0))
    day_score = 0.6 * sat_score + 0.4 * val_score
    label = "day" if day_score >= 0.5 else "night"
    confidence = abs(day_score - 0.5) * 2.0

    metrics = {
        "mean_sat": mean_sat,
        "mean_val": mean_val,
        "sat_score": sat_score,
        "val_score": val_score,
        "day_score": day_score,
    }
    return label, confidence, metrics


def classify_detection(label: str) -> Optional[str]:
    """Map a class label to our category system.

    Returns:
        Category ID ("1", "2", or "3") or None if the class should be ignored.
    """
    label_lower = label.lower()

    # Filter out non-interesting classes (calibration markers, etc.)
    if label_lower in IGNORED_CLASSES:
        return None

    if label_lower in COCO_PERSON_CLASSES or "person" in label_lower:
        return "2"
    elif label_lower in COCO_VEHICLE_CLASSES:
        return "3"
    elif label_lower in COCO_ANIMAL_CLASSES:
        return "1"
    else:
        # Unknown class - log and skip rather than defaulting to animal
        # This prevents false positives from unexpected labels
        return None


def run_model_on_segment(
    model_config: ModelConfig,
    video_path: Path,
    frame_sample: int,
    confidence_threshold: float,
    device: str,
) -> ModelResult:
    """Run a single model on a video segment."""
    result = ModelResult(model_name=model_config.name)
    model = model_config.model

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return result

    frame_idx = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_sample > 1 and frame_idx % frame_sample != 0:
                frame_idx += 1
                continue

            result.frame_count += 1
            h, w = frame.shape[:2]

            # Run inference
            results = model(frame, verbose=False, device=device or None)

            if results and len(results) > 0:
                r = results[0]
                boxes = r.boxes

                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clses = boxes.cls.cpu().numpy()
                    names = model.names if hasattr(model, 'names') else {}

                    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clses):
                        conf_f = float(conf)
                        if conf_f < confidence_threshold:
                            continue

                        class_id = int(cls)
                        label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
                        category = classify_detection(label)

                        # Skip ignored/unknown classes
                        if category is None:
                            continue

                        # Normalize bbox to [x, y, w, h] format
                        bbox = [
                            max(0.0, float(x1) / w),
                            max(0.0, float(y1) / h),
                            min(1.0, float(x2 - x1) / w),
                            min(1.0, float(y2 - y1) / h),
                        ]

                        detection = Detection(
                            frame_idx=frame_idx,
                            category=category,
                            label=label,
                            confidence=conf_f,
                            bbox=bbox,
                        )
                        result.detections.append(detection)
                        result.max_confidence = max(result.max_confidence, conf_f)

            frame_idx += 1

    finally:
        cap.release()

    result.inference_time_ms = (time.time() - start_time) * 1000
    return result


def list_segments(root: Path) -> Iterable[Path]:
    """List all segments in date subdirectories."""
    if not root.exists():
        return []

    segments: List[Path] = []
    for day_dir in sorted(root.iterdir()):
        if not day_dir.is_dir():
            continue
        for ext in (".mp4", ".mkv"):
            segments.extend(sorted(day_dir.glob(f"segment_*{ext}")))
    return segments


def segment_ready(path: Path, min_age: float) -> bool:
    """Check if segment is old enough to process."""
    age = time.time() - path.stat().st_mtime
    return age >= min_age


def get_meta_path(segment_path: Path, segments_root: Path, events_root: Path) -> Path:
    """Get the meta.json path for a segment in the events directory."""
    relative = segment_path.relative_to(segments_root)
    event_dir = events_root / relative.parent / segment_path.stem
    return event_dir / "meta.json"


def promote_event(
    segment_path: Path,
    segments_root: Path,
    events_root: Path,
    model_results: Dict[str, ModelResult],
    promoted_by: str,
    routing_result: Optional[Any] = None,
    detection_timeline: Optional[Dict] = None,
) -> Path:
    """Promote a segment to the events directory with multi-model results."""
    relative = segment_path.relative_to(segments_root)
    event_dir = events_root / relative.parent / segment_path.stem
    event_dir.mkdir(parents=True, exist_ok=True)

    # Copy segment
    segment_target = event_dir / segment_path.name
    if not segment_target.exists():
        shutil.copy2(segment_path, segment_target)

    # Build meta.json with all model results
    meta = {
        "segment": str(relative),
        "created_at": format_utc(utc_now()),
        "promoted_by": promoted_by,
        "models": {name: result.to_dict() for name, result in model_results.items()},
    }

    # Add combined summary
    all_counts: Dict[str, int] = {}
    max_conf_overall = 0.0
    for result in model_results.values():
        max_conf_overall = max(max_conf_overall, result.max_confidence)
        for det in result.detections:
            label = CATEGORY_NAMES.get(det.category, det.category)
            all_counts[label] = all_counts.get(label, 0) + 1

    meta["max_confidence"] = max_conf_overall
    meta["counts"] = all_counts

    # Add detection timeline if available
    if detection_timeline:
        meta["detection_timeline"] = detection_timeline

    # Add routing decision if available
    if routing_result is not None and hasattr(routing_result, "to_dict"):
        meta["routing"] = routing_result.to_dict()

    # Write meta.json
    meta_path = event_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    routing_info = ""
    if routing_result is not None:
        routing_info = f", route={routing_result.decision.value}"
    logging.info(f"Promoted {relative} (by {promoted_by}, max_conf={max_conf_overall:.3f}{routing_info})")

    return event_dir


def process_segment(
    args: argparse.Namespace,
    models_default: List[ModelConfig],
    models_day: List[ModelConfig],
    models_night: List[ModelConfig],
    segment_path: Path,
    pseudo_store: Optional[Any] = None,
    routing_config: Optional[Any] = None,
) -> None:
    """Process a segment with all models."""
    relative = segment_path.relative_to(args.segments_dir)
    meta_path = get_meta_path(segment_path, args.segments_dir, args.events_dir)

    # Skip if already processed
    if meta_path.exists():
        return

    # Pick lighting and models
    lighting_label = args.lighting_mode
    lighting_conf = None
    lighting_metrics: Dict[str, float] = {}
    if args.lighting_mode == "auto" and models_day and models_night:
        lighting_label, lighting_conf, lighting_metrics = detect_lighting(
            segment_path, frame_ratio=args.lighting_frame_ratio
        )
    elif args.lighting_mode in {"day", "night"}:
        lighting_label = args.lighting_mode

    if lighting_label == "day" and models_day:
        active_models = models_day
    elif lighting_label == "night" and models_night:
        active_models = models_night
    else:
        active_models = models_default

    logging.info(
        f"Processing {relative} with lighting={lighting_label} "
        f"(conf={lighting_conf}) using {len(active_models)} model(s)"
    )

    model_results: Dict[str, ModelResult] = {}
    promoted_by: Optional[str] = None

    for model_config in active_models:
        result = run_model_on_segment(
            model_config,
            segment_path,
            args.frame_sample,
            0.1,  # Store low-confidence detections for analysis
            args.device,
        )
        model_results[model_config.name] = result

        # Check if this model triggers promotion
        interesting_detections = [
            d for d in result.detections
            if d.category in INTERESTING_CATEGORIES and d.confidence >= args.event_threshold
        ]

        if interesting_detections and promoted_by is None:
            promoted_by = model_config.name

        logging.info(
            f"  {model_config.name}: {len(result.detections)} detections, "
            f"max_conf={result.max_confidence:.3f}, time={result.inference_time_ms:.0f}ms"
        )

    # Run routing if enabled
    routing_result = None
    if ROUTING_AVAILABLE and args.enable_routing and model_results:
        model_dicts = {name: result.to_dict() for name, result in model_results.items()}
        routing_result = route_detection(model_dicts, routing_config)
        logging.info(f"  Routing: {routing_result.decision.value} ({routing_result.reason})")

    # Check temporal consistency (filter single-frame noise)
    temporal_ok = True
    if not args.disable_temporal_filter and promoted_by:
        temporal_ok = has_temporal_consistency(
            model_results,
            min_detections=args.min_cluster_detections,
            max_frame_gap=args.max_frame_gap,
            min_confidence=args.event_threshold,
        )
        if not temporal_ok:
            logging.info(
                f"  Temporal filter: REJECTED (need {args.min_cluster_detections}+ "
                f"clustered detections within {args.max_frame_gap} frames)"
            )

    # Check bbox size (filter tiny detections from texture patterns)
    bbox_ok = True
    if promoted_by and temporal_ok:
        bbox_ok = check_bbox_size(
            model_results,
            min_confidence=args.event_threshold,
            min_width=args.min_bbox_width,
            min_height=args.min_bbox_height,
        )
        if not bbox_ok:
            logging.info(
                f"  Bbox filter: REJECTED (all detections < {args.min_bbox_width*100:.0f}%×{args.min_bbox_height*100:.0f}%)"
            )

    # Check for static detections (filter fixed scene features)
    static_ok = True
    if promoted_by and temporal_ok and bbox_ok:
        is_static = is_static_detection(
            model_results,
            min_confidence=args.event_threshold,
            max_position_variance=args.static_variance_threshold,
            min_samples=5,
        )
        if is_static:
            static_ok = False
            logging.info(
                f"  Static filter: REJECTED (position variance < {args.static_variance_threshold:.2f})"
            )

    # Check model agreement (filter single-model FPs)
    agreement_ok = True
    if args.require_model_agreement and promoted_by and temporal_ok and bbox_ok and static_ok:
        agreement_ok = check_model_agreement(
            model_results,
            min_confidence=args.event_threshold,
            min_models=args.min_models_agree,
        )
        if not agreement_ok:
            logging.info(
                f"  Agreement filter: REJECTED (< {args.min_models_agree} models agree)"
            )

    # Check edge exclusion (filter corner/edge artifacts)
    edge_ok = True
    if promoted_by and temporal_ok and bbox_ok and static_ok and agreement_ok:
        edge_ok = check_edge_exclusion(
            model_results,
            min_confidence=args.event_threshold,
            edge_margin=args.edge_margin,
        )
        if not edge_ok:
            logging.info(
                f"  Edge filter: REJECTED (all detections within {args.edge_margin*100:.0f}% of frame edge)"
            )

    # Combined filter result
    all_filters_pass = temporal_ok and bbox_ok and static_ok and agreement_ok and edge_ok

    # Calculate detection timeline
    detection_timeline = None
    if promoted_by and all_filters_pass:
        detection_timeline = calculate_detection_timeline(
            model_results,
            fps=15.0,  # Assuming 15fps video
            min_confidence=args.event_threshold,
        )
        if detection_timeline:
            logging.info(
                f"  Timeline: {detection_timeline['first_second']:.1f}s - "
                f"{detection_timeline['last_second']:.1f}s "
                f"(peak @ {detection_timeline['peak_confidence']:.0%})"
            )

    # Promote if any model found something interesting AND passes all filters
    if promoted_by and all_filters_pass:
        event_dir = promote_event(
            segment_path,
            args.segments_dir,
            args.events_dir,
            model_results,
            promoted_by,
            routing_result,
            detection_timeline,
        )

        # Send push notification
        if NOTIFICATIONS_AVAILABLE:
            # Find the best detection for notification
            best_category = "animal"
            best_conf = 0.0
            for result in model_results.values():
                for det in result.detections:
                    if det.confidence > best_conf:
                        best_conf = det.confidence
                        best_category = CATEGORY_NAMES.get(det.category, "animal")

            # Build review URL
            server_ip = os.environ.get("DEERVISION_UBUNTU_HOST", "192.168.68.71")
            review_port = os.environ.get("REVIEW_SERVER_PORT", "8501")
            event_id = f"{relative.parent}/{segment_path.stem}"
            review_url = f"http://{server_ip}:{review_port}/review/{event_id}"

            # Find video path for thumbnail
            video_files = list(event_dir.glob("*.mkv")) + list(event_dir.glob("*.mp4"))
            video_path = video_files[0] if video_files else None

            # Include timing info in notification
            timing_info = None
            if detection_timeline:
                timing_info = (
                    detection_timeline["first_second"],
                    detection_timeline["last_second"],
                )

            try:
                notify_wildlife_detection(
                    event_id=event_id,
                    category=best_category,
                    confidence=best_conf,
                    video_path=video_path,
                    review_url=review_url,
                    timing_info=timing_info,
                )
            except Exception as e:
                logging.warning(f"Notification failed: {e}")

        # Add to pseudo-label store if auto-accepted
        if (
            ROUTING_AVAILABLE
            and pseudo_store is not None
            and routing_result is not None
            and routing_result.decision == RouteDecision.AUTO_ACCEPT
        ):
            model_dicts = {name: result.to_dict() for name, result in model_results.items()}
            pseudo_store.add_clip(str(relative), routing_result, model_dicts)
    else:
        # Still save results for analysis (in detections dir, not events)
        det_dir = args.detections_dir / relative.parent / segment_path.stem
        det_dir.mkdir(parents=True, exist_ok=True)

        # Determine rejection reason (check filters in order they were applied)
        max_conf = max(r.max_confidence for r in model_results.values())
        if promoted_by and not temporal_ok:
            rejection_reason = "temporal_filter"
            rejection_detail = (
                f"Detection found but rejected: need {args.min_cluster_detections}+ "
                f"clustered detections within {args.max_frame_gap} frames"
            )
        elif promoted_by and not bbox_ok:
            rejection_reason = "bbox_size_filter"
            rejection_detail = (
                f"Detection rejected: bbox too small (< {args.min_bbox_width*100:.0f}%×{args.min_bbox_height*100:.0f}%)"
            )
        elif promoted_by and not static_ok:
            rejection_reason = "static_detection_filter"
            rejection_detail = (
                f"Detection rejected: static position (variance < {args.static_variance_threshold:.2f})"
            )
        elif promoted_by and not agreement_ok:
            rejection_reason = "model_agreement_filter"
            rejection_detail = (
                f"Detection rejected: < {args.min_models_agree} models agree"
            )
        elif promoted_by and not edge_ok:
            rejection_reason = "edge_exclusion_filter"
            rejection_detail = (
                f"Detection rejected: all detections within {args.edge_margin*100:.0f}% of frame edge"
            )
        else:
            rejection_reason = "below_threshold"
            rejection_detail = f"max_conf={max_conf:.3f} < threshold={args.event_threshold}"

        meta = {
            "segment": str(relative),
            "created_at": format_utc(utc_now()),
            "promoted": False,
            "rejection_reason": rejection_reason,
            "rejection_detail": rejection_detail,
            "models": {name: result.to_dict() for name, result in model_results.items()},
            "lighting": {
                "label": lighting_label,
                "confidence": lighting_conf,
                "metrics": lighting_metrics,
            },
        }

        # Add routing info even for non-promoted segments
        if routing_result is not None:
            meta["routing"] = routing_result.to_dict()

        (det_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        logging.info(f"  No promotion ({rejection_reason}: {rejection_detail})")


def main() -> int:
    load_env_file()
    args = parse_args()
    configure_logging(args.log_file)

    # Create directories
    args.segments_dir.mkdir(parents=True, exist_ok=True)
    args.detections_dir.mkdir(parents=True, exist_ok=True)
    args.events_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    models_default = load_models(args.models, args.device) if args.models else []
    models_day = load_models(args.models_day, args.device) if args.models_day else []
    models_night = load_models(args.models_night, args.device) if args.models_night else []

    if not any([models_default, models_day, models_night]):
        logging.error("No models loaded, exiting")
        return 1

    # Set up routing if enabled
    pseudo_store = None
    routing_config = None
    if args.enable_routing:
        if not ROUTING_AVAILABLE:
            logging.warning("Routing requested but detection_router module not found")
        else:
            # Set up routing config
            routing_config = RoutingConfig(
                min_models_agree=2,
                min_agreement_confidence=args.auto_accept_threshold,
            )

            # Set up pseudo-label store
            if args.pseudo_label_dir:
                pseudo_label_dir = args.pseudo_label_dir
            else:
                pseudo_label_dir = args.events_dir.parent / "pseudo_labels"

            pseudo_label_dir.mkdir(parents=True, exist_ok=True)
            pseudo_store = PseudoLabelStore(pseudo_label_dir)

            logging.info(f"Routing enabled: auto-accept threshold={args.auto_accept_threshold}")
            logging.info(f"Pseudo-label store: {pseudo_label_dir}")

    logging.info(f"Watching {args.segments_dir} for segments (min age {args.min_segment_age}s)")
    logging.info(
        f"Models default: {[m.name for m in models_default]}, "
        f"day: {[m.name for m in models_day]}, night: {[m.name for m in models_night]}"
    )
    logging.info(f"Event threshold: {args.event_threshold}")

    try:
        while True:
            work_done = False

            for segment_path in list_segments(args.segments_dir):
                if not segment_ready(segment_path, args.min_segment_age):
                    continue

                meta_path = get_meta_path(segment_path, args.segments_dir, args.events_dir)
                if meta_path.exists():
                    continue

                process_segment(
                    args,
                    models_default,
                    models_day,
                    models_night,
                    segment_path,
                    pseudo_store,
                    routing_config,
                )
                work_done = True

            if not work_done:
                time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        logging.info("Stopping on keyboard interrupt")
        if pseudo_store:
            stats = pseudo_store.get_stats()
            logging.info(f"Pseudo-label stats: {stats}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
