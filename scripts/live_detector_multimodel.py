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


def classify_detection(label: str) -> str:
    """Map a class label to our category system."""
    label_lower = label.lower()

    if label_lower in COCO_PERSON_CLASSES or "person" in label_lower:
        return "2"
    elif label_lower in COCO_VEHICLE_CLASSES:
        return "3"
    else:
        # Default to animal for unknown classes (conservative for wildlife detection)
        return "1"


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
    models: List[ModelConfig],
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

    logging.info(f"Processing {relative} with {len(models)} model(s)")

    model_results: Dict[str, ModelResult] = {}
    promoted_by: Optional[str] = None

    for model_config in models:
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

    # Promote if any model found something interesting
    if promoted_by:
        event_dir = promote_event(
            segment_path,
            args.segments_dir,
            args.events_dir,
            model_results,
            promoted_by,
            routing_result,
        )

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
        meta = {
            "segment": str(relative),
            "created_at": format_utc(utc_now()),
            "promoted": False,
            "models": {name: result.to_dict() for name, result in model_results.items()},
        }

        # Add routing info even for non-promoted segments
        if routing_result is not None:
            meta["routing"] = routing_result.to_dict()

        (det_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        max_conf = max(r.max_confidence for r in model_results.values())
        logging.info(f"  No promotion (max_conf={max_conf:.3f} < threshold={args.event_threshold})")


def main() -> int:
    load_env_file()
    args = parse_args()
    configure_logging(args.log_file)

    # Create directories
    args.segments_dir.mkdir(parents=True, exist_ok=True)
    args.detections_dir.mkdir(parents=True, exist_ok=True)
    args.events_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    models = load_models(args.models, args.device)
    if not models:
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
    logging.info(f"Models: {[m.name for m in models]}")
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

                process_segment(args, models, segment_path, pseudo_store, routing_config)
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
