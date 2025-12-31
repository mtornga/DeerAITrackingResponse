#!/usr/bin/env python3
"""
Collect reviewed feedback and prepare data for model training.

Scans events with feedback.json and:
- quality=good + category=deer|person -> Extract frames to golden staging
- quality=bad or category=false_positive -> Copy to hard negatives
- quality=marginal -> Queue for CVAT annotation

Usage:
    python scripts/collect_feedback.py --dry-run
    python scripts/collect_feedback.py --process
    python scripts/collect_feedback.py --summary
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


def _ensure_repo_root_on_path() -> Path:
    """Add repo root to path for imports."""
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

try:
    from env_loader import load_env_file
    load_env_file()
except ImportError:
    pass

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
SHARE_ROOT = Path(os.environ.get("DEER_SHARE_SERVER_PATH", "/srv/deer-share"))
EVENTS_DIR = SHARE_ROOT / "runs/live/events"
GOLDEN_STAGING = SHARE_ROOT / "golden_staging"
HARD_NEGATIVES = SHARE_ROOT / "hard_negatives"
CVAT_QUEUE = SHARE_ROOT / "cvat_queue"


@dataclass
class FeedbackEvent:
    """An event with feedback."""
    event_id: str
    event_dir: Path
    feedback: Dict
    meta: Optional[Dict]
    video_path: Optional[Path]

    @property
    def category(self) -> str:
        return self.feedback.get("category", "unknown")

    @property
    def quality(self) -> str:
        return self.feedback.get("quality", "unknown")

    @property
    def is_processable(self) -> bool:
        """Check if this event can be processed."""
        return self.category != "unknown" and self.quality != "unknown"


def find_feedback_events(events_dir: Path, processed_log: Optional[Path] = None) -> List[FeedbackEvent]:
    """Find all events with feedback that haven't been processed."""
    events = []
    processed_ids = set()

    # Load processed log if exists
    if processed_log and processed_log.exists():
        processed_ids = set(processed_log.read_text().strip().split("\n"))

    if not events_dir.exists():
        return events

    # Walk through date directories
    for date_dir in sorted(events_dir.iterdir()):
        if not date_dir.is_dir():
            continue

        for event_dir in sorted(date_dir.iterdir()):
            if not event_dir.is_dir():
                continue

            feedback_path = event_dir / "feedback.json"
            if not feedback_path.exists():
                continue

            event_id = f"{date_dir.name}/{event_dir.name}"

            if event_id in processed_ids:
                continue

            try:
                feedback = json.loads(feedback_path.read_text())
            except Exception as e:
                logger.warning(f"Error reading feedback for {event_id}: {e}")
                continue

            # Load meta if exists
            meta = None
            meta_path = event_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    pass

            # Find video file
            video_path = None
            for ext in ["mkv", "mp4"]:
                videos = list(event_dir.glob(f"*.{ext}"))
                if videos:
                    video_path = videos[0]
                    break

            events.append(FeedbackEvent(
                event_id=event_id,
                event_dir=event_dir,
                feedback=feedback,
                meta=meta,
                video_path=video_path,
            ))

    return events


def extract_frames(
    video_path: Path,
    output_dir: Path,
    prefix: str,
    frame_interval: int = 15,
    max_frames: int = 20,
) -> List[Path]:
    """Extract frames from video at regular intervals."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    extracted = []
    frame_idx = 0
    saved_count = 0

    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            output_path = output_dir / f"{prefix}_{saved_count:04d}.jpg"
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted.append(output_path)
            saved_count += 1

        frame_idx += 1

    cap.release()
    return extracted


def generate_yolo_labels(
    event: FeedbackEvent,
    frames_dir: Path,
    labels_dir: Path,
) -> int:
    """Generate YOLO labels from detection metadata."""
    if not event.meta:
        return 0

    labels_dir.mkdir(parents=True, exist_ok=True)

    # Map category to class ID (matching training data)
    category_to_class = {
        "deer": 0,
        "animal": 0,
        "person": 1,
    }

    class_id = category_to_class.get(event.category)
    if class_id is None:
        return 0

    # Get detections from meta
    detections = []
    for model_name, model_data in event.meta.get("models", {}).items():
        for hit in model_data.get("hits", []):
            if hit.get("confidence", 0) >= 0.3:
                bbox = hit.get("bbox", [])
                if len(bbox) == 4:
                    detections.append(bbox)

    if not detections:
        return 0

    # Use the best detection (highest confidence, or first one)
    best_bbox = detections[0]  # [x_center, y_center, width, height] normalized

    # Write label for each frame
    label_count = 0
    for frame_path in frames_dir.glob("*.jpg"):
        label_path = labels_dir / f"{frame_path.stem}.txt"
        # YOLO format: class x_center y_center width height
        label_line = f"{class_id} {best_bbox[0]:.6f} {best_bbox[1]:.6f} {best_bbox[2]:.6f} {best_bbox[3]:.6f}\n"
        label_path.write_text(label_line)
        label_count += 1

    return label_count


def process_good_event(event: FeedbackEvent, staging_dir: Path, dry_run: bool = False) -> bool:
    """Process a good quality event for training."""
    if event.category not in ("deer", "person"):
        return False

    if not event.video_path or not event.video_path.exists():
        logger.warning(f"No video for {event.event_id}")
        return False

    # Create output directory
    category_dir = staging_dir / event.category / event.event_id.replace("/", "_")

    if dry_run:
        logger.info(f"[DRY-RUN] Would extract frames to: {category_dir}")
        return True

    # Extract frames
    frames_dir = category_dir / "images"
    frames = extract_frames(
        video_path=event.video_path,
        output_dir=frames_dir,
        prefix=event.event_id.replace("/", "_"),
        frame_interval=15,
        max_frames=20,
    )

    if not frames:
        logger.warning(f"No frames extracted for {event.event_id}")
        return False

    # Generate labels
    labels_dir = category_dir / "labels"
    label_count = generate_yolo_labels(event, frames_dir, labels_dir)

    logger.info(f"Extracted {len(frames)} frames, {label_count} labels for {event.event_id}")

    # Copy feedback and meta
    shutil.copy2(event.event_dir / "feedback.json", category_dir / "feedback.json")
    if (event.event_dir / "meta.json").exists():
        shutil.copy2(event.event_dir / "meta.json", category_dir / "meta.json")

    return True


def process_bad_event(event: FeedbackEvent, negatives_dir: Path, dry_run: bool = False) -> bool:
    """Process a bad/false positive event as hard negative."""
    if not event.video_path or not event.video_path.exists():
        return False

    # Create output directory
    output_dir = negatives_dir / event.event_id.replace("/", "_")

    if dry_run:
        logger.info(f"[DRY-RUN] Would copy hard negative to: {output_dir}")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract a few frames as hard negatives (no labels)
    frames = extract_frames(
        video_path=event.video_path,
        output_dir=output_dir / "images",
        prefix=event.event_id.replace("/", "_"),
        frame_interval=30,
        max_frames=5,
    )

    # Copy feedback for reference
    shutil.copy2(event.event_dir / "feedback.json", output_dir / "feedback.json")

    logger.info(f"Added {len(frames)} hard negative frames for {event.event_id}")
    return True


def process_marginal_event(event: FeedbackEvent, cvat_dir: Path, dry_run: bool = False) -> bool:
    """Queue marginal event for CVAT annotation."""
    if not event.video_path or not event.video_path.exists():
        return False

    output_dir = cvat_dir / event.event_id.replace("/", "_")

    if dry_run:
        logger.info(f"[DRY-RUN] Would queue for CVAT: {output_dir}")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy video for annotation
    shutil.copy2(event.video_path, output_dir / event.video_path.name)

    # Copy metadata
    shutil.copy2(event.event_dir / "feedback.json", output_dir / "feedback.json")
    if (event.event_dir / "meta.json").exists():
        shutil.copy2(event.event_dir / "meta.json", output_dir / "meta.json")

    logger.info(f"Queued for CVAT: {event.event_id}")
    return True


def process_events(
    events: List[FeedbackEvent],
    staging_dir: Path,
    negatives_dir: Path,
    cvat_dir: Path,
    processed_log: Path,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Process all events based on their feedback."""
    stats = {"good": 0, "bad": 0, "marginal": 0, "skipped": 0}

    processed_ids = []

    for event in events:
        if not event.is_processable:
            stats["skipped"] += 1
            continue

        success = False

        if event.quality == "good" and event.category in ("deer", "person"):
            success = process_good_event(event, staging_dir, dry_run)
            if success:
                stats["good"] += 1

        elif event.quality == "bad" or event.category == "false_positive":
            success = process_bad_event(event, negatives_dir, dry_run)
            if success:
                stats["bad"] += 1

        elif event.quality == "marginal":
            success = process_marginal_event(event, cvat_dir, dry_run)
            if success:
                stats["marginal"] += 1

        else:
            stats["skipped"] += 1

        if success and not dry_run:
            processed_ids.append(event.event_id)

    # Update processed log
    if processed_ids and not dry_run:
        with open(processed_log, "a") as f:
            for event_id in processed_ids:
                f.write(event_id + "\n")

    return stats


def print_summary(events: List[FeedbackEvent]) -> None:
    """Print summary of feedback events."""
    by_category = {}
    by_quality = {}

    for event in events:
        cat = event.category
        qual = event.quality
        by_category[cat] = by_category.get(cat, 0) + 1
        by_quality[qual] = by_quality.get(qual, 0) + 1

    print(f"\nTotal events with feedback: {len(events)}")
    print("\nBy category:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")
    print("\nBy quality:")
    for qual, count in sorted(by_quality.items()):
        print(f"  {qual}: {count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-dir", type=Path, default=EVENTS_DIR)
    parser.add_argument("--staging-dir", type=Path, default=GOLDEN_STAGING)
    parser.add_argument("--negatives-dir", type=Path, default=HARD_NEGATIVES)
    parser.add_argument("--cvat-dir", type=Path, default=CVAT_QUEUE)
    parser.add_argument("--dry-run", action="store_true", help="Don't actually process, just show what would happen")
    parser.add_argument("--process", action="store_true", help="Process all pending feedback")
    parser.add_argument("--summary", action="store_true", help="Show summary of feedback")
    parser.add_argument("--include-processed", action="store_true", help="Include already processed events")
    args = parser.parse_args()

    # Find processed log
    processed_log = args.staging_dir / "processed.log"

    # Find events
    events = find_feedback_events(
        args.events_dir,
        processed_log=None if args.include_processed else processed_log,
    )

    if args.summary:
        print_summary(events)
        return 0

    if not events:
        print("No new feedback events to process")
        return 0

    print(f"Found {len(events)} events with feedback")

    if args.dry_run or args.process:
        # Ensure directories exist
        args.staging_dir.mkdir(parents=True, exist_ok=True)
        args.negatives_dir.mkdir(parents=True, exist_ok=True)
        args.cvat_dir.mkdir(parents=True, exist_ok=True)

        stats = process_events(
            events=events,
            staging_dir=args.staging_dir,
            negatives_dir=args.negatives_dir,
            cvat_dir=args.cvat_dir,
            processed_log=processed_log,
            dry_run=args.dry_run,
        )

        print(f"\nProcessed: {stats['good']} good, {stats['bad']} bad/FP, {stats['marginal']} marginal, {stats['skipped']} skipped")
    else:
        print("Use --dry-run to preview or --process to execute")
        print_summary(events)

    return 0


if __name__ == "__main__":
    sys.exit(main())
