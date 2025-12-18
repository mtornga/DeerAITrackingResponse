#!/usr/bin/env python3
"""Generate detection thumbnails for event clips.

Creates a single "best detection" thumbnail for each event showing:
- The frame with highest confidence detection
- Bounding box overlay with label and confidence
- Saved as thumbnail.jpg in the event directory

Usage:
    # Generate thumbnail for a single event
    python scripts/generate_event_thumbnail.py \
        --event-dir /srv/deer-share/runs/live/events/2025-12-17/segment_232808

    # Batch generate for all events missing thumbnails
    python scripts/generate_event_thumbnail.py --batch

    # Force regenerate all thumbnails
    python scripts/generate_event_thumbnail.py --batch --force
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _ensure_repo_root_on_path() -> Path:
    script_path = Path(__file__).resolve()
    for parent in (script_path.parent, *script_path.parents):
        if (parent / ".env").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    fallback = script_path.parents[1]
    if str(fallback) not in sys.path:
        sys.path.insert(0, str(fallback))
    return fallback


REPO_ROOT = _ensure_repo_root_on_path()

try:
    from env_loader import load_env_file
    load_env_file()
except ImportError:
    pass


# Colors for different categories (BGR format for OpenCV)
CATEGORY_COLORS = {
    "animal": (0, 165, 255),    # Orange
    "deer": (0, 165, 255),      # Orange
    "person": (255, 100, 100),  # Blue
    "vehicle": (100, 255, 100), # Green
    "unknown": (128, 128, 128), # Gray
}

THUMBNAIL_WIDTH = 640
THUMBNAIL_HEIGHT = 360


def resolve_share_root() -> Path:
    """Find the deer-share root directory."""
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


def load_event_meta(event_dir: Path) -> Optional[Dict]:
    """Load meta.json from event directory."""
    meta_path = event_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None


def find_best_detection(meta: Dict) -> Optional[Dict]:
    """Find the highest confidence detection across all models."""
    best_hit = None
    best_conf = 0.0

    models_data = meta.get("models", {})
    for model_name, model_result in models_data.items():
        hits = model_result.get("hits", [])
        for hit in hits:
            conf = hit.get("confidence", 0)
            if conf > best_conf:
                best_conf = conf
                best_hit = {
                    **hit,
                    "model": model_name,
                }

    return best_hit


def extract_frame(video_path: Path, frame_number: int) -> Optional[np.ndarray]:
    """Extract a specific frame from video using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None
    return frame


def extract_frame_ffmpeg(video_path: Path, frame_number: int, fps: float = 15.0) -> Optional[np.ndarray]:
    """Extract a specific frame using ffmpeg (faster for seeking in large files)."""
    # Calculate timestamp from frame number
    timestamp = frame_number / fps

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg",
            "-ss", str(timestamp),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            tmp_path,
            "-y",
            "-loglevel", "error",
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            frame = cv2.imread(tmp_path)
            return frame
    except subprocess.CalledProcessError:
        pass
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return None


def draw_detection_box(
    frame: np.ndarray,
    bbox: List[float],
    category: str,
    confidence: float,
    model: str = "",
) -> np.ndarray:
    """Draw bounding box and label on frame.

    bbox format: [x_center, y_center, width, height] normalized 0-1
    """
    h, w = frame.shape[:2]

    # Convert normalized bbox to pixel coordinates
    x_center, y_center, box_w, box_h = bbox
    x1 = int((x_center - box_w / 2) * w)
    y1 = int((y_center - box_h / 2) * h)
    x2 = int((x_center + box_w / 2) * w)
    y2 = int((y_center + box_h / 2) * h)

    # Clamp to frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Get color for category
    color = CATEGORY_COLORS.get(category.lower(), CATEGORY_COLORS["unknown"])

    # Draw rectangle with thick border
    thickness = max(2, int(min(w, h) / 200))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Create label text
    label = f"{category} {confidence:.0%}"
    if model:
        label = f"{label} ({model})"

    # Calculate text size and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(w, h) / 1000)
    text_thickness = max(1, int(font_scale * 2))
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    # Draw label background
    label_y = max(y1 - 5, text_h + 5)
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - 5),
        (x1 + text_w + 10, label_y + 5),
        color,
        -1,  # Filled
    )

    # Draw label text
    cv2.putText(
        frame,
        label,
        (x1 + 5, label_y),
        font,
        font_scale,
        (255, 255, 255),  # White text
        text_thickness,
    )

    return frame


def generate_thumbnail(
    event_dir: Path,
    output_path: Optional[Path] = None,
    force: bool = False,
) -> Optional[Path]:
    """Generate a thumbnail for an event.

    Returns the path to the generated thumbnail, or None if failed.
    """
    if output_path is None:
        output_path = event_dir / "thumbnail.jpg"

    # Skip if already exists (unless force)
    if output_path.exists() and not force:
        return output_path

    # Load metadata
    meta = load_event_meta(event_dir)
    if not meta:
        print(f"  No meta.json found in {event_dir}")
        return None

    # Find the video file
    video_files = list(event_dir.glob("*.mkv")) + list(event_dir.glob("*.mp4"))
    if not video_files:
        print(f"  No video file found in {event_dir}")
        return None
    video_path = video_files[0]

    # Find best detection
    best_hit = find_best_detection(meta)
    if not best_hit:
        print(f"  No detections found in {event_dir}")
        return None

    # Parse frame number from frame name (e.g., "frame_000840" -> 840)
    frame_name = best_hit.get("frame", "frame_000000")
    try:
        frame_number = int(frame_name.replace("frame_", ""))
    except ValueError:
        frame_number = 0

    # Extract frame
    frame = extract_frame_ffmpeg(video_path, frame_number)
    if frame is None:
        # Fallback to OpenCV
        frame = extract_frame(video_path, frame_number)
    if frame is None:
        print(f"  Failed to extract frame {frame_number} from {video_path}")
        return None

    # Draw detection box
    frame = draw_detection_box(
        frame,
        best_hit["bbox"],
        best_hit.get("category", "unknown"),
        best_hit.get("confidence", 0),
        best_hit.get("model", ""),
    )

    # Resize to thumbnail dimensions
    frame = cv2.resize(frame, (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT))

    # Save thumbnail
    cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return output_path


def batch_generate_thumbnails(
    events_root: Path,
    force: bool = False,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    """Generate thumbnails for all events missing them.

    Returns (generated_count, skipped_count).
    """
    generated = 0
    skipped = 0

    # Find all event directories
    event_dirs = []
    for date_dir in sorted(events_root.iterdir(), reverse=True):
        if not date_dir.is_dir() or not date_dir.name.startswith("2"):
            continue
        for event_dir in sorted(date_dir.iterdir(), reverse=True):
            if not event_dir.is_dir():
                continue
            event_dirs.append(event_dir)

    print(f"Found {len(event_dirs)} events to process")

    for event_dir in event_dirs:
        if limit and generated >= limit:
            break

        thumb_path = event_dir / "thumbnail.jpg"
        if thumb_path.exists() and not force:
            skipped += 1
            continue

        print(f"Generating thumbnail for {event_dir.name}...")
        result = generate_thumbnail(event_dir, force=force)
        if result:
            generated += 1
            print(f"  Created: {result}")
        else:
            skipped += 1

    return generated, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--event-dir",
        type=Path,
        help="Path to a specific event directory to process",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all events missing thumbnails",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate thumbnails even if they exist",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of thumbnails to generate (for batch mode)",
    )

    args = parser.parse_args()

    if args.event_dir:
        # Single event mode
        result = generate_thumbnail(args.event_dir, force=args.force)
        if result:
            print(f"Generated: {result}")
            return 0
        else:
            print("Failed to generate thumbnail")
            return 1

    elif args.batch:
        # Batch mode
        share_root = resolve_share_root()
        events_root = share_root / "runs" / "live" / "events"

        if not events_root.exists():
            print(f"Events directory not found: {events_root}")
            return 1

        generated, skipped = batch_generate_thumbnails(
            events_root,
            force=args.force,
            limit=args.limit,
        )
        print(f"\nComplete: {generated} generated, {skipped} skipped")
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
