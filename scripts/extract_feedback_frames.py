#!/usr/bin/env python3
"""Extract frames from reviewed clips for the feedback training cycle.

This script is the first step in the continuous improvement loop:
1. Extract frames from tagged clips (deer, person, false_positive)
2. Organize into categories for training data curation
3. Feed into the deer-vision retraining pipeline

The extracted frames can then be:
- 'animal' tagged: Added to positive training set with deer labels
- 'person' tagged: Added to training set with person labels (or excluded)
- 'false_positive' tagged: Added as hard negatives to reduce FP rate

Usage:
    python scripts/extract_feedback_frames.py --dry-run
    python scripts/extract_feedback_frames.py --output-dir outdoor/deer-vision/data/raw/feedback
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set


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


UTC = timezone.utc


def resolve_share_root() -> Path:
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


@dataclass
class ClipInfo:
    clip_id: str
    path: Path
    tags: List[str]
    max_conf: Optional[float]
    notes: str
    category: str  # 'animal', 'person', 'false_positive', 'unknown'


@dataclass
class ExtractionStats:
    clips_processed: int = 0
    frames_extracted: int = 0
    by_category: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def categorize_clip(tags: List[str]) -> str:
    """Determine the primary category from clip tags."""
    tags_lower = {t.lower() for t in tags}

    if "animal" in tags_lower or "deer" in tags_lower:
        return "animal"
    if "person" in tags_lower:
        return "person"
    if "false_positive" in tags_lower:
        return "false_positive"
    if "nomammalsseen" in tags_lower:
        return "hard_negative"

    return "unknown"


def load_reviewed_clips(share_root: Path) -> List[ClipInfo]:
    """Load all reviewed (done) clips from the index."""

    index_path = share_root / "index" / "daily_review_index.json"
    if not index_path.exists():
        raise SystemExit(f"Index file not found: {index_path}")

    index_data = json.loads(index_path.read_text())
    clips_data = index_data.get("clips", {})

    reviewed: List[ClipInfo] = []
    for clip_id, data in clips_data.items():
        if data.get("review_status") != "done":
            continue

        clip_path = share_root / data.get("path", clip_id)
        tags = data.get("tags", [])

        reviewed.append(ClipInfo(
            clip_id=clip_id,
            path=clip_path,
            tags=tags,
            max_conf=data.get("max_conf"),
            notes=data.get("notes", ""),
            category=categorize_clip(tags),
        ))

    return reviewed


def extract_frames_ffmpeg(
    video_path: Path,
    output_dir: Path,
    prefix: str,
    stride: int = 5,
    max_frames: int = 50,
) -> int:
    """Extract frames from video using ffmpeg.

    Args:
        video_path: Path to input video
        output_dir: Directory for extracted frames
        prefix: Prefix for output filenames
        stride: Extract every Nth frame
        max_frames: Maximum frames to extract

    Returns:
        Number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg to extract frames
    output_pattern = output_dir / f"{prefix}_%04d.jpg"

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"select=not(mod(n\\,{stride}))",
        "-vsync", "vfr",
        "-frames:v", str(max_frames),
        "-q:v", "2",
        str(output_pattern),
        "-y",  # Overwrite
        "-loglevel", "error",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}")

    # Count extracted frames
    frames = list(output_dir.glob(f"{prefix}_*.jpg"))
    return len(frames)


def extract_feedback_frames(
    share_root: Path,
    output_dir: Path,
    stride: int = 5,
    max_frames_per_clip: int = 50,
    categories: Optional[Set[str]] = None,
    dry_run: bool = True,
) -> ExtractionStats:
    """Extract frames from reviewed clips organized by category."""

    stats = ExtractionStats()
    reviewed_clips = load_reviewed_clips(share_root)

    if categories is None:
        categories = {"animal", "person", "false_positive", "hard_negative"}

    print(f"Found {len(reviewed_clips)} reviewed clips")
    print(f"Filtering to categories: {categories}")
    print()

    for clip in reviewed_clips:
        if clip.category not in categories:
            continue

        if not clip.path.exists():
            stats.errors.append(f"Video not found: {clip.path}")
            continue

        # Create category subdirectory
        category_dir = output_dir / clip.category

        # Generate unique prefix from clip path
        date_str = clip.path.parent.name
        stem = clip.path.stem
        prefix = f"{date_str}_{stem}"

        print(f"[{clip.category}] {clip.path.name}")
        print(f"  Tags: {clip.tags}")
        print(f"  Max conf: {clip.max_conf:.3f}" if clip.max_conf else "  Max conf: N/A")
        if clip.notes:
            print(f"  Notes: {clip.notes[:60]}...")

        if dry_run:
            print(f"  -> Would extract to: {category_dir / prefix}_*.jpg")
            stats.clips_processed += 1
            stats.by_category[clip.category] = stats.by_category.get(clip.category, 0) + 1
        else:
            try:
                n_frames = extract_frames_ffmpeg(
                    clip.path,
                    category_dir,
                    prefix,
                    stride=stride,
                    max_frames=max_frames_per_clip,
                )
                stats.clips_processed += 1
                stats.frames_extracted += n_frames
                stats.by_category[clip.category] = stats.by_category.get(clip.category, 0) + 1
                print(f"  -> Extracted {n_frames} frames")
            except Exception as e:
                stats.errors.append(f"Failed to extract {clip.path}: {e}")
                print(f"  -> ERROR: {e}")

        print()

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outdoor" / "deer-vision" / "data" / "raw" / "feedback",
        help="Output directory for extracted frames (default: %(default)s)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Extract every Nth frame (default: %(default)s)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Max frames per clip (default: %(default)s)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["animal", "person", "false_positive", "hard_negative"],
        help="Categories to extract (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without actually running ffmpeg",
    )
    args = parser.parse_args()

    share_root = resolve_share_root()
    print(f"Share root: {share_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXTRACTING'}")
    print()

    stats = extract_feedback_frames(
        share_root=share_root,
        output_dir=args.output_dir,
        stride=args.stride,
        max_frames_per_clip=args.max_frames,
        categories=set(args.categories),
        dry_run=args.dry_run,
    )

    print("=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Clips processed:  {stats.clips_processed}")
    print(f"Frames extracted: {stats.frames_extracted}")
    print()
    print("By category:")
    for cat, count in sorted(stats.by_category.items()):
        print(f"  {cat}: {count} clips")

    if stats.errors:
        print()
        print("ERRORS:")
        for err in stats.errors:
            print(f"  {err}")

    if args.dry_run:
        print()
        print("This was a dry run. Run without --dry-run to extract frames.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
