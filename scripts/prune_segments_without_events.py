#!/usr/bin/env python3
"""Remove recorded segments that never triggered an event."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import sys


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


_ensure_repo_root_on_path()

from env_loader import load_env_file


def default_live_path(relative: str) -> Path:
    """Prefer the shared USB mount when available."""
    overrides = []
    shared_root = os.environ.get("DEER_SHARE_ROOT")
    if shared_root:
        overrides.append(Path(shared_root).expanduser())
    overrides.append(Path("/srv/deer-share"))
    for root in overrides:
        if root.exists():
            return root / relative
    return Path(relative)

SEGMENT_EXTENSIONS = (".mp4", ".mkv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=default_live_path("runs/live/segments"),
        help="Directory that ingests write into (default: %(default)s).",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=default_live_path("runs/live/analysis"),
        help="Directory mirrored for downstream analysis (default: %(default)s).",
    )
    parser.add_argument(
        "--detections-dir",
        type=Path,
        default=default_live_path("runs/live/detections"),
        help="MegaDetector JSON output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=default_live_path("runs/live/events"),
        help="Directory where triggered events are promoted (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without removing any files.",
    )
    return parser.parse_args()


def iter_segments(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    day_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    segments: List[Path] = []
    for day_dir in day_dirs:
        for ext in SEGMENT_EXTENSIONS:
            segments.extend(sorted(day_dir.glob(f"segment_*{ext}")))
    return (path.relative_to(root) for path in segments)


def has_event(rel_path: Path, events_root: Path) -> bool:
    event_dir = events_root / rel_path.parent / rel_path.stem
    return event_dir.exists()


def detection_json_path(rel_path: Path, detections_root: Path) -> Path:
    return detections_root / rel_path.parent / rel_path.stem / "detections.json"


@dataclass
class RemovalStats:
    removed_segments: int = 0
    removed_analysis: int = 0
    removed_detections: int = 0
    pending_segments: int = 0


def remove_segment(rel_path: Path, args: argparse.Namespace, stats: RemovalStats, dry_run: bool) -> None:
    segment_file = args.segments_dir / rel_path
    analysis_file = args.analysis_dir / rel_path
    detections_dir = args.detections_dir / rel_path.parent / rel_path.stem

    if segment_file.exists():
        stats.removed_segments += 1
        if not dry_run:
            segment_file.unlink()

    if analysis_file.exists() and analysis_file != segment_file:
        stats.removed_analysis += 1
        if not dry_run:
            analysis_file.unlink()

    if detections_dir.exists():
        stats.removed_detections += 1
        if not dry_run:
            shutil.rmtree(detections_dir)


def main() -> int:
    load_env_file()
    args = parse_args()

    pruned: List[Path] = []
    stats = RemovalStats()
    for rel_path in iter_segments(args.segments_dir):
        detection_json = detection_json_path(rel_path, args.detections_dir)
        if not detection_json.exists():
            stats.pending_segments += 1
            continue
        if not has_event(rel_path, args.events_dir):
            pruned.append(rel_path)

    if not pruned:
        if stats.pending_segments:
            print(f"No segments ready for pruning (skipped {stats.pending_segments} pending detections).")
        else:
            print("No segments to prune.")
        return 0

    for rel_path in pruned:
        remove_segment(rel_path, args, stats, args.dry_run)

    mode = "DRY RUN" if args.dry_run else "DELETED"
    print(f"{mode}: {len(pruned)} segments without events")
    print(
        f"segments={stats.removed_segments}, analysis={stats.removed_analysis}, detections={stats.removed_detections}"
    )
    if stats.pending_segments:
        print(f"Skipped {stats.pending_segments} segments awaiting detections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
