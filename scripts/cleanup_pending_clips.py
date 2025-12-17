#!/usr/bin/env python3
"""Remove pending (unreviewed) clips from the daily review system.

Preserves all clips that have been reviewed (status='done') and removes
everything else to free disk space. This is safe to run after a review
session where you've tagged all the clips worth keeping.

Usage:
    python scripts/cleanup_pending_clips.py --dry-run   # Preview what would be deleted
    python scripts/cleanup_pending_clips.py             # Actually delete
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set


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


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
class CleanupStats:
    kept_clips: int = 0
    removed_analysis: int = 0
    removed_events: int = 0
    bytes_freed: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def get_clip_sizes(path: Path) -> int:
    """Get total size of a file or directory."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def cleanup_pending_clips(
    share_root: Path,
    dry_run: bool = True,
    verbose: bool = False,
) -> CleanupStats:
    """Remove all pending clips, keeping only reviewed (done) clips."""

    index_path = share_root / "index" / "daily_review_index.json"
    analysis_dir = share_root / "runs" / "live" / "analysis"
    events_dir = share_root / "runs" / "live" / "events"

    stats = CleanupStats()

    if not index_path.exists():
        raise SystemExit(f"Index file not found: {index_path}")

    index_data = json.loads(index_path.read_text())
    clips = index_data.get("clips", {})

    # Identify clips to keep (status='done') vs remove (status='pending' or 'in_progress')
    clips_to_keep: Set[str] = set()
    clips_to_remove: Set[str] = set()

    for clip_id, clip_data in clips.items():
        status = clip_data.get("review_status", "pending")
        if status == "done":
            clips_to_keep.add(clip_id)
        else:
            clips_to_remove.add(clip_id)

    print(f"Index summary: {len(clips)} total clips")
    print(f"  - Keeping: {len(clips_to_keep)} reviewed clips")
    print(f"  - Removing: {len(clips_to_remove)} pending clips")
    print()

    if verbose:
        print("Clips to keep:")
        for clip_id in sorted(clips_to_keep):
            clip = clips[clip_id]
            tags = clip.get("tags", [])
            print(f"  {clip_id} [{', '.join(tags) or 'no tags'}]")
        print()

    stats.kept_clips = len(clips_to_keep)

    # Remove pending clips from analysis and events
    for clip_id in clips_to_remove:
        clip_data = clips[clip_id]
        clip_path = Path(clip_data.get("path", clip_id))

        # Analysis video file (e.g., runs/live/analysis/2025-12-14/segment_123456.mkv)
        analysis_file = share_root / clip_path
        if analysis_file.exists():
            size = get_clip_sizes(analysis_file)
            stats.bytes_freed += size
            if verbose:
                print(f"Remove analysis: {analysis_file} ({size / 1024 / 1024:.1f} MB)")
            if not dry_run:
                try:
                    analysis_file.unlink()
                    stats.removed_analysis += 1
                except Exception as e:
                    stats.errors.append(f"Failed to remove {analysis_file}: {e}")
            else:
                stats.removed_analysis += 1

        # Events directory (e.g., runs/live/events/2025-12-14/segment_123456/)
        date_str = clip_path.parent.name
        stem = clip_path.stem
        events_subdir = events_dir / date_str / stem
        if events_subdir.exists():
            size = get_clip_sizes(events_subdir)
            stats.bytes_freed += size
            if verbose:
                print(f"Remove events: {events_subdir} ({size / 1024 / 1024:.1f} MB)")
            if not dry_run:
                try:
                    shutil.rmtree(events_subdir)
                    stats.removed_events += 1
                except Exception as e:
                    stats.errors.append(f"Failed to remove {events_subdir}: {e}")
            else:
                stats.removed_events += 1

    # Update the index to only contain kept clips
    if not dry_run:
        updated_index = {
            "version": index_data.get("version", 1),
            "updated_at": _now_iso(),
            "tag_vocabulary": index_data.get("tag_vocabulary", []),
            "clips": {
                clip_id: clips[clip_id]
                for clip_id in clips_to_keep
            },
        }

        # Backup old index
        backup_path = index_path.with_suffix(".json.bak")
        shutil.copy2(index_path, backup_path)
        print(f"Backed up index to: {backup_path}")

        index_path.write_text(json.dumps(updated_index, indent=2))
        print(f"Updated index with {len(clips_to_keep)} remaining clips")

    # Clean up empty date directories
    for date_dir in analysis_dir.iterdir():
        if date_dir.is_dir() and not any(date_dir.iterdir()):
            if verbose:
                print(f"Remove empty dir: {date_dir}")
            if not dry_run:
                try:
                    date_dir.rmdir()
                except Exception:
                    pass

    for date_dir in events_dir.iterdir():
        if date_dir.is_dir() and not any(date_dir.iterdir()):
            if verbose:
                print(f"Remove empty dir: {date_dir}")
            if not dry_run:
                try:
                    date_dir.rmdir()
                except Exception:
                    pass

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually removing files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output for each file",
    )
    args = parser.parse_args()

    share_root = resolve_share_root()
    print(f"Share root: {share_root}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE - files will be deleted!'}")
    print()

    stats = cleanup_pending_clips(share_root, dry_run=args.dry_run, verbose=args.verbose)

    print()
    print("=" * 60)
    print("CLEANUP SUMMARY")
    print("=" * 60)
    print(f"Clips kept (reviewed):     {stats.kept_clips}")
    print(f"Analysis files removed:    {stats.removed_analysis}")
    print(f"Event directories removed: {stats.removed_events}")
    print(f"Space freed:               {stats.bytes_freed / 1024 / 1024 / 1024:.2f} GB")

    if stats.errors:
        print()
        print("ERRORS:")
        for err in stats.errors:
            print(f"  {err}")

    if args.dry_run:
        print()
        print("This was a dry run. Run without --dry-run to actually delete files.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
