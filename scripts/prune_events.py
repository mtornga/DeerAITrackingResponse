#!/usr/bin/env python3
"""Prune old events from the pipeline to manage disk space.

This script implements tiered retention:
- Events marked "done" with notes or "keep" tag: kept indefinitely
- Events marked "done" without notes: 14 days
- Events still "pending": 48 hours
- Events with no index entry: 24 hours

Usage:
    python scripts/prune_events.py --events-dir /srv/deer-share/runs/live/events
    python scripts/prune_events.py --dry-run  # Preview what would be deleted
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Set

UTC = timezone.utc

# Retention policies (hours)
RETENTION_KEEP_FOREVER_TAGS = {"keep", "important", "training_candidate"}
RETENTION_REVIEWED_WITH_NOTES_HOURS = 14 * 24  # 14 days
RETENTION_REVIEWED_NO_NOTES_HOURS = 7 * 24     # 7 days
RETENTION_PENDING_HOURS = 48                    # 48 hours
RETENTION_UNINDEXED_HOURS = 24                  # 24 hours


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_review_index(share_root: Path) -> Dict[str, dict]:
    """Load the daily review index to check review status."""
    index_path = share_root / "index" / "daily_review_index.json"
    if not index_path.exists():
        return {}

    try:
        data = json.loads(index_path.read_text())
        return data.get("clips", {})
    except (json.JSONDecodeError, OSError) as e:
        logging.warning("Failed to load review index: %s", e)
        return {}


def get_retention_hours(clip_id: str, index: Dict[str, dict]) -> Optional[float]:
    """Determine retention hours for a clip based on its review status.

    Returns None if the clip should be kept forever.
    """
    entry = index.get(clip_id)

    if entry is None:
        # Not in index - use short retention
        return RETENTION_UNINDEXED_HOURS

    tags = set(entry.get("tags", []))
    notes = (entry.get("notes") or "").strip()
    status = entry.get("review_status", "pending")

    # Check for "keep forever" tags
    if tags & RETENTION_KEEP_FOREVER_TAGS:
        return None  # Keep forever

    if status == "done":
        if notes:
            return RETENTION_REVIEWED_WITH_NOTES_HOURS
        else:
            return RETENTION_REVIEWED_NO_NOTES_HOURS
    else:
        # pending or in_progress
        return RETENTION_PENDING_HOURS


def find_events(events_dir: Path) -> Dict[str, Path]:
    """Find all event directories and map clip_id to path."""
    events = {}

    # Safety check: never prune golden_clips
    if "golden_clips" in str(events_dir):
        logging.warning("Refusing to prune golden_clips directory")
        return {}

    for date_dir in events_dir.iterdir():
        if not date_dir.is_dir() or not date_dir.name.startswith("2"):
            continue

        for event_dir in date_dir.iterdir():
            if not event_dir.is_dir():
                continue

            # Construct clip_id to match index format
            # Index uses: runs/live/analysis/2025-12-17/segment_182217.mkv
            clip_id = f"runs/live/analysis/{date_dir.name}/{event_dir.name}.mkv"
            events[clip_id] = event_dir

    return events


def get_event_age_hours(event_dir: Path) -> float:
    """Get age of event in hours based on meta.json or directory mtime."""
    meta_path = event_dir / "meta.json"

    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            created_at = meta.get("created_at")
            if created_at:
                # Parse ISO format: 2025-12-17T18:49:21Z
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                age = datetime.now(UTC) - dt
                return age.total_seconds() / 3600
    except (json.JSONDecodeError, OSError, ValueError):
        pass

    # Fallback to directory mtime
    mtime = datetime.fromtimestamp(event_dir.stat().st_mtime, UTC)
    age = datetime.now(UTC) - mtime
    return age.total_seconds() / 3600


def prune_events(
    events_dir: Path,
    share_root: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Prune events based on retention policy.

    Returns (deleted_count, kept_count).
    """
    index = load_review_index(share_root)
    events = find_events(events_dir)

    deleted = 0
    kept = 0
    bytes_freed = 0

    for clip_id, event_dir in events.items():
        retention = get_retention_hours(clip_id, index)
        age_hours = get_event_age_hours(event_dir)

        if retention is None:
            # Keep forever
            logging.debug("KEEP (forever): %s", event_dir.name)
            kept += 1
            continue

        if age_hours < retention:
            # Not old enough to delete
            logging.debug(
                "KEEP (%.1fh < %.1fh): %s",
                age_hours, retention, event_dir.name
            )
            kept += 1
            continue

        # Should delete
        dir_size = sum(f.stat().st_size for f in event_dir.rglob("*") if f.is_file())

        if dry_run:
            logging.info(
                "WOULD DELETE (%.1fh > %.1fh, %.1fMB): %s",
                age_hours, retention, dir_size / 1e6, event_dir
            )
        else:
            try:
                shutil.rmtree(event_dir)
                logging.info(
                    "DELETED (%.1fh > %.1fh, %.1fMB): %s",
                    age_hours, retention, dir_size / 1e6, event_dir
                )
                deleted += 1
                bytes_freed += dir_size
            except OSError as e:
                logging.error("Failed to delete %s: %s", event_dir, e)
                kept += 1
                continue

    # Clean up empty date directories
    for date_dir in events_dir.iterdir():
        if date_dir.is_dir() and not any(date_dir.iterdir()):
            if dry_run:
                logging.info("WOULD REMOVE empty dir: %s", date_dir)
            else:
                try:
                    date_dir.rmdir()
                    logging.info("Removed empty directory: %s", date_dir)
                except OSError:
                    pass

    if not dry_run:
        logging.info(
            "Pruning complete: deleted %d events (%.1f MB), kept %d",
            deleted, bytes_freed / 1e6, kept
        )

    return deleted, kept


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune old events to manage disk space")
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=Path("/srv/deer-share/runs/live/events"),
        help="Events directory to prune",
    )
    parser.add_argument(
        "--share-root",
        type=Path,
        default=None,
        help="Share root (defaults to parent of events-dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without actually deleting",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    events_dir = args.events_dir.resolve()
    if not events_dir.exists():
        logging.error("Events directory does not exist: %s", events_dir)
        return 1

    # Infer share root if not provided
    share_root = args.share_root
    if share_root is None:
        # events_dir is typically /srv/deer-share/runs/live/events
        # share_root should be /srv/deer-share
        share_root = events_dir.parent.parent.parent

    if args.dry_run:
        logging.info("DRY RUN - no files will be deleted")

    logging.info("Pruning events in: %s", events_dir)
    logging.info("Using review index from: %s", share_root / "index")

    deleted, kept = prune_events(events_dir, share_root, args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
