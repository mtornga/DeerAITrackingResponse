#!/usr/bin/env python3
"""Promote valuable clips to the protected golden_clips archive.

Copies clips from the live pipeline to /srv/deer-share/golden_clips/ where
they are protected from automatic pruning.

Usage:
    # Promote a single segment
    python scripts/promote_to_golden.py \
        --source runs/live/analysis/2025-12-17/segment_232808.mkv \
        --category buck \
        --notes "Great buck footage"

    # Promote a range of segments (before/after context)
    python scripts/promote_to_golden.py \
        --source runs/live/analysis/2025-12-17/segment_232808.mkv \
        --include-adjacent 1 \
        --category buck \
        --notes "Buck with context clips"

    # Dry run to preview
    python scripts/promote_to_golden.py --source ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

UTC = timezone.utc


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


def parse_segment_time(filename: str) -> Optional[str]:
    """Extract HHMMSS from segment_HHMMSS.mkv filename."""
    match = re.search(r"segment_(\d{6})", filename)
    return match.group(1) if match else None


def find_adjacent_segments(segment_path: Path, count: int) -> List[Path]:
    """Find segments before and after the given segment."""
    parent = segment_path.parent
    if not parent.exists():
        return [segment_path]

    all_segments = sorted(parent.glob("segment_*.mkv"))
    if segment_path not in all_segments:
        # Try with .mp4
        all_segments = sorted(parent.glob("segment_*.mp4"))

    if segment_path not in all_segments:
        return [segment_path]

    idx = all_segments.index(segment_path)
    start = max(0, idx - count)
    end = min(len(all_segments), idx + count + 1)

    return all_segments[start:end]


def load_manifest(golden_root: Path) -> dict:
    """Load the golden clips manifest."""
    manifest_path = golden_root / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {
        "version": 1,
        "description": "Protected archive of high-value wildlife clips",
        "created": datetime.now(UTC).strftime("%Y-%m-%d"),
        "clips": [],
    }


def save_manifest(golden_root: Path, manifest: dict) -> None:
    """Save the golden clips manifest."""
    manifest_path = golden_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def promote_clips(
    source_path: Path,
    category: str,
    notes: str,
    tags: List[str],
    include_adjacent: int,
    share_root: Path,
    dry_run: bool,
) -> None:
    """Promote clips to the golden archive."""
    golden_root = share_root / "golden_clips"

    if not golden_root.exists():
        raise SystemExit(f"Golden clips directory not found: {golden_root}")

    # Resolve source path
    if not source_path.is_absolute():
        source_path = share_root / source_path

    if not source_path.exists():
        raise SystemExit(f"Source file not found: {source_path}")

    # Find all segments to copy
    segments = find_adjacent_segments(source_path, include_adjacent)
    print(f"Segments to promote ({len(segments)}):")
    for seg in segments:
        size_mb = seg.stat().st_size / 1024 / 1024
        print(f"  {seg.name} ({size_mb:.1f} MB)")

    # Generate clip ID and destination
    date_str = source_path.parent.name  # e.g., 2025-12-17
    time_str = parse_segment_time(source_path.name) or "000000"
    clip_id = f"{date_str}_{time_str}_{category}"

    dest_dir = golden_root / category / clip_id
    print(f"\nDestination: {dest_dir}")

    if dest_dir.exists():
        raise SystemExit(f"Destination already exists: {dest_dir}")

    total_size = sum(seg.stat().st_size for seg in segments)
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")

    if dry_run:
        print("\nDRY RUN - no files copied")
        return

    # Create destination and copy files
    dest_dir.mkdir(parents=True, exist_ok=True)

    for seg in segments:
        dest_file = dest_dir / seg.name
        print(f"Copying {seg.name}...")
        shutil.copy2(seg, dest_file)

    # Update manifest
    manifest = load_manifest(golden_root)

    entry = {
        "id": clip_id,
        "category": category,
        "path": f"{category}/{clip_id}",
        "segments": [seg.name for seg in segments],
        "date": date_str,
        "time_utc": f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}",
        "duration_seconds": len(segments) * 60,  # Approximate
        "notes": notes,
        "added_by": os.environ.get("USER", "unknown"),
        "added_date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "source": str(source_path.parent.relative_to(share_root)),
        "tags": tags,
    }

    manifest["clips"].append(entry)
    save_manifest(golden_root, manifest)

    print(f"\nPromoted {len(segments)} clips to golden archive")
    print(f"Clip ID: {clip_id}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the primary segment to promote (relative to share root or absolute)",
    )
    parser.add_argument(
        "--category",
        choices=["buck", "doe", "person", "mixed"],
        required=True,
        help="Category for organizing the clip",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Description of the clip content",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Additional tags (e.g., daytime night high_quality)",
    )
    parser.add_argument(
        "--include-adjacent",
        type=int,
        default=0,
        help="Include N segments before and after the source (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be copied without actually copying",
    )

    args = parser.parse_args()
    share_root = resolve_share_root()

    print(f"Share root: {share_root}")
    print()

    promote_clips(
        source_path=args.source,
        category=args.category,
        notes=args.notes,
        tags=args.tags,
        include_adjacent=args.include_adjacent,
        share_root=share_root,
        dry_run=args.dry_run,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
