#!/usr/bin/env python3
"""Daily review CLI for Deer Vision.

This script is the first skeleton of the "morning review" workflow:
it discovers new/unreviewed clips on the shared storage, maintains a
lightweight index, and prints a prioritized list for the human reviewer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple


UTC = timezone.utc

SEGMENT_EXTENSIONS: Tuple[str, ...] = (".mp4", ".mkv")
ReviewStatus = Literal["pending", "in_progress", "done"]


@dataclass
class ClipEntry:
    """Metadata tracked for each clip in the daily review index."""

    clip_id: str
    path: str
    first_seen: str
    capture_mtime: float
    review_status: ReviewStatus = "pending"
    detector_model: Optional[str] = None
    max_conf: Optional[float] = None
    tags: List[str] = None
    notes: str = ""

    def to_dict(self) -> Dict:
        payload = asdict(self)
        if payload["tags"] is None:
            payload["tags"] = []
        return payload


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_share_root(explicit: Optional[Path] = None) -> Path:
    """Resolve the Samba/shared storage root.

    Preference order:
    - explicit flag
    - DEER_SHARE_SERVER_PATH
    - DEER_SHARE_LOCAL_MOUNT
    - /srv/deer-share
    - ~/DeerShare
    """

    if explicit is not None:
        return explicit

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


def discover_segments(segments_root: Path) -> Iterable[Path]:
    """Yield all segment video files under the given root."""

    if not segments_root.exists():
        return []

    for path in segments_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in SEGMENT_EXTENSIONS:
            yield path


def load_index(index_path: Path) -> Dict[str, ClipEntry]:
    """Load the daily review index from disk, if present."""

    if not index_path.exists():
        return {}

    try:
        raw = json.loads(index_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse daily review index {index_path}: {exc}") from exc

    clips: Dict[str, ClipEntry] = {}
    for clip_id, data in raw.get("clips", {}).items():
        entry = ClipEntry(
            clip_id=clip_id,
            path=data.get("path", clip_id),
            first_seen=data.get("first_seen", _now_iso()),
            capture_mtime=data.get("capture_mtime", 0.0),
            review_status=data.get("review_status", "pending"),
            detector_model=data.get("detector_model"),
            max_conf=data.get("max_conf"),
            tags=data.get("tags") or [],
            notes=data.get("notes", ""),
        )
        clips[clip_id] = entry
    return clips


def save_index(index_path: Path, clips: Dict[str, ClipEntry]) -> None:
    payload = {
        "version": 1,
        "updated_at": _now_iso(),
        "clips": {clip_id: entry.to_dict() for clip_id, entry in clips.items()},
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, indent=2))


def build_or_update_index(
    share_root: Path,
    segments_root: Path,
    index_path: Path,
) -> Dict[str, ClipEntry]:
    """Populate the index with any new segments, preserving existing entries."""

    existing = load_index(index_path)

    for segment_path in discover_segments(segments_root):
        try:
            rel = segment_path.relative_to(share_root)
        except ValueError:
            # If the segment is not under the share root, fall back to segments_root-relative.
            rel = segment_path.relative_to(segments_root)

        clip_id = str(rel)
        if clip_id in existing:
            # Keep existing metadata; update capture_mtime if it changed.
            entry = existing[clip_id]
            mtime = segment_path.stat().st_mtime
            if mtime != entry.capture_mtime:
                entry.capture_mtime = mtime
            continue

        stat = segment_path.stat()
        entry = ClipEntry(
            clip_id=clip_id,
            path=str(rel),
            first_seen=_now_iso(),
            capture_mtime=stat.st_mtime,
            review_status="pending",
            tags=[],
        )
        existing[clip_id] = entry

    return existing


def format_clip_row(entry: ClipEntry) -> str:
    ts = datetime.fromtimestamp(entry.capture_mtime, UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    max_conf_str = "-" if entry.max_conf is None else f"{entry.max_conf:.2f}"
    tags_str = ",".join(entry.tags) if entry.tags else "-"
    return f"{entry.review_status:11s} {max_conf_str:6s} {ts:20s} {tags_str:20s} {entry.path}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="List Deer Vision clips that need human review.",
    )
    parser.add_argument(
        "--share-root",
        type=Path,
        help="Override shared storage root (default: detect via DEER_SHARE_* or /srv/deer-share).",
    )
    parser.add_argument(
        "--status",
        choices=["pending", "in_progress", "done"],
        default="pending",
        help="Filter by review_status (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of clips to display (default: %(default)s).",
    )
    parser.add_argument(
        "--segments-subdir",
        default="runs/live/analysis",
        help="Path under the share root where segments are stored (default: %(default)s).",
    )

    args = parser.parse_args(argv)

    share_root = resolve_share_root(args.share_root)
    segments_root = share_root / args.segments_subdir
    index_path = share_root / "index" / "daily_review_index.json"

    if not segments_root.exists():
        print(f"No segment directory found at {segments_root}", file=sys.stderr)
        return 1

    clips = build_or_update_index(share_root, segments_root, index_path)
    save_index(index_path, clips)

    # Filter and sort for display.
    filtered = [
        entry for entry in clips.values() if entry.review_status == args.status  # type: ignore[comparison-overlap]
    ]
    filtered.sort(key=lambda e: e.capture_mtime, reverse=True)

    print(
        f"Daily review index: {len(clips)} clip(s) tracked "
        f"({len(filtered)} with status={args.status})"
    )
    print("status       max   capture_time          tags                 path")
    print("-" * 80)

    for entry in filtered[: args.limit]:
        print(format_clip_row(entry))

    if not filtered:
        print(f"\nNo clips with status={args.status!r} found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

