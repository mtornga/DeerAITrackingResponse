#!/usr/bin/env python3
"""
Normalize golden_clips layout to:
golden_clips/<day|night>/<category>/<clip_id>/

Updates golden_clips/manifest.json entries in place.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _ensure_repo_root_on_path() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / ".env").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    fallback = script_path.parents[1]
    if str(fallback) not in sys.path:
        sys.path.insert(0, str(fallback))
    return fallback


REPO_ROOT = _ensure_repo_root_on_path()

if str(REPO_ROOT / "agents") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agents"))

from training_loop_utils import load_json, resolve_share_root


ALLOWED_CATEGORIES = {"deer", "person", "other_wildlife", "false_positive", "unknown"}


@dataclass
class MoveRecord:
    clip_id: str
    source: str
    dest: str
    status: str
    reason: Optional[str] = None


@dataclass
class RunLog:
    run_id: str
    started_at: str
    finished_at: str
    dry_run: bool
    moves: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def format_utc(ts: datetime) -> str:
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_category(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    value = raw.strip().lower()
    if value in {"buck", "doe", "deer", "animal"}:
        return "deer"
    if value in {"person", "people", "human"}:
        return "person"
    if value in {"other_wildlife", "other-wildlife", "wildlife"}:
        return "other_wildlife"
    if value in {"false_positive", "false-positive", "falsepositive", "false"}:
        return "false_positive"
    if value in {"unknown", "misc", "other"}:
        return "unknown"
    return value if value in ALLOWED_CATEGORIES else None


def normalize_lighting(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    value = raw.strip().lower()
    if value in {"day", "daytime", "sun", "sunny"}:
        return "day"
    if value in {"night", "nighttime", "ir", "infrared", "dark"}:
        return "night"
    return None


def infer_lighting_from_path(path: Path) -> Optional[str]:
    parts = [part.lower() for part in path.parts]
    if "night" in parts or "ir" in parts:
        return "night"
    if "day" in parts or "daytime" in parts:
        return "day"
    return None


def infer_category_from_path(path: Path) -> Optional[str]:
    parts = [part.lower() for part in path.parts]
    for part in parts:
        if part in {"buck", "doe", "deer", "animal"}:
            return "deer"
        if part in {"person", "people", "human"}:
            return "person"
        if part in {"other_wildlife", "wildlife"}:
            return "other_wildlife"
        if part in {"false_positive", "false"}:
            return "false_positive"
        if part in {"unknown", "misc"}:
            return "unknown"
    return None


def resolve_clip_dir(entry: Dict[str, Any], golden_root: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for key in ("dest_dir", "source_path", "path"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = golden_root / candidate
            candidates.append(candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def unique_destination(dest: Path) -> Path:
    if not dest.exists():
        return dest
    idx = 1
    while True:
        candidate = dest.parent / f"{dest.name}__dup{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share-root", help="Override share root (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true", help="Plan moves without writing")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    share_root = resolve_share_root(args.share_root)
    golden_root = share_root / "golden_clips"
    if not golden_root.exists():
        raise SystemExit(f"Golden clips directory not found: {golden_root}")

    manifest_path = golden_root / "manifest.json"
    manifest = load_json(manifest_path) or {}
    clips = manifest.get("clips", [])
    if not isinstance(clips, list):
        raise SystemExit("manifest.json missing clips list")

    run_id = datetime.now(UTC).strftime("REORG-%Y%m%d-%H%M%S")
    started_at = datetime.now(UTC)
    log = RunLog(run_id=run_id, started_at=format_utc(started_at), finished_at="", dry_run=args.dry_run)

    for entry in clips:
        if not isinstance(entry, dict):
            continue
        clip_id = entry.get("id") or entry.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id.strip():
            log.errors.append("Missing clip id in manifest entry")
            continue
        clip_id = clip_id.strip()

        clip_dir = resolve_clip_dir(entry, golden_root)
        if not clip_dir or not clip_dir.exists():
            log.errors.append(f"Clip dir missing for {clip_id}")
            continue

        category = normalize_category(entry.get("category"))
        if not category:
            category = infer_category_from_path(clip_dir)
        if not category:
            log.errors.append(f"Unable to infer category for {clip_id}")
            continue

        lighting = normalize_lighting(entry.get("lighting"))
        if not lighting:
            lighting = infer_lighting_from_path(clip_dir)
        if not lighting:
            log.errors.append(f"Unable to infer lighting for {clip_id}")
            continue

        dest_dir = golden_root / lighting / category / clip_id
        dest_dir = unique_destination(dest_dir)

        if clip_dir.resolve() != dest_dir.resolve():
            if args.dry_run:
                log.moves.append(
                    MoveRecord(
                        clip_id=clip_id,
                        source=str(clip_dir),
                        dest=str(dest_dir),
                        status="planned",
                    ).__dict__
                )
            else:
                dest_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(clip_dir), str(dest_dir))
                log.moves.append(
                    MoveRecord(
                        clip_id=clip_id,
                        source=str(clip_dir),
                        dest=str(dest_dir),
                        status="moved",
                    ).__dict__
                )
        else:
            log.moves.append(
                MoveRecord(
                    clip_id=clip_id,
                    source=str(clip_dir),
                    dest=str(dest_dir),
                    status="unchanged",
                ).__dict__
            )

        entry["category"] = category
        entry["lighting"] = lighting
        try:
            entry["path"] = str(dest_dir.relative_to(golden_root))
        except ValueError:
            entry["path"] = str(dest_dir)
        entry["dest_dir"] = str(dest_dir)
        entry["layout_version"] = 2

    if not args.dry_run:
        manifest["layout_version"] = 2
        manifest["updated"] = format_utc(datetime.now(UTC))
        manifest_path.write_text(json.dumps(manifest, indent=2))

    finished_at = datetime.now(UTC)
    log.finished_at = format_utc(finished_at)
    logs_dir = REPO_ROOT / "runs" / "logs" / "training_loop"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"reorg-{run_id}.json"
    if not args.dry_run:
        log_path.write_text(json.dumps(log.__dict__, indent=2))
    else:
        log_path.write_text(json.dumps(log.__dict__, indent=2))

    print(f"[normalize] Finished {run_id} (dry_run={args.dry_run})")
    print(f"[normalize] Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
