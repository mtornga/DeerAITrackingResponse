#!/usr/bin/env python3
"""
Classify golden clips as day or night (IR) and optionally move them into
per-lighting subdirectories.

Default layout after running (category-first for compatibility):
    golden_clips/<category>/<day|night>/<clip_id>/

Classification uses a frame-based heuristic (mean saturation/brightness) and
optionally blends a timestamp prior extracted from the clip name
(_HHMMSS). It updates manifest entries with lighting metadata when found.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

VIDEO_EXTS = {".mkv", ".mp4"}
UTC = timezone.utc


def resolve_share_root(explicit: Optional[str] = None) -> Path:
    """Locate shared storage root."""
    if explicit:
        return Path(explicit).expanduser()

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


def sample_frame(video_path: Path, frame_ratio: float = 0.3) -> Optional[np.ndarray]:
    """Grab a representative frame from the video with fallbacks if the target index fails."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # Try multiple ratios to avoid dead keyframe seeks (e.g., middle of some MKVs)
    ratios = []
    for r in (frame_ratio, 0.5, 0.25, 0.75, 0.1):
        if r not in ratios:
            ratios.append(r)

    frame: Optional[np.ndarray] = None
    for r in ratios:
        target_idx = int(frame_count * r)
        if target_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, candidate = cap.read()
        if ok:
            frame = candidate
            break

    cap.release()
    return frame


@dataclass
class LightingDecision:
    label: str  # "day" or "night"
    confidence: float
    day_score: float
    frame_metrics: Dict[str, float]
    prior_day: Optional[float]
    combined_day_score: float


def classify_frame(frame: np.ndarray) -> Tuple[str, float, float, Dict[str, float]]:
    """Heuristic day/night classification from a single frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
    mean_sat = float(mean_hsv[1])
    mean_val = float(mean_hsv[2])

    sat_score = float(np.clip((mean_sat - 15.0) / 70.0, 0.0, 1.0))
    val_score = float(np.clip((mean_val - 60.0) / 140.0, 0.0, 1.0))
    day_score = 0.6 * sat_score + 0.4 * val_score
    label = "day" if day_score >= 0.5 else "night"
    confidence = abs(day_score - 0.5) * 2.0

    metrics = {
        "mean_sat": mean_sat,
        "mean_val": mean_val,
        "sat_score": sat_score,
        "val_score": val_score,
    }
    return label, day_score, confidence, metrics


def timestamp_prior(
    name: str, tz_offset_hours: int, day_start: int, day_end: int
) -> Optional[float]:
    """Return a day prior (0.8 for day window, 0.2 for night) from _HHMMSS in the name."""
    match = re.search(r"_(\d{6})", name)
    if not match:
        return None
    hh = int(match.group(1)[:2])
    local_hour = (hh + tz_offset_hours) % 24
    return 0.8 if day_start <= local_hour < day_end else 0.2


def classify_clip(
    clip_dir: Path,
    tz_offset_hours: int,
    day_start: int,
    day_end: int,
    frame_ratio: float,
    prior_weight: float,
) -> LightingDecision:
    videos = [
        p for p in clip_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    if not videos:
        raise SystemExit(f"No video files found in {clip_dir}")

    frame = sample_frame(videos[0], frame_ratio=frame_ratio)
    if frame is None:
        # Default to night with zero confidence; caller can flag
        return LightingDecision(
            label="night",
            confidence=0.0,
            day_score=0.0,
            frame_metrics={},
            prior_day=None,
            combined_day_score=0.0,
        )

    frame_label, frame_day_score, frame_conf, metrics = classify_frame(frame)
    prior_day = timestamp_prior(clip_dir.name, tz_offset_hours, day_start, day_end)
    combined_day = (
        prior_weight * prior_day + (1 - prior_weight) * frame_day_score
        if prior_day is not None
        else frame_day_score
    )
    final_label = "day" if combined_day >= 0.5 else "night"
    combined_conf = abs(combined_day - 0.5) * 2.0

    return LightingDecision(
        label=final_label,
        confidence=combined_conf,
        day_score=frame_day_score,
        frame_metrics=metrics | {"frame_label": frame_label, "frame_confidence": frame_conf},
        prior_day=prior_day,
        combined_day_score=combined_day,
    )


def iter_clip_dirs(category_dir: Path, include_day_night: bool) -> Iterable[Path]:
    """Yield clip directories under a category, optionally including existing day/night splits."""
    if include_day_night:
        for lighting_dir in category_dir.iterdir():
            if lighting_dir.is_dir() and lighting_dir.name in {"day", "night"}:
                yield from (p for p in lighting_dir.iterdir() if p.is_dir())
        # Also consider any stray clips directly under category
        for p in category_dir.iterdir():
            if p.is_dir() and p.name not in {"day", "night"}:
                yield p
    else:
        for p in category_dir.iterdir():
            if p.is_dir() and p.name not in {"day", "night"}:
                yield p


def update_manifest_entry(
    manifest: dict,
    golden_root: Path,
    old_dir: Path,
    new_dir: Path,
    decision: LightingDecision,
) -> bool:
    """Update manifest entries that reference the old directory."""
    updated = False
    new_rel = str(new_dir.relative_to(golden_root))
    old_rel = str(old_dir.relative_to(golden_root))

    for entry in manifest.get("clips", []):
        dest_dir = entry.get("dest_dir")
        path = entry.get("path")
        source_path = entry.get("source_path")

        if dest_dir and Path(dest_dir) == old_dir:
            entry["dest_dir"] = str(new_dir)
            updated = True
        if path and old_rel in path:
            entry["path"] = new_rel
            updated = True
        if source_path and Path(source_path).parent == old_dir:
            entry["source_path"] = str(new_dir / Path(source_path).name)
            updated = True

        if updated:
            entry["lighting"] = decision.label
            entry["lighting_confidence"] = round(decision.confidence, 3)
            entry["lighting_details"] = {
                **decision.frame_metrics,
                "prior_day": decision.prior_day,
                "combined_day_score": round(decision.combined_day_score, 3),
            }
            break

    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--share-root",
        help="Override share root (defaults to env/auto-detect)",
    )
    parser.add_argument(
        "--tz-offset-hours",
        type=int,
        default=0,
        help="Offset to convert segment HHMMSS (assumed UTC) to local hour (e.g., -6 for CST).",
    )
    parser.add_argument(
        "--day-start",
        type=int,
        default=6,
        help="Local hour when day window starts (inclusive).",
    )
    parser.add_argument(
        "--day-end",
        type=int,
        default=20,
        help="Local hour when day window ends (exclusive).",
    )
    parser.add_argument(
        "--prior-weight",
        type=float,
        default=0.5,
        help="Weight of timestamp prior vs. frame heuristic (0-1).",
    )
    parser.add_argument(
        "--frame-ratio",
        type=float,
        default=0.5,
        help="Which portion of the clip to sample (0=start, 0.5=middle).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.15,
        help="Below this confidence, flag clip for manual review.",
    )
    parser.add_argument(
        "--reclassify-all",
        action="store_true",
        help="Re-evaluate lighting for all clips (including already classified day/night).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not move files; just report classifications.",
    )
    args = parser.parse_args()

    share_root = resolve_share_root(args.share_root)
    golden_root = share_root / "golden_clips"
    manifest_path = golden_root / "manifest.json"

    if not golden_root.exists():
        raise SystemExit(f"Golden clips directory not found: {golden_root}")

    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    classified: List[Tuple[str, Path, LightingDecision]] = []
    manual: List[str] = []

    categories = [
        p for p in golden_root.iterdir() if p.is_dir() and p.name not in {"lighting", ".git"}
    ]

    for category_dir in categories:
        for clip_dir in iter_clip_dirs(category_dir, include_day_night=args.reclassify_all):
            decision = classify_clip(
                clip_dir=clip_dir,
                tz_offset_hours=args.tz_offset_hours,
                day_start=args.day_start,
                day_end=args.day_end,
                frame_ratio=args.frame_ratio,
                prior_weight=args.prior_weight,
            )
            classified.append((category_dir.name, clip_dir, decision))
            if decision.confidence < args.min_confidence:
                manual.append(str(clip_dir))

    if not classified:
        print("No unclassified clips found.")
        return 0

    print(f"Classified {len(classified)} clips:")
    day_count = sum(1 for _, _, d in classified if d.label == "day")
    night_count = len(classified) - day_count
    print(f"  Day: {day_count}, Night: {night_count}")
    if manual:
        print(f"  Low-confidence ({len(manual)}):")
        for clip in manual:
            print(f"    {clip}")

    if args.dry_run:
        return 0

    manifest_backup = manifest_path.with_suffix(
        ".json.bak-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    )
    if manifest_path.exists():
        shutil.copy2(manifest_path, manifest_backup)
        print(f"Manifest backed up to {manifest_backup}")

    for category, clip_dir, decision in classified:
        dest_dir = golden_root / category / decision.label / clip_dir.name
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        if dest_dir.resolve() != clip_dir.resolve():
            print(
                f"Moving {clip_dir} -> {dest_dir} "
                f"({decision.label}, conf={decision.confidence:.2f})"
            )
            shutil.move(str(clip_dir), dest_dir)
        else:
            print(f"Updating metadata for {clip_dir} ({decision.label}, conf={decision.confidence:.2f})")
        update_manifest_entry(manifest, golden_root, clip_dir, dest_dir, decision)

    manifest["updated"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("Manifest updated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
