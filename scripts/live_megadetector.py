#!/usr/bin/env python3
"""Monitor recorded segments, run MegaDetector, and archive interesting clips."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

def _ensure_repo_root_on_path() -> Path:
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


_ensure_repo_root_on_path()

from env_loader import load_env_file


DEFAULT_SEGMENTS_DIR = Path("runs/live/analysis")
DEFAULT_DETECTIONS_DIR = Path("runs/live/detections")
DEFAULT_EVENTS_DIR = Path("runs/live/events")
DEFAULT_MODEL_PATH = Path("models/md_v5a.0.0.pt")
DEFAULT_MEGADETECTOR_SCRIPT = Path("tmp/MegaDetector/detection/process_video.py")
DEFAULT_LOG_PATH = Path("logs/live_megadetector.log")
DEFAULT_EVENTS_LOG = Path("runs/live/events.log")
PYTHONPATH_APPEND = os.pathsep.join(["tmp/MegaDetector", "tmp/ai4eutils", "tmp/yolov5"])
INTERESTING_CATEGORIES = {"1", "2"}  # 1=animal, 2=person


def utc_now() -> datetime:
    return datetime.now(UTC)


def format_utc(ts: datetime) -> str:
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MegaDetector on new segments and retain any person/animal clips."
    )
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--detections-dir", type=Path, default=DEFAULT_DETECTIONS_DIR)
    parser.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR)
    parser.add_argument("--events-log", type=Path, default=DEFAULT_EVENTS_LOG)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--megadetector-script",
        type=Path,
        default=DEFAULT_MEGADETECTOR_SCRIPT,
        help="Path to tmp/MegaDetector/detection/process_video.py",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python interpreter to run MegaDetector with (default: current interpreter).",
    )
    parser.add_argument(
        "--frame-sample",
        type=int,
        default=15,
        help="Process every Nth frame (default: %(default)s).",
    )
    parser.add_argument(
        "--json-confidence",
        type=float,
        default=0.1,
        help="Minimum confidence stored in the MegaDetector JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=0.35,
        help="Confidence threshold to treat a detection as interesting (default: %(default)s).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds to wait when no new segments are available (default: %(default)s).",
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=20.0,
        help="Expected segment length in seconds (used to enforce a minimum file age).",
    )
    parser.add_argument(
        "--segment-age-slack",
        type=float,
        default=5.0,
        help="Extra seconds to wait beyond the segment length before processing.",
    )
    parser.add_argument(
        "--min-segment-age",
        type=float,
        default=0.0,
        help="Absolute minimum file age (overrides auto-calculated age when larger).",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=1,
        help="Number of CPU cores to expose to MegaDetector (default: %(default)s).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path for the detection log (default: %(default)s).",
    )
    return parser.parse_args()


SEGMENT_EXTENSIONS = (".mp4", ".mkv")


def list_segments(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    day_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    segments: List[Path] = []
    for day_dir in day_dirs:
        for ext in SEGMENT_EXTENSIONS:
            segments.extend(sorted(day_dir.glob(f"segment_*{ext}")))
    return segments


def detection_output_path(segment_path: Path, segments_root: Path, detections_root: Path) -> Path:
    relative = segment_path.relative_to(segments_root)
    date_dir = relative.parent
    stem = segment_path.stem
    target_dir = detections_root / date_dir / stem
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / "detections.json"


def event_directory(segment_path: Path, segments_root: Path, events_root: Path) -> Path:
    relative = segment_path.relative_to(segments_root)
    return events_root / relative.parent / segment_path.stem


def segment_ready(segment_path: Path, min_age_seconds: float) -> bool:
    age = time.time() - segment_path.stat().st_mtime
    return age >= min_age_seconds


def run_megadetector(
    args: argparse.Namespace, segment_path: Path, output_json: Path
) -> bool:
    cmd: List[str] = [
        args.python_executable,
        str(args.megadetector_script),
        str(args.model_path),
        str(segment_path),
        "--output_json_file",
        str(output_json),
        "--json_confidence_threshold",
        str(args.json_confidence),
        "--n_cores",
        str(args.n_cores),
    ]
    if args.frame_sample:
        cmd.extend(["--frame_sample", str(args.frame_sample)])

    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        PYTHONPATH_APPEND if not existing_path else f"{PYTHONPATH_APPEND}{os.pathsep}{existing_path}"
    )

    logging.info("Running MegaDetector on %s", segment_path)
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        logging.error("MegaDetector failed for %s (exit %s)", segment_path, result.returncode)
        return False
    return True


def load_detection_json(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_detections(
    payload: Dict, interesting_categories: Sequence[str], threshold: float
) -> Dict[str, object]:
    interesting = []
    max_conf = 0.0
    counts: Dict[str, int] = {}
    category_lookup = payload.get("detection_categories", {})

    for image in payload.get("images", []):
        frame_id = image.get("file")
        for det in image.get("detections", []):
            cat = str(det.get("category"))
            conf = float(det.get("conf", 0.0))
            if conf > max_conf:
                max_conf = conf
            if cat not in interesting_categories or conf < threshold:
                continue
            label = category_lookup.get(cat, cat)
            counts[label] = counts.get(label, 0) + 1
            interesting.append(
                {
                    "frame": frame_id,
                    "category": label,
                    "confidence": conf,
                    "bbox": det.get("bbox"),
                }
            )

    return {
        "hits": interesting,
        "counts": counts,
        "max_confidence": max_conf,
        "triggered": bool(interesting),
    }


def write_event_log(
    log_path: Path, event_time: datetime, segment_relative: str, counts: Dict[str, int], max_conf: float
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_counts = "|".join(f"{label}={count}" for label, count in counts.items()) or "none"
    line = f"{format_utc(event_time)},{segment_relative},{serialized_counts},{max_conf:.3f}\n"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def promote_event(
    segment_path: Path,
    segment_relative: Path,
    detection_json: Path,
    meta: Dict[str, object],
    events_root: Path,
) -> None:
    target_dir = events_root / segment_relative.parent / segment_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    segment_target = target_dir / segment_path.name
    detection_target = target_dir / "detections.json"
    shutil.copy2(segment_path, segment_target)
    shutil.copy2(detection_json, detection_target)
    meta_path = target_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("Promoted %s to %s", segment_relative, target_dir)


def relative_segment_path(segment_path: Path, segments_root: Path) -> Path:
    return segment_path.relative_to(segments_root)


def process_segment(args: argparse.Namespace, segment_path: Path, output_json: Path) -> None:
    if output_json.exists():
        return
    if not run_megadetector(args, segment_path, output_json):
        return
    data = load_detection_json(output_json)
    summary = summarize_detections(data, INTERESTING_CATEGORIES, args.event_threshold)
    segment_relative = relative_segment_path(segment_path, args.segments_dir)

    if summary["triggered"]:
        now = utc_now()
        meta = {
            "segment": str(segment_relative),
            "created_at": format_utc(now),
            "counts": summary["counts"],
            "max_confidence": summary["max_confidence"],
            "hits": summary["hits"][:20],
        }
        promote_event(segment_path, segment_relative, output_json, meta, args.events_dir)
        write_event_log(
            args.events_log,
            now,
            str(segment_relative),
            summary["counts"],
            summary["max_confidence"],
        )
    else:
        logging.info("No events in %s (max conf %.3f)", segment_relative, summary["max_confidence"])


def main() -> int:
    load_env_file()
    args = parse_args()
    configure_logging(args.log_file)

    args.segments_dir.mkdir(parents=True, exist_ok=True)
    args.detections_dir.mkdir(parents=True, exist_ok=True)
    args.events_dir.mkdir(parents=True, exist_ok=True)

    effective_min_age = max(args.segment_length + args.segment_age_slack, args.min_segment_age)
    logging.info(
        "Watching %s for new segments (min file age %.1fs)",
        args.segments_dir,
        effective_min_age,
    )

    try:
        while True:
            work_done = False
            for segment_path in list_segments(args.segments_dir):
                if not segment_ready(segment_path, effective_min_age):
                    continue
                output_json = detection_output_path(
                    segment_path, args.segments_dir, args.detections_dir
                )
                if output_json.exists():
                    continue
                process_segment(args, segment_path, output_json)
                work_done = True
            if not work_done:
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        logging.info("Stopping on keyboard interrupt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
