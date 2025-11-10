#!/usr/bin/env python3
"""Run MegaDetector on a single video segment and store the output JSON."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


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


REPO_ROOT = _ensure_repo_root_on_path()

from env_loader import load_env_file  # noqa: E402

DEFAULT_SEGMENTS_ROOT = Path("runs/live/analysis")
DEFAULT_DETECTIONS_ROOT = Path("runs/live/detections")
DEFAULT_MODEL_PATH = Path("models/md_v5a.0.0.pt")
DEFAULT_MEGADETECTOR_SCRIPT = Path("tmp/MegaDetector/detection/process_video.py")
PYTHONPATH_APPEND = os.pathsep.join(["tmp/MegaDetector", "tmp/ai4eutils", "tmp/yolov5"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MegaDetector on a single segment file.")
    parser.add_argument("segment", type=Path, help="Path to the video segment to process.")
    parser.add_argument(
        "--segments-root",
        type=Path,
        default=DEFAULT_SEGMENTS_ROOT,
        help="Root directory containing mirrored segments (default: %(default)s).",
    )
    parser.add_argument(
        "--detections-root",
        type=Path,
        default=DEFAULT_DETECTIONS_ROOT,
        help="Root directory to store detection JSONs (default: %(default)s).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MegaDetector weights (default: %(default)s).",
    )
    parser.add_argument(
        "--megadetector-script",
        type=Path,
        default=DEFAULT_MEGADETECTOR_SCRIPT,
        help="Path to tmp/MegaDetector/detection/process_video.py (default: %(default)s).",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python interpreter to invoke MegaDetector (default: current interpreter).",
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
        default=0.2,
        help="Confidence floor to store in the JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Explicit output path. When omitted, mirrors the segments_root structure under detections_root.",
    )
    return parser.parse_args()


def compute_output_path(segment: Path, segments_root: Path, detections_root: Path) -> Path:
    try:
        relative = segment.resolve().relative_to(segments_root.resolve())
        date_dir = relative.parent
        stem = segment.stem
        return detections_root / date_dir / stem / "detections.json"
    except ValueError:
        stem = segment.stem
        return detections_root / f"{stem}.json"


def main() -> int:
    load_env_file()
    args = parse_args()

    segment_path = args.segment.resolve()
    if not segment_path.exists():
        print(f"Segment not found: {segment_path}", file=sys.stderr)
        return 1

    output_json = args.output_json
    if output_json is None:
        output_json = compute_output_path(segment_path, args.segments_root, args.detections_root)
    output_json = output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_executable,
        str(args.megadetector_script),
        str(args.model_path),
        str(segment_path),
        "--output_json_file",
        str(output_json),
        "--json_confidence_threshold",
        str(args.json_confidence),
    ]
    if args.frame_sample:
        cmd.extend(["--frame_sample", str(args.frame_sample)])

    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        PYTHONPATH_APPEND if not existing else f"{PYTHONPATH_APPEND}{os.pathsep}{existing}"
    )

    print(f"Running MegaDetector on {segment_path} -> {output_json}")
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        print(f"MegaDetector failed with exit code {result.returncode}", file=sys.stderr)
        return result.returncode

    print(f"Wrote detections to {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
