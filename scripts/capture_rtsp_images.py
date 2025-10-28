#!/usr/bin/env python3
"""
Capture single-frame snapshots from an RTSP stream into class-specific folders.

Each run collects one JPEG per class defined in CLASS_OUTPUTS and writes it into
the corresponding `datasets/raw/<class>/images` directory.
"""

import argparse
import datetime
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

from env_loader import load_env_file, require_env  # noqa: E402

load_env_file()
DEFAULT_STREAM = os.getenv("WYZE_TABLETOP_RTSP")

# Map class names to their image output directories.
CLASS_OUTPUTS = {
    "Horse": Path("TableTopSimulation/datasets/raw/Horse/images"),
    "Alien_Maggie": Path("TableTopSimulation/datasets/raw/Alien_Maggie/images"),
}


def capture_frame(stream_url: str, output_dir: Path, overwrite: bool) -> Path:
    """Capture a single frame from the RTSP stream and save it into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = output_dir / f"{timestamp}.jpg"

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        stream_url,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-y" if overwrite else "-n",
        str(destination),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed for {output_dir}") from exc

    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture single frames from an RTSP stream into YOLO class folders."
    )
    parser.add_argument(
        "--stream-url",
        default=DEFAULT_STREAM,
        help="RTSP stream URL (default: value from WYZE_TABLETOP_RTSP)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting files when timestamps collide.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target paths without invoking ffmpeg.",
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    successes = []

    stream_url = args.stream_url or require_env("WYZE_TABLETOP_RTSP")

    for class_name, relative_path in CLASS_OUTPUTS.items():
        target_dir = root / relative_path
        destination = target_dir / f"{datetime.datetime.now():%Y%m%d_%H%M%S}.jpg"

        if args.dry_run:
            print(f"[DRY RUN] {class_name}: {destination}")
            successes.append(destination)
            continue

        try:
            saved_path = capture_frame(stream_url, target_dir, args.overwrite)
            print(f"{class_name}: saved {saved_path}")
            successes.append(saved_path)
        except RuntimeError as err:
            print(err, file=sys.stderr)
            return 1

    return 0 if successes else 1


if __name__ == "__main__":
    sys.exit(main())
