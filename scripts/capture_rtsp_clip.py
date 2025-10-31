#!/usr/bin/env python3
"""
Capture a fixed-duration clip from the RTSP stream for manual labeling.

The resulting MP4 is stored in the CVAT_inputs directory so it can be uploaded
directly to CVAT.
"""

import argparse
import datetime
import os
import subprocess
import sys
import time
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
DEFAULT_DURATION = 15  # seconds
DEFAULT_OUTPUT = Path("/Users/marktornga/Movies/CVAT_clips")


def capture_clip(stream_url: str, duration: int, output_dir: Path, overwrite: bool) -> Path:
    """Record a clip of `duration` seconds and save it in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = output_dir / f"{timestamp}.mp4"

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        stream_url,
        "-t",
        str(duration),
        "-c:v",
        "copy",
        "-an",
        "-movflags",
        "+faststart",
        "-y" if overwrite else "-n",
        str(destination),
    ]

    time.sleep(2)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed creating clip at {destination}")

    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture a fixed-duration RTSP clip for CVAT labeling."
    )
    parser.add_argument(
        "--stream-url",
        default=DEFAULT_STREAM,
        help="RTSP stream URL (default: value from WYZE_TABLETOP_RTSP)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help="Clip length in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Destination directory for clips (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting when timestamps collide.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the intended target path without invoking ffmpeg.",
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    stream_url = args.stream_url or require_env("WYZE_TABLETOP_RTSP")

    if args.dry_run:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[DRY RUN] {output_dir / (timestamp + '.mp4')}")
        return 0

    try:
        clip_path = capture_clip(stream_url, args.duration, output_dir, args.overwrite)
    except RuntimeError as err:
        print(err, file=sys.stderr)
        return 1

    print(f"Saved clip to {clip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
