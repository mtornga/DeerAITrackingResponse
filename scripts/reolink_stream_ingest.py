#!/usr/bin/env python3
"""Continuously pull the Reolink RTSP feed into timestamped segments."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional


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

from env_loader import load_env_file, require_env


DEFAULT_OUTPUT_ROOT = Path("runs/live/segments")
DEFAULT_LOG_PATH = Path("logs/reolink_stream_ingest.log")


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def unique_segment_path(root: Path, suffix: str) -> Path:
    now = datetime.now(UTC)
    date_dir = root / now.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    base = f"segment_{now.strftime('%H%M%S')}"
    candidate = date_dir / f"{base}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = date_dir / f"{base}_{counter:02d}{suffix}"
        counter += 1
    return candidate


def prune_segments(root: Path, retention_hours: float) -> None:
    if retention_hours <= 0:
        return
    cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
    for day_dir in sorted(root.glob("*/")):
        if not day_dir.is_dir():
            continue
        for video_path in day_dir.glob("segment_*"):
            if video_path.suffix.lower() not in {".mp4", ".mkv"}:
                continue
            mtime = datetime.fromtimestamp(video_path.stat().st_mtime, UTC)
            if mtime < cutoff:
                try:
                    video_path.unlink()
                    logging.info("Pruned %s", video_path)
                except OSError as exc:
                    logging.warning("Unable to delete %s: %s", video_path, exc)
        try:
            if not any(day_dir.iterdir()):
                day_dir.rmdir()
        except OSError:
            continue


def build_ffmpeg_command(
    stream_url: str,
    duration: int,
    codec: str,
    bitrate: str,
    container: str,
    destination: Path,
) -> list[str]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-rtsp_transport",
        "tcp",
        "-fflags",
        "+genpts+discardcorrupt",
        "-avoid_negative_ts",
        "make_zero",
        "-use_wallclock_as_timestamps",
        "1",
        "-flags",
        "low_delay",
        "-max_delay",
        "5000000",
        "-i",
        stream_url,
        "-t",
        str(duration),
    ]

    if codec == "copy":
        cmd.extend(["-c:v", "copy"])
    else:
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-b:v",
                bitrate,
            ]
        )

    cmd.extend(["-an"])

    if container == "mp4":
        cmd.extend(["-movflags", "+faststart"])
    else:
        cmd.extend(["-f", "matroska"])

    cmd.extend(["-y", str(destination)])
    return cmd


def capture_segment(
    stream_url: str,
    duration: int,
    destination: Path,
    codec: str,
    bitrate: str,
    container: str,
) -> int:
    cmd = build_ffmpeg_command(stream_url, duration, codec, bitrate, container, destination)
    logging.info("Capturing segment %s (codec=%s)", destination, codec)
    result = subprocess.run(cmd, check=False)
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously cut the Reolink RTSP stream into segments.")
    parser.add_argument("--stream-url", help="RTSP URL to record (defaults to REOLINK_3_RTSP in .env).")
    parser.add_argument("--segment-length", type=int, default=20, help="Clip length in seconds.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Destination for segments.")
    parser.add_argument("--retention-hours", type=float, default=6.0, help="Hours of footage to retain.")
    parser.add_argument("--warmup-seconds", type=int, default=2, help="Delay before each ffmpeg run.")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH, help="Path for the ingest log.")
    parser.add_argument(
        "--video-codec",
        choices=["copy", "libx264"],
        default="libx264",
        help="Codec used for segments. 'copy' keeps the source bitstream.",
    )
    parser.add_argument(
        "--video-bitrate",
        type=str,
        default="4M",
        help="Bitrate when re-encoding with libx264 (ignored when codec=copy).",
    )
    parser.add_argument(
        "--container",
        choices=["mp4", "mkv"],
        default="mkv",
        help="Container/extension for captured clips (MKV tolerates dropouts better).",
    )
    return parser.parse_args()


def main() -> int:
    load_env_file()
    args = parse_args()
    configure_logging(args.log_file)

    stream_url = args.stream_url or require_env("REOLINK_3_RTSP")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{args.container}"

    logging.info("Starting ingest loop -> %s", output_dir)

    stop_requested = False

    def _handle_signal(signum: int, _frame: Optional[object]) -> None:
        nonlocal stop_requested
        logging.info("Received signal %s, exiting after current segment", signum)
        stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    while not stop_requested:
        if args.warmup_seconds:
            time.sleep(args.warmup_seconds)

        destination = unique_segment_path(output_dir, suffix)
        rc = capture_segment(
            stream_url,
            args.segment_length,
            destination,
            args.video_codec,
            args.video_bitrate,
            args.container,
        )

        if rc != 0:
            logging.error("ffmpeg exited with status %s", rc)
            if destination.exists():
                destination.unlink(missing_ok=True)
            time.sleep(5)
        else:
            prune_segments(output_dir, args.retention_hours)

    logging.info("Ingest loop stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
