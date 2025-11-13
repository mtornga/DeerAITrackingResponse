#!/usr/bin/env python3
"""Package clips into the CVAT share area for easy task creation."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

try:
    from env_loader import load_env_file
except ModuleNotFoundError:  # running outside repo root fallback
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from env_loader import load_env_file  # type: ignore


DEFAULT_SOURCE = Path("runs/live/analysis")


def default_share_path() -> Path:
    share_root = os.environ.get("CVAT_SHARE_ROOT")
    return Path(share_root) if share_root else Path("cvat-share")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zip selected clips into the CVAT share folder.")
    parser.add_argument(
        "clips",
        nargs="+",
        help="Clip identifiers to package (e.g. 2025-11-11/segment_190458).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Root directory containing mirrored clips (default: %(default)s).",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=None,
        help="Destination (the mounted CVAT share). Defaults to $CVAT_SHARE_ROOT or ./cvat-share.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("tmp/megadetector_frames"),
        help="Root for extracted frames (default: %(default)s/<clip>/).",
    )
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="Also copy the raw MP4 alongside the frame zip.",
    )
    parser.add_argument(
        "--reextract",
        action="store_true",
        help="Force re-extraction of frames even if the folder already exists.",
    )
    return parser.parse_args()


def extract_frames(clip_path: Path, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(clip_path),
        "-vf",
        "scale=1280:-2",
        str(output_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)


def zip_frames(frames_path: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    shutil.make_archive(archive_path.with_suffix(""), "zip", frames_path)


def resolve_clip_path(clip: str, source_dir: Path) -> Path:
    candidate = Path(clip)

    def variants(path: Path) -> List[Path]:
        opts = [path]
        if path.suffix == "":
            opts.append(path.with_suffix(".mp4"))
            opts.append(path.with_suffix(".mkv"))
        return opts

    probes: List[Path] = []
    if candidate.is_absolute():
        probes.extend(variants(candidate))
    else:
        for var in variants(candidate):
            probes.append(source_dir / var)
            probes.append(var)
    for probe in probes:
        if probe.exists():
            return probe.resolve()
    raise FileNotFoundError(f"Clip not found: {candidate}")


def process_clip(
    clip: str,
    source_dir: Path,
    dest_dir: Path,
    frames_root: Path,
    include_video: bool,
    reextract: bool,
) -> None:
    clip_path = resolve_clip_path(clip, source_dir)

    date_dir = clip_path.parent.name
    frames_path = frames_root / date_dir / clip_path.stem
    if frames_path.exists() and reextract:
        shutil.rmtree(frames_path)

    if not frames_path.exists():
        extract_frames(clip_path, frames_path)

    dest_subdir = dest_dir / date_dir
    dest_subdir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_subdir / f"{clip_path.stem}_frames.zip"
    zip_frames(frames_path, archive_path)

    if include_video:
        shutil.copy2(clip_path, dest_subdir / clip_path.name)

    print(f"Packaged {clip} -> {archive_path}")


def main() -> int:
    load_env_file()
    args = parse_args()
    if args.dest_dir is None:
        args.dest_dir = default_share_path()
    for clip in args.clips:
        process_clip(
            clip,
            args.source_dir,
            args.dest_dir,
            args.frames_dir,
            include_video=args.include_video,
            reextract=args.reextract,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
