"""Extract frames from raw clips into data/raw/frames."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from utils import copy_image  # noqa: E402


@dataclass
class ExtractionStats:
    clip: str
    frames_written: int = 0
    fps: float = 0.0
    stride: int = 1
    interval_seconds: float | None = None


def extract_clip(
    clip_path: Path,
    dst_root: Path,
    stride: int,
    max_per_clip: int | None,
    interval_seconds: float | None,
) -> ExtractionStats:
    """Extract frames from a single clip."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open clip {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stats = ExtractionStats(
        clip=clip_path.name,
        fps=fps,
        stride=stride,
        interval_seconds=interval_seconds,
    )
    out_dir = dst_root / clip_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    write_idx = 0
    next_capture = 0
    capture_every = int(interval_seconds * fps) if interval_seconds else stride

    with tqdm(
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        desc=f"frames:{clip_path.stem}",
    ) as progress:
        while True:
            success, frame = cap.read()
            if not success:
                break
            if frame_idx >= next_capture:
                filename = out_dir / f"frame_{write_idx:06d}.jpg"
                cv2.imwrite(str(filename), frame)
                write_idx += 1
                stats.frames_written += 1
                next_capture += capture_every
                if max_per_clip and stats.frames_written >= max_per_clip:
                    break
            frame_idx += 1
            progress.update(1)
    cap.release()
    metadata = out_dir / "metadata.json"
    metadata.write_text(json.dumps(asdict(stats), indent=2))
    return stats


def snapshot_images(src: Path, dst: Path) -> List[ExtractionStats]:
    """Copy standalone still images into the frames directory."""
    stats: List[ExtractionStats] = []
    for image in src.glob("*.jpg"):
        target_dir = dst / image.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        copy_image(image, target_dir / image.name)
        stats.append(ExtractionStats(clip=image.name, frames_written=1, stride=1))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from clips.")
    parser.add_argument("--src", type=Path, default=Path("data/raw/clips"), help="Directory containing source clips.")
    parser.add_argument("--dst", type=Path, default=Path("data/raw/frames"), help="Output directory for frame folders.")
    parser.add_argument("--stride", type=int, default=5, help="Sample every N frames (ignored if --interval is set).")
    parser.add_argument("--interval", type=float, default=None, help="Capture frame every N seconds instead of stride.")
    parser.add_argument("--max-per-clip", type=int, default=300, help="Maximum frames per clip (0 for unlimited).")
    parser.add_argument("--include-stills", type=Path, help="Optional folder of standalone JPGs to ingest verbatim.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats: List[ExtractionStats] = []
    clips = sorted(p for p in args.src.glob("*.mp4"))
    if not clips:
        print(f"No clips found in {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        stats.append(
            extract_clip(
                clip,
                args.dst,
                stride=max(1, args.stride),
                max_per_clip=args.max_per_clip or None,
                interval_seconds=args.interval,
            )
        )
    if args.include_stills:
        stats.extend(snapshot_images(args.include_stills, args.dst))
    summary = [asdict(stat) for stat in stats]
    (args.dst / "extraction_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
