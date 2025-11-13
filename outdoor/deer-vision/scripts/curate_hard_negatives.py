"""Collect hard-negative frames for training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from utils import copy_image, iter_image_files, load_yolo_file, save_yolo_file  # noqa: E402


def load_manifest(manifest_path: Path) -> List[Path]:
    entries: List[Path] = []
    for raw in manifest_path.read_text().splitlines():
        raw = raw.strip()
        if raw:
            entries.append(Path(raw))
    return entries


def find_empty_frames(images_root: Path, labels_root: Path) -> List[Path]:
    candidates: List[Path] = []
    for image in iter_image_files(images_root):
        label = labels_root / image.relative_to(images_root).with_suffix(".txt")
        annotations = load_yolo_file(label)
        if not annotations:
            candidates.append(image)
    return candidates


def curate(
    images_root: Path,
    labels_root: Path,
    dst_root: Path,
    manifest: Path | None,
    limit: int | None,
) -> List[str]:
    dst_root.mkdir(parents=True, exist_ok=True)
    if manifest and manifest.exists():
        rel_paths = load_manifest(manifest)
        candidates = [images_root / rel for rel in rel_paths]
    else:
        candidates = find_empty_frames(images_root, labels_root)
    curated: List[str] = []
    for idx, image in enumerate(candidates):
        if limit and idx >= limit:
            break
        rel = image.relative_to(images_root)
        dest = dst_root / rel
        copy_image(image, dest)
        save_yolo_file(dst_root / "labels" / rel.with_suffix(".txt"), [])
        curated.append(str(rel))
    (dst_root / "manifest.json").write_text(json.dumps({"count": len(curated), "items": curated}, indent=2))
    return curated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate hard negative samples.")
    parser.add_argument("--src", type=Path, default=Path("data/interim/autolabel/images"), help="Source images root.")
    parser.add_argument("--labels", type=Path, default=Path("data/interim/autolabel/labels"), help="Labels root.")
    parser.add_argument("--dst", type=Path, default=Path("data/interim/hard_negatives"), help="Destination folder.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional text file listing relative paths to include.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on curated samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curated = curate(args.src, args.labels, args.dst, args.manifest, args.limit)
    print(f"Curated {len(curated)} hard negatives.")


if __name__ == "__main__":
    main()
