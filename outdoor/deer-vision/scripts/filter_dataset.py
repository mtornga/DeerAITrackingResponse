"""Filter YOLO datasets by minimum bbox size (area fraction, short side)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Source labels root")
    parser.add_argument("dst", type=Path, help="Destination labels root")
    parser.add_argument("--img-root", type=Path, help="Images root to mirror structure")
    parser.add_argument("--min-area", type=float, default=0.004, help="Minimum normalized area fraction")
    parser.add_argument("--min-short", type=float, default=0.02, help="Minimum normalized short side")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def keep_line(parts: list[str], min_area: float, min_short: float) -> bool:
    _, _, _, w, h = parts
    w = float(w)
    h = float(h)
    area = w * h
    short = min(w, h)
    return area >= min_area and short >= min_short


def main() -> None:
    args = parse_args()
    src_files = sorted(args.src.rglob("*.txt"))
    for src in src_files:
        rel = src.relative_to(args.src)
        dst = args.dst / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        lines = src.read_text().strip().splitlines()
        kept = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            if keep_line(parts, args.min_area, args.min_short):
                kept.append(line)
        if kept:
            if not args.dry_run:
                dst.write_text("\n".join(kept) + "\n")
        else:
            if dst.exists() and not args.dry_run:
                dst.unlink()
        if args.img_root and kept:
            img_src = args.img_root / rel.with_suffix(".jpg")
            img_dst = args.dst.parent / args.img_root.name / rel.with_suffix(".jpg")
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            if img_src.exists() and not args.dry_run:
                shutil.copy2(img_src, img_dst)


if __name__ == "__main__":
    main()
