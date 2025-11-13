"""Tile high-resolution frames to boost small-object recall."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import sys

import cv2
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from utils import (  # noqa: E402
    YoloAnnotation,
    clip_box_to_window,
    iter_image_files,
    load_yolo_file,
    xyxy_to_yolo,
    yolo_to_xyxy,
    save_yolo_file,
)


def parse_grid(value: str) -> tuple[int, int]:
    try:
        rows, cols = value.lower().split("x")
        return int(rows), int(cols)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError("Grid must be formatted like 2x2") from exc


def tile_image(
    image_path: Path,
    images_root: Path,
    labels_root: Path,
    out_root: Path,
    rows: int,
    cols: int,
    overlap: float,
    min_overlap: float,
) -> int:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to load image {image_path}")
    height, width = img.shape[:2]
    tile_h = height / rows
    tile_w = width / cols
    step_y = int(tile_h * (1 - overlap)) or int(tile_h)
    step_x = int(tile_w * (1 - overlap)) or int(tile_w)
    rel_path = image_path.relative_to(images_root)
    annotations = load_yolo_file(labels_root / rel_path.with_suffix(".txt"))
    tiles_created = 0
    tile_index = 0
    row = 0
    while row < height:
        next_row = min(height, row + int(tile_h))
        if next_row - row < tile_h * 0.5 and row != 0:
            row = max(0, height - int(tile_h))
            next_row = height
        col = 0
        while col < width:
            next_col = min(width, col + int(tile_w))
            if next_col - col < tile_w * 0.5 and col != 0:
                col = max(0, width - int(tile_w))
                next_col = width
            x0, y0 = int(col), int(row)
            x1, y1 = int(next_col), int(next_row)
            window = (x0, y0, x1, y1)
            tile = img[y0:y1, x0:x1]
            tile_boxes: List[YoloAnnotation] = []
            for ann in annotations:
                box_xyxy = yolo_to_xyxy(ann, width, height)
                clipped = clip_box_to_window(box_xyxy, window)
                orig_area = (box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1])
                clipped_area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
                if orig_area <= 0 or clipped_area / orig_area < min_overlap:
                    continue
                rel_box = (
                    clipped[0] - x0,
                    clipped[1] - y0,
                    clipped[2] - x0,
                    clipped[3] - y0,
                )
                tile_boxes.append(xyxy_to_yolo(rel_box, x1 - x0, y1 - y0, ann.cls))
            if not tile_boxes:
                col += step_x
                continue
            rel_dir = out_root / rel_path.parent
            rel_dir.mkdir(parents=True, exist_ok=True)
            tile_name = rel_dir / f"{image_path.stem}_tile_{tile_index:02d}.jpg"
            cv2.imwrite(str(tile_name), tile)
            save_yolo_file(tile_name.with_suffix(".txt"), tile_boxes)
            tile_index += 1
            tiles_created += 1
            col += step_x
        row += step_y
    return tiles_created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile large frames for small-object detection.")
    parser.add_argument("--src", type=Path, default=Path("data/interim/autolabel/images"), help="Source images root.")
    parser.add_argument("--labels", type=Path, default=Path("data/interim/autolabel/labels"), help="Labels root.")
    parser.add_argument("--out", type=Path, default=Path("data/interim/tiles"), help="Output directory.")
    parser.add_argument("--grid", type=parse_grid, default="2x2", help="Grid definition like 2x2 or 3x3.")
    parser.add_argument("--overlap", type=float, default=0.1, help="Fractional overlap between tiles (0-0.9).")
    parser.add_argument("--min-overlap", type=float, default=0.2, help="Minimum fraction of the box that must remain.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, cols = args.grid if isinstance(args.grid, tuple) else parse_grid(args.grid)
    total_tiles = 0
    images = list(iter_image_files(args.src))
    for image_path in tqdm(images, desc="tiling"):
        total_tiles += tile_image(image_path, args.src, args.labels, args.out, rows, cols, args.overlap, args.min_overlap)
    print(f"Tiled {total_tiles} crops.")


if __name__ == "__main__":
    main()
