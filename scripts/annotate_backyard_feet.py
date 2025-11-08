#!/usr/bin/env python3
"""
Interactively capture “foot point” pixels for the BackyardMe evaluation stills.

Usage:
    python scripts/annotate_backyard_feet.py \
        --images outdoor/calibration/BackyardMe16Feet.jpg \
                 outdoor/calibration/BackyardMe32feet.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        nargs="+",
        default=[
            "outdoor/calibration/BackyardMe16Feet.jpg",
            "outdoor/calibration/BackyardMe32feet.jpg",
            "outdoor/calibration/BackyardMe48feetbydeck.jpg",
            "outdoor/calibration/BackyardMe48feetuphill.jpg",
        ],
        help="List of JPGs to annotate.",
    )
    parser.add_argument(
        "--output-json",
        default="calibration/outdoor/backyard_feet_manual.json",
        help="Destination JSON file with {image: [x, y]} mappings.",
    )
    parser.add_argument(
        "--annot-prefix",
        default="_foot_manual",
        help="Suffix inserted before the extension for annotated copies.",
    )
    return parser.parse_args()


def annotate_image(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")

    window = "click-foot"
    point: dict[str, int | None] = {"x": None, "y": None}

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            point["x"], point["y"] = x, y

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, int(img.shape[1] * 0.7), int(img.shape[0] * 0.7))
    cv2.setMouseCallback(window, on_mouse)

    while True:
        display = img.copy()
        if point["x"] is not None and point["y"] is not None:
            cv2.circle(
                display,
                (int(point["x"]), int(point["y"])),
                25,
                (0, 0, 255),
                thickness=6,
            )
            cv2.putText(
                display,
                f"({point['x']}, {point['y']})",
                (int(point["x"]) + 10, int(point["y"]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            display,
            "Click feet | ENTER=accept | R=reset | Q=quit",
            (30, display.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            display,
            "Click feet | ENTER=accept | R=reset | Q=quit",
            (30, display.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, ord(" ")):  # Enter or Space accepts selection
            if point["x"] is not None and point["y"] is not None:
                cv2.destroyWindow(window)
                return int(point["x"]), int(point["y"])
        elif key in (ord("r"), ord("R")):
            point["x"] = point["y"] = None
        elif key in (ord("q"), ord("Q"), 27):  # ESC
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("Annotation aborted by user.")


def save_annotation(path: Path, point: tuple[int, int], suffix: str) -> None:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Failed to load image for saving: {path}")
    cv2.circle(img, point, 25, (0, 0, 255), 6)
    cv2.putText(
        img,
        f"{point}",
        (point[0] + 10, point[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )

    annotated_path = path.with_name(
        f"{path.stem}{suffix}{path.suffix}"
    )
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(annotated_path), img)


def main() -> None:
    args = parse_args()
    images: Iterable[Path] = [Path(p).resolve() for p in args.images]
    results: dict[str, list[int]] = {}

    try:
        for img_path in images:
            print(f"\nAnnotating {img_path} …")
            point = annotate_image(img_path)
            print(f"  -> recorded pixel {point}")
            save_annotation(img_path, point, args.annot_prefix)
            results[str(img_path)] = [point[0], point[1]]
    finally:
        cv2.destroyAllWindows()

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path} with {len(results)} entries.")


if __name__ == "__main__":
    main()
