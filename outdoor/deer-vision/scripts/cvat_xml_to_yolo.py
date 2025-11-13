#!/usr/bin/env python3
"""Convert CVAT XML annotations into YOLO label files."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from utils import save_yolo_file, xyxy_to_yolo  # noqa: E402

DEFAULT_CLASS_MAP = {
    "deer": 0,
    "unknown_animal": 1,
    "person": 2,
    "apriltag": 3,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path, help="Path to CVAT annotations.xml export.")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write YOLO label files into.")
    parser.add_argument(
        "--class-map",
        type=str,
        default=None,
        help="Optional mapping like deer:0,unknown_animal:1 (defaults to deer/person/apriltag schema).",
    )
    return parser.parse_args()


def load_class_map(raw: str | None) -> Dict[str, int]:
    if not raw:
        return dict(DEFAULT_CLASS_MAP)
    mapping: Dict[str, int] = {}
    for pair in raw.split(","):
        if not pair.strip():
            continue
        name, _, value = pair.partition(":")
        mapping[name.strip()] = int(value.strip())
    return mapping


def main() -> int:
    args = parse_args()
    class_map = load_class_map(args.class_map)
    tree = ET.parse(args.xml)
    root = tree.getroot()

    size_node = root.find("./meta/task/size")
    width_node = root.find("./meta/task/original_size/width")
    height_node = root.find("./meta/task/original_size/height")
    if width_node is None or height_node is None:
        raise SystemExit("Unable to determine frame dimensions from annotations.xml")
    frame_width = float(width_node.text)
    frame_height = float(height_node.text)
    total_frames = int(size_node.text) if size_node is not None else 0

    frames: Dict[int, List[YoloAnnotation]] = defaultdict(list)

    def add_box(frame_idx: int, cls_name: str, xtl: float, ytl: float, xbr: float, ybr: float) -> None:
        if cls_name not in class_map:
            return
        cls_id = class_map[cls_name]
        x0 = clamp(xtl, 0.0, frame_width)
        y0 = clamp(ytl, 0.0, frame_height)
        x1 = clamp(xbr, 0.0, frame_width)
        y1 = clamp(ybr, 0.0, frame_height)
        if x1 <= x0 or y1 <= y0:
            return
        frames[frame_idx].append(
            xyxy_to_yolo((x0, y0, x1, y1), int(frame_width), int(frame_height), int(cls_id))
        )

    for track in root.findall("track"):
        label = track.attrib.get("label", "")
        for box in track.findall("box"):
            if box.attrib.get("outside") == "1":
                continue
            frame_idx = int(box.attrib["frame"])
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])
            add_box(frame_idx, label, xtl, ytl, xbr, ybr)

    for image_tag in root.findall("image"):
        label = image_tag.attrib.get("name", "")
        if label not in class_map:
            continue
        frame_idx = int(image_tag.attrib.get("id", "0"))
        for box in image_tag.findall("box"):
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])
            add_box(frame_idx, box.attrib.get("label", label), xtl, ytl, xbr, ybr)

    args.output.mkdir(parents=True, exist_ok=True)
    num_frames = total_frames if total_frames > 0 else max(frames.keys(), default=-1) + 1
    for frame_idx in range(num_frames):
        label_path = args.output / f"frame_{frame_idx:06d}.txt"
        save_yolo_file(label_path, frames.get(frame_idx, []))
    print(f"Wrote labels for {num_frames} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
