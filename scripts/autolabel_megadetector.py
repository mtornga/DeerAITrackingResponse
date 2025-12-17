#!/usr/bin/env python3
"""Autolabel frames using MegaDetector for wildlife detection.

MegaDetector is trained on wildlife data and detects:
- Class 1: Animal (maps to our class 0: deer for now, since these are deer clips)
- Class 2: Person (maps to our class 2: person)
- Class 3: Vehicle (ignored)

Usage:
    python scripts/autolabel_megadetector.py \
        --src outdoor/deer-vision/data/raw/feedback/animal \
        --out outdoor/deer-vision/data/interim/feedback_autolabel \
        --model models/md_v5a.0.0.pt \
        --conf 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm


def _ensure_repo_root_on_path() -> Path:
    script_path = Path(__file__).resolve()
    for parent in (script_path.parent, *script_path.parents):
        if (parent / ".env").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    fallback = script_path.parents[1]
    if str(fallback) not in sys.path:
        sys.path.insert(0, str(fallback))
    return fallback


REPO_ROOT = _ensure_repo_root_on_path()


# MegaDetector class mapping to our dataset classes
# MegaDetector: 0=animal, 1=person, 2=vehicle
# Our dataset: 0=deer, 1=unknown_animal, 2=person, 3=apriltag
MD_CLASS_MAP = {
    0: 0,  # animal -> deer (we know these are deer clips)
    1: 2,  # person -> person
    # 2: vehicle - ignored
}


@dataclass
class YoloAnnotation:
    cls: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float = 0.0


@dataclass
class AutoLabelStats:
    image_count: int = 0
    detections: int = 0
    animals: int = 0
    persons: int = 0


def iter_image_files(src: Path):
    """Yield image files from directory."""
    if src.is_file():
        yield src
        return
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        yield from src.glob(ext)
        yield from src.rglob(ext)


def save_yolo_file(path: Path, annotations: List[YoloAnnotation]) -> None:
    """Save annotations in YOLO format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for ann in annotations:
        # YOLO format: class x_center y_center width height
        lines.append(f"{ann.cls} {ann.x_center:.6f} {ann.y_center:.6f} {ann.width:.6f} {ann.height:.6f}")
    path.write_text("\n".join(lines))


def copy_image(src: Path, dst: Path) -> None:
    """Copy image file."""
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run_autolabel_megadetector(
    src: Path,
    out_dir: Path,
    model_path: Path,
    conf: float,
    imgsz: int,
    device: str | None,
) -> AutoLabelStats:
    """Run MegaDetector autolabeling on frames."""

    # Load MegaDetector model
    print(f"Loading MegaDetector from {model_path}...")
    model = torch.hub.load(
        str(REPO_ROOT / "external" / "yolov5"),
        "custom",
        path=str(model_path),
        source="local",
        force_reload=False,
    )
    model.conf = conf
    model.iou = 0.45

    if device:
        model.to(device)

    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    stats = AutoLabelStats()

    image_files = list(iter_image_files(src))
    print(f"Found {len(image_files)} images to process")

    for image_path in tqdm(image_files, desc="autolabel"):
        # Get relative path for output
        try:
            rel = image_path.relative_to(src)
        except ValueError:
            rel = Path(image_path.name)

        # Copy image
        dest_image = images_out / rel
        copy_image(image_path, dest_image)

        # Run inference
        results = model(str(image_path), size=imgsz)

        # Parse results
        annotations: List[YoloAnnotation] = []
        detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]

        if len(detections) > 0:
            # Get image dimensions from results
            h, w = results.ims[0].shape[:2]

            for det in detections.cpu().numpy():
                x1, y1, x2, y2, score, md_cls = det
                md_cls = int(md_cls)

                # Map MegaDetector class to our class
                if md_cls not in MD_CLASS_MAP:
                    continue

                our_cls = MD_CLASS_MAP[md_cls]

                # Convert to YOLO format (normalized x_center, y_center, width, height)
                box_w = (x2 - x1) / w
                box_h = (y2 - y1) / h
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h

                annotations.append(YoloAnnotation(
                    cls=our_cls,
                    x_center=float(x_center),
                    y_center=float(y_center),
                    width=float(box_w),
                    height=float(box_h),
                    confidence=float(score),
                ))

                if our_cls == 0:
                    stats.animals += 1
                elif our_cls == 2:
                    stats.persons += 1

        # Save labels
        save_yolo_file(labels_out / rel.with_suffix(".txt"), annotations)

        stats.image_count += 1
        stats.detections += len(annotations)

    # Save summary
    summary = {
        "image_count": stats.image_count,
        "detections": stats.detections,
        "animals": stats.animals,
        "persons": stats.persons,
        "model": str(model_path),
        "conf_threshold": conf,
    }
    (out_dir / "autolabel_summary.json").write_text(json.dumps(summary, indent=2))

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autolabel frames using MegaDetector.")
    parser.add_argument("--src", type=Path, required=True, help="Source directory with images.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "models" / "md_v5a.0.0.pt",
        help="MegaDetector model path.",
    )
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference resolution.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda:0, cpu, etc.).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stats = run_autolabel_megadetector(
        args.src,
        args.out,
        args.model,
        args.conf,
        args.imgsz,
        args.device,
    )

    print(f"\nAutolabeling complete:")
    print(f"  Images processed: {stats.image_count}")
    print(f"  Total detections: {stats.detections}")
    print(f"  Animals (deer): {stats.animals}")
    print(f"  Persons: {stats.persons}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
