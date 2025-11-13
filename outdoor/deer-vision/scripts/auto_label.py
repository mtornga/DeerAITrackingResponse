"""Run a pretrained YOLO model to bootstrap labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List
import sys

from tqdm import tqdm
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from utils import (  # noqa: E402
    YoloAnnotation,
    ensure_relative,
    iter_image_files,
    copy_image,
    save_yolo_file,
)


@dataclass
class AutoLabelStats:
    image_count: int = 0
    detections: int = 0


def run_autolabel(
    src: Path,
    out_dir: Path,
    model_path: Path,
    conf: float,
    imgsz: int,
    device: str | None,
) -> AutoLabelStats:
    model = YOLO(str(model_path))
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    stats = AutoLabelStats()
    for image_path in tqdm(list(iter_image_files(src)), desc="autolabel"):
        rel = ensure_relative(image_path, src)
        dest_image = images_out / rel
        copy_image(image_path, dest_image)
        results = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, device=device, verbose=False)
        annotations: List[YoloAnnotation] = []
        for result in results:
            if result.boxes is None or result.boxes.xywhn is None:
                continue
            for cls, box, score in zip(
                result.boxes.cls.tolist(),
                result.boxes.xywhn.tolist(),
                result.boxes.conf.tolist(),
            ):
                annotations.append(
                    YoloAnnotation(
                        cls=int(cls),
                        x_center=float(box[0]),
                        y_center=float(box[1]),
                        width=float(box[2]),
                        height=float(box[3]),
                        confidence=float(score),
                    )
                )
        save_yolo_file(labels_out / rel.with_suffix(".txt"), annotations)
        stats.image_count += 1
        stats.detections += len(annotations)
    (out_dir / "autolabel_summary.json").write_text(json.dumps(asdict(stats), indent=2))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label frames using a pretrained YOLO model.")
    parser.add_argument("--src", type=Path, default=Path("data/raw/frames"), help="Root directory containing extracted frames.")
    parser.add_argument("--out", type=Path, default=Path("data/interim/autolabel"), help="Directory for auto-label outputs.")
    parser.add_argument("--model", type=Path, default=Path("yolov8n.pt"), help="Pretrained weights to run.")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold for predictions.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference resolution.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override (cpu, mps, cuda:0, ...).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run_autolabel(args.src, args.out, args.model, args.conf, args.imgsz, args.device)


if __name__ == "__main__":
    main()
