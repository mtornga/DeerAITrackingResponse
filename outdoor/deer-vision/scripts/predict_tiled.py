"""Run YOLO predictions with tiled inference to boost tiny-object recall."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from utils import compute_iou, load_yolo_file, yolo_to_xyxy  # noqa: E402


def generate_offsets(size: int, tile: int, overlap: int) -> List[int]:
    if tile >= size:
        return [0]
    stride = max(tile - overlap, 1)
    coords = list(range(0, size - tile + 1, stride))
    if coords[-1] + tile < size:
        coords.append(size - tile)
    return coords


def run_tiled_inference(
    model: YOLO,
    image: np.ndarray,
    tile: int,
    overlap: int,
    conf: float,
    imgsz: int,
    device: str | None,
    nms_iou: float,
    agnostic_nms: bool,
    max_det: int,
) -> List[dict]:
    h, w = image.shape[:2]
    boxes: List[Tuple[float, float, float, float]] = []
    scores: List[float] = []
    classes: List[int] = []

    tile_h = min(tile, h)
    tile_w = min(tile, w)

    for y in generate_offsets(h, tile_h, overlap):
        for x in generate_offsets(w, tile_w, overlap):
            patch = image[y : y + tile_h, x : x + tile_w]
            results = model.predict(
                source=patch,
                conf=conf,
                imgsz=imgsz,
                device=device,
                verbose=False,
                agnostic_nms=agnostic_nms,
                iou=nms_iou,
                max_det=max_det,
            )
            if not results:
                continue
            result = results[0]
            if not result.boxes:
                continue
            for cls, xyxy, score in zip(
                result.boxes.cls.tolist(),
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
            ):
                x0 = x + xyxy[0]
                y0 = y + xyxy[1]
                x1 = x + xyxy[2]
                y1 = y + xyxy[3]
                boxes.append((x0, y0, x1, y1))
                scores.append(float(score))
                classes.append(int(cls))

    if not boxes:
        return []

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    keep = nms(boxes_tensor, scores_tensor, nms_iou)

    kept = []
    for idx in keep:
        b = boxes_tensor[idx].tolist()
        kept.append({"box": b, "score": float(scores_tensor[idx]), "cls": classes[int(idx)]})
    return kept


def draw_boxes(image: np.ndarray, detections: List[dict], class_names: List[str]) -> np.ndarray:
    canvas = image.copy()
    for det in detections:
        x0, y0, x1, y1 = map(int, det["box"])
        cls = det["cls"]
        label = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        caption = f"{label} {det['score']:.2f}"
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(canvas, caption, (x0, max(15, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(canvas, caption, (x0, max(15, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiled YOLO predictions to improve tiny-object recall.")
    parser.add_argument("source", type=Path, help="Image or directory of images.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/tiled_predict"), help="Directory to save annotated frames.")
    parser.add_argument("--tile", type=int, default=640, help="Tile size in pixels.")
    parser.add_argument("--overlap", type=int, default=160, help="Overlap between tiles in pixels.")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--classes", nargs="*", default=["deer", "unknown", "person", "apriltag"], help="Class names for labels.")
    parser.add_argument("--json", type=Path, help="Optional path to dump raw detections as JSON.")
    parser.add_argument("--labels", type=Path, help="Optional labels root for FP/FN stats.")
    parser.add_argument("--eval-classes", type=int, nargs="*", default=[0, 1, 2, 3], help="Classes to include when computing FP/FN.")
    parser.add_argument("--match-iou", type=float, default=0.35, help="IoU threshold for TP when labels provided.")
    return parser.parse_args()


def collect_images(source: Path) -> List[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"})


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    images = collect_images(args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    json_records = {}
    stats = {"frames": 0, "tp": 0, "fp": 0, "fn": 0}

    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        stats["frames"] += 1
        detections = run_tiled_inference(
            model,
            frame,
            tile=args.tile,
            overlap=args.overlap,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            nms_iou=args.nms_iou,
            agnostic_nms=args.agnostic_nms,
            max_det=args.max_det,
        )
        annotated = draw_boxes(frame, detections, args.classes)
        rel = image_path.name
        cv2.imwrite(str(args.out / rel), annotated)
        json_records[rel] = detections

        if args.labels:
            label_path = args.labels / Path(rel).with_suffix(".txt")
            gt_boxes = load_yolo_file(label_path)
            matched = [False] * len(gt_boxes)
            for det in detections:
                det_cls = det["cls"]
                if det_cls not in args.eval_classes:
                    continue
                box = det["box"]
                best_iou = 0.0
                best_idx = -1
                for idx, gt in enumerate(gt_boxes):
                    if matched[idx]:
                        continue
                    if gt.cls != det_cls:
                        continue
                    iou = compute_iou(box, yolo_to_xyxy(gt, frame.shape[1], frame.shape[0]))
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_idx >= 0 and best_iou >= args.match_iou:
                    matched[best_idx] = True
                    stats["tp"] += 1
                else:
                    stats["fp"] += 1
            stats["fn"] += sum((not m) and (gt.cls in args.eval_classes) for m, gt in zip(matched, gt_boxes))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(json_records, indent=2))
    if args.labels:
        summary_path = args.out / "summary.json"
        summary_path.write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
