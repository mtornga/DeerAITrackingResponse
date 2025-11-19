"""Visualize false positives/negatives for qualitative review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List
import sys

import cv2
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from utils import compute_iou, iter_image_files, load_yolo_file, yolo_to_xyxy  # noqa: E402


def draw_flag(img, box, label, color) -> None:
    """Overlay label text with a solid background for readability."""
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    text = label.upper()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    text_origin = (x0, max(0, y0 - 5))
    cv2.rectangle(
        img,
        (text_origin[0], text_origin[1] - th - 4),
        (text_origin[0] + tw + 4, text_origin[1]),
        color,
        thickness=-1,
    )
    cv2.putText(
        img,
        text,
        (text_origin[0] + 2, text_origin[1] - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def annotate_image(
    image_path: Path,
    images_root: Path,
    model: YOLO,
    labels_root: Path,
    out_path: Path,
    conf: float,
    imgsz: int,
    device: str | None,
    match_iou: float,
    nms_iou: float,
    agnostic_nms: bool,
    max_det: int,
    eval_classes: list[int] | None,
    pred_classes: list[int] | None,
) -> dict:
    relative = image_path.relative_to(images_root)
    label_path = labels_root / relative.with_suffix(".txt")
    gt_boxes = load_yolo_file(label_path)
    if eval_classes:
        gt_boxes = [b for b in gt_boxes if b.cls in set(eval_classes)]
    results = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
        iou=nms_iou,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
    )
    annotations: dict = {"tp": 0, "fp": 0, "fn": 0}
    if not results:
        return annotations
    result = results[0]
    img = result.plot()  # includes predictions
    h, w = result.orig_shape
    matched_gt = [False] * len(gt_boxes)
    if result.boxes:
        box_cls = result.boxes.cls.tolist() if result.boxes.cls is not None else []
        box_xywhn = result.boxes.xywhn.tolist() if result.boxes.xywhn is not None else []
        box_conf = result.boxes.conf.tolist() if result.boxes.conf is not None else []
        for cls, xywhn, score in zip(box_cls, box_xywhn, box_conf):
            if pred_classes and int(cls) not in set(pred_classes):
                continue
            box = (
                (xywhn[0] - xywhn[2] / 2) * w,
                (xywhn[1] - xywhn[3] / 2) * h,
                (xywhn[0] + xywhn[2] / 2) * w,
                (xywhn[1] + xywhn[3] / 2) * h,
            )
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if matched_gt[idx]:
                    continue
                iou = compute_iou(box, yolo_to_xyxy(gt, w, h))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= match_iou:
                matched_gt[best_idx] = True
                annotations["tp"] += 1
            else:
                annotations["fp"] += 1
                draw_flag(img, box, "FP", (255, 0, 255))
    for idx, gt in enumerate(gt_boxes):
        if matched_gt[idx]:
            continue
        annotations["fn"] += 1
        box = yolo_to_xyxy(gt, w, h)
        draw_flag(img, box, "FN", (0, 0, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize top detector errors.")
    parser.add_argument("--images", type=Path, default=Path("data/eval/night_hard/images"), help="Images root.")
    parser.add_argument("--labels", type=Path, default=Path("data/eval/night_hard/labels"), help="Labels root.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model weights.")
    parser.add_argument("--out", type=Path, default=Path("data/interim/visualizations"), help="Output directory.")
    parser.add_argument("--limit", type=int, default=25, help="Max frames to annotate.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold to count a detection as TP.")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold passed to YOLO.")
    parser.add_argument("--agnostic-nms", action="store_true", help="Enable class-agnostic NMS.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per frame.")
    parser.add_argument(
        "--eval-classes",
        type=int,
        nargs="*",
        default=None,
        help="Only evaluate these GT classes (by id). If omitted, all GT are used.",
    )
    parser.add_argument(
        "--pred-classes",
        type=int,
        nargs="*",
        default=None,
        help="Only count predictions for these classes (by id).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    stats: List[dict] = []
    for idx, image_path in enumerate(iter_image_files(args.images)):
        if args.limit and idx >= args.limit:
            break
        out_path = args.out / image_path.relative_to(args.images)
        out_path = out_path.with_suffix(".jpg")
        stats.append(
            annotate_image(
                image_path,
                args.images,
                model,
                args.labels,
                out_path,
                args.conf,
                args.imgsz,
                args.device,
                args.match_iou,
                args.nms_iou,
                args.agnostic_nms,
                args.max_det,
                args.eval_classes,
                args.pred_classes,
            )
        )
    summary = {
        "frames": len(stats),
        "fp": sum(item["fp"] for item in stats),
        "fn": sum(item["fn"] for item in stats),
        "tp": sum(item["tp"] for item in stats),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {len(stats)} visualizations to {args.out}")


if __name__ == "__main__":
    main()
