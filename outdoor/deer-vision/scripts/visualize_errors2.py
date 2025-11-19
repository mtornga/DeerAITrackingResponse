"""Alternative error visualizer that always writes overlays reliably.

Draws FP/FN labels directly onto the original image using OpenCV, with
class-filtering for GT and predictions. Matches GT to predictions by IoU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import sys

import cv2
import yaml
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from utils import compute_iou, iter_image_files, load_yolo_file, yolo_to_xyxy  # type: ignore  # noqa: E402


def draw_box(img: np.ndarray, box: Tuple[float, float, float, float], color: Tuple[int, int, int], label: str) -> None:
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    text = label.upper()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    bx0, by0 = x0, max(0, y0 - 6)
    cv2.rectangle(img, (bx0, by0 - th - 4), (bx0 + tw + 4, by0), color, thickness=-1)
    cv2.putText(img, text, (bx0 + 2, by0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


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
    eval_classes: List[int] | None,
    pred_classes: List[int] | None,
    min_gt_area_frac: float | None,
    min_gt_short_px: int | None,
) -> dict:
    relative = image_path.relative_to(images_root)
    label_path = labels_root / relative.with_suffix(".txt")
    gt_boxes = load_yolo_file(label_path)
    if eval_classes:
        allowed = set(eval_classes)
        gt_boxes = [b for b in gt_boxes if b.cls in allowed]

    img = cv2.imread(str(image_path))
    if img is None:
        return {"tp": 0, "fp": 0, "fn": 0}
    h, w = img.shape[:2]
    # Filter tiny GT by policy
    if min_gt_area_frac or min_gt_short_px:
        keep = []
        for b in gt_boxes:
            x0, y0, x1, y1 = yolo_to_xyxy(b, w, h)
            bw, bh = max(0.0, x1 - x0), max(0.0, y1 - y0)
            area_frac = (bw * bh) / float(w * h) if w and h else 0.0
            short_px = min(bw, bh)
            if min_gt_area_frac and area_frac < min_gt_area_frac:
                continue
            if min_gt_short_px and short_px < float(min_gt_short_px):
                continue
            keep.append(b)
        gt_boxes = keep

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

    annotations: dict = {"tp": 0, "fp": 0, "fn": 0, "pred": 0}
    preds_xyxy: List[Tuple[float, float, float, float]] = []
    preds_cls: List[int] = []

    if results:
        r = results[0]
        if r.boxes is not None and r.boxes.xyxy is not None:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            cls = r.boxes.cls.cpu().numpy().astype(int).tolist() if r.boxes.cls is not None else [0] * len(xyxy)
            for b, c in zip(xyxy, cls):
                if pred_classes and c not in set(pred_classes):
                    continue
                x0, y0, x1, y1 = map(float, b[:4])
                preds_xyxy.append((x0, y0, x1, y1))
                preds_cls.append(int(c))
    annotations["pred"] = len(preds_xyxy)

    matched_gt = [False] * len(gt_boxes)
    for p_box in preds_xyxy:
        # find best GT match
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if matched_gt[idx]:
                continue
            iou = compute_iou(p_box, yolo_to_xyxy(gt, w, h))
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= match_iou:
            matched_gt[best_idx] = True
            annotations["tp"] += 1
            draw_box(img, p_box, (0, 200, 0), "TP")
        else:
            annotations["fp"] += 1
            draw_box(img, p_box, (255, 0, 255), "FP")

    for idx, gt in enumerate(gt_boxes):
        if matched_gt[idx]:
            continue
        annotations["fn"] += 1
        draw_box(img, yolo_to_xyxy(gt, w, h), (0, 0, 255), "FN")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize detector errors with robust overlays.")
    parser.add_argument("--images", type=Path, required=True, help="Images root.")
    parser.add_argument("--labels", type=Path, required=True, help="Labels root.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model weights.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--limit", type=int, default=0, help="Max frames to annotate (0 = all).")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--match-iou", type=float, default=0.45, help="IoU threshold to count a detection as TP.")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="NMS IoU threshold passed to YOLO.")
    parser.add_argument("--agnostic-nms", action="store_true", help="Enable class-agnostic NMS.")
    parser.add_argument("--max-det", type=int, default=10, help="Maximum detections per frame.")
    parser.add_argument("--eval-classes", type=int, nargs="*", default=None, help="Evaluate only these GT classes.")
    parser.add_argument("--pred-classes", type=int, nargs="*", default=None, help="Count predictions only for these classes.")
    parser.add_argument("--policy", type=Path, default=None, help="QC policy YAML with eval thresholds and ignore rules.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load policy if provided
    if args.policy and args.policy.exists():
        with args.policy.open() as f:
            cfg = yaml.safe_load(f) or {}
        ev = (cfg or {}).get("eval", {})
        # Apply defaults if not explicitly overridden via flags
        args.conf = ev.get("conf", args.conf)
        args.nms_iou = ev.get("iou", args.nms_iou)
        args.agnostic_nms = ev.get("agnostic_nms", args.agnostic_nms)
        min_gt_area_frac = ev.get("min_gt_area_frac")
        min_gt_short_px = ev.get("min_gt_short_px")
    else:
        min_gt_area_frac = None
        min_gt_short_px = None

    # Default to deer-only if not specified
    if args.eval_classes is None:
        args.eval_classes = [0]
    if args.pred_classes is None:
        args.pred_classes = [0]

    model = YOLO(str(args.model))
    stats: List[dict] = []
    frames: dict[str, dict] = {}
    for idx, image_path in enumerate(iter_image_files(args.images)):
        if args.limit and idx >= args.limit:
            break
        out_path = args.out / image_path.relative_to(args.images)
        out_path = out_path.with_suffix(".jpg")
        per = annotate_image(
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
            min_gt_area_frac,
            min_gt_short_px,
        )
        stats.append(per)
        rel = str(image_path.relative_to(args.images).with_suffix(".jpg"))
        frames[rel] = per
    summary = {
        "frames": len(stats),
        "fp": sum(item["fp"] for item in stats),
        "fn": sum(item["fn"] for item in stats),
        "tp": sum(item["tp"] for item in stats),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    # Per-frame details for downstream A/B tools and UI
    (args.out / "frames.json").write_text(json.dumps(frames, indent=2))
    print(f"Wrote {len(stats)} visualizations to {args.out}")


if __name__ == "__main__":
    main()
