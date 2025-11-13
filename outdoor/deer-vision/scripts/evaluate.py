"""Evaluate detector checkpoints against frozen packs."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import yaml
from ultralytics import YOLO
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from utils import compute_iou, iter_image_files, load_yolo_file, yolo_to_xyxy  # noqa: E402


@dataclass
class Detection:
    image_id: str
    cls: int
    score: float
    box: tuple[float, float, float, float]


@dataclass
class GroundTruth:
    image_id: str
    cls: int
    box: tuple[float, float, float, float]
    matched: bool = False


@dataclass
class PackMetrics:
    name: str
    ap50: float
    recall: float
    fp_per_min: float
    latency_ms_p95: float
    images: int
    fps: float
    detail: Dict[str, float] = field(default_factory=dict)


def compute_ap(tp: Sequence[int], fp: Sequence[int], total_gt: int) -> float:
    if total_gt == 0 or not tp:
        return 0.0
    tp_cum = []
    fp_cum = []
    t_sum = 0
    f_sum = 0
    for t, f in zip(tp, fp):
        t_sum += t
        f_sum += f
        tp_cum.append(t_sum)
        fp_cum.append(f_sum)
    precisions = []
    recalls = []
    for t_val, f_val in zip(tp_cum, fp_cum):
        denom = t_val + f_val
        precisions.append(t_val / denom if denom else 0.0)
        recalls.append(t_val / total_gt if total_gt else 0.0)
    ap = 0.0
    prev_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        ap += precision * max(0.0, recall - prev_recall)
        prev_recall = recall
    return ap


def match_detections(detections: List[Detection], ground_truths: List[GroundTruth], iou_thr: float) -> tuple[List[int], List[int], int]:
    gt_lookup: Dict[str, List[GroundTruth]] = {}
    for gt in ground_truths:
        gt_lookup.setdefault(gt.image_id, []).append(gt)
    detections_sorted = sorted(detections, key=lambda det: det.score, reverse=True)
    tp_flags: List[int] = []
    fp_flags: List[int] = []
    matched = 0
    for det in detections_sorted:
        gts = gt_lookup.get(det.image_id, [])
        best_iou = 0.0
        best_gt: GroundTruth | None = None
        for gt in gts:
            if gt.cls != det.cls or gt.matched:
                continue
            iou = compute_iou(det.box, gt.box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
        if best_gt and best_iou >= iou_thr:
            best_gt.matched = True
            tp_flags.append(1)
            fp_flags.append(0)
            matched += 1
        else:
            tp_flags.append(0)
            fp_flags.append(1)
    return tp_flags, fp_flags, matched


def percentile(data: Sequence[float], q: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    index = int(round((len(data_sorted) - 1) * q))
    return data_sorted[min(index, len(data_sorted) - 1)]


def evaluate_pack(
    model: YOLO,
    pack_dir: Path,
    conf: float,
    imgsz: int,
    device: str | None,
    fps: float,
) -> PackMetrics:
    images_dir = pack_dir / "images"
    labels_dir = pack_dir / "labels"
    detections: List[Detection] = []
    ground_truths: List[GroundTruth] = []
    latencies: List[float] = []
    image_files = list(iter_image_files(images_dir))
    for image_path in image_files:
        rel_id = image_path.relative_to(images_dir).as_posix()
        label_path = labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")
        gt_boxes = load_yolo_file(label_path)
        start = time.perf_counter()
        results = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, device=device, verbose=False)
        duration_ms = (time.perf_counter() - start) * 1000
        if results:
            result = results[0]
            latency = result.speed.get("inference", duration_ms)
            latencies.append(latency)
            h, w = result.orig_shape
        else:
            latencies.append(duration_ms)
            img = cv2.imread(str(image_path))
            if img is None:
                raise RuntimeError(f"Unable to read {image_path}")
            h, w = img.shape[:2]
            result = None
        for gt in gt_boxes:
            ground_truths.append(GroundTruth(image_id=rel_id, cls=gt.cls, box=yolo_to_xyxy(gt, w, h)))
        if result is None or result.boxes is None:
            continue
        for cls, xywhn, score in zip(
            result.boxes.cls.tolist(),
            result.boxes.xywhn.tolist(),
            result.boxes.conf.tolist(),
        ):
            center_x, center_y, width_n, height_n = xywhn
            box = (
                (center_x - width_n / 2) * w,
                (center_y - height_n / 2) * h,
                (center_x + width_n / 2) * w,
                (center_y + height_n / 2) * h,
            )
            detections.append(Detection(image_id=rel_id, cls=int(cls), score=float(score), box=box))
    tp, fp, matched = match_detections(detections, ground_truths, iou_thr=0.5)
    total_gt = len(ground_truths)
    ap50 = compute_ap(tp, fp, total_gt)
    recall = matched / total_gt if total_gt else 0.0
    duration_minutes = (len(image_files) / fps) / 60 if fps else 0.0
    fp_count = sum(fp)
    fp_per_min = fp_count / duration_minutes if duration_minutes else 0.0
    latency_p95 = percentile(latencies, 0.95)
    return PackMetrics(
        name=pack_dir.name,
        ap50=ap50,
        recall=recall,
        fp_per_min=fp_per_min,
        latency_ms_p95=latency_p95,
        images=len(image_files),
        fps=fps,
        detail={
            "detections": len(detections),
            "ground_truth": total_gt,
            "fps_assumed": fps,
        },
    )


def aggregate_metrics(packs: List[PackMetrics]) -> Dict[str, float]:
    valid_packs = [pack for pack in packs if pack.images and pack.detail.get("ground_truth", 0)]
    if not valid_packs:
        return {"map50": 0.0, "recall": 0.0, "latency_ms_p95": 0.0}
    map50 = sum(pack.ap50 for pack in valid_packs) / len(valid_packs)
    recall = sum(pack.recall for pack in valid_packs) / len(valid_packs)
    latency = max(pack.latency_ms_p95 for pack in packs)
    return {"map50": map50, "recall": recall, "latency_ms_p95": latency}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO checkpoint against frozen packs.")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained weights (e.g., models/yolov8n_det_v01/weights/best.pt).")
    parser.add_argument("--eval-packs", type=Path, default=Path("data/eval"), help="Directory containing eval packs.")
    parser.add_argument("--output", type=Path, default=Path("models/metrics.json"), help="Where to write aggregated metrics.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for evaluation.")
    parser.add_argument("--imgsz", type=int, default=960, help="Eval resolution.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    parser.add_argument("--fps", type=float, default=30.0, help="Assumed frame rate when computing FP/min.")
    parser.add_argument("--thresholds", type=Path, default=Path("configs/eval_thresholds.yaml"), help="Quality thresholds for reference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    packs = []
    for pack_dir in args.eval_packs.iterdir():
        if not pack_dir.is_dir():
            continue
        images_dir = pack_dir / "images"
        labels_dir = pack_dir / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue
        packs.append(evaluate_pack(model, pack_dir, args.conf, args.imgsz, args.device, args.fps))
    aggregate = aggregate_metrics(packs)
    metrics = {
        **aggregate,
        "fp_per_min": {pack.name: pack.fp_per_min for pack in packs},
        "packs": [asdict(pack) for pack in packs],
    }
    if args.thresholds.exists():
        metrics["thresholds"] = yaml.safe_load(args.thresholds.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
