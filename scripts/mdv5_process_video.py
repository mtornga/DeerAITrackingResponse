#!/usr/bin/env python3
"""Lightweight MegaDetector-style video processor using Ultralytics YOLO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import sys
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
YOLOV5_ROOT = REPO_ROOT / "tmp" / "yolov5"
if YOLOV5_ROOT.exists():
    sys.path.insert(0, str(YOLOV5_ROOT))

from models.common import DetectMultiBackend  # type: ignore
from utils.augmentations import letterbox  # type: ignore
from utils.general import non_max_suppression, scale_boxes  # type: ignore
from utils.torch_utils import select_device  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a video with md_v5 style weights and emit a MegaDetector JSON."
    )
    parser.add_argument("model_path", type=Path)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--output_json_file", type=Path, required=True)
    parser.add_argument("--json_confidence_threshold", type=float, default=0.2)
    parser.add_argument("--frame_sample", type=int, default=15, help="Process every Nth frame.")
    parser.add_argument("--n_cores", type=int, default=1, help="Ignored; kept for compatibility.")
    parser.add_argument("--device", default="")
    return parser.parse_args()


MD_CATEGORIES = {"1": "animal", "2": "person", "3": "vehicle"}


def load_model(path: Path, device: str) -> tuple[DetectMultiBackend, torch.device, List[str], int]:
    device_sel = select_device(device)
    model = DetectMultiBackend(str(path), device=device_sel, dnn=False)
    stride = int(model.stride)
    names = model.names
    return model, device_sel, names, stride


def main() -> int:
    args = parse_args()
    model, device, names, stride = load_model(args.model_path, args.device)
    imgsz = (640, 640)
    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise SystemExit(f"Unable to open video: {args.video_path}")

    images = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.frame_sample > 1 and frame_idx % args.frame_sample != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            detections = []
            max_conf = 0.0
            img = letterbox(frame, imgsz, stride=stride, auto=model.pt)[0]
            img = img.transpose((2, 0, 1))
            img = np.ascontiguousarray(img)
            im = torch.from_numpy(img).to(device)
            im = im.float() / 255.0
            if im.ndimension() == 3:
                im = im.unsqueeze(0)

            pred = model(im)
            det = non_max_suppression(
                pred,
                conf_thres=0.001,
                iou_thres=0.45,
                max_det=300,
            )[0]

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    conf = float(conf)
                    if conf < args.json_confidence_threshold:
                        continue
                    class_id = int(cls)
                    label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
                    if "person" in label.lower():
                        category = "2"
                    elif "vehicle" in label.lower():
                        category = "3"
                    else:
                        category = "1"
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    bbox = [
                        max(0.0, x1 / w),
                        max(0.0, y1 / h),
                        min(1.0, (x2 - x1) / w),
                        min(1.0, (y2 - y1) / h),
                    ]
                    detections.append({"category": category, "conf": conf, "bbox": bbox})
                    max_conf = max(max_conf, conf)

            images.append(
                {
                    "file": f"frame_{frame_idx:06d}",
                    "max_detection_conf": max_conf,
                    "detections": detections,
                }
            )
            frame_idx += 1
    finally:
        cap.release()

    payload = {
        "detection_categories": MD_CATEGORIES,
        "images": images,
        "info": {
            "model": args.model_path.name,
            "video": str(args.video_path),
            "frame_sample": args.frame_sample,
        },
    }
    args.output_json_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_json_file.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
