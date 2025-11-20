#!/usr/bin/env python3
"""Lightweight MegaDetector-style video processor using Ultralytics YOLO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO


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


def main() -> int:
    args = parse_args()
    # Let Ultralytics handle device selection; optional override via args.device.
    model = YOLO(str(args.model_path))
    if isinstance(model.names, dict):
        names: Dict[int, str] = {int(k): v for k, v in model.names.items()}
    else:
        names = {i: n for i, n in enumerate(model.names)}

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
            # Run a single-image prediction. Ultralytics handles preprocessing internally.
            results = model(frame, verbose=False, device=args.device or None)
            if results:
                r = results[0]
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clses = boxes.cls.cpu().numpy()

                    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clses):
                        conf_f = float(conf)
                        if conf_f < args.json_confidence_threshold:
                            continue
                        class_id = int(cls)
                        label = names.get(class_id, str(class_id))
                        label_lower = label.lower()
                        if "person" in label_lower:
                            category = "2"
                        elif any(k in label_lower for k in ("vehicle", "car", "truck")):
                            category = "3"
                        else:
                            category = "1"
                        x1_f, y1_f, x2_f, y2_f = float(x1), float(y1), float(x2), float(y2)
                        bbox = [
                            max(0.0, x1_f / w),
                            max(0.0, y1_f / h),
                            min(1.0, (x2_f - x1_f) / w),
                            min(1.0, (y2_f - y1_f) / h),
                        ]
                        detections.append({"category": category, "conf": conf_f, "bbox": bbox})
                        max_conf = max(max_conf, conf_f)

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
