#!/usr/bin/env python3
"""
Generate an image→world calibration matrix for the tabletop simulation.

The script detects a set of known anchor objects, pairs their image locations
with measured real-world (tabletop) coordinates, and solves for an affine
transform. The resulting matrix is stored as JSON and can be consumed by the
top-down tracker to improve world-space accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def _ensure_repo_root_on_path() -> Path:
    script_path = Path(__file__).resolve()
    for parent in (script_path.parent, *script_path.parents):
        candidate = parent / ".env"
        if candidate.exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    root = script_path.parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_path()

from env_loader import load_env_file  # noqa: E402

load_env_file()


DEFAULT_MODEL = Path("runs/detect/train4/weights/best.pt")
DEFAULT_OUTPUT = Path("calibration/tabletop_affine.json")

# Default anchor measurements (inches from the tabletop's top-left corner)
DEFAULT_ANCHORS_IN = {
    "horse": (4.0, 4.0),
    "cutebot": (12.75, 12.75),
    "alien_maggie": (7.5, 15.25),
}


@dataclass
class Anchor:
    name: str
    x_ft: float
    y_ft: float

    @classmethod
    def from_inches(cls, name: str, x_in: float, y_in: float) -> "Anchor":
        return cls(name=name, x_ft=x_in / 12.0, y_ft=y_in / 12.0)


def parse_anchor_override(raw: str) -> Tuple[str, float, float]:
    """
    Parse an anchor override of the form `class=x_in,y_in`.
    """
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Anchor override must look like class=x_in,y_in")
    name, coords = raw.split("=", 1)
    if "," not in coords:
        raise argparse.ArgumentTypeError(f"Missing comma in anchor override: {raw!r}")
    try:
        x_str, y_str = coords.split(",", 1)
        x_val = float(x_str.strip())
        y_val = float(y_str.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid numeric values in {raw!r}") from exc
    return name.strip().lower(), x_val, y_val


def build_anchor_table(overrides: Iterable[str]) -> Dict[str, Anchor]:
    anchors: Dict[str, Anchor] = {
        name.lower(): Anchor.from_inches(name.lower(), *vals)
        for name, vals in DEFAULT_ANCHORS_IN.items()
    }
    for raw in overrides:
        name, xin, yin = parse_anchor_override(raw)
        anchors[name] = Anchor.from_inches(name, xin, yin)
    return anchors


def run_detection(model: YOLO, image: np.ndarray, conf: float, iou: float) -> Dict[str, Tuple[np.ndarray, float]]:
    """
    Run the detector on an image and return the best bottom-center point for each class.
    """
    results = model.predict(source=image, conf=conf, iou=iou, verbose=False)
    res = results[0]
    detected: Dict[str, Tuple[np.ndarray, float]] = {}

    if res.boxes is None or len(res.boxes) == 0:
        return detected

    names = model.names
    boxes = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()

    for xyxy, cls_idx, score in zip(boxes, cls, confs):
        label = names.get(int(cls_idx), str(cls_idx)).lower()
        x1, y1, x2, y2 = xyxy
        bottom_center = np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)
        prior = detected.get(label)
        if prior is None or score > prior[1]:
            detected[label] = (bottom_center, float(score))

    return detected


def compute_affine(image_pts: np.ndarray, world_pts: np.ndarray) -> np.ndarray:
    """
    Compute a 3x3 affine matrix mapping image pixel coords -> world-space feet.
    """
    if image_pts.shape != (3, 2) or world_pts.shape != (3, 2):
        raise ValueError("Exactly three non-collinear anchor points are required for affine calibration.")
    affine_2x3 = cv2.getAffineTransform(image_pts.astype(np.float32), world_pts.astype(np.float32))
    return np.vstack([affine_2x3, np.array([0.0, 0.0, 1.0], dtype=np.float32)])


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate tabletop image->world transform using YOLO detections.")
    parser.add_argument("--image", required=True, help="Path to a captured tabletop snapshot.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO weights for inference.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Destination JSON for the calibration matrix.")
    parser.add_argument("--anchor", action="append", default=[], help="Override anchor measurement in inches (class=x_in,y_in).")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="Detection IoU threshold.")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"Model weights not found: {model_path}")

    anchors = build_anchor_table(args.anchor)
    required_names = sorted(anchors.keys())

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Failed to load image: {image_path}")

    model = YOLO(str(model_path))
    detections = run_detection(model, image, args.conf, args.iou)

    missing = [name for name in required_names if name not in detections]
    if missing:
        raise SystemExit(f"Missing detections for anchors: {', '.join(missing)}")

    src_pts = []
    dst_pts = []
    anchor_summary = {}
    for name in required_names:
        bottom_center, score = detections[name]
        src_pts.append(bottom_center)
        anchor = anchors[name]
        dst_pts.append([anchor.x_ft, anchor.y_ft])
        anchor_summary[name] = {
            "image_bottom_center": bottom_center.tolist(),
            "confidence": score,
            "world_feet": [anchor.x_ft, anchor.y_ft],
        }

    src = np.vstack(src_pts)
    dst = np.vstack(dst_pts)

    matrix = compute_affine(src, dst)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "type": "affine",
        "image": str(image_path),
        "model": str(model_path),
        "anchors": anchor_summary,
        "matrix": matrix.tolist(),
        "notes": "Affine matrix maps image pixel bottom-centers to world feet coordinates.",
    }

    output_path.write_text(json.dumps(data, indent=2))
    print(f"[info] Saved calibration matrix to {output_path}")

    # Report residuals for quick sanity check
    src_h = cv2.convertPointsToHomogeneous(src.reshape(-1, 1, 2)).reshape(-1, 3)
    projected = (matrix @ src_h.T).T
    projected[:, 0] /= projected[:, 2]
    projected[:, 1] /= projected[:, 2]
    residuals = np.linalg.norm(projected[:, :2] - dst, axis=1)

    for idx, name in enumerate(required_names):
        world_xy = dst[idx]
        print(f"[info] {name}: world={world_xy.tolist()} residual={residuals[idx]:.4f} ft")

    return 0


if __name__ == "__main__":
    sys.exit(main())
