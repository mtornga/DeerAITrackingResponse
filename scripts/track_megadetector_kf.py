#!/usr/bin/env python
"""
Dynamic Kalman+appearance tracker for MegaDetector outputs.

Given a MegaDetector detection JSON and the corresponding frame directory,
assign consistent IDs across frames using a constant-velocity Kalman filter
and cosine-similarity matching from the MegaDescriptor embedding network.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision import transforms


def parse_class_map(values: List[str]) -> Dict[str, str]:
    """
    Parse CLI class mappings in the form of ["1=deer", "2=person"].
    """
    mapping = {}
    for item in values:
        try:
            key, prefix = item.split("=", 1)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid --class-map entry '{item}', expected format id=prefix"
            ) from exc
        mapping[key.strip()] = prefix.strip()
    return mapping


class KalmanFilter:
    """
    Constant-velocity Kalman filter on bounding-box center/size.
    State vector: [cx, cy, w, h, vx, vy, vw, vh]
    """

    def __init__(self, dt: float = 1.0, std_pos: float = 1e-2, std_vel: float = 1e-3) -> None:
        self.std_pos = std_pos
        self.std_vel = std_vel
        self.motion_mat = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.motion_mat[i, i + 4] = dt
        self.update_mat = np.eye(4, 8, dtype=np.float32)

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(8, dtype=np.float32)
        mean[:4] = measurement
        covariance = np.eye(8, dtype=np.float32)
        covariance[:4, :4] *= self.std_pos**2
        covariance[4:, 4:] *= (self.std_vel**2) * 10000.0
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        motion_cov = np.diag([self.std_pos**2] * 4 + [self.std_vel**2] * 4).astype(np.float32)
        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        meas_cov = np.diag([self.std_pos**2] * 4).astype(np.float32)
        proj_mean = self.update_mat @ mean
        proj_cov = self.update_mat @ covariance @ self.update_mat.T + meas_cov
        return proj_mean, proj_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        proj_mean, proj_cov = self.project(mean, covariance)
        kalman_gain = covariance @ self.update_mat.T @ np.linalg.inv(proj_cov)
        innovation = measurement - proj_mean
        mean = mean + kalman_gain @ innovation
        covariance = (np.eye(8, dtype=np.float32) - kalman_gain @ self.update_mat) @ covariance
        return mean, covariance

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> float:
        proj_mean, proj_cov = self.project(mean, covariance)
        diff = measurement - proj_mean
        try:
            chol = np.linalg.cholesky(proj_cov)
            z = np.linalg.solve(chol, diff)
            return float(z @ z)
        except np.linalg.LinAlgError:
            return float(diff @ np.linalg.inv(proj_cov) @ diff)


def bbox_to_measure(bbox: List[float]) -> np.ndarray:
    xmin, ymin, w, h = bbox
    return np.array([xmin + w / 2, ymin + h / 2, w, h], dtype=np.float32)


def measure_to_bbox(measure: np.ndarray) -> np.ndarray:
    cx, cy, w, h = measure
    return np.array([cx - w / 2, cy - h / 2, w, h], dtype=np.float32)


def clamp_bbox(bbox: np.ndarray) -> np.ndarray:
    xmin, ymin, w, h = bbox
    xmin = np.clip(float(xmin), 0.0, 1.0)
    ymin = np.clip(float(ymin), 0.0, 1.0)
    w = np.clip(float(w), 0.0, 1.0)
    h = np.clip(float(h), 0.0, 1.0)
    xmax = np.clip(xmin + w, 0.0, 1.0)
    ymax = np.clip(ymin + h, 0.0, 1.0)
    return np.array([xmin, ymin, xmax - xmin, ymax - ymin], dtype=np.float32)


def load_descriptor(weights_path: Path) -> Tuple[torch.nn.Module, transforms.Compose]:
    state = torch.load(weights_path, map_location="cpu")["model"]
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
    model.load_state_dict(state)
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((288, 288)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, transform


def embed_crop(
    image: Image.Image, bbox: List[float], model: torch.nn.Module, transform: transforms.Compose, expand: float = 0.2
) -> np.ndarray:
    width, height = image.size
    xmin, ymin, w, h = bbox
    cx, cy = xmin + w / 2, ymin + h / 2
    new_w, new_h = w * (1 + expand), h * (1 + expand)
    x1 = max(0, (cx - new_w / 2) * width)
    y1 = max(0, (cy - new_h / 2) * height)
    x2 = min(width, (cx + new_w / 2) * width)
    y2 = min(height, (cy + new_h / 2) * height)
    crop = image.crop((x1, y1, x2, y2))
    tensor = transform(crop).unsqueeze(0)
    with torch.no_grad():
        vec = model(tensor).numpy().squeeze(0)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def assign_colors() -> Tuple[int, int, int]:
    return tuple(int(x) for x in np.random.randint(64, 255, size=3))


def run_tracker(args: argparse.Namespace) -> None:
    descriptor, transform = load_descriptor(args.descriptor_weights)
    kf = KalmanFilter()

    # load detections
    with args.detections_json.open() as f:
        det_data = json.load(f)
    frame_entries = {img["file"]: img for img in det_data["images"]}
    frame_names = sorted(frame_entries.keys())

    class_map = parse_class_map(args.class_map) if args.class_map else {"1": "deer"}
    next_ids = defaultdict(lambda: 1)  # per category id counter
    tracks = defaultdict(list)  # category id -> list of track dicts
    label_pool = {}
    free_labels = defaultdict(set)
    label_colors: Dict[str, Tuple[int, int, int]] = {}

    if args.max_tracks_per_class:
        for cat_id, prefix in class_map.items():
            pool = [f"{prefix}{i + 1}" for i in range(args.max_tracks_per_class)]
            label_pool[cat_id] = pool
            free_labels[cat_id] = set(pool)
    else:
        for cat_id in class_map:
            label_pool[cat_id] = None

    results = {}

    def allocate_label(cat_id: str, prefix: str) -> str:
        if args.max_tracks_per_class:
            if free_labels[cat_id]:
                label = sorted(free_labels[cat_id])[0]
                free_labels[cat_id].remove(label)
                return label
            raise RuntimeError(
                "No available labels to allocate; consider increasing --max-tracks-per-class."
            )
        label = f"{prefix}{next_ids[cat_id]}"
        next_ids[cat_id] += 1
        return label

    for frame in frame_names:
        frame_path = args.frames_dir / frame
        if not frame_path.exists():
            results[frame] = []
            continue
        image = Image.open(frame_path).convert("RGB")

        # predict step
        for cat_tracks in tracks.values():
            for trk in cat_tracks:
                trk["mean"], trk["covariance"] = kf.predict(trk["mean"], trk["covariance"])
                trk["bbox"] = clamp_bbox(measure_to_bbox(trk["mean"][:4]))

        detections = frame_entries[frame].get("detections", [])
        outputs_this_frame = []

        # process per category
        for category_id, prefix in class_map.items():
            dets = [d for d in detections if d.get("category") == category_id and d.get("conf", 0.0) >= args.conf_threshold]
            if not dets:
                for trk in tracks[category_id]:
                    trk["missed"] += 1
                continue
            if args.max_detections_per_frame:
                dets = dets[: args.max_detections_per_frame]

            embeddings = [embed_crop(image, det["bbox"], descriptor, transform) for det in dets]
            measures = [bbox_to_measure(det["bbox"]) for det in dets]
            confidences = [det["conf"] for det in dets]

            active_tracks = [trk for trk in tracks[category_id] if trk["missed"] <= args.max_missed]
            cost_matrix = np.full((len(active_tracks), len(dets)), fill_value=1e3, dtype=np.float32)

            for i, trk in enumerate(active_tracks):
                for j, (emb, meas) in enumerate(zip(embeddings, measures)):
                    gating = kf.gating_distance(trk["mean"], trk["covariance"], meas)
                    if gating > args.gating_threshold:
                        continue
                    appearance = 1 - float(np.clip(np.dot(trk["embedding"], emb), -1, 1)) if trk["embedding"] is not None else 1
                    pred_bbox = trk["bbox"] if trk["bbox"] is not None else clamp_bbox(measure_to_bbox(trk["mean"][:4]))
                    det_bbox = clamp_bbox(measure_to_bbox(meas))
                    inter_x1 = max(pred_bbox[0], det_bbox[0])
                    inter_y1 = max(pred_bbox[1], det_bbox[1])
                    inter_x2 = min(pred_bbox[0] + pred_bbox[2], det_bbox[0] + det_bbox[2])
                    inter_y2 = min(pred_bbox[1] + pred_bbox[3], det_bbox[1] + det_bbox[3])
                    inter_w = max(0.0, inter_x2 - inter_x1)
                    inter_h = max(0.0, inter_y2 - inter_y1)
                    inter = inter_w * inter_h
                    union = pred_bbox[2] * pred_bbox[3] + det_bbox[2] * det_bbox[3] - inter
                    spatial = 1 - (inter / union if union > 0 else 0)
                    cost_matrix[i, j] = args.appearance_weight * appearance + (1 - args.appearance_weight) * spatial

            assigned_tracks = set()
            assigned_dets = set()
            if active_tracks and dets:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] > args.cost_threshold:
                        continue
                    trk = active_tracks[r]
                    trk["mean"], trk["covariance"] = kf.update(trk["mean"], trk["covariance"], measures[c])
                    trk["embedding"] = embeddings[c]
                    trk["bbox"] = clamp_bbox(measure_to_bbox(trk["mean"][:4]))
                    trk["conf"] = confidences[c]
                    trk["missed"] = 0
                    assigned_tracks.add(trk["id"])
                    assigned_dets.add(c)

            # mark unmatched
            for trk in tracks[category_id]:
                if trk["id"] not in assigned_tracks:
                    trk["missed"] += 1

            # new tracks
            for idx in range(len(dets)):
                if idx in assigned_dets:
                    continue
                if args.max_tracks_per_class and len(tracks[category_id]) >= args.max_tracks_per_class:
                    # reuse the stalest track if any, otherwise drop detection
                    candidate = max(tracks[category_id], key=lambda t: t["missed"], default=None)
                    if candidate and candidate["missed"] > 0:
                        candidate["mean"], candidate["covariance"] = kf.initiate(measures[idx])
                        candidate["embedding"] = embeddings[idx]
                        candidate["bbox"] = clamp_bbox(measure_to_bbox(candidate["mean"][:4]))
                        candidate["conf"] = confidences[idx]
                        candidate["missed"] = 0
                        assigned_tracks.add(candidate["id"])
                        if candidate["id"] in free_labels[category_id]:
                            free_labels[category_id].discard(candidate["id"])
                        continue
                    else:
                        continue
                new_id = allocate_label(category_id, prefix)
                if new_id not in label_colors:
                    label_colors[new_id] = assign_colors()
                mean, covariance = kf.initiate(measures[idx])
                tracks[category_id].append(
                    {
                        "id": new_id,
                        "mean": mean,
                        "covariance": covariance,
                        "embedding": embeddings[idx],
                        "bbox": clamp_bbox(measure_to_bbox(mean[:4])),
                        "conf": confidences[idx],
                        "missed": 0,
                        "color": label_colors[new_id],
                    }
                )
                assigned_tracks.add(new_id)

            # prune stale
            pruned = []
            for trk in tracks[category_id]:
                if trk["missed"] <= args.max_missed:
                    pruned.append(trk)
                else:
                    if args.max_tracks_per_class:
                        free_labels[category_id].add(trk["id"])
            tracks[category_id] = pruned

            # record frame outputs
            for trk in tracks[category_id]:
                if trk["missed"] == 0 and trk["bbox"] is not None:
                    outputs_this_frame.append(
                        {
                            "category": category_id,
                            "id": trk["id"],
                            "bbox": trk["bbox"].tolist(),
                            "conf": float(trk.get("conf", 0.0)),
                        }
                    )

        results[frame] = outputs_this_frame

    # save JSON
    tracks_json = {"frames": results}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(tracks_json, f, indent=2)

    # render video
    frame_paths = [args.frames_dir / name for name in frame_names if (args.frames_dir / name).exists()]
    if not frame_paths:
        print("No frames found for video rendering; skipping.")
        return
    first_frame = cv2.imread(str(frame_paths[0]))
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )

    color_cache = label_colors.copy()
    for cat_tracks in tracks.values():
        for trk in cat_tracks:
            color_cache.setdefault(trk["id"], trk["color"])

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        for det in results.get(frame_path.name, []):
            xmin = int(det["bbox"][0] * width)
            ymin = int(det["bbox"][1] * height)
            w = int(det["bbox"][2] * width)
            h = int(det["bbox"][3] * height)
            color = color_cache.get(det["id"], (0, 255, 0))
            cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, args.box_thickness)
            label_text = f"{det['id']} {det['conf']:.2f}"
            cv2.putText(
                frame,
                label_text,
                (xmin, max(0, ymin - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                args.text_scale,
                color,
                args.text_thickness,
                cv2.LINE_AA,
            )
        writer.write(frame)

    writer.release()
    print("Saved tracking JSON:", args.output_json)
    print("Saved annotated video:", args.output_video)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track MegaDetector detections with Kalman+appearance matching.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory of extracted frames.")
    parser.add_argument("--detections-json", type=Path, required=True, help="MegaDetector JSON output.")
    parser.add_argument(
        "--descriptor-weights",
        type=Path,
        default=Path("models/MegaDescriptor-T-CNN-288/pytorch_model.bin"),
        help="Path to MegaDescriptor weights (.bin).",
    )
    parser.add_argument(
        "--class-map",
        action="append",
        default=None,
        help="CategoryID=prefix mapping (e.g. 1=deer). Repeat for multiple classes.",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.2, help="Detection confidence threshold.")
    parser.add_argument("--appearance-weight", type=float, default=0.7, help="Weight for appearance vs spatial cost.")
    parser.add_argument("--cost-threshold", type=float, default=0.85, help="Maximum assignment cost.")
    parser.add_argument("--gating-threshold", type=float, default=9.4877, help="Chi-square gating threshold (4 DOF).")
    parser.add_argument("--max-missed", type=int, default=8, help="Frames to keep a track alive without matches.")
    parser.add_argument(
        "--max-detections-per-frame",
        type=int,
        default=None,
        help="Optional limit on detections per class per frame.",
    )
    parser.add_argument(
        "--max-tracks-per-class",
        type=int,
        default=None,
        help="Optional cap on simultaneously active tracks per category.",
    )
    parser.add_argument("--fps", type=float, default=17.0, help="Output video FPS.")
    parser.add_argument("--box-thickness", type=int, default=3)
    parser.add_argument("--text-scale", type=float, default=0.7)
    parser.add_argument("--text-thickness", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True, help="Path to save track JSON.")
    parser.add_argument("--output-video", type=Path, required=True, help="Path to save annotated video.")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_tracker(args)
