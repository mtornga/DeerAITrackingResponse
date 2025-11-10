#!/usr/bin/env python3
"""Run the Kalman + embedding tracker described in docs/deer_tracking_pipeline.md."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
import timm
from torchvision import transforms


TrackID = str


def load_descriptor(weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Load the MegaDescriptor EfficientNet-B3 backbone."""
    weights = torch.load(weights_path, map_location=device)
    if isinstance(weights, dict) and "model" in weights:
        weights = weights["model"]
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model.to(device)


def build_preprocess() -> transforms.Compose:
    """Return the descriptor pre-processing pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((288, 288)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return 1 - cosine similarity for two embeddings."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU for two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter = inter_w * inter_h
    area_a = max(0.0, (box_a[2] - box_a[0])) * max(0.0, (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0])) * max(0.0, (box_b[3] - box_b[1]))
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


class KalmanBoxTracker:
    """Constant-velocity Kalman filter operating on [cx, cy, w, h]."""

    def __init__(self) -> None:
        dt = 1.0
        self._state_dim = 8
        self._meas_dim = 4
        self.F = np.eye(self._state_dim, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = dt
        self.H = np.zeros((self._meas_dim, self._state_dim), dtype=np.float32)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0
        self.Q = np.eye(self._state_dim, dtype=np.float32) * 1e-2
        self.R = np.eye(self._meas_dim, dtype=np.float32) * 10.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        state = np.zeros((self._state_dim, 1), dtype=np.float32)
        state[:4, 0] = measurement.reshape(-1)
        covariance = np.eye(self._state_dim, dtype=np.float32)
        covariance[:4, :4] *= 10.0
        covariance[4:, 4:] *= 1000.0
        return state, covariance

    def predict(self, state: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        state = self.F @ state
        covariance = self.F @ covariance @ self.F.T + self.Q
        return state, covariance

    def update(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        z = measurement.reshape(-1, 1)
        S = self.H @ covariance @ self.H.T + self.R
        K = covariance @ self.H.T @ np.linalg.inv(S)
        y = z - (self.H @ state)
        state = state + K @ y
        covariance = (np.eye(self._state_dim) - K @ self.H) @ covariance
        return state, covariance

    def gating_distance(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> float:
        z = measurement.reshape(-1, 1)
        S = self.H @ covariance @ self.H.T + self.R
        y = z - (self.H @ state)
        maha = (y.T @ np.linalg.inv(S) @ y).item()
        return float(maha)


@dataclass
class Track:
    """Container for a single deer track."""

    track_id: TrackID
    state: np.ndarray
    covariance: np.ndarray
    embedding: Optional[np.ndarray] = None
    missed: int = 0
    last_conf: float = 0.0
    history: List[Tuple[int, np.ndarray]] = field(default_factory=list)

    def bounding_box(self) -> np.ndarray:
        """Return [x1,y1,x2,y2] in pixels from the current state."""
        cx, cy, w, h = self.state[:4, 0]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def measurement_vector(self) -> np.ndarray:
        return self.state[:4, 0]


def load_detections(detections_json: Path) -> Dict[str, List[Dict]]:
    """Load MegaDetector JSON into a filename -> detections map."""
    payload = json.loads(detections_json.read_text())
    mapping: Dict[str, List[Dict]] = {}
    for img in payload.get("images", []):
        mapping[img["file"]] = img.get("detections", [])
    return mapping


def compute_embedding(
    descriptor: torch.nn.Module,
    preprocess: transforms.Compose,
    device: torch.device,
    crop_bgr: np.ndarray,
) -> np.ndarray:
    """Return a 1536-D descriptor for the provided crop."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)
    tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = descriptor(tensor).cpu().numpy().reshape(-1)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    return embedding


def clamp_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp box coordinates to the image bounds."""
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def normalize_bbox(box: np.ndarray, width: int, height: int) -> List[float]:
    """Convert [x1,y1,x2,y2] pixels to YOLO-normalized [cx,cy,w,h]."""
    cx = ((box[0] + box[2]) / 2.0) / width
    cy = ((box[1] + box[3]) / 2.0) / height
    w = (box[2] - box[0]) / width
    h = (box[3] - box[1]) / height
    return [float(cx), float(cy), float(w), float(h)]


def draw_tracks(
    frame: np.ndarray,
    tracks: Sequence[Track],
    font_color=(0, 255, 0),
) -> np.ndarray:
    """Render the tracked boxes on the frame."""
    out = frame.copy()
    for track in tracks:
        box = track.bounding_box()
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), font_color, 2)
        label = f"{track.track_id} ({track.last_conf:.2f})"
        cv2.putText(out, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, font_color, 2)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track deer using Kalman + appearance embeddings.")
    parser.add_argument("--frames", required=True, type=Path, help="Directory of frames extracted by MegaDetector.")
    parser.add_argument("--detections", required=True, type=Path, help="MegaDetector JSON file for the clip.")
    parser.add_argument(
        "--descriptor",
        required=True,
        type=Path,
        help="Path to MegaDescriptor weights (pytorch_model.bin).",
    )
    parser.add_argument("--output-json", required=True, type=Path, help="Where to store tracking JSON.")
    parser.add_argument("--output-video", type=Path, help="Optional annotated MP4 output path.")
    parser.add_argument("--max-tracks", type=int, default=2, help="Maximum simultaneous tracks.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Embedding vs IoU weighting factor.")
    parser.add_argument("--max-missed", type=int, default=8, help="Frames before a track is dropped.")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="MegaDetector conf minimum.")
    parser.add_argument("--top-k", type=int, default=2, help="Detections to keep per frame.")
    parser.add_argument("--gating-threshold", type=float, default=16.0, help="Chi-square gating threshold.")
    parser.add_argument("--device", default="cpu", help="Torch device (e.g. cpu or mps).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames_dir = args.frames
    assert frames_dir.is_dir(), f"{frames_dir} does not exist"

    detections_map = load_detections(args.detections)

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No JPG frames found in {frames_dir}")

    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        raise RuntimeError(f"Unable to read {frame_paths[0]}")
    height, width = sample.shape[:2]

    device = torch.device(args.device)
    descriptor = load_descriptor(args.descriptor, device)
    preprocess = build_preprocess()
    kf = KalmanBoxTracker()

    tracks: List[Track] = []
    track_ids = [f"deer{i+1}" for i in range(args.max_tracks)]
    available_ids = track_ids.copy()
    frame_results: List[Dict] = []

    video_writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(args.output_video), fourcc, 15, (width, height))

    for frame_idx, frame_path in enumerate(frame_paths):
        frame_name = frame_path.name
        detections = detections_map.get(frame_name, [])
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        det_entries: List[Dict] = []
        for det in detections:
            if det.get("category") != "1":
                continue
            if float(det.get("conf", 0.0)) < args.conf_threshold:
                continue
            cx = det["bbox"][0] * width
            cy = det["bbox"][1] * height
            w = det["bbox"][2] * width
            h = det["bbox"][3] * height
            box = np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)
            det_entries.append(
                {
                    "conf": float(det["conf"]),
                    "measurement": np.array([cx, cy, w, h], dtype=np.float32),
                    "bbox": box,
                }
            )

        det_entries.sort(key=lambda d: d["conf"], reverse=True)
        det_entries = det_entries[: args.top_k]

        for track in tracks:
            track.state, track.covariance = kf.predict(track.state, track.covariance)

        for track in tracks:
            track.history.append((frame_idx, track.bounding_box().copy()))

        if det_entries and tracks:
            embeddings = []
            for det in det_entries:
                x1, y1, x2, y2 = clamp_bbox(*det["bbox"], width, height)
                crop = img[y1:y2, x1:x2]
                emb = compute_embedding(descriptor, preprocess, device, crop)
                embeddings.append(emb)
                det["embedding"] = emb

            cost_matrix = np.full((len(tracks), len(det_entries)), fill_value=1e6, dtype=np.float32)
            for i, track in enumerate(tracks):
                track_box = track.bounding_box()
                for j, det in enumerate(det_entries):
                    gating = kf.gating_distance(track.state, track.covariance, det["measurement"])
                    if gating > args.gating_threshold:
                        continue
                    embedding_cost = 1.0
                    if track.embedding is not None and "embedding" in det:
                        embedding_cost = cosine_distance(track.embedding, det["embedding"])
                    iou_cost = 1.0 - bbox_iou(track_box, det["bbox"])
                    cost = args.alpha * embedding_cost + (1.0 - args.alpha) * iou_cost
                    cost_matrix[i, j] = cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_tracks = set()
            matched_dets = set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] >= 1e5:
                    continue
                track = tracks[r]
                det = det_entries[c]
                track.state, track.covariance = kf.update(track.state, track.covariance, det["measurement"])
                track.embedding = det.get("embedding")
                track.missed = 0
                track.last_conf = det["conf"]
                matched_tracks.add(r)
                matched_dets.add(c)

        else:
            matched_tracks = set()
            matched_dets = set()

        for idx, track in enumerate(tracks):
            if idx not in matched_tracks:
                track.missed += 1

        tracks = [track for track in tracks if track.missed <= args.max_missed]
        available_ids = [tid for tid in track_ids if tid not in {t.track_id for t in tracks}]

        for idx, det in enumerate(det_entries):
            if idx in matched_dets:
                continue
            if not available_ids:
                continue
            new_id = available_ids.pop(0)
            state, covariance = kf.initiate(det["measurement"])
            track = Track(
                track_id=new_id,
                state=state,
                covariance=covariance,
                embedding=det.get("embedding"),
                last_conf=det["conf"],
            )
            tracks.append(track)

        frame_tracks = []
        for track in tracks:
            bbox_norm = normalize_bbox(track.bounding_box(), width, height)
            frame_tracks.append(
                {"id": track.track_id, "bbox": bbox_norm, "confidence": track.last_conf}
            )

        frame_results.append({"frame": frame_name, "tracks": frame_tracks})

        if video_writer:
            annotated = draw_tracks(img, tracks)
            video_writer.write(annotated)

    if video_writer:
        video_writer.release()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "frames_dir": str(frames_dir),
            "detections": str(args.detections),
            "descriptor": str(args.descriptor),
            "image_size": {"width": width, "height": height},
        },
        "frames": frame_results,
    }
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote tracking results to {args.output_json}")
    if args.output_video:
        print(f"Annotated video saved to {args.output_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
