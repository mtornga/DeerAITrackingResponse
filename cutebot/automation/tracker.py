from __future__ import annotations

import asyncio
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from calibration.tabletop_geometry import image_to_world_homography
from env_loader import load_env_file, require_env


def _load_calibration_matrix(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    matrix = np.asarray(data.get("matrix"), dtype=np.float32)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    if matrix.shape != (3, 3):
        raise ValueError(f"Calibration matrix must be 3x3. Got shape={matrix.shape!r}")
    return matrix


def _apply_homography(matrix: np.ndarray, point_xy: np.ndarray) -> np.ndarray:
    vec = np.array([point_xy[0], point_xy[1], 1.0], dtype=np.float32)
    world = matrix @ vec
    if world[2] == 0.0:
        raise ZeroDivisionError("Homography projection produced zero w component.")
    return world[:2] / world[2]


@dataclass
class CutebotPose:
    x_ft: float
    y_ft: float
    confidence: float
    timestamp: float

    @property
    def x_in(self) -> float:
        return self.x_ft * 12.0

    @property
    def y_in(self) -> float:
        return self.y_ft * 12.0

    def as_inches(self) -> tuple[float, float]:
        return self.x_in, self.y_in


class TopDownCutebotTracker:
    """
    Helper that reuses the YOLO + calibration stack to localise the Cutebot.
    """

    def __init__(
        self,
        *,
        model_path: Path | str = Path("runs/detect/ultraYOLODetection1_v13/weights/best.pt"),
        calibration_path: Path | str | None = None,
        csv_path: Path | str = Path("detections_world.csv"),
        rtsp_url: Optional[str] = None,
        target_class: str = "cutebot",
        conf: float = 0.25,
        iou: float = 0.45,
        backend: str = "csv",
    ) -> None:
        load_env_file()
        self.model_path = Path(model_path)
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.csv_path = Path(csv_path)
        self.rtsp_url = rtsp_url
        self.target_class = target_class.lower()
        self.conf = conf
        self.iou = iou
        backend = backend.lower().strip()
        if backend not in {"csv", "yolo"}:
            raise ValueError("backend must be 'csv' or 'yolo'")
        self.backend = backend

        self._model = None
        self._matrix: Optional[np.ndarray] = None
        self._capture: Optional[cv2.VideoCapture] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_csv_sig: Optional[tuple] = None

    @property
    def camera_source(self) -> str:
        if self.rtsp_url:
            return self.rtsp_url
        return require_env("WYZE_TABLETOP_RTSP")

    def start(self) -> None:
        if self.backend == "csv":
            return

        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        if self.calibration_path and not self.calibration_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {self.calibration_path}")

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Ultralytics YOLO is unavailable. Install it or run the tracker with "
                "--tracker-backend csv while demo/topdown_tracker.py is logging detections."
            ) from exc

        self._model = YOLO(str(self.model_path))
        if self.calibration_path:
            self._matrix = _load_calibration_matrix(self.calibration_path)
        else:
            self._matrix = image_to_world_homography()

        source = self.camera_source
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open tabletop source: {source}")
        self._capture = capture

    def close(self) -> None:
        if self.backend == "csv":
            self._last_csv_sig = None
            return

        if self._capture:
            self._capture.release()
        self._capture = None
        self._model = None
        self._matrix = None
        self._last_frame = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _grab_frame(self) -> Optional[np.ndarray]:
        if not self._capture:
            raise RuntimeError("Tracker not started. Call start() first.")
        ok, frame = self._capture.read()
        if not ok:
            return None
        self._last_frame = frame
        return frame

    def detect_once(self) -> Optional[CutebotPose]:
        if self.backend == "csv":
            return self._detect_from_csv()

        if self._model is None or self._matrix is None:
            raise RuntimeError("Tracker not started. Call start() first.")

        frame = self._grab_frame()
        if frame is None:
            return None

        results = self._model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        if not results:
            return None
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None

        boxes = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        names = self._model.names

        best_pose: Optional[CutebotPose] = None
        for xyxy, cls_idx, score in zip(boxes, cls, confs):
            label = names.get(int(cls_idx), str(cls_idx)).lower()
            if label != self.target_class:
                continue
            x1, y1, x2, y2 = map(float, xyxy)
            pixel_bottom_center = np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)
            world_xy = _apply_homography(self._matrix, pixel_bottom_center)
            pose = CutebotPose(
                x_ft=float(world_xy[0]),
                y_ft=float(world_xy[1]),
                confidence=float(score),
                timestamp=time.time(),
            )
            if best_pose is None or pose.confidence > best_pose.confidence:
                best_pose = pose
        return best_pose

    def _detect_from_csv(self) -> Optional[CutebotPose]:
        if not self.csv_path.exists():
            return None

        latest_row = None
        try:
            with self.csv_path.open("r", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if row.get("class", "").lower() == self.target_class:
                        latest_row = row
        except Exception:
            return None

        if not latest_row:
            return None

        try:
            x_ft = float(latest_row.get("world_x_ft"))
            y_ft = float(latest_row.get("world_y_ft"))
            conf = float(latest_row.get("conf", 1.0))
            ts_val = latest_row.get("ts")
            ts = float(ts_val) if ts_val not in (None, "") else time.time()
            track_id = latest_row.get("track_id")
        except (TypeError, ValueError):
            return None

        signature = (
            int(ts) if ts_val not in (None, "") else int(time.time()),
            track_id,
            round(x_ft, 4),
            round(y_ft, 4),
        )
        if self._last_csv_sig == signature:
            return None
        self._last_csv_sig = signature
        return CutebotPose(
            x_ft=x_ft,
            y_ft=y_ft,
            confidence=conf,
            timestamp=ts,
        )

    async def get_pose(
        self,
        *,
        retries: int = 6,
        delay_sec: float = 0.25,
        min_confidence: float = 0.0,
    ) -> CutebotPose:
        """
        Attempt to detect the Cutebot pose, retrying a few frames if necessary.
        """
        for attempt in range(retries):
            pose = await asyncio.to_thread(self.detect_once)
            if pose and pose.confidence >= min_confidence:
                return pose
            await asyncio.sleep(delay_sec)
        raise RuntimeError("Failed to detect Cutebot after multiple attempts.")

    @property
    def last_frame(self) -> Optional[np.ndarray]:
        return self._last_frame
