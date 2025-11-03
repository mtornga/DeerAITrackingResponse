#!/usr/bin/env python3
"""
Interactively capture the tabletop corner coordinates and store them in
calibration/tabletop_homography.json.

Usage examples:
    python scripts/tabletop_calibrate_corners.py --rtsp "$WYZE_TABLETOP_RTSP"
    python scripts/tabletop_calibrate_corners.py --snapshot my_frame.jpg

Click the four corners in this order:
    1. Top-left
    2. Top-right
    3. Bottom-right
    4. Bottom-left
Right-click removes the last point if you make a mistake.
Press Enter to accept the selection once all four points are marked.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


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

from env_loader import load_env_file, require_env  # noqa: E402

load_env_file()

DEFAULT_OUTPUT = REPO_ROOT / "calibration" / "tabletop_homography.json"


@dataclass
class CornerCapture:
    points: List[Tuple[float, float]]
    confirmed: bool = False

    def add(self, xy: Tuple[float, float]) -> None:
        if len(self.points) < 4:
            self.points.append(xy)

    def remove_last(self) -> None:
        if self.points:
            self.points.pop()


def draw_overlay(frame: np.ndarray, capture: CornerCapture) -> np.ndarray:
    canvas = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, (x, y) in enumerate(capture.points):
        cv2.circle(canvas, (int(x), int(y)), 6, (0, 165, 255), -1)
        cv2.putText(
            canvas,
            str(idx + 1),
            (int(x) + 8, int(y) - 8),
            font,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(idx + 1),
            (int(x) + 8, int(y) - 8),
            font,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if len(capture.points) == 4:
        pts = np.array(capture.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 200, 0), thickness=2)
    instructions = [
        "Click corners in order: top-left, top-right, bottom-right, bottom-left.",
        "Right-click: undo last point. ENTER: save. ESC: cancel.",
    ]
    y0 = 24
    for line in instructions:
        cv2.putText(canvas, line, (16, y0), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (16, y0), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y0 += 28
    return canvas


def grab_frame_from_rtsp(rtsp_url: str, timeout_sec: float = 5.0) -> np.ndarray:
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP stream: {rtsp_url}")
    start = time.time()
    frame = None
    while time.time() - start < timeout_sec:
        ok, img = cap.read()
        if ok:
            frame = img
            break
    cap.release()
    if frame is None:
        raise RuntimeError("Failed to capture frame from RTSP stream.")
    return frame


def interactive_capture(frame: np.ndarray) -> List[Tuple[float, float]]:
    capture = CornerCapture(points=[])

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(capture.points) < 4:
            capture.add((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            capture.remove_last()

    window = "Tabletop Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        overlay = draw_overlay(frame, capture)
        cv2.imshow(window, overlay)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            capture.confirmed = False
            break
        if key in (13, 10):  # Enter
            if len(capture.points) == 4:
                capture.confirmed = True
                break
        # allow space bar as alternate confirm
        if key == ord(" "):
            if len(capture.points) == 4:
                capture.confirmed = True
                break

    cv2.destroyWindow(window)
    if not capture.confirmed or len(capture.points) != 4:
        raise RuntimeError("Corner selection cancelled or incomplete.")
    return capture.points


def write_config(
    output_path: Path,
    board_width_in: float,
    board_height_in: float,
    image_points: List[Tuple[float, float]],
) -> None:
    data = {
        "board_width_in": float(board_width_in),
        "board_height_in": float(board_height_in),
        "image_points": [[float(x), float(y)] for (x, y) in image_points],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    print(f"[info] Wrote calibration to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactively mark tabletop corners for homography calibration.")
    parser.add_argument("--snapshot", type=str, help="Path to a saved image to annotate instead of an RTSP capture.")
    parser.add_argument("--rtsp", type=str, help="RTSP URL for live capture (defaults to WYZE_TABLETOP_RTSP).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination JSON file.")
    parser.add_argument("--board-width", type=float, default=24.0, help="Board width in inches.")
    parser.add_argument("--board-height", type=float, default=30.0, help="Board height in inches.")
    args = parser.parse_args()

    if args.snapshot:
        frame = cv2.imread(args.snapshot)
        if frame is None:
            raise SystemExit(f"Failed to load snapshot image: {args.snapshot}")
    else:
        rtsp_url = args.rtsp or require_env("WYZE_TABLETOP_RTSP")
        frame = grab_frame_from_rtsp(rtsp_url)

    try:
        points = interactive_capture(frame)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    write_config(args.output, args.board_width, args.board_height, points)
    print("[info] Remember to restart demo/topdown_tracker.py to use the new calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
