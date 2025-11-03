from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).with_name("tabletop_homography.json")

DEFAULT_CONFIG = {
    "board_width_in": 24.0,
    "board_height_in": 30.0,
    # Order: top-left, top-right, bottom-right, bottom-left
    "image_points": [
        [475.0, 390.0],
        [1433.0, 424.0],
        [1914.0, 1100.0],
        [0.0, 1040.0],
    ],
}


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse {CONFIG_PATH}: {exc}") from exc
        return {**DEFAULT_CONFIG, **data}
    return DEFAULT_CONFIG.copy()


CONFIG = _load_config()

BOARD_WIDTH_IN = float(CONFIG["board_width_in"])
BOARD_HEIGHT_IN = float(CONFIG["board_height_in"])

BOARD_WIDTH_FT = BOARD_WIDTH_IN / 12.0
BOARD_HEIGHT_FT = BOARD_HEIGHT_IN / 12.0

IMAGE_POINTS = np.array(CONFIG["image_points"], dtype=np.float32)
if IMAGE_POINTS.shape != (4, 2):
    raise ValueError(
        f"Expected 4 corner image points in {CONFIG_PATH} (top-left, top-right, "
        f"bottom-right, bottom-left); got shape {IMAGE_POINTS.shape}"
    )

WORLD_POINTS_FT = np.array(
    [
        [0.0, BOARD_HEIGHT_FT],          # top-left
        [BOARD_WIDTH_FT, BOARD_HEIGHT_FT],  # top-right
        [BOARD_WIDTH_FT, 0.0],           # bottom-right
        [0.0, 0.0],                      # bottom-left
    ],
    dtype=np.float32,
)


@lru_cache(maxsize=1)
def image_to_world_homography() -> np.ndarray:
    """Return the 3x3 homography mapping image pixels → world feet."""
    H, _ = cv2.findHomography(IMAGE_POINTS, WORLD_POINTS_FT, method=0)
    if H is None:
        raise RuntimeError("Failed to compute image->world homography for tabletop.")
    return H


@lru_cache(maxsize=1)
def world_to_image_homography() -> np.ndarray:
    """Return the inverse mapping (world → image)."""
    H = image_to_world_homography()
    return np.linalg.inv(H)


def project_image_to_world(pixel_xy: np.ndarray) -> np.ndarray:
    """Project N×2 array of image pixels into world feet."""
    pts = np.asarray(pixel_xy, dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, image_to_world_homography()).reshape(-1, 2)
    return proj


def project_world_to_image(world_xy: np.ndarray) -> np.ndarray:
    """Project N×2 array of world feet into image pixel coordinates."""
    pts = np.asarray(world_xy, dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, world_to_image_homography()).reshape(-1, 2)
    return proj


def board_extent_feet() -> Tuple[float, float]:
    """Return (width_ft, height_ft)."""
    return BOARD_WIDTH_FT, BOARD_HEIGHT_FT


def as_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p.resolve()
