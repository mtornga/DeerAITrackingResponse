"""Shared helpers for the deer-vision detector workflow."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Sequence

import cv2


IMAGE_EXTENSIONS: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class YoloAnnotation:
    """Single YOLO-format annotation."""

    cls: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float | None = None

    def as_line(self) -> str:
        parts = [
            str(self.cls),
            f"{self.x_center:.6f}",
            f"{self.y_center:.6f}",
            f"{self.width:.6f}",
            f"{self.height:.6f}",
        ]
        if self.confidence is not None:
            parts.append(f"{self.confidence:.4f}")
        return " ".join(parts)


def iter_image_files(root: Path) -> Generator[Path, None, None]:
    """Yield image files underneath root."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_yolo_file(path: Path) -> List[YoloAnnotation]:
    """Parse YOLO-format labels, returning an empty list if the file is missing."""
    if not path.exists():
        return []
    annotations: List[YoloAnnotation] = []
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 5:
                raise ValueError(f"Malformed label line in {path}: '{raw}'")
            cls, x_c, y_c, w, h, *rest = parts
            conf = float(rest[0]) if rest else None
            annotations.append(
                YoloAnnotation(
                    cls=int(cls),
                    x_center=float(x_c),
                    y_center=float(y_c),
                    width=float(w),
                    height=float(h),
                    confidence=conf,
                )
            )
    return annotations


def save_yolo_file(path: Path, annotations: Iterable[YoloAnnotation]) -> None:
    """Write YOLO-format labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for ann in annotations:
            handle.write(f"{ann.as_line()}\n")


def ensure_relative(path: Path, root: Path) -> Path:
    """Return path relative to root, raising if that is not possible."""
    try:
        return path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{path} is not inside {root}") from exc


def copy_image(src: Path, dst: Path) -> None:
    """Copy an image preserving metadata."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) for the image."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Unable to read image {path}")
    height, width = img.shape[:2]
    return width, height


def yolo_to_xyxy(box: YoloAnnotation, width: int, height: int) -> tuple[float, float, float, float]:
    """Convert YOLO normalized box to top-left/bottom-right pixel coordinates."""
    box_width = box.width * width
    box_height = box.height * height
    center_x = box.x_center * width
    center_y = box.y_center * height
    x0 = center_x - box_width / 2
    y0 = center_y - box_height / 2
    x1 = center_x + box_width / 2
    y1 = center_y + box_height / 2
    return x0, y0, x1, y1


def xyxy_to_yolo(coords: tuple[float, float, float, float], width: int, height: int, cls: int) -> YoloAnnotation:
    """Convert absolute pixel coords back to a YOLO annotation."""
    x0, y0, x1, y1 = coords
    box_width = max(x1 - x0, 0.0)
    box_height = max(y1 - y0, 0.0)
    center_x = x0 + box_width / 2
    center_y = y0 + box_height / 2
    return YoloAnnotation(
        cls=cls,
        x_center=(center_x / width) if width else 0.0,
        y_center=(center_y / height) if height else 0.0,
        width=(box_width / width) if width else 0.0,
        height=(box_height / height) if height else 0.0,
    )


def clip_box_to_window(box: tuple[float, float, float, float], window: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Clip a bounding box to the provided window; may return an invalid box if there is no overlap."""
    x0, y0, x1, y1 = box
    wx0, wy0, wx1, wy1 = window
    clipped = (max(x0, wx0), max(y0, wy0), min(x1, wx1), min(y1, wy1))
    return clipped


def box_area(box: tuple[float, float, float, float]) -> float:
    """Compute area of a box."""
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union between two boxes."""
    intersection = clip_box_to_window(box_a, box_b)
    inter_area = box_area(intersection)
    if inter_area <= 0:
        return 0.0
    return inter_area / (box_area(box_a) + box_area(box_b) - inter_area)
