"""Tools for querying GPT-5 with the Reolink E1 snapshot.

This module loads the static `ReolinkE1Test.jpg` image, sends it to the
`gpt-5` model, and returns a structured description of the CuteBot pose.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
from openai import OpenAI

from env_loader import require_env


# Board dimensions in pixels taken directly from the reference test image.
REOLINK_REFERENCE_IMAGE = Path(__file__).with_name("ReolinkE1Test.jpg")
IMAGE_WIDTH_PIXELS = 2560
IMAGE_HEIGHT_PIXELS = 1920

BOARD_LENGTH_INCHES = 30.0  # distance from top (plants) to bottom edges in inches
BOARD_WIDTH_INCHES = 24.0  # distance from left to right edges in inches

SYSTEM_PROMPT = (
    "You are a vision assistant that estimates the CuteBot pose on a tabletop robot arena. "
    "The camera is positioned on a wall about 32 inches above the tabletop looking down."
    "Reolink E1 frame (Y increases to the right, X increases downward). Only consider the black playing surface; ignore the surrounding cardboard when locating corners."
)

USER_PROMPT = (
    "Identify the CuteBot robot (white body with blue chassis) and estimate the pose of the very tip of "
    "its nose. The robot now carries a 3/4‑inch square black‑and‑white AprilTag (Tag36h11, ID 0) centered on the top front deck; when visible, use the tag’s center as the nose location."
    "The nose is the forward-most point on the white bumper directly beneath the green arrow tip; "
    "it sits centrally between the headlights and is on the opposite side of the two big black wheels. The black rectangle sitting on top of "
    "the cardboard is the active board surface; it measures exactly 24 inches wide (left-to-right) by "
    "30 inches long (back-to-front). The board is perfectly flat and rectangular. It is angled to match the compass points, and the camera is not directly above the board, which both distort the view."
    " The labelled corners refer to the inch coordinated of the black rectangle only: "
    "top-left = (0, 0), top-right = (0, 24), bottom-left = (30, 0), bottom-right = (30, 24).  (0,12), (15,24), (30,12), and (15,0) are midpoints of the edges of the rectangle"
    " There is no label but the center of the board is (15,12) and is marked with a green tape plus sign." 
    "In this inch coordinate system, x increases from the back edge toward the camera (top to bottom of "
    "the image) and y increases from the board's left edge to the right edge. "
    
    "\n"
    "Work through these steps internally (do not expose them in the final output):\n"
    "1. Carefully locate the inch coordinates of the four black board corners listed above, as well as the 4 edge midpoints. Locate the green plus sign in the board center as well. Make thorough use of the 8 coordinate tags"
     "surrounding the board plus the marker at board center to sanity check your estimated location. \n"
    "2. Using these landmarks, determine the inch coordinate system location of the CuteBot nose tip while respecting the 30 in by 24 in dimensions. Focus on the AprilTag or bumper centre under the arrow tip (not the wheels) to locate the bot's nose. "
    "\n"
    "Respond with a JSON object containing exactly these fields:\n"
    "coordinate_system: always set to \"inches\".\n"
    "origin: short string describing the origin for operators.\n"
    "cutebot_nose_inches: object with numeric x and y location for the nose tip in board inches.\n"
    "confidence: numeric value between 0 and 1.\n"
    "notes: short string with reasoning, observed ambiguities, and suggestions to make your visual locating of cutebot more accurate in future iterations.\n"
    "Return only the JSON object—no additional text or Markdown."
)


@dataclass
class CutebotObservation:
    """Structured pose estimate returned by the vision model."""

    coordinate_system: str
    origin: str
    units: str
    board_corners_pixels: Dict[str, Dict[str, float]]
    cutebot_nose_pixels_x: Optional[float]
    cutebot_nose_pixels_y: Optional[float]
    cutebot_nose_inches_x: float
    cutebot_nose_inches_y: float
    cutebot_nose_inches_projected_x: float
    cutebot_nose_inches_projected_y: float
    arrow_tail_pixels_x: Optional[float]
    arrow_tail_pixels_y: Optional[float]
    arrow_tip_pixels_x: Optional[float]
    arrow_tip_pixels_y: Optional[float]
    arrow_heading_degrees: Optional[float]
    heading_model_degrees: float
    heading_degrees: float
    confidence: float
    notes: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CutebotObservation":
        """Validate and normalise a JSON payload from the LLM."""
        try:
            coordinate_system = str(payload["coordinate_system"])
            origin = str(payload["origin"])
        except KeyError as exc:
            raise ValueError(f"Missing required field: {exc}") from exc

        units = str(payload.get("units", "inches"))

        corners: Dict[str, Dict[str, float]] = {}
        corners_node = payload.get("board_corners_pixels")
        if isinstance(corners_node, Mapping):
            for corner_name in ("top_left", "top_right", "bottom_left", "bottom_right"):
                corner_value = corners_node.get(corner_name)
                if not isinstance(corner_value, Mapping):
                    continue
                try:
                    corner_x = float(corner_value["x"])
                    corner_y = float(corner_value["y"])
                except (KeyError, TypeError, ValueError):
                    continue
                corners[corner_name] = {"x": corner_x, "y": corner_y}

        nose_pixels_x: Optional[float] = None
        nose_pixels_y: Optional[float] = None
        nose_pixels_node = payload.get("cutebot_nose_pixels")
        if isinstance(nose_pixels_node, Mapping):
            try:
                nose_pixels_x = float(nose_pixels_node["x"])
                nose_pixels_y = float(nose_pixels_node["y"])
            except (KeyError, TypeError, ValueError):
                nose_pixels_x = nose_pixels_y = None

        nose_inches_node = payload.get("cutebot_nose_inches")
        if not isinstance(nose_inches_node, Mapping):
            raise ValueError("cutebot_nose_inches must be an object with x and y numbers")
        try:
            nose_in_x = float(nose_inches_node["x"])
            nose_in_y = float(nose_inches_node["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("cutebot_nose_inches must contain numeric x and y") from exc

        projected_node = payload.get("cutebot_nose_inches_projected") or nose_inches_node
        if isinstance(projected_node, Mapping):
            try:
                projected_x = float(projected_node["x"])
                projected_y = float(projected_node["y"])
            except (KeyError, TypeError, ValueError):
                projected_x, projected_y = nose_in_x, nose_in_y
        else:
            projected_x, projected_y = nose_in_x, nose_in_y

        confidence_raw = payload.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be numeric") from exc

        notes = str(payload.get("notes", ""))

        aspect_ratio_error = 0.0
        if len(corners) == 4:
            try:
                aspect_ratio_error = _validate_corner_geometry(corners)
            except Exception:
                aspect_ratio_error = 0.0

        arrow_tail = _normalise_optional_point(payload.get("arrow_tail_pixels"))
        arrow_tip = _normalise_optional_point(payload.get("arrow_tip_pixels"))
        arrow_heading = None
        if arrow_tail and arrow_tip:
            try:
                arrow_heading = _compute_heading_from_arrow(
                    arrow_tail[0], arrow_tail[1], arrow_tip[0], arrow_tip[1]
                )
            except ValueError:
                arrow_heading = None

        heading_model_raw = payload.get("heading_model_degrees")
        heading_raw = payload.get("heading_degrees")
        try:
            heading_model = float(heading_model_raw) if heading_model_raw is not None else 0.0
        except (TypeError, ValueError):
            heading_model = 0.0
        try:
            heading = float(heading_raw) if heading_raw is not None else heading_model
        except (TypeError, ValueError):
            heading = heading_model
        if arrow_heading is not None:
            heading = arrow_heading

        return cls(
            coordinate_system=coordinate_system,
            origin=origin,
            units=units,
            board_corners_pixels=corners,
            cutebot_nose_pixels_x=nose_pixels_x,
            cutebot_nose_pixels_y=nose_pixels_y,
            cutebot_nose_inches_x=nose_in_x,
            cutebot_nose_inches_y=nose_in_y,
            cutebot_nose_inches_projected_x=projected_x,
            cutebot_nose_inches_projected_y=projected_y,
            arrow_tail_pixels_x=arrow_tail[0] if arrow_tail else None,
            arrow_tail_pixels_y=arrow_tail[1] if arrow_tail else None,
            arrow_tip_pixels_x=arrow_tip[0] if arrow_tip else None,
            arrow_tip_pixels_y=arrow_tip[1] if arrow_tip else None,
            arrow_heading_degrees=arrow_heading,
            heading_model_degrees=heading_model,
            heading_degrees=heading,
            confidence=confidence,
            notes=_append_geometry_notes(notes, aspect_ratio_error, arrow_heading, heading_model),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return the observation as a JSON-serialisable dictionary."""
        nose_payload = (
            {"x": self.cutebot_nose_pixels_x, "y": self.cutebot_nose_pixels_y}
            if self.cutebot_nose_pixels_x is not None and self.cutebot_nose_pixels_y is not None
            else None
        )

        return {
            "coordinate_system": self.coordinate_system,
            "origin": self.origin,
            "units": self.units,
            "board_corners_pixels": self.board_corners_pixels,
            "cutebot_nose_pixels": nose_payload,
            "cutebot_nose_inches": {"x": self.cutebot_nose_inches_x, "y": self.cutebot_nose_inches_y},
            "cutebot_nose_inches_projected": {
                "x": self.cutebot_nose_inches_projected_x,
                "y": self.cutebot_nose_inches_projected_y,
            },
            "arrow_tail_pixels": None
            if self.arrow_tail_pixels_x is None
            else {"x": self.arrow_tail_pixels_x, "y": self.arrow_tail_pixels_y},
            "arrow_tip_pixels": None
            if self.arrow_tip_pixels_x is None
            else {"x": self.arrow_tip_pixels_x, "y": self.arrow_tip_pixels_y},
            "arrow_heading_degrees": self.arrow_heading_degrees,
            "heading_model_degrees": self.heading_model_degrees,
            "heading_degrees": self.heading_degrees,
            "confidence": self.confidence,
            "notes": self.notes,
        }


def _encode_image_as_data_url(image_path: Path) -> str:
    """Load the image file and return a data URL string acceptable by the API."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _create_client() -> OpenAI:
    """Instantiate the OpenAI client using the project environment configuration."""
    api_key = require_env("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def query_cutebot_pose(image_path: Path = REOLINK_REFERENCE_IMAGE, client: Optional[OpenAI] = None) -> CutebotObservation:
    """Send the provided image to gpt-5 and parse the structured pose estimate."""
    image_data_url = _encode_image_as_data_url(image_path)
    client = client or _create_client()

    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": USER_PROMPT},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            },
        ],
    )

    response_text = response.output_text
    if not response_text and response.output:
        # Fallback for SDK versions that expose content segments only.
        segments = []
        for item in response.output:
            if item.type != "message":
                continue
            for content_item in item.content:
                if getattr(content_item, "type", None) == "output_text" and getattr(
                    content_item, "text", None
                ):
                    segments.append(content_item.text)
        response_text = "\n".join(segments)

    if not response_text:
        raise RuntimeError("Model response did not include any text output")
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Model response was not valid JSON") from exc

    return CutebotObservation.from_dict(payload)


def _validate_corner_geometry(corners: Mapping[str, Mapping[str, float]]) -> float:
    """Return the absolute difference between observed and expected aspect ratios."""
    top_left = corners["top_left"]
    top_right = corners["top_right"]
    bottom_left = corners["bottom_left"]
    bottom_right = corners["bottom_right"]

    width_top = math.hypot(top_right["x"] - top_left["x"], top_right["y"] - top_left["y"])
    width_bottom = math.hypot(bottom_right["x"] - bottom_left["x"], bottom_right["y"] - bottom_left["y"])
    height_left = math.hypot(bottom_left["x"] - top_left["x"], bottom_left["y"] - top_left["y"])
    height_right = math.hypot(bottom_right["x"] - top_right["x"], bottom_right["y"] - top_right["y"])

    width = max((width_top + width_bottom) / 2.0, 1e-6)
    height = (height_left + height_right) / 2.0
    expected_ratio = BOARD_LENGTH_INCHES / BOARD_WIDTH_INCHES
    observed_ratio = height / width
    return abs(observed_ratio - expected_ratio)


def _append_geometry_notes(
    notes: str, aspect_ratio_error: float, arrow_heading: Optional[float], model_heading: float
) -> str:
    extra_parts = []
    if aspect_ratio_error > 0.1:
        extra_parts.append(f"corner_ratio_delta≈{aspect_ratio_error:.2f}")
    if arrow_heading is not None:
        extra_parts.append(f"arrow_heading≈{arrow_heading:.1f}°")
        delta = min((arrow_heading - model_heading) % 360, (model_heading - arrow_heading) % 360)
        if delta > 5.0:
            extra_parts.append(f"arrow_vs_model≈{delta:.1f}°")
    if not extra_parts:
        return notes
    prefix = f"{notes} | " if notes else ""
    return prefix + "; ".join(extra_parts)


def _normalise_optional_point(point: Any) -> Optional[tuple[float, float]]:
    if not isinstance(point, Mapping):
        return None
    try:
        x = float(point["x"])
        y = float(point["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0.0 <= x <= IMAGE_WIDTH_PIXELS and 0.0 <= y <= IMAGE_HEIGHT_PIXELS):
        return None
    return (x, y)


def _compute_heading_from_arrow(tail_x: float, tail_y: float, tip_x: float, tip_y: float) -> float:
    dx = tip_x - tail_x
    dy = tip_y - tail_y
    if dx == 0.0 and dy == 0.0:
        raise ValueError("Arrow vector has zero length")
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0.0:
        angle += 360.0
    return angle


def _project_nose_inches(
    corners: Mapping[str, Mapping[str, float]], nose_px_x: float, nose_px_y: float
) -> tuple[float, float]:
    pixel_points = np.array(
        [
            [corners["top_left"]["x"], corners["top_left"]["y"]],
            [corners["top_right"]["x"], corners["top_right"]["y"]],
            [corners["bottom_left"]["x"], corners["bottom_left"]["y"]],
            [corners["bottom_right"]["x"], corners["bottom_right"]["y"]],
        ],
        dtype=float,
    )
    board_points = np.array(
        [
            [0.0, 0.0],
            [0.0, BOARD_WIDTH_INCHES],
            [BOARD_LENGTH_INCHES, 0.0],
            [BOARD_LENGTH_INCHES, BOARD_WIDTH_INCHES],
        ],
        dtype=float,
    )

    homography = _solve_homography(pixel_points, board_points)
    homogeneous = homography @ np.array([nose_px_x, nose_px_y, 1.0])
    if abs(homogeneous[2]) < 1e-8:
        raise ValueError("Homography projection produced invalid scaling factor")
    return homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2]


def _solve_homography(pixel_points: np.ndarray, board_points: np.ndarray) -> np.ndarray:
    if pixel_points.shape != (4, 2) or board_points.shape != (4, 2):
        raise ValueError("Exactly four corner correspondences are required")

    A = []
    b = []
    for (px, py), (ix, iy) in zip(pixel_points, board_points, strict=True):
        A.append([px, py, 1.0, 0.0, 0.0, 0.0, -ix * px, -ix * py])
        b.append(ix)
        A.append([0.0, 0.0, 0.0, px, py, 1.0, -iy * px, -iy * py])
        b.append(iy)

    A_matrix = np.array(A, dtype=float)
    b_vector = np.array(b, dtype=float)
    solution, _, rank, _ = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
    if rank < 8:
        raise ValueError("Degenerate corner configuration; homography could not be solved")

    h11, h12, h13, h21, h22, h23, h31, h32 = solution
    return np.array(
        [
            [h11, h12, h13],
            [h21, h22, h23],
            [h31, h32, 1.0],
        ],
        dtype=float,
    )




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query gpt-5 with the Reolink E1 snapshot to obtain a CuteBot pose estimate."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=REOLINK_REFERENCE_IMAGE,
        help="Path to the JPEG frame captured from the Reolink E1 camera.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to store the JSON response.",
    )
    args = parser.parse_args()

    observation = query_cutebot_pose(args.image)
    output_json = json.dumps(observation.as_dict(), indent=2)

    if args.output:
        args.output.write_text(output_json, encoding="utf-8")
    print(output_json)


if __name__ == "__main__":
    main()
