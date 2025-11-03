"""Tools for querying GPT-5-mini with the Reolink E1 snapshot.

This module loads the static `ReolinkE1Test.jpg` image, sends it to the
`gpt-5-mini` model, and returns a structured description of the CuteBot pose.
"""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

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
    "The camera views the board from the front; the edge with the plants is the back of the board. "
    "Heading angles are measured clockwise from this back edge: 0° faces the plants (top of the image), "
    "90° faces the right edge of the image, 180° faces the camera/foreground (bottom of the image), "
    "and 270° faces the left edge of the image. The pixel origin is at the top-left corner of the raw "
    "Reolink E1 frame (x increases to the right, y increases downward)."
)

USER_PROMPT = (
    "Identify the CuteBot robot (white body with blue chassis) and estimate the pose of the very tip of "
    "its nose (the foremost point between the two front wheels). The black rectangle sitting on top of "
    "the cardboard is the active board surface; it measures exactly 24 inches wide (left-to-right) by "
    "30 inches long (back-to-front). The labelled corners refer to this black rectangle only: "
    "top-left = (0, 0), top-right = (0, 24), bottom-left = (30, 0), bottom-right = (30, 24). "
    "In this inch coordinate system, x increases from the back edge toward the camera (top to bottom of "
    "the image) and y increases from the board's left edge to the right edge. A green arrow decal sits on "
    "the robot chassis and runs from the rear toward the front; the arrow tip points in the direction the "
    "CuteBot is facing.\n"
    "\n"
    "Work through these steps internally (do not expose them in the final output):\n"
    "1. Locate the pixel coordinates of the four black board corners listed above.\n"
    "2. Using those corner pixels, determine the pixel location of the CuteBot nose tip and convert that "
    "point into board inches.\n"
    "3. Determine the CuteBot heading using the angle convention described earlier.\n"
    "\n"
    "Respond with a JSON object containing exactly these fields:\n"
    "coordinate_system: always set to \"image_pixels\".\n"
    "origin: short string describing the origin for operators.\n"
    "units: units for the pixel coordinates, use \"pixels\".\n"
    "board_corners_pixels: object holding top_left, top_right, bottom_left, and bottom_right entries, each containing x and y pixel coordinates.\n"
    "cutebot_nose_pixels: object with numeric x and y properties for the nose tip in image pixels.\n"
    "cutebot_nose_inches: object with numeric x and y properties for the nose tip in board inches.\n"
    "heading_degrees: numeric value (use the clockwise convention from the back edge).\n"
    "confidence: numeric value between 0 and 1.\n"
    "notes: short string with reasoning or observed ambiguities.\n"
    "Return only the JSON object—no additional text or Markdown."
)


@dataclass
class CutebotObservation:
    """Structured pose estimate returned by the vision model."""

    coordinate_system: str
    origin: str
    units: str
    board_corners_pixels: Dict[str, Dict[str, float]]
    cutebot_nose_pixels_x: float
    cutebot_nose_pixels_y: float
    cutebot_nose_inches_x: float
    cutebot_nose_inches_y: float
    heading_degrees: float
    confidence: float
    notes: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CutebotObservation":
        """Validate and normalise a JSON payload from the LLM."""
        try:
            coordinate_system = str(payload["coordinate_system"])
            origin = str(payload["origin"])
            units = str(payload["units"])
            corners = payload["board_corners_pixels"]
            nose_pixels = payload["cutebot_nose_pixels"]
            nose_inches = payload["cutebot_nose_inches"]
            heading = float(payload["heading_degrees"])
            confidence = float(payload["confidence"])
            notes = str(payload["notes"])
        except KeyError as exc:
            raise ValueError(f"Missing required field: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid field types in observation payload") from exc

        if not isinstance(corners, Mapping):
            raise ValueError("board_corners_pixels must be an object with corner entries")

        normalised_corners: Dict[str, Dict[str, float]] = {}
        for corner_name in ("top_left", "top_right", "bottom_left", "bottom_right"):
            corner_value = corners.get(corner_name)
            if not isinstance(corner_value, Mapping):
                raise ValueError(f"board_corners_pixels.{corner_name} must be an object with x and y numbers")
            try:
                corner_x = float(corner_value["x"])
                corner_y = float(corner_value["y"])
            except KeyError as exc:
                raise ValueError(f"board_corners_pixels.{corner_name} missing coordinate: {exc}") from exc
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"board_corners_pixels.{corner_name}.x and .y must be numbers"
                ) from exc

            if not (0.0 <= corner_x <= IMAGE_WIDTH_PIXELS and 0.0 <= corner_y <= IMAGE_HEIGHT_PIXELS):
                raise ValueError(f"board_corners_pixels.{corner_name} lies outside the image bounds")
            normalised_corners[corner_name] = {"x": corner_x, "y": corner_y}

        if not isinstance(nose_pixels, Mapping):
            raise ValueError("cutebot_nose_pixels must be an object with x and y numbers")

        if not isinstance(nose_inches, Mapping):
            raise ValueError("cutebot_nose_inches must be an object with x and y numbers")

        try:
            nose_px_x = float(nose_pixels["x"])
            nose_px_y = float(nose_pixels["y"])
        except KeyError as exc:
            raise ValueError(f"cutebot_nose_pixels missing coordinate: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("cutebot_nose_pixels.x and cutebot_nose_pixels.y must be numbers") from exc

        try:
            nose_in_x = float(nose_inches["x"])
            nose_in_y = float(nose_inches["y"])
        except KeyError as exc:
            raise ValueError(f"cutebot_nose_inches missing coordinate: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("cutebot_nose_inches.x and cutebot_nose_inches.y must be numbers") from exc

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        if not (0.0 <= nose_px_x <= IMAGE_WIDTH_PIXELS and 0.0 <= nose_px_y <= IMAGE_HEIGHT_PIXELS):
            raise ValueError("cutebot_nose_pixels lies outside the reference image bounds")

        if not (0.0 <= nose_in_x <= BOARD_LENGTH_INCHES and 0.0 <= nose_in_y <= BOARD_WIDTH_INCHES):
            raise ValueError("cutebot_nose_inches lies outside the expected arena bounds")

        return cls(
            coordinate_system=coordinate_system,
            origin=origin,
            units=units,
            board_corners_pixels=normalised_corners,
            cutebot_nose_pixels_x=nose_px_x,
            cutebot_nose_pixels_y=nose_px_y,
            cutebot_nose_inches_x=nose_in_x,
            cutebot_nose_inches_y=nose_in_y,
            heading_degrees=heading,
            confidence=confidence,
            notes=notes,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return the observation as a JSON-serialisable dictionary."""
        return {
            "coordinate_system": self.coordinate_system,
            "origin": self.origin,
            "units": self.units,
            "board_corners_pixels": self.board_corners_pixels,
            "cutebot_nose_pixels": {"x": self.cutebot_nose_pixels_x, "y": self.cutebot_nose_pixels_y},
            "cutebot_nose_inches": {"x": self.cutebot_nose_inches_x, "y": self.cutebot_nose_inches_y},
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
    """Send the provided image to gpt-5-mini and parse the structured pose estimate."""
    image_data_url = _encode_image_as_data_url(image_path)
    client = client or _create_client()

    response = client.responses.create(
        model="gpt-5-mini",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query gpt-5-mini with the Reolink E1 snapshot to obtain a CuteBot pose estimate."
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
