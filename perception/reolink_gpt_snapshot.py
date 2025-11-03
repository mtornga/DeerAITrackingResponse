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

BOARD_LENGTH_INCHES = 30.0  # distance from top to bottom edges in the image frame
BOARD_WIDTH_INCHES = 24.0  # distance from left to right edges in the image frame

SYSTEM_PROMPT = (
    "You are a vision assistant that estimates the CuteBot pose on a tabletop robot arena. "
    "The camera views the board from the side; the end of the board with the plants is the back. "
    "Heading is expressed in degrees clockwise from the back edge (0° faces the plants at the back, "
    "90° faces the viewer). The image coordinate system origin is in the top-left corner of the "
    "raw Reolink E1 frame (x increases to the right, y increases downward)."
)

USER_PROMPT = (
    "Identify the CuteBot robot (white body with blue chassis) and estimate its center location "
    "and heading relative to the provided coordinate system. "
    "The black rectangle sitting on top of the cardboard is the active board surface; it measures "
    "exactly 24 inches wide (left-to-right) by 30 inches long (front-to-back). The labelled corners "
    "represent the black rectangle only: top-left = (0, 0), top-right = (0, 24), bottom-left = (30, 0), "
    "bottom-right = (30, 24). Use these labels when estimating real-world coordinates. "
    "Respond with a JSON object containing exactly these fields:\n"
    "coordinate_system: always set to \"image_pixels\".\n"
    "origin: short string describing the origin for operators.\n"
    "units: units for x/y values, use \"pixels\".\n"
    "cutebot_center: object with numeric x and y properties (image pixels).\n"
    "board_coordinates: object with numeric x and y properties measured in inches using the labels above.\n"
    "heading_degrees: numeric value (0° faces the plants/back, 90° faces the camera/viewer).\n"
    "confidence: numeric value between 0 and 1.\n"
    "notes: short string with reasoning or observed ambiguities.\n"
    "Do not include any other text or Markdown."
)


@dataclass
class CutebotObservation:
    """Structured pose estimate returned by the vision model."""

    coordinate_system: str
    origin: str
    units: str
    cutebot_center_x: float
    cutebot_center_y: float
    board_center_x: float
    board_center_y: float
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
            center = payload["cutebot_center"]
            heading = float(payload["heading_degrees"])
            confidence = float(payload["confidence"])
            notes = str(payload["notes"])
        except KeyError as exc:
            raise ValueError(f"Missing required field: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid field types in observation payload") from exc

        if not isinstance(center, Mapping):
            raise ValueError("cutebot_center must be an object with x and y numbers")

        board_coords = payload.get("board_coordinates")
        if not isinstance(board_coords, Mapping):
            raise ValueError("board_coordinates must be an object with x and y numbers")

        try:
            center_x = float(center["x"])
            center_y = float(center["y"])
        except KeyError as exc:
            raise ValueError(f"cutebot_center missing coordinate: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("cutebot_center.x and cutebot_center.y must be numbers") from exc

        try:
            board_x = float(board_coords["x"])
            board_y = float(board_coords["y"])
        except KeyError as exc:
            raise ValueError(f"board_coordinates missing coordinate: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("board_coordinates.x and board_coordinates.y must be numbers") from exc

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        if not (0.0 <= center_x <= IMAGE_WIDTH_PIXELS and 0.0 <= center_y <= IMAGE_HEIGHT_PIXELS):
            raise ValueError("cutebot_center lies outside the reference image bounds")

        if not (0.0 <= board_x <= BOARD_LENGTH_INCHES and 0.0 <= board_y <= BOARD_WIDTH_INCHES):
            raise ValueError("board_coordinates lie outside the expected arena bounds")

        return cls(
            coordinate_system=coordinate_system,
            origin=origin,
            units=units,
            cutebot_center_x=center_x,
            cutebot_center_y=center_y,
            board_center_x=board_x,
            board_center_y=board_y,
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
            "cutebot_center": {"x": self.cutebot_center_x, "y": self.cutebot_center_y},
            "board_coordinates": {"x": self.board_center_x, "y": self.board_center_y},
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
