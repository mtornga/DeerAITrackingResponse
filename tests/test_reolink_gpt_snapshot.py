from __future__ import annotations

import os
from pathlib import Path

import pytest

from perception.reolink_gpt_snapshot import (
    BOARD_LENGTH_INCHES,
    BOARD_WIDTH_INCHES,
    CutebotObservation,
    IMAGE_HEIGHT_PIXELS,
    IMAGE_WIDTH_PIXELS,
    REOLINK_REFERENCE_IMAGE,
    query_cutebot_pose,
)


def _make_payload(
    *,
    x: float = 1200.0,
    y: float = 900.0,
    board_x: float = 15.0,
    board_y: float = 12.0,
    heading: float = 45.0,
    confidence: float = 0.8,
) -> dict:
    return {
        "coordinate_system": "image_pixels",
        "origin": "Top-left of Reolink E1 frame",
        "units": "pixels",
        "cutebot_center": {"x": x, "y": y},
        "board_coordinates": {"x": board_x, "y": board_y},
        "heading_degrees": heading,
        "confidence": confidence,
        "notes": "Example payload for tests",
    }


def test_cutebot_observation_from_dict_accepts_valid_payload() -> None:
    payload = _make_payload()
    observation = CutebotObservation.from_dict(payload)

    assert observation.coordinate_system == "image_pixels"
    assert observation.units == "pixels"
    assert observation.cutebot_center_x == pytest.approx(payload["cutebot_center"]["x"])
    assert observation.cutebot_center_y == pytest.approx(payload["cutebot_center"]["y"])
    assert observation.board_center_x == pytest.approx(payload["board_coordinates"]["x"])
    assert observation.board_center_y == pytest.approx(payload["board_coordinates"]["y"])
    assert observation.heading_degrees == pytest.approx(payload["heading_degrees"])
    assert observation.confidence == pytest.approx(payload["confidence"])


@pytest.mark.parametrize(
    "x,y",
    [
        (-1.0, 500.0),
        (IMAGE_WIDTH_PIXELS + 1.0, 500.0),
        (500.0, -5.0),
        (500.0, IMAGE_HEIGHT_PIXELS + 10.0),
    ],
)
def test_cutebot_observation_rejects_out_of_bounds_coordinates(x: float, y: float) -> None:
    payload = _make_payload(x=x, y=y)
    with pytest.raises(ValueError):
        CutebotObservation.from_dict(payload)


def test_cutebot_observation_rejects_board_coordinates_out_of_bounds() -> None:
    payload = _make_payload(board_x=BOARD_LENGTH_INCHES + 5.0, board_y=10.0)
    with pytest.raises(ValueError):
        CutebotObservation.from_dict(payload)

def test_cutebot_observation_rejects_confidence_outside_bounds() -> None:
    payload = _make_payload(confidence=1.2)
    with pytest.raises(ValueError):
        CutebotObservation.from_dict(payload)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("RUN_GPT_LIVE_TEST"),
    reason="Set RUN_GPT_LIVE_TEST=1 to call the OpenAI API.",
)
def test_query_cutebot_pose_live_round_trip(tmp_path: Path) -> None:
    """Live evaluation that ensures the model returns a well-formed observation."""
    observation = query_cutebot_pose(REOLINK_REFERENCE_IMAGE)

    assert 0.0 <= observation.cutebot_center_x <= IMAGE_WIDTH_PIXELS
    assert 0.0 <= observation.cutebot_center_y <= IMAGE_HEIGHT_PIXELS
    assert 0.0 <= observation.board_center_x <= BOARD_LENGTH_INCHES
    assert 0.0 <= observation.board_center_y <= BOARD_WIDTH_INCHES
    assert -180.0 <= observation.heading_degrees <= 360.0
    assert 0.0 <= observation.confidence <= 1.0
    assert observation.notes
