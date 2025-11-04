from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(slots=True)
class HeadingSample:
    raw_degrees: float
    calibrated_degrees: Optional[float] = None
    source: str = "magnetometer"
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class GPTPoseSample:
    x_in: float
    y_in: float
    confidence: float
    raw_heading: Optional[float]
    notes: str
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class ControllerCommand:
    iteration: int
    action: str
    left_speed: int
    right_speed: int
    duration_ms: int
    pose_xy: Tuple[float, float]
    pose_heading: Optional[float]
    lateral_error: Optional[float]
    forward_error: Optional[float]
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class LogMessage:
    text: str
    level: str = "info"
    timestamp: float = field(default_factory=time.time)
