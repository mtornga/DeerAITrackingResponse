from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from cutebot.controller_auto import CutebotUARTSession

from .tracker import CutebotPose, TopDownCutebotTracker


@dataclass
class TargetPose:
    x_in: float
    y_in: float
    heading_degrees: Optional[float] = None


@dataclass
class ExecutedCommand:
    iteration: int
    left_speed: int
    right_speed: int
    duration_ms: int
    reason: str


class CutebotFeedbackLoop:
    """
    Closed-loop controller that nudges the Cutebot towards a target pose.
    """

    def __init__(
        self,
        controller: CutebotUARTSession,
        tracker: TopDownCutebotTracker,
        target: TargetPose,
        *,
        forward_speed: int = 28,
        pivot_speed: int = 26,
        pos_tolerance_in: float = 0.8,
        lateral_tolerance_in: float = 0.8,
        max_iterations: int = 20,
        turn_gain: float = 2.5,
        max_turn_delta: int = 12,
        pivot_duration_ms: int = 220,
        settle_sec: float = 0.3,
        max_advance_in: float = 3.0,
        min_duration_ms: int = 160,
        max_duration_ms: int = 360,
        pose_retries: int = 30,
        pose_delay_sec: float = 0.45,
        min_confidence: float = 0.25,
        pivot_first_threshold_in: float = 4.0,
        session_log_dir: Path = Path("logs/cutebot_sessions"),
    ) -> None:
        self.controller = controller
        self.tracker = tracker
        self.target = target
        self.forward_speed = forward_speed
        self.pivot_speed = pivot_speed
        self.pos_tolerance_in = pos_tolerance_in
        self.lateral_tolerance_in = lateral_tolerance_in
        self.max_iterations = max_iterations
        self.turn_gain = turn_gain
        self.max_turn_delta = max_turn_delta
        self.pivot_duration_ms = pivot_duration_ms
        self.settle_sec = settle_sec
        self.max_advance_in = max_advance_in
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.pose_retries = pose_retries
        self.pose_delay_sec = pose_delay_sec
        self.min_confidence = min_confidence
        self.pivot_first_threshold_in = pivot_first_threshold_in

        self.history: List[CutebotPose] = []
        self.commands: List[ExecutedCommand] = []
        self._session_started_at = datetime.utcnow()
        self._log_dir = session_log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        ts = self._session_started_at.strftime("%Y%m%dT%H%M%S")
        self._log_path = self._log_dir / f"session_{ts}.json"

    async def move_to_target(self) -> CutebotPose:
        """
        Iteratively move the Cutebot until the pose error is within tolerance.
        """
        await self.controller.connect()
        self.tracker.start()

        status: str = "failed"
        failure_reason: Optional[str] = None
        try:
            for iteration in range(1, self.max_iterations + 1):
                pose = await self.tracker.get_pose(
                    retries=self.pose_retries,
                    delay_sec=self.pose_delay_sec,
                    min_confidence=self.min_confidence,
                )
                self.history.append(pose)

                lateral_error = pose.x_in - self.target.x_in
                forward_error = self.target.y_in - pose.y_in

                print(
                    f"[loop] iter={iteration} pose=({pose.x_in:.2f}, {pose.y_in:.2f})\" "
                    f"error(lateral={lateral_error:+.2f}, forward={forward_error:+.2f})\" "
                    f"conf={pose.confidence:.2f}"
                )

                if self._within_tolerance(lateral_error, forward_error):
                    print("[loop] Target tolerance achieved. Issuing stop command.")
                    await self.controller.stop()
                    status = "success"
                    return pose

                if abs(lateral_error) >= self.pivot_first_threshold_in:
                    await self._pivot_adjust(iteration, lateral_error)
                    await asyncio.sleep(self.settle_sec)
                    continue

                if forward_error > self.pos_tolerance_in:
                    await self._drive_towards(iteration, lateral_error, forward_error)
                elif forward_error < -self.pos_tolerance_in:
                    msg = (
                        f"Overshoot detected (forward_error={forward_error:+.2f}\"). "
                        "Reduce speed or duration."
                    )
                    print("[loop] Warning: overshoot detected; stopping to avoid moving past target.")
                    await self.controller.stop()
                    failure_reason = msg
                    raise RuntimeError(msg)
                else:
                    await self._pivot_adjust(iteration, lateral_error)

                await asyncio.sleep(self.settle_sec)

            failure_reason = f"Failed to reach target after {self.max_iterations} iterations."
            raise RuntimeError(failure_reason)
        except Exception as exc:
            failure_reason = str(exc)
            raise
        finally:
            self._write_session_log(status=status, failure_reason=failure_reason)

    def _within_tolerance(self, lateral_error: float, forward_error: float) -> bool:
        return (
            abs(lateral_error) <= self.lateral_tolerance_in
            and abs(forward_error) <= self.pos_tolerance_in
        )

    def _clamp_speed(self, value: int) -> int:
        return max(0, min(100, value))

    def _pick_duration(self, distance_in: float) -> int:
        """
        Translate desired travel distance into a conservative pulse duration.
        """
        scaled = min(distance_in, self.max_advance_in) / max(self.max_advance_in, 1e-6)
        span = self.max_duration_ms - self.min_duration_ms
        return int(self.min_duration_ms + scaled * span)

    def _compute_turn_delta(self, x_error: float) -> int:
        delta = int(round(x_error * self.turn_gain))
        return max(-self.max_turn_delta, min(self.max_turn_delta, delta))

    async def _drive_towards(
        self,
        iteration: int,
        lateral_error: float,
        forward_error: float,
    ) -> None:
        distance_in = min(abs(forward_error) * 0.6, self.max_advance_in)
        duration_ms = self._pick_duration(distance_in)
        base_speed = self.forward_speed
        turn_delta = self._compute_turn_delta(lateral_error)

        left = self._clamp_speed(base_speed - turn_delta)
        right = self._clamp_speed(base_speed + turn_delta)

        reason = (
            f"forward_error {forward_error:+.2f}\" with lateral_error {lateral_error:+.2f}\" "
            f"(distance {distance_in:.2f}\")"
        )
        current_pose = self.history[-1] if self.history else None
        await self._execute_drive(
            iteration,
            left,
            right,
            duration_ms,
            reason,
            pose=current_pose,
            lateral_error=lateral_error,
            forward_error=forward_error,
            action="drive",
        )

    async def _pivot_adjust(self, iteration: int, lateral_error: float) -> None:
        if abs(lateral_error) <= self.lateral_tolerance_in:
            return
        direction = -1 if lateral_error > 0 else 1
        if direction < 0:
            left = self._clamp_speed(self.pivot_speed)
            right = 0
        else:
            left = 0
            right = self._clamp_speed(self.pivot_speed)
        reason = f"pivot adjust for lateral_error {lateral_error:+.2f}\""
        current_pose = self.history[-1] if self.history else None
        await self._execute_drive(
            iteration,
            left,
            right,
            self.pivot_duration_ms,
            reason,
            pose=current_pose,
            lateral_error=lateral_error,
            forward_error=None,
            action="pivot",
        )

    async def _execute_drive(
        self,
        iteration: int,
        left: int,
        right: int,
        duration_ms: int,
        reason: str,
        *,
        pose: Optional[CutebotPose],
        lateral_error: Optional[float],
        forward_error: Optional[float],
        action: str,
    ) -> None:
        print(
            f"[loop] command iter={iteration} left={left} right={right} "
            f"duration={duration_ms}ms :: {reason}"
        )
        if pose is not None:
            loc = f"{pose.x_in:.1f}\", {pose.y_in:.1f}\""
        else:
            loc = "unknown pose"
        if forward_error is not None:
            fwd = f"{forward_error:+.2f}\" forward error"
        else:
            fwd = "forward error n/a"
        if lateral_error is not None:
            lat = f"{lateral_error:+.2f}\" lateral error"
        else:
            lat = "lateral error n/a"
        action_desc = "nudging forward" if action == "drive" else "pivoting to realign"
        print(
            f"[cutebot] I'm around ({loc}). {fwd}, {lat}. "
            f"I'm {action_desc} by running left={left}% right={right}% for {duration_ms}ms."
        )
        await self.controller.drive_timed(left, right, duration_ms)
        self.commands.append(
            ExecutedCommand(
                iteration=iteration,
                left_speed=left,
                right_speed=right,
                duration_ms=duration_ms,
                reason=reason,
            )
        )
        await self.controller.stop()

    def _write_session_log(self, *, status: str, failure_reason: Optional[str]) -> None:
        data = {
            "session_started_at": self._session_started_at.isoformat() + "Z",
            "status": status,
            "failure_reason": failure_reason,
            "target": asdict(self.target),
            "settings": {
                "forward_speed": self.forward_speed,
                "pivot_speed": self.pivot_speed,
                "pos_tolerance_in": self.pos_tolerance_in,
                "lateral_tolerance_in": self.lateral_tolerance_in,
                "max_iterations": self.max_iterations,
                "turn_gain": self.turn_gain,
                "max_turn_delta": self.max_turn_delta,
                "pivot_duration_ms": self.pivot_duration_ms,
                "settle_sec": self.settle_sec,
                "max_advance_in": self.max_advance_in,
                "min_duration_ms": self.min_duration_ms,
                "max_duration_ms": self.max_duration_ms,
                "pose_retries": self.pose_retries,
                "pose_delay_sec": self.pose_delay_sec,
                "min_confidence": self.min_confidence,
            },
            "commands": [
                asdict(cmd)
                for cmd in self.commands
            ],
            "poses": [
                {
                    "iteration": idx + 1,
                    "x_in": pose.x_in,
                    "y_in": pose.y_in,
                    "confidence": pose.confidence,
                    "timestamp": pose.timestamp,
                    "heading_degrees": pose.heading_degrees,
                }
                for idx, pose in enumerate(self.history)
            ],
        }
        try:
            self._log_path.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            print(f"[loop] Warning: failed to write session log {self._log_path}: {exc}")
