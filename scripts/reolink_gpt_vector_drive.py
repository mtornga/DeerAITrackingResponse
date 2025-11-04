from __future__ import annotations

import argparse
import asyncio
import math
from pathlib import Path
from typing import Optional

from cutebot.controller_auto import CutebotUARTSession
from cutebot.automation.tracker import ReolinkGPTTracker
from cutebot.automation.loop_runner import (
    _compute_heading_calibration,
    _calibrate_heading,
    _normalize_heading_diff,
    _align_heading,
)
from cutebot.dashboard.bus import GLOBAL_EVENT_BUS
from cutebot.dashboard.events import ControllerCommand, GPTPoseSample, HeadingSample, LogMessage
from cutebot.dashboard.ui import run_dashboard


async def drive_to_target(
    target_x: float,
    target_y: float,
    *,
    final_heading: Optional[float],
    heading_calibration: Optional[tuple[float, float]],
    position_tolerance: float,
    heading_tolerance: float,
    max_iterations: int,
    tracker: ReolinkGPTTracker,
    controller: CutebotUARTSession,
    forward_speed: int,
    pivot_speed: int,
    drive_duration_ms: int,
    pivot_duration_ms: int,
    settle_sec: float,
    pose_retries: int,
    pose_delay: float,
    min_confidence: float,
    lock_axis: Optional[str],
) -> None:
    for iteration in range(1, max_iterations + 1):
        pose = await tracker.get_pose(
            retries=pose_retries,
            delay_sec=pose_delay,
            min_confidence=min_confidence,
        )
        if pose.raw_payload:
            calibrated = pose.raw_payload.get("calibrated_inches")
            if isinstance(calibrated, dict):
                display_x = float(calibrated.get("forward", pose.y_in * 12.0))
                display_y = float(calibrated.get("lateral", pose.x_in * 12.0))
            else:
                projected = pose.raw_payload.get("cutebot_nose_inches_projected")
                if isinstance(projected, dict):
                    display_x = float(projected.get("x", pose.y_in * 12.0))
                    display_y = float(projected.get("y", pose.x_in * 12.0))
                else:
                    display_x = pose.y_in * 12.0
                    display_y = pose.x_in * 12.0
            await GLOBAL_EVENT_BUS.publish(
                GPTPoseSample(
                    x_in=display_x,
                    y_in=display_y,
                    confidence=pose.confidence,
                    raw_heading=pose.heading_degrees,
                    notes=pose.raw_payload.get("notes", ""),
                )
            )
        dx = target_x - pose.x_in
        dy = target_y - pose.y_in
        if lock_axis == "y":
            dy = 0.0
        elif lock_axis == "x":
            dx = 0.0
        distance = math.hypot(dx, dy)
        print(
            f"[vector] iter={iteration} pose=({pose.x_in:.2f}, {pose.y_in:.2f})" +
            f" dx={dx:+.2f} dy={dy:+.2f} dist={distance:.2f}"
        )

        if distance <= position_tolerance:
            print("[vector] Position tolerance achieved.")
            break

        heading_raw = await controller.request_heading(timeout=pose_delay)
        if heading_raw is None:
            print("[vector] Warning: heading unavailable; skipping iteration.")
            await GLOBAL_EVENT_BUS.publish(LogMessage("Heading read timed out", level="warn"))
            await asyncio.sleep(settle_sec)
            continue

        heading = _calibrate_heading(heading_raw, heading_calibration)
        await GLOBAL_EVENT_BUS.publish(
            HeadingSample(
                raw_degrees=heading_raw,
                calibrated_degrees=None if heading_calibration is None else heading,
            )
        )
        desired_heading = math.degrees(math.atan2(dy, dx)) % 360.0
        heading_diff = _normalize_heading_diff(desired_heading, heading)
        print(
            f"[vector] raw_heading={heading_raw:.1f}° calibrated={heading:.1f}° "
            f"desired={desired_heading:.1f}° diff={heading_diff:+.1f}°"
        )

        if abs(heading_diff) > heading_tolerance:
            direction = 1 if heading_diff > 0 else -1
            left = CutebotUARTSession._clamp_speed(direction * pivot_speed)
            right = CutebotUARTSession._clamp_speed(-direction * pivot_speed)
            await GLOBAL_EVENT_BUS.publish(
                ControllerCommand(
                    iteration=iteration,
                    action="pivot",
                    left_speed=left,
                    right_speed=right,
                    duration_ms=pivot_duration_ms,
                    pose_xy=(pose.x_in, pose.y_in),
                    pose_heading=heading,
                    lateral_error=-dx,
                    forward_error=-dy,
                    reason=f"align diff {heading_diff:+.1f}°",
                )
            )
            await controller.drive_timed(left, right, pivot_duration_ms)
        else:
            step = min(distance, 2.0)
            duration_ms = max(120, int(drive_duration_ms * (step / 2.0)))
            speed = CutebotUARTSession._clamp_speed(forward_speed)
            await GLOBAL_EVENT_BUS.publish(
                ControllerCommand(
                    iteration=iteration,
                    action="drive",
                    left_speed=speed,
                    right_speed=speed,
                    duration_ms=duration_ms,
                    pose_xy=(pose.x_in, pose.y_in),
                    pose_heading=heading,
                    lateral_error=-dx,
                    forward_error=dy,
                    reason=f"progress {step:.2f}\"",
                )
            )
            await controller.drive_timed(speed, speed, duration_ms)

        await asyncio.sleep(settle_sec)

    if final_heading is not None:
        await _align_heading(
            controller,
            tracker,
            target_heading=final_heading,
            tolerance=heading_tolerance,
            max_iterations=8,
            settle_sec=settle_sec,
            duration_ms=pivot_duration_ms,
            pose_retries=pose_retries,
            pose_delay=pose_delay,
            min_confidence=min_confidence,
            pivot_speed=pivot_speed,
            heading_calibration=heading_calibration,
        )


async def main_async(args: argparse.Namespace) -> None:
    if args.heading_calibration:
        if args.heading_calibration.lower() == "none":
            calibration = None
        else:
            samples = [float(s.strip()) for s in args.heading_calibration.split(",") if s.strip()]
            if len(samples) != 4:
                raise SystemExit("--heading-calibration must contain four values")
            calibration = _compute_heading_calibration(samples)
    else:
        calibration = None

    transform_path: Optional[Path]
    if args.transform is None:
        transform_path = None
    else:
        transform_arg = str(args.transform)
        if transform_arg.lower() == "none":
            transform_path = None
        else:
            transform_path = Path(transform_arg)

    tracker = ReolinkGPTTracker(
        rtsp_url=args.rtsp,
        snapshot_dir=args.snapshot_dir,
        ffmpeg_path=args.ffmpeg,
        transport=args.transport,
        cleanup_snapshots=not args.keep_snapshots,
        capture_timeout_sec=args.capture_timeout,
        transform_path=transform_path,
    )

    stop_event = asyncio.Event()
    dashboard_task = asyncio.create_task(run_dashboard(stop_event))

    async with CutebotUARTSession(verbose=args.verbose) as controller:
        await controller.enable_heading_stream(True)
        try:
            await drive_to_target(
                args.target_x,
                args.target_y,
                final_heading=args.target_heading,
                heading_calibration=calibration,
                position_tolerance=args.position_tolerance,
                heading_tolerance=args.heading_tolerance,
                max_iterations=args.iterations,
                tracker=tracker,
                controller=controller,
                forward_speed=args.forward_speed,
                pivot_speed=args.pivot_speed,
                drive_duration_ms=args.drive_duration_ms,
                pivot_duration_ms=args.pivot_duration_ms,
                settle_sec=args.settle_sec,
                pose_retries=args.pose_retries,
                pose_delay=args.pose_delay,
                min_confidence=args.min_confidence,
                lock_axis=args.lock_axis,
            )
        finally:
            await controller.enable_heading_stream(False)
            tracker.close()
            stop_event.set()
            await dashboard_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drive Cutebot using Reolink GPT pose estimates.")
    parser.add_argument("--target-x", type=float, required=True)
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--target-heading", type=float, default=None)
    parser.add_argument("--position-tolerance", type=float, default=0.7)
    parser.add_argument("--heading-tolerance", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--forward-speed", type=int, default=20)
    parser.add_argument("--pivot-speed", type=int, default=22)
    parser.add_argument("--drive-duration-ms", type=int, default=220)
    parser.add_argument("--pivot-duration-ms", type=int, default=160)
    parser.add_argument("--settle-sec", type=float, default=0.6)
    parser.add_argument("--pose-retries", type=int, default=4)
    parser.add_argument("--pose-delay", type=float, default=1.0)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument(
        "--lock-axis",
        choices=("x", "y", "none"),
        default="none",
        help="Lock movement along a particular axis (useful when one coordinate is unreliable).",
    )
    parser.add_argument("--rtsp", default=None)
    parser.add_argument("--snapshot-dir", type=Path, default=Path("tmp/reolink_snapshots"))
    parser.add_argument("--keep_snapshots", action="store_true")
    parser.add_argument("--transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--capture-timeout", type=float, default=20.0)
    parser.add_argument(
        "--heading-calibration",
        default=None,
        help="Comma-separated raw headings for markers 0,90,180,270 (or 'none' to disable calibration).",
    )
    parser.add_argument(
        "--transform",
        default="calibration/reolink_gpt/transform.json",
        help="Affine transform JSON produced by reolink_gpt_fit_transform.py (use 'none' to disable).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
