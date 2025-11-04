from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Optional

from cutebot.controller_auto import CutebotUARTSession
from cutebot.dashboard.bus import GLOBAL_EVENT_BUS
from cutebot.dashboard.events import HeadingSample, LogMessage

from .feedback import CutebotFeedbackLoop, TargetPose
from .tracker import TopDownCutebotTracker, ReolinkGPTTracker


def _compute_heading_calibration(samples: list[float]) -> tuple[float, float]:
    real = [0.0, 90.0, 180.0, 270.0]
    measured = samples
    sum_m = sum(measured)
    sum_r = sum(real)
    sum_mm = sum(m * m for m in measured)
    sum_mr = sum(m * r for m, r in zip(measured, real))
    n = len(measured)
    denom = n * sum_mm - sum_m * sum_m
    if abs(denom) < 1e-6:
        return 0.0, 1.0
    b = (n * sum_mr - sum_m * sum_r) / denom
    a = (sum_r - b * sum_m) / n
    return a, b


def _calibrate_heading(raw: float, calibration: Optional[tuple[float, float]]) -> float:
    if calibration is None:
        return raw % 360.0
    a, b = calibration
    return (a + b * raw) % 360.0


def _normalize_heading_diff(target: float, current: float) -> float:
    diff = (target - current + 180.0) % 360.0 - 180.0
    return diff


async def _align_heading(
    controller: CutebotUARTSession,
    tracker,
    *,
    target_heading: float,
    tolerance: float,
    max_iterations: int,
    settle_sec: float,
    duration_ms: int,
    pose_retries: int,
    pose_delay: float,
    min_confidence: float,
    pivot_speed: int,
    heading_calibration: Optional[tuple[float, float]],
) -> bool:
    for attempt in range(1, max_iterations + 1):
        heading = await controller.request_heading(timeout=pose_delay)
        if heading is None:
            print("[heading] Unable to read heading from Cutebot; retrying.")
            await GLOBAL_EVENT_BUS.publish(LogMessage("Heading read timed out"))
            await asyncio.sleep(pose_delay)
            continue
        calibrated = _calibrate_heading(heading, heading_calibration)
        diff = _normalize_heading_diff(target_heading, calibrated)
        print(
            f"[heading] attempt={attempt} raw={heading:.1f}° calibrated={calibrated:.1f}° diff={diff:+.1f}°"
        )
        await GLOBAL_EVENT_BUS.publish(
            HeadingSample(
                raw_degrees=heading,
                calibrated_degrees=None if heading_calibration is None else calibrated,
            )
        )

        if abs(diff) <= tolerance:
            await controller.stop()
            print("[heading] Heading within tolerance.")
            await GLOBAL_EVENT_BUS.publish(LogMessage("Heading aligned"))
            return True

        direction = 1 if diff > 0 else -1
        if direction > 0:
            left = CutebotUARTSession._clamp_speed(pivot_speed)
            right = CutebotUARTSession._clamp_speed(-pivot_speed)
        else:
            left = CutebotUARTSession._clamp_speed(-pivot_speed)
            right = CutebotUARTSession._clamp_speed(pivot_speed)

        await controller.drive_timed(left, right, duration_ms)
        await controller.stop()
        await asyncio.sleep(settle_sec)

    print("[heading] Failed to reach target heading within allotted iterations.")
    await GLOBAL_EVENT_BUS.publish(LogMessage("Heading alignment failed", level="warn"))
    return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Closed-loop Cutebot controller using top-down tracker feedback."
    )
    parser.add_argument(
        "--target-x",
        type=float,
        default=12.5,
        help="Target X position in inches from the tabletop's left edge.",
    )
    parser.add_argument(
        "--target-y",
        type=float,
        default=6.0,
        help="Target Y position in inches from the tabletop's top edge.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of move→feedback cycles to execute.",
    )
    parser.add_argument(
        "--rest-sec",
        type=float,
        default=1.0,
        help="Pause between cycles to allow the Cutebot to settle.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/detect/ultraYOLODetection1_v13/weights/best.pt"),
        help="Path to the YOLO weights used for detection.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration/tabletop_affine.json"),
        help="Path to the tabletop calibration matrix JSON.",
    )
    parser.add_argument(
        "--rtsp",
        default=None,
        help="Override RTSP source for the tracker (defaults to WYZE_TABLETOP_RTSP).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Detection IoU threshold.",
    )
    parser.add_argument(
        "--pos-tol",
        type=float,
        default=0.75,
        help="Position tolerance in inches for the Y axis.",
    )
    parser.add_argument(
        "--lat-tol",
        type=float,
        default=0.75,
        help="Lateral (X axis) tolerance in inches.",
    )
    parser.add_argument(
        "--controller-verbose",
        action="store_true",
        help="Enable verbose logging from the BLE controller.",
    )
    parser.add_argument(
        "--forward-speed",
        type=int,
        default=28,
        help="Base forward PWM (0-100). Lower for gentler moves.",
    )
    parser.add_argument(
        "--pivot-speed",
        type=int,
        default=26,
        help="Pivot PWM (0-100).",
    )
    parser.add_argument(
        "--tracker-backend",
        choices=("csv", "yolo", "reolink"),
        default="csv",
        help="Source of pose estimates. 'csv' tails detections_world.csv; 'yolo' runs detection live; 'reolink' queries the GPT-based Reolink tracker.",
    )
    parser.add_argument(
        "--tracker-csv",
        type=Path,
        default=Path("detections_world.csv"),
        help="Path to the CSV written by demo/topdown_tracker.py when using the csv backend.",
    )
    parser.add_argument(
        "--reolink-snapshot-dir",
        type=Path,
        default=Path("tmp/reolink_snapshots"),
        help="Directory for temporary snapshots when using the Reolink GPT tracker.",
    )
    parser.add_argument(
        "--reolink-keep-snapshots",
        action="store_true",
        help="Keep captured Reolink snapshots on disk for debugging.",
    )
    parser.add_argument(
        "--reolink-transport",
        choices=("tcp", "udp"),
        default="tcp",
        help="RTSP transport protocol for Reolink snapshot capture.",
    )
    parser.add_argument(
        "--reolink-ffmpeg",
        default="ffmpeg",
        help="Path to the ffmpeg binary used to capture Reolink frames.",
    )
    parser.add_argument(
        "--reolink-capture-timeout",
        type=float,
        default=20.0,
        help="Timeout in seconds when capturing a Reolink snapshot.",
    )
    parser.add_argument(
        "--target-heading",
        type=float,
        default=None,
        help="Desired final heading in degrees (0°=right, 90°=toward camera). Requires the Reolink tracker.",
    )
    parser.add_argument(
        "--heading-tolerance",
        type=float,
        default=12.0,
        help="Tolerance in degrees when aligning the final heading.",
    )
    parser.add_argument(
        "--heading-max-iterations",
        type=int,
        default=8,
        help="Maximum heading adjustment iterations after reaching the target position.",
    )
    parser.add_argument(
        "--heading-settle-sec",
        type=float,
        default=1.0,
        help="Settle time between heading adjustment pulses.",
    )
    parser.add_argument(
        "--heading-duration-ms",
        type=int,
        default=220,
        help="Drive pulse duration in milliseconds for heading adjustments.",
    )
    parser.add_argument(
        "--heading-calibration",
        default=None,
        help="Comma-separated magnetometer readings measured at real headings 0°,90°,180°,270°.",
    )
    parser.add_argument(
        "--pose-retries",
        type=int,
        default=20,
        help="Number of attempts to grab a pose before aborting the cycle.",
    )
    parser.add_argument(
        "--pose-delay",
        type=float,
        default=0.4,
        help="Delay between pose attempts in seconds.",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.15,
        help="Minimum detection confidence to accept (set to 0 to accept everything).",
    )
    parser.add_argument(
        "--max-advance",
        type=float,
        default=3.0,
        help="Maximum forward distance (inches) for a single drive pulse.",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=160,
        help="Minimum drive pulse duration in milliseconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=360,
        help="Maximum drive pulse duration in milliseconds.",
    )
    parser.add_argument(
        "--pivot-first-threshold",
        type=float,
        default=4.0,
        help="If lateral error exceeds this (inches), pivot before moving forward.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/cutebot_sessions"),
        help="Directory where feedback sessions are logged as JSON.",
    )
    return parser


async def run_cycle(
    cycle_index: int,
    args: argparse.Namespace,
) -> None:
    if args.tracker_backend == "reolink":
        tracker = ReolinkGPTTracker(
            rtsp_url=args.rtsp,
            snapshot_dir=args.reolink_snapshot_dir,
            ffmpeg_path=args.reolink_ffmpeg,
            transport=args.reolink_transport,
            cleanup_snapshots=not args.reolink_keep_snapshots,
            capture_timeout_sec=args.reolink_capture_timeout,
        )
    else:
        tracker = TopDownCutebotTracker(
            model_path=args.model,
            calibration_path=args.calibration,
            csv_path=args.tracker_csv,
            rtsp_url=args.rtsp,
            conf=args.conf,
            iou=args.iou,
            backend=args.tracker_backend,
        )

    target = TargetPose(x_in=args.target_x, y_in=args.target_y, heading_degrees=args.target_heading)

    async with CutebotUARTSession(verbose=args.controller_verbose) as controller:
        heading_stream_enabled = False
        if args.target_heading is not None:
            await controller.enable_heading_stream(True)
            heading_stream_enabled = True

        feedback = CutebotFeedbackLoop(
        controller=controller,
        tracker=tracker,
        target=target,
        forward_speed=args.forward_speed,
        pivot_speed=args.pivot_speed,
        pos_tolerance_in=args.pos_tol,
        lateral_tolerance_in=args.lat_tol,
        pose_retries=args.pose_retries,
        pose_delay_sec=args.pose_delay,
        min_confidence=args.min_conf,
        max_advance_in=args.max_advance,
        min_duration_ms=args.min_duration,
        max_duration_ms=args.max_duration,
        pivot_first_threshold_in=args.pivot_first_threshold,
        session_log_dir=args.log_dir,
    )
        try:
            final_pose = await feedback.move_to_target()
            print(
                f"[cycle {cycle_index}] Final pose "
                f"({final_pose.x_in:.2f}, {final_pose.y_in:.2f})\" "
                f"after {len(feedback.history)} measurements."
            )

            if (
                args.target_heading is not None
                and getattr(tracker, "supports_heading", False)
            ):
                await _align_heading(
                    controller,
                    tracker,
                    target_heading=args.target_heading,
                    tolerance=args.heading_tolerance,
                    max_iterations=args.heading_max_iterations,
                    settle_sec=args.heading_settle_sec,
                    duration_ms=args.heading_duration_ms,
                    pose_retries=args.pose_retries,
                    pose_delay=args.pose_delay,
                    min_confidence=args.min_conf,
                    pivot_speed=args.pivot_speed,
                    heading_calibration=args.heading_calibration_tuple,
                )
        finally:
            tracker.close()
            if heading_stream_enabled:
                try:
                    await controller.enable_heading_stream(False)
                except Exception:
                    pass


async def main_async() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.heading_calibration:
        try:
            samples = [float(x.strip()) for x in args.heading_calibration.split(",") if x.strip()]
        except ValueError as exc:
            parser.error(f"Invalid heading calibration values: {exc}")
        if len(samples) != 4:
            parser.error("--heading-calibration requires four values for 0°,90°,180°,270°")
    else:
        samples = [0.0, 90.0, 180.0, 270.0]
    args.heading_calibration_tuple = _compute_heading_calibration(samples)

    for i in range(1, args.cycles + 1):
        print(f"[cycle {i}] Starting move towards ({args.target_x}, {args.target_y})\"")
        try:
            await run_cycle(i, args)
        except Exception as exc:
            print(f"[cycle {i}] Failed: {exc}")
            break
        if i < args.cycles and args.rest_sec > 0:
            await asyncio.sleep(args.rest_sec)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
