from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from cutebot.controller_auto import CutebotUARTSession

from .feedback import CutebotFeedbackLoop, TargetPose
from .tracker import TopDownCutebotTracker


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
        "--tracker-backend",
        choices=("csv", "yolo"),
        default="csv",
        help="Source of pose estimates. 'csv' tails detections_world.csv produced by demo/topdown_tracker.py.",
    )
    parser.add_argument(
        "--tracker-csv",
        type=Path,
        default=Path("detections_world.csv"),
        help="Path to the CSV written by demo/topdown_tracker.py when using the csv backend.",
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
    tracker = TopDownCutebotTracker(
        model_path=args.model,
        calibration_path=args.calibration,
        csv_path=args.tracker_csv,
        rtsp_url=args.rtsp,
        conf=args.conf,
        iou=args.iou,
        backend=args.tracker_backend,
    )
    target = TargetPose(x_in=args.target_x, y_in=args.target_y)

    async with CutebotUARTSession(verbose=args.controller_verbose) as controller:
        feedback = CutebotFeedbackLoop(
            controller=controller,
            tracker=tracker,
            target=target,
            pos_tolerance_in=args.pos_tol,
            lateral_tolerance_in=args.lat_tol,
            pose_retries=args.pose_retries,
            pose_delay_sec=args.pose_delay,
            min_confidence=args.min_conf,
            session_log_dir=args.log_dir,
        )
        try:
            final_pose = await feedback.move_to_target()
            print(
                f"[cycle {cycle_index}] Final pose "
                f"({final_pose.x_in:.2f}, {final_pose.y_in:.2f})\" "
                f"after {len(feedback.history)} measurements."
            )
        finally:
            tracker.close()


async def main_async() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

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
