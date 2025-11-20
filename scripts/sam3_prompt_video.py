#!/usr/bin/env python3
"""
Run the official SAM 3 video predictor with a text prompt and render a tracked video.

This script mirrors the Playground flow:
1. Load the clip through Meta's SAM 3 video predictor.
2. Submit a text prompt (e.g., "deer") on one frame.
3. Propagate the track(s) across the clip.
4. Overlay the predicted masks/boxes and export an annotated MP4.

Usage:
    python scripts/sam3_prompt_video.py \
        --video-path outdoor/deer-vision/data/eval/mixed_weather/clips/segment_092754.mp4 \
        --prompt "deer" \
        --output-video runs/sam3/segment_092754_prompt.mp4 \
        --prompt-frame 0 \
        --gpus 0

Requirements:
- Access to the gated `facebook/sam3` checkpoint on Hugging Face.
- A GPU with ≥12 GB VRAM (the official tracker does not support CPU mode).
- The upstream SAM3 repo cloned under `external/sam3` (already present on Ubuntu).
"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

compat_file = PROJECT_ROOT / "sitecustomize.py"
if compat_file.exists():
    spec = importlib.util.spec_from_file_location("deer_sitecustomize", compat_file)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

SAM3_REPO = PROJECT_ROOT / "external" / "sam3"
if not SAM3_REPO.exists():
    raise RuntimeError(
        "SAM3 repository not found under external/sam3. "
        "Clone https://github.com/facebookresearch/sam3 into that directory first."
    )

sys.path.append(str(SAM3_REPO))

import sam3.model_builder as sam3_model_builder  # type: ignore  # noqa: E402

_ORIG_BUILD_VIDEO_MODEL = sam3_model_builder.build_sam3_video_model


def _build_video_model_fp16(*args, **kwargs):
    model = _ORIG_BUILD_VIDEO_MODEL(*args, **kwargs)
    sample_param = next(model.parameters(), None)
    device = sample_param.device if sample_param is not None else torch.device("cpu")
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.float16)
    return model


sam3_model_builder.build_sam3_video_model = _build_video_model_fp16  # type: ignore[attr-defined]

from sam3.model_builder import build_sam3_video_predictor  # type: ignore  # noqa: E402
from sam3.visualization_utils import render_masklet_frame  # type: ignore  # noqa: E402


def autocast_context():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)  # type: ignore[arg-type]
    return nullcontext()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 video predictor with a text prompt.",
    )
    parser.add_argument("--video-path", type=Path, required=True, help="Input MP4 path.")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='Text prompt (e.g., "deer", "a deer", "a moving deer").',
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        required=True,
        help="Where to write the annotated MP4.",
    )
    parser.add_argument(
        "--prompt-frame",
        type=int,
        default=0,
        help="Frame index where the text prompt should be anchored.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="both",
        choices=["both", "forward", "backward"],
        help="Propagation direction for the tracker.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames to propagate (defaults to entire clip).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of GPU ids to use (defaults to current CUDA device).",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.6,
        help="Mask overlay alpha when rendering (0 = transparent, 1 = opaque).",
    )
    return parser.parse_args()


def ensure_video_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")


def build_predictor(gpus: Optional[List[int]] = None):
    if gpus is not None:
        return build_sam3_video_predictor(gpus_to_use=gpus)
    return build_sam3_video_predictor()


def collect_track_outputs(
    predictor,
    video_path: Path,
    prompt: str,
    prompt_frame: int,
    direction: str,
    max_frames: Optional[int],
) -> Dict[int, Dict[str, np.ndarray]]:
    # 1) Start session
    with autocast_context():
        start_resp = predictor.handle_request(
            {"type": "start_session", "resource_path": str(video_path)}
        )
    session_id = start_resp["session_id"]

    # 2) Add prompt
    with autocast_context():
        add_resp = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": prompt_frame,
                "text": prompt,
            }
        )
    prompt_frame_idx = add_resp.get("frame_index", prompt_frame)

    # 3) Propagate
    outputs_by_frame: Dict[int, Dict[str, np.ndarray]] = {}
    try:
        stream_req = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": direction,
            "start_frame_index": prompt_frame_idx,
        }
        if max_frames is not None:
            stream_req["max_frame_num_to_track"] = max_frames

        with autocast_context():
            for event in predictor.handle_stream_request(stream_req):
                frame_index = int(event["frame_index"])
                outputs = event["outputs"]
                if not outputs:
                    continue
                outputs_by_frame[frame_index] = outputs
    finally:
        with autocast_context():
            predictor.handle_request({"type": "close_session", "session_id": session_id})

    return outputs_by_frame


def render_tracked_video(
    video_path: Path,
    frame_outputs: Dict[int, Dict[str, np.ndarray]],
    output_path: Path,
    mask_alpha: float,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            outputs = frame_outputs.get(frame_idx)
            if outputs:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                overlay_rgb = render_masklet_frame(
                    frame_rgb, outputs, frame_idx=frame_idx, alpha=mask_alpha
                )
                frame_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


def main() -> None:
    args = parse_args()
    ensure_video_exists(args.video_path)

    predictor = build_predictor(gpus=args.gpus)

    outputs_by_frame = collect_track_outputs(
        predictor=predictor,
        video_path=args.video_path,
        prompt=args.prompt,
        prompt_frame=args.prompt_frame,
        direction=args.direction,
        max_frames=args.max_frames,
    )

    if not outputs_by_frame:
        raise RuntimeError(
            "Tracker did not return any masks—check the prompt text or frame index."
        )

    render_tracked_video(
        video_path=args.video_path,
        frame_outputs=outputs_by_frame,
        output_path=args.output_video,
        mask_alpha=args.mask_alpha,
    )


if __name__ == "__main__":
    main()
