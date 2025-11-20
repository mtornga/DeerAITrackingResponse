#!/usr/bin/env python3
"""
Run Meta's SAM 3 model on a video clip to generate per-frame segmentation masks.

This script is intentionally simple: it uses the Hugging Face Transformers
`mask-generation` pipeline with the `facebook/sam3` checkpoint and overlays
the generated masks on each frame of the input video.

Usage (from the repo root, after creating the virtualenv and installing
constraints + requirements):

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --no-cache-dir --force-reinstall -r constraints.txt
    pip install --no-cache-dir -r requirements.txt

    python scripts/sam3_process_video.py \
        --video-path outdoor/deer-vision/data/eval/mixed_weather/clips/segment_092754.mp4 \
        --output-video runs/sam3/segment_092754_annotated.mp4 \
        --device cuda

You must already have access to `facebook/sam3` on Hugging Face and be
authenticated on the machine (for example via `huggingface-cli login`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
from transformers.pipelines import Pipeline

try:
    import torch
except ImportError:  # pragma: no cover - torch is required by constraints.txt
    torch = None  # type: ignore[assignment]


if torch is not None and not hasattr(torch, "compiler"):
    class _TorchCompilerCompat:
        @staticmethod
        def is_compiling() -> bool:
            return False

    torch.compiler = _TorchCompilerCompat()  # type: ignore[attr-defined]


def build_mask_generator(device: str, model_id: str) -> Pipeline:
    """
    Construct a Hugging Face `mask-generation` pipeline for SAM 3.

    The user must have access to `facebook/sam3` and be logged in locally.
    """
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{device}', expected 'cpu' or 'cuda'.")

    pipeline_kwargs: dict[str, Any] = {
        "model": model_id,
        "task": "mask-generation",
        "trust_remote_code": True,
    }

    if device == "cuda":
        if torch is None or not torch.cuda.is_available():  # type: ignore[truthy-function]
            raise RuntimeError("CUDA requested but not available; check your PyTorch install.")
        pipeline_kwargs["device"] = 0
        pipeline_kwargs["torch_dtype"] = torch.float16  # type: ignore[attr-defined]

    return pipeline(**pipeline_kwargs)


def _normalize_masks(masks_obj: Any) -> np.ndarray:
    """
    Convert the pipeline output `masks` field into a uint8 array of shape (N, H, W).

    The mask-generation pipeline typically returns either:
    - A dict with key ``\"masks\"`` → list/array of masks
    - A list with a single dict for the image
    """
    if isinstance(masks_obj, np.ndarray):
        if masks_obj.ndim == 2:
            return masks_obj[None, ...].astype(np.uint8)
        if masks_obj.ndim == 3:
            return masks_obj.astype(np.uint8)
        raise ValueError(f"Unexpected mask array shape {masks_obj.shape}.")

    if isinstance(masks_obj, list):
        if not masks_obj:
            return np.zeros((0, 1, 1), dtype=np.uint8)
        # If this is a list of dicts (one per image), unwrap the first element.
        if isinstance(masks_obj[0], dict) and "masks" in masks_obj[0]:
            return _normalize_masks(masks_obj[0]["masks"])

        stacked = [np.array(m, dtype=np.uint8) for m in masks_obj]
        # Ensure we always have shape (N, H, W)
        if stacked[0].ndim == 2:
            return np.stack(stacked, axis=0)
        if stacked[0].ndim == 3:
            return np.stack([m.squeeze() for m in stacked], axis=0)
        raise ValueError(f"Unexpected mask element shape {stacked[0].shape}.")

    raise TypeError(f"Unsupported masks object type: {type(masks_obj)}")


def overlay_masks(frame_bgr: np.ndarray, masks: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay colored masks on top of a BGR frame.

    Each mask in `masks` is expected to be a binary array (0/1) of shape (H, W).
    """
    if masks.size == 0:
        return frame_bgr

    overlay = frame_bgr.copy()
    height, width = frame_bgr.shape[:2]
    num_masks = masks.shape[0]

    colors = _generate_palette(num_masks)

    for mask_index in range(num_masks):
        mask = masks[mask_index].astype(bool)
        if not mask.any():
            continue

        color = colors[mask_index]
        colored_layer = np.zeros_like(overlay, dtype=np.uint8)
        colored_layer[mask] = color

        overlay = cv2.addWeighted(colored_layer, alpha, overlay, 1.0 - alpha, 0.0)

    return overlay


def _generate_palette(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate a simple, deterministic color palette in BGR space.

    The palette is not fancy, just distinct enough for a handful of tracks.
    """
    base_colors: List[Tuple[int, int, int]] = [
        (0, 255, 0),
        (0, 165, 255),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (255, 128, 0),
    ]

    if n <= len(base_colors):
        return base_colors[:n]

    colors: List[Tuple[int, int, int]] = []
    for i in range(n):
        color = base_colors[i % len(base_colors)]
        # Slightly jitter brightness based on index so repeated colors differ.
        factor = 0.8 + 0.2 * ((i // len(base_colors)) % 3) / 2.0
        jittered = tuple(int(min(255, c * factor)) for c in color)
        colors.append(jittered)
    return colors


def iter_video_frames(video_path: Path) -> Iterable[Tuple[int, np.ndarray]]:
    """
    Yield (frame_index, frame_bgr) for each frame in the video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()


def process_video(
    video_path: Path,
    output_video_path: Path,
    device: str,
    model_id: str,
    max_frames: int | None = None,
    min_area_ratio: float = 0.0005,
) -> None:
    """
    Run SAM 3 mask generation on every frame of `video_path` and write an annotated video.

    Parameters
    ----------
    video_path:
        Input video path.
    output_video_path:
        Where to write the annotated MP4.
    device:
        'cpu' or 'cuda'.
    max_frames:
        Optional cap on the number of frames to process (for quick tests).
    min_area_ratio:
        Fraction of the frame area below which masks are discarded as noise.
    """
    mask_generator = build_mask_generator(device, model_id=model_id)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    try:
        frame_index = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            outputs = mask_generator(
                pil_image,
                points_per_batch=128,
            )

            if isinstance(outputs, dict):
                masks_obj = outputs.get("masks", [])
            elif isinstance(outputs, list) and outputs:
                maybe_dict = outputs[0]
                masks_obj = maybe_dict.get("masks", []) if isinstance(maybe_dict, dict) else []
            else:
                masks_obj = []

            masks = _normalize_masks(masks_obj)

            if masks.size > 0:
                frame_area = float(height * width)
                areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(float)
                keep = areas >= (min_area_ratio * frame_area)
                masks = masks[keep]

            annotated = overlay_masks(frame_bgr, masks)
            writer.write(annotated)

            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break
    finally:
        cap.release()
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Meta SAM 3 on a video and write an annotated MP4.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
        help="Path to the input video (e.g. outdoor/deer-vision/data/eval/.../segment_092754.mp4).",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        required=True,
        help="Where to save the annotated video (MP4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Inference device for the SAM 3 pipeline.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on number of frames to process (for quick tests).",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.0005,
        help="Discard masks smaller than this fraction of the frame area.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/sam3",
        help="Hugging Face model id to use (default: facebook/sam3). "
        "For local testing without gated access, you can try facebook/sam-vit-base.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {args.video_path}")

    process_video(
        video_path=args.video_path,
        output_video_path=args.output_video,
        device=args.device,
        model_id=args.model_id,
        max_frames=args.max_frames,
        min_area_ratio=args.min_area_ratio,
    )


if __name__ == "__main__":
    main()
