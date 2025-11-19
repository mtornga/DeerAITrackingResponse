#!/usr/bin/env python3
"""Generate A/B review overlays for recent event clips.

For each segment under an events date folder, extract sampled frames and
render overlays for multiple parameter variants. Writes results to a review
folder with an index.json summary suitable for a Streamlit UI.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


@dataclass
class Variant:
    name: str
    conf: float
    iou: float
    max_det: int = 10
    match_iou: float = 0.45
    imgsz: int = 960


def run_ffmpeg_extract(src_mp4: Path, dst_dir: Path, every_n: int = 5) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Extract every nth frame
    # Using select=not(mod(n\,%d)) to sample
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src_mp4),
        "-vf",
        f"select='not(mod(n\\,{every_n}))',setpts=N/TB",
        "-vsync",
        "vfr",
        str(dst_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events-root", type=Path, default=Path("/srv/deer-share/runs/live/events"))
    p.add_argument("--date", type=str, default=datetime.utcnow().strftime("%Y-%m-%d"))
    p.add_argument("--model", type=Path, default=REPO_ROOT / "models/yolov8m_det_v04/weights/best.pt")
    p.add_argument("--out-root", type=Path, default=Path("/srv/deer-share/runs/review"))
    p.add_argument("--policy", type=Path, default=REPO_ROOT / "outdoor/deer-vision/configs/qc_policies.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--frame-stride", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    day_dir = args.events_root / args.date
    if not day_dir.exists():
        print(f"No events under {day_dir}")
        return 0

    segments = sorted(p for p in day_dir.iterdir() if p.is_dir())
    out_day = args.out_root / args.date
    out_day.mkdir(parents=True, exist_ok=True)

    # Define A/B variants
    variants: List[Variant] = [
        Variant(name="conf030_iou045", conf=0.30, iou=0.45, imgsz=960),
        Variant(name="conf035_iou040", conf=0.35, iou=0.40, imgsz=960),
        Variant(name="conf030_iou045_1280", conf=0.30, iou=0.45, imgsz=1280),
    ]

    # index structure: {segment: {variant: {frames,tp,fp,fn,conf,iou,imgsz, frames_with_preds:int, top_pred_frames:[str], disagree_frames:[str]}}}
    index: Dict[str, Dict[str, Dict[str, float]]] = {}

    for seg_dir in segments:
        mp4s = list(seg_dir.glob("*.mp4"))
        if not mp4s:
            continue
        clip = mp4s[0]
        seg_name = seg_dir.name
        work_frames = out_day / seg_name / "frames_sampled"
        if work_frames.exists():
            shutil.rmtree(work_frames)
        work_frames.mkdir(parents=True, exist_ok=True)
        try:
            run_ffmpeg_extract(clip, work_frames, every_n=max(1, args.frame_stride))
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg failed for {clip}: {e}")
            continue

        index[seg_name] = {}
        # Collect per-variant frame stats to compute disagreements
        variant_frames: Dict[str, Dict[str, Dict]] = {}
        for v in variants:
            out_dir = out_day / seg_name / v.name
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts/visualize_errors2.py"),
                "--images",
                str(work_frames),
                "--labels",
                str(work_frames),  # no GT; will render predictions only with FP label logic not counting
                "--model",
                str(args.model),
                "--out",
                str(out_dir),
                "--limit",
                "0",
                "--conf",
                str(v.conf),
                "--nms-iou",
                str(v.iou),
                "--max-det",
                str(v.max_det),
                "--match-iou",
                str(v.match_iou),
                "--imgsz",
                str(v.imgsz),
                "--agnostic-nms",
                "--pred-classes",
                "0",
                "--eval-classes",
                "0",
                "--policy",
                str(args.policy),
            ]
            if args.device:
                cmd.extend(["--device", args.device])
            # Run and capture summary
            subprocess.run(cmd, check=True)
            summary_path = out_dir / "summary.json"
            metrics = {"fp": 0, "fn": 0, "tp": 0, "frames": 0}
            if summary_path.exists():
                try:
                    metrics.update(json.loads(summary_path.read_text()))
                except Exception:
                    pass
            # load per-frame details for detection counts
            frames_path = out_dir / "frames.json"
            frames = {}
            if frames_path.exists():
                try:
                    frames = json.loads(frames_path.read_text())
                except Exception:
                    frames = {}
            variant_frames[v.name] = frames
            # quick count of frames with any predictions
            frames_with_preds = sum(1 for k, d in frames.items() if int(d.get("pred", 0)) > 0)
            # top frames by prediction count
            top_pred_frames = [k for k, d in sorted(frames.items(), key=lambda kv: int(kv[1].get("pred", 0)), reverse=True) if int(d.get("pred", 0)) > 0][:50]
            metrics.update({
                "frames_with_preds": frames_with_preds,
                "top_pred_frames": top_pred_frames,
            })
            index[seg_name][v.name] = metrics

        # compute disagreement frames across variants (any vs zero predictions)
        all_keys = set()
        for fr in variant_frames.values():
            all_keys.update(fr.keys())
        disagree: List[str] = []
        for key in sorted(all_keys):
            preds = [int(variant_frames[v.name].get(key, {}).get("pred", 0)) for v in variants]
            if any(p > 0 for p in preds) and not all(p > 0 for p in preds):
                disagree.append(key)
        disagree = disagree[:50]
        # store the same disagreement list into each variant entry for convenience
        for v in variants:
            if v.name in index[seg_name]:
                index[seg_name][v.name]["disagree_frames"] = disagree

    (out_day / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Review artifacts written to {out_day}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
