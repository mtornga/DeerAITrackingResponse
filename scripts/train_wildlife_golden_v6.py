#!/usr/bin/env python3
"""Train day/night YOLOv8n and RT-DETR models on wildlife_golden_v6 datasets."""

from __future__ import annotations

from pathlib import Path
import shutil

from ultralytics import YOLO, RTDETR


def train_yolo(data_path: Path, name: str, batch: int = 8) -> Path:
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_path),
        epochs=80,
        batch=batch,
        imgsz=960,
        device=0,
        project="runs/train",
        name=name,
        exist_ok=True,
    )
    best = Path("runs/train") / name / "weights" / "best.pt"
    return best


def train_rtdetr(data_path: Path, name: str) -> Path:
    model = RTDETR("rtdetr-l.pt")
    model.train(
        data=str(data_path),
        epochs=80,
        batch=2,  # Avoid OOM
        imgsz=960,
        device=0,
        project="runs/train",
        name=name,
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.0001,
        patience=20,
    )
    best = Path("runs/train") / name / "weights" / "best.pt"
    return best


def copy_best(best: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, target)
    print(f"Saved {target}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    day_yaml = repo / "outdoor/deer-vision/data/datasets/wildlife_golden_v6_day/data.yaml"
    night_yaml = repo / "outdoor/deer-vision/data/datasets/wildlife_golden_v6_night/data.yaml"

    # YOLO day/night
    best = train_yolo(day_yaml, "yolov8n_golden_v6_day")
    copy_best(best, repo / "models/yolov8n_wildlife_v6_day.pt")

    best = train_yolo(night_yaml, "yolov8n_golden_v6_night")
    copy_best(best, repo / "models/yolov8n_wildlife_v6_night.pt")

    # RT-DETR day/night (batch=2)
    best = train_rtdetr(day_yaml, "rtdetr_golden_v6_day")
    copy_best(best, repo / "models/rtdetr_wildlife_v6_day.pt")

    best = train_rtdetr(night_yaml, "rtdetr_golden_v6_night")
    copy_best(best, repo / "models/rtdetr_wildlife_v6_night.pt")


if __name__ == "__main__":
    main()
