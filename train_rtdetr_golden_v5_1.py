#!/usr/bin/env python3
"""Train RT-DETR on wildlife_golden_v5 dataset (v5.1 models).

Uses a small batch size to avoid OOM on the 10GB GPU.
"""
from pathlib import Path
import shutil

from ultralytics import RTDETR


def main() -> None:
    print("Loading RT-DETR-L base model...")
    model = RTDETR("rtdetr-l.pt")

    print("Starting training on wildlife_golden_v5 (batch=2 for OOM avoidance)...")
    model.train(
        data="outdoor/deer-vision/data/datasets/wildlife_golden_v5/data.yaml",
        epochs=80,
        batch=2,
        imgsz=960,
        device=0,
        project="runs/train",
        name="rtdetr_golden_v5.1",
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.0001,
        patience=20,
    )

    print("Training complete!")
    best_path = Path("runs/train/rtdetr_golden_v5.1/weights/best.pt")
    print(f"Best model: {best_path}")

    target = Path("models/rtdetr_wildlife_v5.1.pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_path, target)
    print(f"Copied to {target}")


if __name__ == "__main__":
    main()
