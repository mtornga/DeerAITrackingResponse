"""Train YOLO with repository-specific defaults using the Python API."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import ast
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO detector with pinned hyperparameters.")
    parser.add_argument("--data", type=Path, default=Path("configs/data_deer.yaml"), help="YOLO dataset YAML.")
    parser.add_argument("--hyp", type=Path, default=Path("configs/hyp_det.yaml"), help="Hyperparameters file (YAML).")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights.")
    parser.add_argument("--name", type=str, default="yolov8n_det_v01", help="Training run name.")
    parser.add_argument("--project", type=Path, default=Path("models"), help="Ultralytics project output directory.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override (cpu, mps, cuda:0, ...).")
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Optional extra overrides in key=value form (e.g., momentum=0.9).",
    )
    return parser.parse_args()


def load_hyperparameters(hyp_path: Path) -> Dict[str, Any]:
    if not hyp_path.exists():
        return {}
    with hyp_path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{hyp_path} must contain a mapping of hyperparameters.")
    return data


def parse_extra(extra_args: list[str] | None) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if not extra_args:
        return overrides
    for kv in extra_args:
        if "=" not in kv:
            raise ValueError(f"Extra argument must use key=value format, got '{kv}'")
        key, value = kv.split("=", 1)
        try:
            overrides[key] = ast.literal_eval(value)
        except Exception:
            overrides[key] = value
    return overrides


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))

    train_kwargs: Dict[str, Any] = {
        "data": str(args.data),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "project": str(args.project),
        "name": args.name,
        "workers": args.workers,
        "patience": 20,
        "pretrained": True,
    }
    if args.device:
        train_kwargs["device"] = args.device

    train_kwargs.update(load_hyperparameters(args.hyp))
    train_kwargs.update(parse_extra(args.extra))

    print("Starting YOLO training with arguments:")
    for key, value in train_kwargs.items():
        print(f"  {key}={value}")

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
