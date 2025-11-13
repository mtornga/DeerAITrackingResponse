"""Export trained checkpoints to ONNX/TorchScript."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO weights for edge deployment.")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained weights (best.pt).")
    parser.add_argument("--formats", nargs="+", default=["onnx", "torchscript"], help="Formats to export.")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size for export.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        "yolo",
        "export",
        f"model={args.model}",
        f"format={','.join(args.formats)}",
        f"imgsz={args.imgsz}",
    ]
    if args.device:
        cmd.append(f"device={args.device}")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
