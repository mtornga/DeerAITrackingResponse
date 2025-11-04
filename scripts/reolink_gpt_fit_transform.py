#!/usr/bin/env python3
"""
Compute the affine transform that maps GPT-projected inches to real board inches.

Usage (defaults to the standard five-cutebot poses):
    PYTHONPATH=. python scripts/reolink_gpt_fit_transform.py

Optionally supply custom calibration JSONs (format: ACTUAL_X,ACTUAL_Y:path.json):
    PYTHONPATH=. python scripts/reolink_gpt_fit_transform.py \
        --calibration 0,0:calibration/reolink_gpt/calib_00.json \
        --calibration 30,0:calibration/reolink_gpt/calib_3000.json \
        --calibration 0,24:calibration/reolink_gpt/calib_0024.json \
        --calibration 30,24:calibration/reolink_gpt/calib_3024.json \
        --calibration 15,12:calibration/reolink_gpt/calib_1512.json
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

DEFAULT_CALIB_DIR = Path("calibration/reolink_gpt")
DEFAULT_OUTPUT = DEFAULT_CALIB_DIR / "transform.json"


@dataclass(frozen=True)
class CalibrationSample:
    """Pair an observed GPT projection with the true board coordinate."""

    actual_xy: Tuple[float, float]
    projected_xy: Tuple[float, float]
    source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit an affine transform to map GPT inches into real board inches."
    )
    parser.add_argument(
        "--calibration",
        dest="calibrations",
        action="append",
        default=None,
        metavar="X,Y:PATH",
        help="Calibration sample in the form actual_x,actual_y:/path/to/json. "
        "Repeat for each pose. When omitted, the script scans the default calibration folder.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIB_DIR,
        help="Directory containing calibration JSON files (used when --calibration is omitted).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination path for the fitted transform JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing transform file.",
    )
    return parser.parse_args()


def _parse_sample_spec(spec: str) -> Tuple[Tuple[float, float], Path]:
    try:
        coords_part, path_part = spec.split(":", 1)
        x_str, y_str = coords_part.split(",", 1)
        actual_xy = (float(x_str), float(y_str))
        path = Path(path_part)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError(f"Invalid calibration spec '{spec}': expected X,Y:/path.json") from exc
    return actual_xy, path


def _read_projected_inches(json_path: Path) -> Tuple[float, float]:
    try:
        payload = json.loads(json_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Calibration file not found: {json_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Calibration file is not valid JSON: {json_path}") from exc

    node = payload.get("cutebot_nose_inches_projected") or payload.get("cutebot_nose_inches")
    if not isinstance(node, dict) or "x" not in node or "y" not in node:
        raise ValueError(f"Calibration JSON missing projected inches: {json_path}")
    return float(node["x"]), float(node["y"])


def _infer_default_calibrations(calib_dir: Path) -> List[Tuple[Tuple[float, float], Path]]:
    pairs: List[Tuple[Tuple[float, float], Path]] = []
    for json_path in sorted(calib_dir.glob("calib_*.json")):
        stem = json_path.stem  # e.g. calib_1512
        digits = stem.replace("calib_", "")
        if len(digits) < 2 or not digits.isdigit():
            continue
        y_val = float(int(digits[-2:]))
        x_val = float(int(digits[:-2] or "0"))
        pairs.append(((x_val, y_val), json_path))
    if not pairs:
        raise FileNotFoundError(
            f"No calibration JSON files found in {calib_dir}. "
            "Supply explicit samples with --calibration."
        )
    return pairs


def load_samples(args: argparse.Namespace) -> List[CalibrationSample]:
    if args.calibrations:
        raw_pairs = [_parse_sample_spec(spec) for spec in args.calibrations]
    else:
        raw_pairs = _infer_default_calibrations(args.calibration_dir)

    samples: List[CalibrationSample] = []
    for actual_xy, path in raw_pairs:
        projected = _read_projected_inches(path)
        samples.append(CalibrationSample(actual_xy=actual_xy, projected_xy=projected, source=path))
    if len(samples) < 3:
        raise ValueError("Need at least three calibration samples to fit an affine transform.")
    return samples


def fit_affine(samples: Sequence[CalibrationSample]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solve for matrix A and offset b such that A * gpt + b ~= actual."""
    rows: List[List[float]] = []
    rhs: List[float] = []
    for sample in samples:
        gx, gy = sample.projected_xy
        rx, ry = sample.actual_xy
        rows.append([gx, gy, 1.0, 0.0, 0.0, 0.0])
        rows.append([0.0, 0.0, 0.0, gx, gy, 1.0])
        rhs.extend([rx, ry])

    A = np.asarray(rows, dtype=float)
    B = np.asarray(rhs, dtype=float)
    params, *_ = np.linalg.lstsq(A, B, rcond=None)

    matrix = params[[0, 1, 3, 4]].reshape(2, 2)
    offset = params[[2, 5]]

    # Compute root-mean-square error in inches for sanity checking.
    total_sq_err = 0.0
    for sample in samples:
        predicted = matrix @ np.asarray(sample.projected_xy, dtype=float) + offset
        err_x = predicted[0] - sample.actual_xy[0]
        err_y = predicted[1] - sample.actual_xy[1]
        total_sq_err += err_x**2 + err_y**2
    rmse = math.sqrt(total_sq_err / len(samples))
    return matrix, offset, rmse


def write_transform(path: Path, matrix: np.ndarray, offset: np.ndarray, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Use --overwrite to replace the existing transform."
        )
    payload = {
        "matrix": matrix.tolist(),
        "offset": offset.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    samples = load_samples(args)
    matrix, offset, rmse = fit_affine(samples)
    write_transform(args.output, matrix, offset, overwrite=args.overwrite)

    print(f"Wrote transform to {args.output}")
    print("Affine matrix:")
    print(matrix)
    print(f"Offset: {offset}")
    print(f"RMSE: {rmse:.3f} inches across {len(samples)} samples")
    for sample in samples:
        pred = matrix @ np.asarray(sample.projected_xy, dtype=float) + offset
        err_x = pred[0] - sample.actual_xy[0]
        err_y = pred[1] - sample.actual_xy[1]
        print(
            f"  {sample.source}: GPT=({sample.projected_xy[0]:.3f}, {sample.projected_xy[1]:.3f})"
            f" -> actual=({sample.actual_xy[0]:.2f}, {sample.actual_xy[1]:.2f});"
            f" error=({err_x:+.3f}, {err_y:+.3f})"
        )


if __name__ == "__main__":
    main()

