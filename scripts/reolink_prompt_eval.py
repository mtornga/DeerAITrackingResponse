#!/usr/bin/env python3
"""
Evaluate GPT prompt performance against Reolink calibration snapshots.

The script scans calibration JSON files (default: calibration/reolink_gpt/calib_*.json),
compares GPT-projected inches to the known ground-truth encoded in each filename,
and prints a colourised summary with aggregate error statistics.

Each run appends its metrics to calibration/reolink_gpt/prompt_eval_history.json
so subsequent runs can report deltas relative to the previous iteration.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


# ---------- Configuration ----------
DEFAULT_CALIB_DIR = Path("calibration/reolink_gpt")
DEFAULT_PATTERN = "calib_*.json"
HISTORY_PATH = DEFAULT_CALIB_DIR / "prompt_eval_history.json"
PROMPT_FILE = Path("perception/reolink_gpt_snapshot.py")

# ---------- ANSI helpers ----------
RESET = "\033[0m"
COLORS = {
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "bold": "\033[1m",
}


def colour(text: str, name: str) -> str:
    prefix = COLORS.get(name.lower())
    if not prefix:
        return text
    return f"{prefix}{text}{RESET}"


# ---------- Data classes ----------
@dataclass(frozen=True)
class SampleResult:
    name: str
    actual: Tuple[float, float]
    projected: Tuple[float, float]

    @property
    def error(self) -> Tuple[float, float]:
        return (self.projected[0] - self.actual[0], self.projected[1] - self.actual[1])

    @property
    def abs_error(self) -> Tuple[float, float]:
        ex, ey = self.error
        return abs(ex), abs(ey)

    @property
    def combined_abs(self) -> float:
        ex, ey = self.abs_error
        return math.hypot(ex, ey)


@dataclass(frozen=True)
class AggregateMetrics:
    samples: List[SampleResult]

    @property
    def coord_count(self) -> int:
        return len(self.samples) * 2

    @property
    def total_abs_error(self) -> float:
        return sum(sum(sample.abs_error) for sample in self.samples)

    @property
    def mean_abs_error(self) -> float:
        if not self.samples:
            return 0.0
        return self.total_abs_error / self.coord_count

    @property
    def median_abs_error(self) -> float:
        values = [v for sample in self.samples for v in sample.abs_error]
        return statistics.median(values) if values else 0.0

    def tolerance_count(self, threshold: float) -> int:
        return sum(
            1
            for sample in self.samples
            for value in sample.abs_error
            if value <= threshold
        )

    def sorted_by_error(self) -> List[SampleResult]:
        return sorted(self.samples, key=lambda s: s.combined_abs, reverse=True)


# ---------- Prompt hashing ----------
def _load_prompt_hashes(path: Path) -> Tuple[str, str]:
    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ("", "")

    system_prompt = ""
    user_prompt = ""

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        tree = None

    if tree:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name == "SYSTEM_PROMPT":
                    try:
                        system_prompt = ast.literal_eval(node.value)
                    except Exception:
                        system_prompt = ""
                elif name == "USER_PROMPT":
                    try:
                        user_prompt = ast.literal_eval(node.value)
                    except Exception:
                        user_prompt = ""

    system_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest() if system_prompt else ""
    user_hash = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest() if user_prompt else ""
    return system_hash, user_hash


# ---------- Calibration utilities ----------
COORD_RE = re.compile(r"calib_(\d{2})(\d{2})")


def _parse_actual_from_name(path: Path) -> Optional[Tuple[float, float]]:
    match = COORD_RE.search(path.stem)
    if not match:
        return None
    try:
        x = float(int(match.group(1)))
        y = float(int(match.group(2)))
    except ValueError:
        return None
    return x, y


def _load_sample(path: Path) -> Optional[SampleResult]:
    actual = _parse_actual_from_name(path)
    if actual is None:
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    proj_node = data.get("cutebot_nose_inches_projected")
    if not isinstance(proj_node, dict):
        return None
    try:
        proj = (float(proj_node["x"]), float(proj_node["y"]))
    except (KeyError, TypeError, ValueError):
        return None

    return SampleResult(name=path.name, actual=actual, projected=proj)


def load_samples(directory: Path, pattern: str) -> List[SampleResult]:
    samples: List[SampleResult] = []
    for path in sorted(directory.glob(pattern)):
        sample = _load_sample(path)
        if sample:
            samples.append(sample)
    return samples


# ---------- History handling ----------
def read_history(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def append_history(path: Path, entry: dict) -> None:
    history = read_history(path)
    history.append(entry)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


# ---------- Reporting ----------
def format_delta(value: float) -> str:
    arrow = "↑" if value > 0 else ("↓" if value < 0 else "→")
    colour_name = "red" if value > 0 else ("green" if value < 0 else "yellow")
    return colour(f"{arrow}{abs(value):.3f}", colour_name)


def format_delta_count(value: int) -> str:
    if value > 0:
        return colour(f"↑{value}", "green")
    if value < 0:
        return colour(f"↓{abs(value)}", "red")
    return colour("→0", "yellow")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GPT prompt accuracy over calibration snapshots."
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=DEFAULT_CALIB_DIR,
        help="Directory containing calibration JSON files.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Filename glob to select calibration JSON files (default: calib_*.json).",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=HISTORY_PATH,
        help="Path to evaluation history JSON file.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of highest-error samples to display.",
    )
    args = parser.parse_args()

    samples = load_samples(args.calib_dir, args.pattern)
    if not samples:
        raise SystemExit("No calibration samples found. Check --calib-dir and --pattern.")

    metrics = AggregateMetrics(samples)
    system_hash, user_hash = _load_prompt_hashes(PROMPT_FILE)
    timestamp = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    tolerance_1 = metrics.tolerance_count(1.0)
    tolerance_2 = metrics.tolerance_count(2.0)
    total_coords = metrics.coord_count

    history = read_history(args.history)
    previous = history[-1] if history else None

    entry = {
        "timestamp": timestamp,
        "calibration_dir": str(args.calib_dir),
        "pattern": args.pattern,
        "num_samples": len(samples),
        "total_coords": total_coords,
        "total_abs_error": metrics.total_abs_error,
        "mean_abs_error": metrics.mean_abs_error,
        "median_abs_error": metrics.median_abs_error,
        "tolerance": {
            "1in": tolerance_1,
            "2in": tolerance_2,
            "total": total_coords,
        },
        "prompt_hash": {"system": system_hash, "user": user_hash},
        "samples": [
            {
                "name": sample.name,
                "actual": sample.actual,
                "projected": sample.projected,
                "error": sample.error,
            }
            for sample in samples
        ],
    }

    append_history(args.history, entry)

    # ----- Output -----
    print(colour("\nPrompt Evaluation", "bold"))
    print(f"{colour('Timestamp:', 'cyan')} {timestamp}")
    print(f"{colour('Samples:', 'cyan')} {len(samples)} ({args.pattern})")
    print(f"{colour('Prompt hash:', 'cyan')} system={system_hash[:8]} user={user_hash[:8]}")

    mean_str = f"{metrics.mean_abs_error:.3f}\""
    median_str = f"{metrics.median_abs_error:.3f}\""
    total_str = f"{metrics.total_abs_error:.3f}\""

    if previous:
        delta_mean = metrics.mean_abs_error - previous.get("mean_abs_error", 0.0)
        delta_total = metrics.total_abs_error - previous.get("total_abs_error", 0.0)
        delta_t1 = tolerance_1 - previous.get("tolerance", {}).get("1in", 0)
        delta_t2 = tolerance_2 - previous.get("tolerance", {}).get("2in", 0)
    else:
        delta_mean = delta_total = 0.0
        delta_t1 = delta_t2 = 0

    print(
        f"{colour('Total abs error:', 'magenta')} {total_str} "
        f"({format_delta(delta_total)} vs last run)"
    )
    print(
        f"{colour('Mean abs error:', 'magenta')} {mean_str} "
        f"({format_delta(delta_mean)} vs last run)"
    )
    print(f"{colour('Median abs error:', 'magenta')} {median_str}")

    tol_line = (
        f"{colour('Tolerance:', 'cyan')} "
        f"≤1\" {tolerance_1}/{total_coords} "
        f"({format_delta_count(delta_t1)})  "
        f"≤2\" {tolerance_2}/{total_coords} "
        f"({format_delta_count(delta_t2)})"
    )
    print(tol_line)

    print(colour("\nLargest Errors", "bold"))
    top_n = min(args.top, len(samples))
    for sample in metrics.sorted_by_error()[:top_n]:
        ex, ey = sample.error
        ae_x, ae_y = sample.abs_error
        combined = sample.combined_abs
        name = colour(sample.name, "yellow")
        print(
            f"  {name:<25} actual=({sample.actual[0]:5.2f}, {sample.actual[1]:5.2f}) "
            f"projected=({sample.projected[0]:5.2f}, {sample.projected[1]:5.2f}) "
            f"err=({ex:+5.2f}, {ey:+5.2f}) |abs|=({ae_x:4.2f}, {ae_y:4.2f}) "
            f"combined={combined:5.2f}"
        )

    print(colour("\nPer-sample Summary", "bold"))
    for sample in samples:
        ex, ey = sample.error
        status = colour("OK", "green")
        if max(abs(ex), abs(ey)) > 2.0:
            status = colour(">2in", "red")
        elif max(abs(ex), abs(ey)) > 1.0:
            status = colour(">1in", "yellow")
        print(
            f"  {sample.name:<25} "
            f"actual=({sample.actual[0]:4.1f}, {sample.actual[1]:4.1f}) "
            f"proj=({sample.projected[0]:4.1f}, {sample.projected[1]:4.1f}) "
            f"err=({ex:+4.1f}, {ey:+4.1f}) [{status}]"
        )

    print()


if __name__ == "__main__":
    main()
