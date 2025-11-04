#!/usr/bin/env python3
"""
Regenerate GPT pose JSON files for all Reolink calibration snapshots.

Usage:
    PYTHONPATH=. python scripts/reolink_gpt_refresh_prompts.py

Options allow selecting a glob pattern, controlling concurrency, and
limiting retries. The script queries GPT for each JPEG via
`perception.reolink_gpt_snapshot.query_cutebot_pose`, writes the updated
JSON next to the image, and prints a coloured summary.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.reolink_gpt_snapshot import query_cutebot_pose  # noqa: E402


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


@dataclass
class RefreshResult:
    image: Path
    json_path: Path
    success: bool
    error: Optional[str] = None


async def refresh_one(
    image_path: Path,
    *,
    retries: int,
    semaphore: asyncio.Semaphore,
    verbose: bool = False,
) -> RefreshResult:
    json_path = image_path.with_suffix(".json")

    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                observation = await asyncio.to_thread(query_cutebot_pose, image_path)
                payload = observation.as_dict()
                json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                if verbose:
                    proj = payload.get("cutebot_nose_inches_projected")
                    print(
                        f"{colour('✓', 'green')} {image_path.name} "
                        f"-> {json_path.name} proj={proj}",
                        flush=True,
                    )
                return RefreshResult(image=image_path, json_path=json_path, success=True)
            except Exception as exc:  # noqa: BLE001 - want to capture any failure
                if attempt >= retries:
                    return RefreshResult(
                        image=image_path,
                        json_path=json_path,
                        success=False,
                        error=str(exc),
                    )
                if verbose:
                    print(
                        colour(
                            f"Retry {attempt}/{retries} for {image_path.name}: {exc}",
                            "yellow",
                        ),
                        flush=True,
                    )
                await asyncio.sleep(1.5 * attempt)

    return RefreshResult(image=image_path, json_path=json_path, success=False, error="Unknown failure")


def scan_images(directory: Path, pattern: str) -> list[Path]:
    return sorted(p for p in directory.glob(pattern) if p.is_file())


async def main_async(args: argparse.Namespace) -> int:
    image_paths = scan_images(args.calib_dir, args.pattern)
    if not image_paths:
        print(colour("No matching calibration images found.", "red"), file=sys.stderr)
        return 1

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(
            refresh_one(
                path,
                retries=args.retries,
                semaphore=semaphore,
                verbose=args.verbose,
            )
        )
        for path in image_paths
    ]

    results = await asyncio.gather(*tasks)

    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    print(colour("\nRefresh Summary", "bold"))
    print(f"{colour('Total images:', 'cyan')} {len(results)}")
    print(f"{colour('Succeeded:', 'green')} {len(successes)}")
    print(f"{colour('Failed:', 'red')} {len(failures)}")

    if successes:
        print(colour("\nUpdated files:", "magenta"))
        for res in successes:
            print(f"  {res.json_path}")

    if failures:
        print(colour("\nFailures:", "red"))
        for res in failures:
            print(f"  {res.image.name}: {res.error}")
        return 2

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate GPT JSON outputs for Reolink calibration snapshots."
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=Path("calibration/reolink_gpt"),
        help="Directory containing Reolink calibration images (default: calibration/reolink_gpt).",
    )
    parser.add_argument(
        "--pattern",
        default="calib_*.jpg",
        help="Glob pattern to select calibration images (default: calib_*.jpg).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Maximum number of concurrent GPT requests (default: 3).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries per image on failure (default: 3).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image progress updates.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        exit_code = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print(colour("\nAborted by user.", "yellow"))
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
