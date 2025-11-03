from __future__ import annotations

import argparse
import asyncio
from statistics import mean, stdev
from typing import List

from cutebot.controller_auto import CutebotUARTSession


async def collect_headings(count: int, interval: float) -> List[float]:
    readings: List[float] = []
    async with CutebotUARTSession(verbose=False) as controller:
        await controller.enable_heading_stream(True)
        try:
            while len(readings) < count:
                heading = controller.last_heading
                if heading is not None:
                    readings.append(heading)
                    print(f"HEADING {len(readings):03d}: {heading:.1f}°")
                await asyncio.sleep(interval)
        finally:
            await controller.enable_heading_stream(False)
    return readings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream magnetometer heading readings from the Cutebot micro:bit."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to capture (default: 50).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="Seconds between samples (default: 0.25).",
    )
    args = parser.parse_args()

    readings = asyncio.run(collect_headings(args.samples, args.interval))
    if not readings:
        print("No heading samples were recorded.")
        return

    avg = mean(readings)
    sd = stdev(readings) if len(readings) > 1 else 0.0
    print(f"\nAverage heading: {avg:.2f}° (σ={sd:.2f}°) from {len(readings)} samples.")


if __name__ == "__main__":
    main()
