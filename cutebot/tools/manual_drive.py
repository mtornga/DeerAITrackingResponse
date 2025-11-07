from __future__ import annotations

import argparse
import asyncio
import re
from typing import Optional, Tuple, List

from cutebot.controller_auto import CutebotUARTSession

DURATION_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*(ms|s)?", re.IGNORECASE)


def parse_duration_token(token: str) -> Optional[int]:
    token = token.strip().lower()
    if not token:
        return None
    multiplier = 1.0
    if token.endswith("ms"):
        token = token[:-2]
        multiplier = 1.0
    elif token.endswith("s"):
        token = token[:-1]
        multiplier = 1000.0
    try:
        value = float(token)
    except ValueError:
        return None
    return max(0, int(round(value * multiplier)))


def match_to_duration(match: re.Match[str]) -> Optional[int]:
    unit = match.group(2) or ""
    return parse_duration_token(f"{match.group(1)}{unit}")


def parse_drive_line(line: str, default_duration: int) -> Optional[Tuple[int, int, int]]:
    matches = list(DURATION_RE.finditer(line))
    if not matches:
        return None

    speeds: List[int] = []
    duration: Optional[int] = None
    for match in matches:
        unit = match.group(2)
        value_str = match.group(1)
        if unit:
            maybe = match_to_duration(match)
            if maybe is not None:
                duration = maybe
            continue
        if len(speeds) < 2:
            try:
                speeds.append(int(round(float(value_str))))
                continue
            except ValueError:
                continue
        if duration is None:
            maybe = parse_duration_token(value_str)
            if maybe is not None:
                duration = maybe

    if len(speeds) < 2:
        return None
    if duration is None:
        duration = default_duration
    left, right = speeds[:2]
    return left, right, duration


def render_help(default_duration: int) -> None:
    lines = [
        "Commands:",
        f"  forward [duration_ms]         Drive forward at the configured forward speed "
        f"(default {default_duration} ms).",
        "  forward [speed] [duration_ms] Override the forward speed and/or duration for one move.",
        "  L=<left> R=<right> <duration> Explicit wheel speeds (commas or spaces allowed).",
        "  <left> <right> <duration>     Shorthand for explicit wheel speeds.",
        "  speed <value>                 Update the default forward speed (clamped to ±100).",
        "  heading                       Request a single magnetometer heading sample.",
        "  stop                          Send an immediate stop command.",
        "  wait <duration>               Pause locally (accepts ms or s suffix).",
        "  help                          Show this message again.",
        "  quit                          Exit the controller.",
    ]
    for line in lines:
        print(line)


async def interactive_loop(
    controller: CutebotUARTSession,
    *,
    state: dict,
    default_duration: int,
    wait_default_ms: int,
) -> None:
    loop = asyncio.get_running_loop()

    def prompt() -> Optional[str]:
        try:
            return input("cutebot> ")
        except EOFError:
            return None
        except KeyboardInterrupt:
            return None

    print("Type 'help' for available commands. Use Ctrl+D or Ctrl+C to exit.")

    while True:
        line = await loop.run_in_executor(None, prompt)
        if line is None:
            print("Exiting controller.")
            break
        line = line.strip()
        if not line:
            continue

        lowered = line.lower()
        if lowered in {"quit", "exit", "q"}:
            break
        if lowered in {"help", "?"}:
            render_help(default_duration)
            continue
        if lowered.startswith("speed"):
            matches = list(DURATION_RE.finditer(line))
            if matches:
                try:
                    target = int(round(float(matches[0].group(1))))
                    clamped = CutebotUARTSession._clamp_speed(target)
                    state["forward_speed"] = clamped
                    print(f"[config] Forward speed set to {clamped}.")
                except ValueError:
                    print("[config] Could not parse speed.")
            else:
                print(f"[config] Forward speed is {state['forward_speed']}.")
            continue
        if lowered.startswith("forward"):
            tail = line[len("forward") :].strip()
            forward_speed = state["forward_speed"]
            duration_ms = default_duration
            matches = list(DURATION_RE.finditer(tail))
            if matches:
                if len(matches) >= 2:
                    try:
                        forward_speed = int(round(float(matches[0].group(1))))
                    except ValueError:
                        pass
                    maybe = match_to_duration(matches[1])
                    if maybe is not None:
                        duration_ms = maybe
                else:
                    match = matches[0]
                    if "speed" in tail.lower() and match.group(2) is None:
                        try:
                            forward_speed = int(round(float(match.group(1))))
                        except ValueError:
                            pass
                    else:
                        maybe = match_to_duration(match)
                        if maybe is not None:
                            duration_ms = maybe
            clamped = CutebotUARTSession._clamp_speed(forward_speed)
            print(f"[drive] Forward L={clamped} R={clamped} for {duration_ms} ms")
            try:
                await controller.drive_timed(clamped, clamped, duration_ms)
            except Exception as exc:
                print(f"[drive] Failed: {exc}")
            continue
        if lowered.startswith("stop") or lowered == "s":
            try:
                await controller.stop()
                print("[drive] Stop command sent.")
            except Exception as exc:
                print(f"[drive] Stop failed: {exc}")
            continue
        if lowered.startswith("heading"):
            try:
                sample = await controller.request_heading(timeout=3.0)
            except Exception as exc:
                print(f"[heading] Request failed: {exc}")
                continue
            if sample is None:
                print("[heading] Timed out waiting for response.")
            else:
                print(f"[heading] {sample:.1f}°")
            continue
        if lowered.startswith("wait"):
            matches = list(DURATION_RE.finditer(line))
            if matches:
                maybe = match_to_duration(matches[0])
                if maybe is None:
                    print("[wait] Could not parse duration.")
                    continue
                wait_sec = maybe / 1000.0
            else:
                wait_sec = wait_default_ms / 1000.0
            print(f"[wait] Sleeping for {wait_sec:.2f} s")
            await asyncio.sleep(wait_sec)
            continue

        drive_cmd = parse_drive_line(line, default_duration)
        if drive_cmd:
            left, right, duration_ms = drive_cmd
            cl = CutebotUARTSession._clamp_speed(left)
            cr = CutebotUARTSession._clamp_speed(right)
            print(f"[drive] L={cl} R={cr} for {duration_ms} ms")
            try:
                await controller.drive_timed(cl, cr, duration_ms)
            except Exception as exc:
                print(f"[drive] Failed: {exc}")
            continue

        print("Unrecognised command. Type 'help' to list options.")


async def run_cli(args: argparse.Namespace) -> None:
    forward_speed = CutebotUARTSession._clamp_speed(args.forward_speed)
    default_duration = max(20, int(args.default_duration_ms))
    wait_default_ms = max(50, int(args.wait_default_ms))

    def message_handler(msg: str) -> None:
        clean = msg.strip()
        if clean:
            print(f"[bot] {clean}")

    state = {"forward_speed": forward_speed}

    async with CutebotUARTSession(
        message_handler=message_handler,
        timeout=args.timeout,
        verbose=args.verbose,
        name_prefix=args.name_prefix,
    ) as controller:
        heading_stream_enabled = False
        try:
            if args.enable_heading_stream:
                await controller.enable_heading_stream(True)
                heading_stream_enabled = True
            await interactive_loop(
                controller,
                state=state,
                default_duration=default_duration,
                wait_default_ms=wait_default_ms,
            )
        finally:
            if heading_stream_enabled:
                try:
                    await controller.enable_heading_stream(False)
                except Exception:
                    pass
            try:
                await controller.stop()
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive Cutebot controller for manual wheel commands."
    )
    parser.add_argument(
        "--forward-speed",
        type=int,
        default=25,
        help="Default forward speed applied by the 'forward' command (default: 25).",
    )
    parser.add_argument(
        "--default-duration-ms",
        type=int,
        default=300,
        help="Fallback duration (ms) when a drive command omits it (default: 300).",
    )
    parser.add_argument(
        "--wait-default-ms",
        type=int,
        default=500,
        help="Default wait duration (ms) when using the 'wait' command without arguments.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="BLE connection timeout in seconds (default: 15).",
    )
    parser.add_argument(
        "--name-prefix",
        default="BBC micro:bit",
        help="BLE device name prefix to search for (default: 'BBC micro:bit').",
    )
    parser.add_argument(
        "--enable-heading-stream",
        action="store_true",
        help="Enable continuous heading stream while the controller runs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging from the BLE session.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        asyncio.run(run_cli(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")


if __name__ == "__main__":
    main()
