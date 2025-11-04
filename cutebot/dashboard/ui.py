from __future__ import annotations

import asyncio
from collections import deque
from typing import Deque

from rich.console import RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .bus import GLOBAL_EVENT_BUS
from .events import ControllerCommand, GPTPoseSample, HeadingSample, LogMessage


def _render_heading_pane(samples: Deque[HeadingSample]) -> RenderableType:
    table = Table.grid(expand=True)
    table.add_column(justify="right", style="cyan")
    table.add_column(justify="right", style="white")
    for sample in reversed(samples):
        calibrated = (
            f"{sample.calibrated_degrees:.1f}°" if sample.calibrated_degrees is not None else "-"
        )
        table.add_row(f"{sample.raw_degrees:.1f}°", calibrated)
    if not samples:
        table.add_row("—", "—")
    return Panel(table, title="Heading (raw → calibrated)")


def _render_gpt_pane(samples: Deque[GPTPoseSample]) -> RenderableType:
    table = Table.grid(expand=True)
    table.add_column(style="magenta", justify="right")
    table.add_column(style="white", justify="right")
    for sample in reversed(samples):
        table.add_row(
            f"({sample.x_in:.2f}, {sample.y_in:.2f})\"",
            f"conf={sample.confidence:.2f}",
        )
    if not samples:
        table.add_row("—", "—")
    return Panel(table, title="GPT Nose (projected inches)")


def _render_command_pane(samples: Deque[ControllerCommand]) -> RenderableType:
    table = Table.grid(expand=True)
    table.add_column(style="green", justify="left")
    table.add_column(style="white", justify="left")
    for cmd in reversed(samples):
        pose = f"{cmd.pose_xy[0]:.2f}, {cmd.pose_xy[1]:.2f}"
        heading = f"{cmd.pose_heading:.1f}°" if cmd.pose_heading is not None else "-"
        table.add_row(
            f"#{cmd.iteration} {cmd.action}",
            f"L={cmd.left_speed} R={cmd.right_speed} {cmd.duration_ms}ms | pose=({pose}) heading={heading}",
        )
    if not samples:
        table.add_row("—", "—")
    return Panel(table, title="Controller Commands")


def _render_log_pane(samples: Deque[LogMessage]) -> RenderableType:
    table = Table.grid(expand=True)
    table.add_column(style="yellow")
    for msg in reversed(samples):
        table.add_row(msg.text)
    if not samples:
        table.add_row("Logs will appear here.")
    return Panel(table, title="Logs")


async def run_dashboard(stop_event: asyncio.Event, max_lines: int = 20) -> None:
    heading_queue = GLOBAL_EVENT_BUS.subscribe(HeadingSample)
    gpt_queue = GLOBAL_EVENT_BUS.subscribe(GPTPoseSample)
    command_queue = GLOBAL_EVENT_BUS.subscribe(ControllerCommand)
    log_queue = GLOBAL_EVENT_BUS.subscribe(LogMessage)

    headings: Deque[HeadingSample] = deque(maxlen=max_lines)
    gpt_samples: Deque[GPTPoseSample] = deque(maxlen=max_lines)
    commands: Deque[ControllerCommand] = deque(maxlen=max_lines)
    logs: Deque[LogMessage] = deque(maxlen=max_lines)

    async def consume(queue, container):
        while not stop_event.is_set():
            item = await queue.get()
            container.append(item)

    consumers = [
        asyncio.create_task(consume(heading_queue, headings)),
        asyncio.create_task(consume(gpt_queue, gpt_samples)),
        asyncio.create_task(consume(command_queue, commands)),
        asyncio.create_task(consume(log_queue, logs)),
    ]

    layout = Layout()
    layout.split_row(
        Layout(name="heading"),
        Layout(name="gpt"),
        Layout(name="commands"),
        Layout(name="logs"),
    )

    with Live(layout, refresh_per_second=4, screen=False) as live:
        try:
            while not stop_event.is_set():
                layout["heading"].update(_render_heading_pane(headings))
                layout["gpt"].update(_render_gpt_pane(gpt_samples))
                layout["commands"].update(_render_command_pane(commands))
                layout["logs"].update(_render_log_pane(logs))
                await asyncio.sleep(0.25)
        finally:
            stop_event.set()
            for consumer in consumers:
                consumer.cancel()
