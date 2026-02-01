"""Queue Enricher run logging helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QueueEnricherRunLog:
    started_at: str
    finished_at: str
    status: str
    summary: str
    stats: Dict[str, Any]
    clips: List[Dict[str, Any]]


def format_utc(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_daily_log_path(logs_dir: Path, when: datetime) -> Path:
    day = when.astimezone(UTC).strftime("%Y-%m-%d")
    return logs_dir / f"queue-enricher-{day}.json"


def load_daily_log(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)


def append_run_log(
    logs_dir: Path,
    run_log: QueueEnricherRunLog,
    when: Optional[datetime] = None,
    write_latest: bool = True,
) -> Path:
    timestamp = when or datetime.now(UTC)
    daily_path = build_daily_log_path(logs_dir, timestamp)
    runs = load_daily_log(daily_path)
    runs.append(run_log.__dict__)
    _atomic_write(daily_path, runs)

    if write_latest:
        latest_path = logs_dir / "queue-enricher-latest.json"
        _atomic_write(latest_path, run_log.__dict__)

    return daily_path
