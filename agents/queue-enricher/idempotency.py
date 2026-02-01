"""Queue Enricher idempotency helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class IdempotencyCheck:
    should_skip: bool
    reasons: List[str]


def load_queue_meta(queue_meta_path: Path) -> Dict[str, Any]:
    if not queue_meta_path.exists():
        return {}
    try:
        payload = json.loads(queue_meta_path.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def has_enrichment(queue_meta: Dict[str, Any]) -> bool:
    enrichment = queue_meta.get("enrichment") if isinstance(queue_meta, dict) else None
    if not isinstance(enrichment, dict):
        return False
    return bool(enrichment.get("enriched_at"))


def preview_frames_exist(queue_dir: Path, count: int = 3) -> bool:
    matches = sorted(queue_dir.glob("preview_*.jpg"))
    if count <= 0:
        return bool(matches)
    return len(matches) >= count


def parse_timestamp(value: str) -> Optional[datetime]:
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def lock_is_recent(lock_path: Path, max_age_seconds: int) -> bool:
    if not lock_path.exists():
        return False
    if max_age_seconds <= 0:
        return True
    threshold = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
    ts = datetime.fromtimestamp(lock_path.stat().st_mtime, tz=timezone.utc)
    return ts >= threshold


def should_skip_clip(
    queue_dir: Path,
    queue_meta_path: Path,
    lock_path: Optional[Path] = None,
    lock_max_age_seconds: int = 1800,
    preview_count: int = 3,
) -> IdempotencyCheck:
    reasons: List[str] = []
    queue_meta = load_queue_meta(queue_meta_path)

    if has_enrichment(queue_meta):
        reasons.append("enrichment_complete")

    if preview_frames_exist(queue_dir, count=preview_count):
        reasons.append("preview_frames_present")

    if lock_path and lock_is_recent(lock_path, lock_max_age_seconds):
        reasons.append("lock_active")

    return IdempotencyCheck(should_skip=bool(reasons), reasons=reasons)
