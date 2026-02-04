#!/usr/bin/env python3
"""Build a CVAT video-task queue from golden_clips/manifest.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _ensure_repo_root_on_path() -> Path:
    script_path = Path(__file__).resolve()
    for parent in (script_path.parent, *script_path.parents):
        if (parent / ".env").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    fallback = script_path.parents[1]
    if str(fallback) not in sys.path:
        sys.path.insert(0, str(fallback))
    return fallback


REPO_ROOT = _ensure_repo_root_on_path()
if str(REPO_ROOT / "agents") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agents"))

from training_loop_utils import load_json, resolve_share_root  # noqa: E402


TARGET_DEFAULTS = {
    "night_deer": 1500,
    "day_deer": 600,
    "day_person": 400,
    "night_person": 150,
}


@dataclass
class ClipCandidate:
    clip_id: str
    clip_path: Path
    lighting: str
    category: str
    priority: float
    model_hint: str
    max_confidence: Optional[float] = None
    deer_count: Optional[int] = None


def format_utc(ts: datetime) -> str:
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_env_vars() -> Dict[str, str]:
    env_path = REPO_ROOT / ".env"
    env_vars: Dict[str, str] = {}
    if not env_path.exists():
        return env_vars
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export "):]
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        env_vars[key.strip()] = value
    return env_vars


def normalize_category(raw: Optional[str]) -> str:
    if not raw:
        return "unknown"
    lowered = raw.lower().strip()
    if lowered in {"deer", "buck", "doe", "animal"}:
        return "deer"
    if lowered in {"person", "people", "human"}:
        return "person"
    return lowered


def infer_lighting(text: str) -> Optional[str]:
    lowered = text.lower()
    if "night" in lowered or "ir" in lowered:
        return "night"
    if "day" in lowered or "daytime" in lowered:
        return "day"
    return None


def resolve_clip_dir(entry: Dict[str, Any], golden_root: Path) -> Optional[Path]:
    path_str = entry.get("dest_dir") or entry.get("path") or entry.get("source_path") or entry.get("id")
    if not path_str:
        return None
    clip_path = Path(str(path_str))
    if not clip_path.is_absolute():
        clip_path = golden_root / clip_path
    if not clip_path.exists():
        return None
    return clip_path


def extract_meta_stats(meta_path: Path) -> Tuple[Optional[float], Optional[int]]:
    if not meta_path.exists():
        return None, None
    try:
        payload = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None, None

    max_conf = None
    deer_count = None

    raw_max = payload.get("max_confidence")
    if isinstance(raw_max, (int, float)):
        max_conf = float(raw_max)

    counts = payload.get("counts")
    if isinstance(counts, dict):
        deer_count_val = counts.get("deer") or counts.get("animal")
        try:
            deer_count = int(deer_count_val) if deer_count_val is not None else None
        except (TypeError, ValueError):
            deer_count = None

    models = payload.get("models")
    if isinstance(models, dict):
        if max_conf is None:
            best = None
            for model_data in models.values():
                hits = model_data.get("hits") if isinstance(model_data, dict) else None
                if not isinstance(hits, list):
                    continue
                for hit in hits:
                    if not isinstance(hit, dict):
                        continue
                    conf = hit.get("confidence") or hit.get("conf")
                    try:
                        conf_val = float(conf)
                    except (TypeError, ValueError):
                        continue
                    if best is None or conf_val > best:
                        best = conf_val
            max_conf = best
        if deer_count is None:
            deer_hits = 0
            for model_data in models.values():
                hits = model_data.get("hits") if isinstance(model_data, dict) else None
                if not isinstance(hits, list):
                    continue
                for hit in hits:
                    if not isinstance(hit, dict):
                        continue
                    category = str(hit.get("category") or "").lower()
                    if category in {"deer", "buck", "doe", "animal"}:
                        deer_hits += 1
            if deer_hits:
                deer_count = deer_hits

    return max_conf, deer_count


def compute_priority(lighting: str, category: str, max_conf: Optional[float], deer_count: Optional[int]) -> float:
    base = 4.0
    if lighting == "night" and category == "deer":
        base = 0.0
    elif lighting == "night" and category == "person":
        base = 1.0
    elif lighting == "day" and category == "deer":
        base = 2.0
    elif lighting == "day" and category == "person":
        base = 3.0

    if deer_count is not None and deer_count >= 2:
        base -= 0.4
    if max_conf is not None and max_conf < 0.5:
        base -= 0.2

    return max(base, 0.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share-root", help="Override share root (default: auto-detect)")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "cvat" / "queue" / "golden_v9" / "queue.json"),
        help="Output queue path",
    )
    parser.add_argument("--target-night-deer", type=int, default=TARGET_DEFAULTS["night_deer"])
    parser.add_argument("--target-day-deer", type=int, default=TARGET_DEFAULTS["day_deer"])
    parser.add_argument("--target-day-person", type=int, default=TARGET_DEFAULTS["day_person"])
    parser.add_argument("--target-night-person", type=int, default=TARGET_DEFAULTS["night_person"])
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    share_root = resolve_share_root(args.share_root)
    golden_root = share_root / "golden_clips"
    manifest_path = golden_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    payload = load_json(manifest_path)
    clips = payload.get("clips", []) if payload else []
    if not isinstance(clips, list):
        raise SystemExit("Manifest clips not found or invalid.")

    env_vars = load_env_vars()
    day_model = (env_vars.get("DETECTOR_MODELS_DAY") or env_vars.get("DETECTOR_MODELS") or "").split(",")[0].strip()
    night_model = (env_vars.get("DETECTOR_MODELS_NIGHT") or env_vars.get("DETECTOR_MODELS") or "").split(",")[0].strip()

    candidates: List[ClipCandidate] = []
    for entry in clips:
        if not isinstance(entry, dict):
            continue
        clip_dir = resolve_clip_dir(entry, golden_root)
        if not clip_dir:
            continue
        clip_id = entry.get("id") or entry.get("segment_id") or clip_dir.name
        category = normalize_category(entry.get("category"))
        if category not in {"deer", "person"}:
            continue
        lighting = entry.get("lighting")
        if not isinstance(lighting, str):
            lighting = infer_lighting(str(clip_dir)) or "day"
        lighting = lighting if lighting in {"day", "night"} else "day"

        max_conf, deer_count = extract_meta_stats(clip_dir / "meta.json")
        priority = compute_priority(lighting, category, max_conf, deer_count)
        model_hint = night_model if lighting == "night" else day_model

        candidates.append(
            ClipCandidate(
                clip_id=str(clip_id),
                clip_path=clip_dir,
                lighting=lighting,
                category=category,
                priority=priority,
                model_hint=model_hint,
                max_confidence=max_conf,
                deer_count=deer_count,
            )
        )

    targets = {
        "night_deer": args.target_night_deer,
        "day_deer": args.target_day_deer,
        "day_person": args.target_day_person,
        "night_person": args.target_night_person,
    }

    selected_counts = {key: 0 for key in targets}
    selected: List[Dict[str, Any]] = []

    def key_for(candidate: ClipCandidate) -> Optional[str]:
        if candidate.lighting == "night" and candidate.category == "deer":
            return "night_deer"
        if candidate.lighting == "day" and candidate.category == "deer":
            return "day_deer"
        if candidate.lighting == "day" and candidate.category == "person":
            return "day_person"
        if candidate.lighting == "night" and candidate.category == "person":
            return "night_person"
        return None

    candidates_sorted = sorted(
        candidates,
        key=lambda item: (item.priority, item.max_confidence if item.max_confidence is not None else 1.0),
    )

    for candidate in candidates_sorted:
        bucket = key_for(candidate)
        if not bucket:
            continue
        if selected_counts[bucket] >= targets[bucket]:
            continue
        selected_counts[bucket] += 1
        selected.append(
            {
                "clip_id": candidate.clip_id,
                "clip_path": str(candidate.clip_path),
                "lighting": candidate.lighting,
                "category": candidate.category,
                "priority": candidate.priority,
                "model_hint": candidate.model_hint,
                "max_confidence": candidate.max_confidence,
                "deer_count": candidate.deer_count,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": format_utc(datetime.now(UTC)),
        "share_root": str(share_root),
        "targets": targets,
        "selected_counts": selected_counts,
        "clips": selected,
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"[queue] Wrote {len(selected)} clips to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
