#!/usr/bin/env python3
"""
Build configs/golden_v8_manifest.json from labeled sources and TC-manifested clips.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _ensure_repo_root_on_path() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
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

from training_loop_utils import load_json, resolve_share_root


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class ManifestEntry:
    id: str
    klass: str
    lighting: str
    source: str
    image_path: str
    label_path: Optional[str]
    priority: int
    label_status: str


def format_utc(ts: datetime) -> str:
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(source: str, path: str) -> str:
    digest = hashlib.md5(f"{source}:{path}".encode("utf-8")).hexdigest()[:12]
    return f"{source}:{digest}"


def _tokenize(parts: List[str]) -> List[str]:
    tokens: List[str] = []
    for part in parts:
        tokens.append(part)
        for separator in ("_", "-"):
            if separator in part:
                tokens.extend([p for p in part.split(separator) if p])
    return tokens


def infer_lighting(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    tokens = _tokenize(parts)
    if "night" in tokens or "ir" in tokens:
        return "night"
    if "day" in tokens or "daytime" in tokens:
        return "day"
    return None


def infer_class(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    tokens = _tokenize(parts)
    for token in tokens:
        if token in {"buck", "doe", "deer", "animal"}:
            return "deer"
        if token in {"person", "people", "human"}:
            return "person"
        if token in {"other_wildlife", "wildlife"}:
            return "other_wildlife"
        if token in {"false_positive", "false"}:
            return "false_positive"
        if token in {"unknown", "misc"}:
            return "unknown"
    return None


def source_priority(source: str, path: Path) -> int:
    lowered = source.lower()
    if "cvat" in lowered or "cvat" in " ".join(path.parts).lower():
        return 0
    if "golden_clips_frames" in lowered:
        return 1
    if "nestcam_frames" in lowered:
        return 2
    return 3


def find_image_for_label(label_path: Path) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        candidate = label_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    if label_path.parent.name in {"labels", "label"}:
        sibling = label_path.parent.parent / "images" / label_path.name
        for ext in IMAGE_EXTS:
            candidate = sibling.with_suffix(ext)
            if candidate.exists():
                return candidate
    if label_path.parent.parent.name == "labels":
        images_dir = label_path.parent.parent.parent / "images" / label_path.parent.name
        sibling = images_dir / label_path.name
        for ext in IMAGE_EXTS:
            candidate = sibling.with_suffix(ext)
            if candidate.exists():
                return candidate
    return None


def scan_labels(source_root: Path, source_name: str) -> Iterable[ManifestEntry]:
    for label_path in sorted(source_root.rglob("*.txt")):
        if label_path.name.startswith("."):
            continue
        image_path = find_image_for_label(label_path)
        if not image_path:
            continue
        klass = infer_class(label_path)
        lighting = infer_lighting(label_path)
        if not klass or not lighting:
            continue
        label_status = "ok"
        yield ManifestEntry(
            id=stable_id(source_name, str(image_path)),
            klass=klass,
            lighting=lighting,
            source=source_name,
            image_path=str(image_path),
            label_path=str(label_path),
            priority=source_priority(source_name, label_path),
            label_status=label_status,
        )


def iter_tc_manifest_entries(manifest: Dict[str, Any], golden_root: Path) -> Iterable[ManifestEntry]:
    clips = manifest.get("clips", [])
    if not isinstance(clips, list):
        return []
    for entry in clips:
        if not isinstance(entry, dict):
            continue
        clip_id = entry.get("id")
        if not isinstance(clip_id, str) or not clip_id.strip():
            continue
        klass = infer_class(Path(str(entry.get("category", "")))) or infer_class(Path(clip_id))
        lighting = infer_lighting(Path(str(entry.get("lighting", ""))))
        if not klass:
            klass = infer_class(Path(str(entry.get("path", "")))) or "unknown"
        if not lighting:
            lighting = infer_lighting(Path(str(entry.get("path", "")))) or "day"

        path_str = entry.get("dest_dir") or entry.get("path") or entry.get("source_path") or clip_id
        clip_path = Path(path_str)
        if not clip_path.is_absolute():
            clip_path = golden_root / clip_path
        yield ManifestEntry(
            id=f"golden_clips:{clip_id}",
            klass=klass,
            lighting=lighting,
            source="golden_clips",
            image_path=str(clip_path),
            label_path=None,
            priority=3,
            label_status="missing",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share-root", help="Override share root (default: auto-detect)")
    parser.add_argument("--output", help="Override output path (default configs/golden_v8_manifest.json)")
    parser.add_argument("--dry-run", action="store_true", help="Report summary without writing")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    share_root = resolve_share_root(args.share_root)
    config_path = REPO_ROOT / "configs" / "training_loop.json"
    config = load_json(config_path) or {}
    sources = config.get("golden_sources", ["nestcam_frames", "golden_clips_frames", "golden_clips/manifest.json"])
    if not isinstance(sources, list):
        sources = ["nestcam_frames", "golden_clips_frames", "golden_clips/manifest.json"]

    entries: List[ManifestEntry] = []

    for source in sources:
        if not isinstance(source, str):
            continue
        if source.endswith("manifest.json"):
            manifest_path = share_root / source
            manifest = load_json(manifest_path) or {}
            entries.extend(iter_tc_manifest_entries(manifest, share_root / "golden_clips"))
            continue
        source_root = share_root / source
        if not source_root.exists():
            continue
        entries.extend(scan_labels(source_root, source))

    output_path = Path(args.output) if args.output else (REPO_ROOT / "configs" / "golden_v8_manifest.json")
    payload = {
        "version": 1,
        "generated_at": format_utc(datetime.now(UTC)),
        "entries": [
            {
                "id": entry.id,
                "class": entry.klass,
                "lighting": entry.lighting,
                "source": entry.source,
                "image_path": entry.image_path,
                "label_path": entry.label_path,
                "priority": entry.priority,
                "label_status": entry.label_status,
            }
            for entry in entries
        ],
    }

    if args.dry_run:
        print(json.dumps({"entry_count": len(entries), "output": str(output_path)}, indent=2))
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"[manifest] Wrote {len(entries)} entries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
