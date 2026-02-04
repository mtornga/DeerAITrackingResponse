#!/usr/bin/env python3
"""Build golden_frames_v9 dataset from verified CVAT exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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

from training_loop_utils import choose_split, load_json, resolve_share_root  # noqa: E402

ALLOWED_CLASS_IDS = {0, 1}


@dataclass
class FrameEntry:
    entry_id: str
    klass: str
    lighting: str
    source: str
    image_path: Path
    label_path: Path


@dataclass
class BuildStats:
    total_entries: int = 0
    selected_entries: int = 0
    copied_images: int = 0
    copied_labels: int = 0
    empty_labels: int = 0
    per_group: Dict[str, int] = field(default_factory=dict)


def format_utc(ts: datetime) -> str:
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_bucket(value: str) -> float:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def manifest_hash(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.md5(path.read_bytes()).hexdigest()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_manifest_entries(path: Path) -> List[FrameEntry]:
    payload = load_json(path) or {}
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return []
    results: List[FrameEntry] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("label_status") != "verified":
            continue
        image_path = entry.get("image_path")
        label_path = entry.get("label_path")
        if not isinstance(image_path, str) or not isinstance(label_path, str):
            continue
        image = resolve_path(image_path)
        label = resolve_path(label_path)
        if not image.exists() or not label.exists():
            continue
        klass = entry.get("class")
        lighting = entry.get("lighting")
        source = entry.get("source", "cvat")
        if not isinstance(klass, str) or not isinstance(lighting, str):
            continue
        results.append(
            FrameEntry(
                entry_id=str(entry.get("id", image_path)),
                klass=klass,
                lighting=lighting,
                source=str(source),
                image_path=image,
                label_path=label,
            )
        )
    return results


def normalize_group_key(klass: str, lighting: str) -> Optional[str]:
    if klass in {"deer", "person"}:
        return f"{lighting}_{klass}"
    if klass == "background":
        return f"{lighting}_background"
    return None


def filtered_label_lines(label_path: Path, allowed_ids: Iterable[int]) -> List[str]:
    allowed = {str(int(value)) for value in allowed_ids}
    lines: List[str] = []
    for raw in label_path.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        if parts[0] in allowed:
            lines.append(raw.strip())
    return lines


def copy_entry(entry: FrameEntry, images_dir: Path, labels_dir: Path) -> bool:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stem_hash = hashlib.md5(str(entry.image_path).encode("utf-8")).hexdigest()[:8]
    dest_name = f"{entry.image_path.stem}_{stem_hash}{entry.image_path.suffix.lower()}"
    dest_image = images_dir / dest_name
    dest_label = labels_dir / f"{Path(dest_name).stem}.txt"

    shutil.copy2(entry.image_path, dest_image)

    label_lines = filtered_label_lines(entry.label_path, ALLOWED_CLASS_IDS)
    dest_label.write_text("\n".join(label_lines).strip() + ("\n" if label_lines else ""))
    return bool(label_lines)


def write_dataset_yaml(dataset_root: Path, lighting: str) -> None:
    data_path = dataset_root / f"{lighting}.yaml"
    data = (
        f"path: {dataset_root / lighting}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: deer\n"
        "  1: person\n"
    )
    data_path.write_text(data)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share-root", help="Override share root (default: auto-detect)")
    parser.add_argument("--manifest", help="Override manifest path (default configs/golden_v9_manifest.json)")
    parser.add_argument("--output-root", help="Override dataset root (default from config)")
    parser.add_argument("--clean", action="store_true", help="Remove existing dataset before rebuild")
    parser.add_argument("--dry-run", action="store_true", help="Plan without writing files")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    share_root = resolve_share_root(args.share_root)

    config_path = REPO_ROOT / "configs" / "training_loop.json"
    config = load_json(config_path) or {}

    manifest_path = Path(args.manifest) if args.manifest else (REPO_ROOT / "configs" / "golden_v9_manifest.json")
    entries = load_manifest_entries(manifest_path)
    stats = BuildStats(total_entries=len(entries))

    if not entries:
        print("[frames] No manifest entries found.")
        return 1

    caps_cfg = config.get("golden_caps", {})
    caps: Dict[str, int] = {}
    if isinstance(caps_cfg, dict):
        for key, value in caps_cfg.items():
            try:
                caps[key] = int(value)
            except (TypeError, ValueError):
                continue

    output_root = Path(args.output_root) if args.output_root else config.get("golden_frames_root", "golden_frames_v9")
    if not isinstance(output_root, Path):
        output_root = Path(str(output_root))
    if not output_root.is_absolute():
        output_root = share_root / output_root

    include_negatives = bool(config.get("include_negatives", False))

    grouped: Dict[str, List[FrameEntry]] = {}
    for entry in entries:
        group = normalize_group_key(entry.klass, entry.lighting)
        if not group:
            continue
        if group.endswith("background") and not include_negatives:
            continue
        grouped.setdefault(group, []).append(entry)

    selected: List[FrameEntry] = []
    for group, group_entries in grouped.items():
        cap = caps.get(group)
        sorted_entries = sorted(group_entries, key=lambda item: stable_bucket(item.image_path.as_posix()))
        if cap is not None:
            sorted_entries = sorted_entries[:cap]
        selected.extend(sorted_entries)
        stats.per_group[group] = len(sorted_entries)

    stats.selected_entries = len(selected)

    if args.dry_run:
        print(json.dumps({"selected": stats.selected_entries, "groups": stats.per_group}, indent=2))
        return 0

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    val_split = float(config.get("val_split", 0.15))
    for entry in selected:
        split = choose_split(entry.entry_id, val_split)
        images_dir = output_root / entry.lighting / "images" / split
        labels_dir = output_root / entry.lighting / "labels" / split
        has_labels = copy_entry(entry, images_dir, labels_dir)
        stats.copied_images += 1
        stats.copied_labels += 1
        if not has_labels:
            stats.empty_labels += 1

    write_dataset_yaml(output_root, "day")
    write_dataset_yaml(output_root, "night")

    build_meta = {
        "generated_at": format_utc(datetime.now(UTC)),
        "manifest_hash": manifest_hash(manifest_path),
        "stats": stats.__dict__,
    }
    meta_path = output_root / ".build_meta.json"
    meta_path.write_text(json.dumps(build_meta, indent=2))

    print(f"[frames] Built dataset at {output_root} (entries={stats.selected_entries})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
