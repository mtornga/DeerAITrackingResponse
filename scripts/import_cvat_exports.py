#!/usr/bin/env python3
"""Import CVAT YOLO exports into a normalized verified dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".PNG")


@dataclass
class ManifestEntry:
    entry_id: str
    klass: str
    lighting: str
    source: str
    image_path: str
    label_path: str
    label_status: str


def format_utc(ts: datetime) -> str:
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def infer_lighting(name: str) -> Optional[str]:
    lowered = name.lower()
    if "night" in lowered or "ir" in lowered:
        return "night"
    if "day" in lowered or "daytime" in lowered:
        return "day"
    return None


def normalize_class(name: str) -> Optional[str]:
    lowered = name.lower()
    if any(token in lowered for token in ("deer", "buck", "doe")):
        return "deer"
    if any(token in lowered for token in ("person", "human")):
        return "person"
    return None


def find_obj_train_data(root: Path) -> Optional[Path]:
    candidates = [root / "obj_train_data", root / "obj_train_data" / "data"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in root.rglob("obj_train_data"):
        if candidate.exists():
            return candidate
    return None


def find_obj_names(root: Path) -> Optional[Path]:
    for candidate in (root / "obj.names", root / "obj_names", root / "obj.names.txt"):
        if candidate.exists():
            return candidate
    for candidate in root.rglob("obj.names"):
        if candidate.exists():
            return candidate
    return None


def load_class_map(obj_names: Optional[Path]) -> Dict[int, Optional[int]]:
    mapping: Dict[int, Optional[int]] = {}
    if not obj_names or not obj_names.exists():
        return mapping
    names = [line.strip() for line in obj_names.read_text().splitlines() if line.strip()]
    for idx, name in enumerate(names):
        klass = normalize_class(name)
        if klass == "deer":
            mapping[idx] = 0
        elif klass == "person":
            mapping[idx] = 1
        else:
            mapping[idx] = None
    return mapping


def load_export_roots(exports_dir: Path) -> Iterable[Tuple[str, Path]]:
    for item in sorted(exports_dir.iterdir()):
        if item.is_dir():
            yield item.stem, item
        elif item.suffix.lower() == ".zip":
            yield item.stem, item


def extract_zip(zip_path: Path, temp_root: Path) -> Path:
    extract_dir = temp_root / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def find_image_for_label(labels_dir: Path, label_path: Path) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        candidate = labels_dir / f"{label_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def remap_label_lines(label_path: Path, class_map: Dict[int, Optional[int]]) -> Tuple[List[str], bool, bool]:
    deer_found = False
    person_found = False
    lines_out: List[str] = []
    for raw in label_path.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            src_class = int(float(parts[0]))
        except ValueError:
            continue
        if src_class not in class_map:
            continue
        mapped = class_map.get(src_class)
        if mapped is None:
            continue
        if mapped == 0:
            deer_found = True
        if mapped == 1:
            person_found = True
        parts[0] = str(mapped)
        lines_out.append(" ".join(parts))
    return lines_out, deer_found, person_found


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=REPO_ROOT / "cvat" / "exports" / "golden_v9",
        help="Directory containing CVAT YOLO exports (zip or folders)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "cvat" / "processed" / "golden_v9",
        help="Output directory for normalized images/labels",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=REPO_ROOT / "configs" / "golden_v9_manifest.json",
        help="Output manifest path",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    exports_dir = args.exports_dir
    if not exports_dir.exists():
        raise SystemExit(f"Exports directory not found: {exports_dir}")

    output_dir = args.output_dir
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[ManifestEntry] = []

    with tempfile.TemporaryDirectory() as tmp:
        temp_root = Path(tmp)
        for export_name, export_path in load_export_roots(exports_dir):
            if export_path.is_dir():
                root = export_path
            else:
                root = extract_zip(export_path, temp_root)

            labels_root = find_obj_train_data(root)
            if not labels_root:
                print(f"[import] Skipping {export_name}: obj_train_data not found")
                continue

            class_map = load_class_map(find_obj_names(root))
            lighting = infer_lighting(export_name) or infer_lighting(str(root)) or "day"

            for label_path in sorted(labels_root.glob("*.txt")):
                image_path = find_image_for_label(labels_root, label_path)
                if not image_path:
                    continue

                lines_out, deer_found, person_found = remap_label_lines(label_path, class_map)

                if deer_found:
                    klass = "deer"
                elif person_found:
                    klass = "person"
                else:
                    klass = "background"

                stem = f"{export_name}_{label_path.stem}"
                dest_image = images_out / f"{stem}{image_path.suffix.lower()}"
                dest_label = labels_out / f"{stem}.txt"

                shutil.copy2(image_path, dest_image)
                dest_label.write_text("\n".join(lines_out).strip() + ("\n" if lines_out else ""))

                rel_image = dest_image.relative_to(REPO_ROOT) if dest_image.is_relative_to(REPO_ROOT) else dest_image
                rel_label = dest_label.relative_to(REPO_ROOT) if dest_label.is_relative_to(REPO_ROOT) else dest_label

                manifest_entries.append(
                    ManifestEntry(
                        entry_id=f"cvat:{export_name}:{label_path.stem}",
                        klass=klass,
                        lighting=lighting,
                        source=f"cvat:{export_name}",
                        image_path=str(rel_image),
                        label_path=str(rel_label),
                        label_status="verified",
                    )
                )

    manifest_payload = {
        "version": 1,
        "generated_at": format_utc(datetime.now(UTC)),
        "entries": [
            {
                "id": entry.entry_id,
                "class": entry.klass,
                "lighting": entry.lighting,
                "source": entry.source,
                "image_path": entry.image_path,
                "label_path": entry.label_path,
                "label_status": entry.label_status,
            }
            for entry in manifest_entries
        ],
    }

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest_payload, indent=2))
    print(f"[import] Wrote {len(manifest_entries)} entries to {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
