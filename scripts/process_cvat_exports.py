#!/usr/bin/env python3
"""Process rescued CVAT exports and prepare for training.

This script:
1. Extracts all CVAT export zips
2. Consolidates labels in YOLO format
3. Creates manifest of what's needed
4. Prepares for frame extraction when clips become available

Usage:
    python scripts/process_cvat_exports.py \
        --exports-dir /Users/marktornga/Downloads/RescuedCVATexports \
        --output-dir outdoor/deer-vision/data/raw/cvat_rescued
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


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


@dataclass
class CVATExport:
    """Metadata for a CVAT export."""
    zip_name: str
    segment_id: str | None
    label_count: int
    classes: List[str]
    frames_referenced: List[str]


def extract_segment_id(zip_name: str) -> str | None:
    """Extract segment ID from zip filename."""
    # Patterns: 055549.zip, segment_092645.zip, nighthard1.zip
    name = Path(zip_name).stem
    if name.isdigit():
        return f"segment_{name}"
    elif name.startswith("segment_"):
        return name
    else:
        # Generic name like "nighthard1" - use as-is
        return name


def process_cvat_export(zip_path: Path, output_dir: Path) -> CVATExport:
    """Extract and process a CVAT export."""
    zip_name = zip_path.name
    segment_id = extract_segment_id(zip_name)

    # Create extraction directory
    extract_dir = output_dir / segment_id
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Extract zip
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    # Read metadata
    obj_names_path = extract_dir / "obj.names"
    classes = []
    if obj_names_path.exists():
        classes = obj_names_path.read_text().strip().split("\n")

    # Count labels
    labels_dir = extract_dir / "obj_train_data"
    label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []

    # Get frame references from train.txt
    train_txt = extract_dir / "train.txt"
    frames = []
    if train_txt.exists():
        frames = train_txt.read_text().strip().split("\n")

    return CVATExport(
        zip_name=zip_name,
        segment_id=segment_id,
        label_count=len(label_files),
        classes=classes,
        frames_referenced=frames
    )


def create_manifest(exports: List[CVATExport], output_path: Path) -> Dict:
    """Create manifest documenting all exports."""
    manifest = {
        "total_exports": len(exports),
        "total_labels": sum(e.label_count for e in exports),
        "unique_classes": list(set(cls for e in exports for cls in e.classes)),
        "exports": [asdict(e) for e in exports],
        "clips_needed": []
    }

    # Identify which clips need frames extracted
    for export in exports:
        if export.segment_id and export.segment_id.startswith("segment_"):
            manifest["clips_needed"].append({
                "segment_id": export.segment_id,
                "frames_count": len(export.frames_referenced),
                "label_count": export.label_count
            })

    output_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Process rescued CVAT exports.")
    parser.add_argument(
        "--exports-dir",
        type=Path,
        required=True,
        help="Directory with CVAT export zips"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for processed exports"
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process all zips
    print(f"Processing CVAT exports from {args.exports_dir}")
    zip_files = list(args.exports_dir.glob("*.zip"))
    print(f"Found {len(zip_files)} zip files")

    exports = []
    for zip_path in sorted(zip_files):
        print(f"\nProcessing {zip_path.name}...")
        export = process_cvat_export(zip_path, args.output_dir)
        exports.append(export)
        print(f"  Segment: {export.segment_id}")
        print(f"  Labels: {export.label_count}")
        print(f"  Classes: {', '.join(export.classes)}")
        print(f"  Frames referenced: {len(export.frames_referenced)}")

    # Create manifest
    manifest_path = args.output_dir / "manifest.json"
    manifest = create_manifest(exports, manifest_path)

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total exports: {manifest['total_exports']}")
    print(f"  Total labels: {manifest['total_labels']}")
    print(f"  Unique classes: {', '.join(manifest['unique_classes'])}")
    print(f"  Clips needing frames: {len(manifest['clips_needed'])}")
    print(f"\nManifest written to: {manifest_path}")
    print(f"\nNext steps:")
    print(f"  1. Locate original video clips for segments:")
    for clip in manifest['clips_needed']:
        print(f"     - {clip['segment_id']}: {clip['frames_count']} frames needed")
    print(f"  2. Extract frames using extract_feedback_frames.py or similar")
    print(f"  3. Match frames with labels and merge into training dataset")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
