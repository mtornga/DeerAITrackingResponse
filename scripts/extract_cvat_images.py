#!/usr/bin/env python3
"""Extract images from CVAT docker container and pair with labels.

This script:
1. Copies images from CVAT task directories
2. Matches them with existing YOLO labels
3. Creates a consolidated dataset ready for training

Usage:
    python scripts/extract_cvat_images.py \
        --cvat-labels-dir outdoor/deer-vision/data/raw/cvat_rescued \
        --output-dir outdoor/deer-vision/data/raw/cvat_with_images
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
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

# Task ID to segment name mapping
TASK_MAPPING = {
    3: "nighthard1",  # 302 frames
    4: "PersonDayTimeEasy1",  # 302 frames
    8: "segment_092645",  # 1002 frames
    10: "segment_092708",  # 302 frames
    12: "segment_092754",  # 505 frames
}


def copy_images_from_cvat(task_id: int, segment_name: str, output_dir: Path) -> int:
    """Copy images from CVAT docker container to local filesystem."""
    print(f"\nCopying images for task {task_id} ({segment_name})...")

    # Create output directories
    images_dir = output_dir / segment_name / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Docker cp command to copy images
    container_path = f"cvat_server:/home/django/data/data/{task_id}/raw"
    temp_dir = output_dir / f"temp_task_{task_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["docker", "cp", f"{container_path}/.", str(temp_dir)]

    try:
        subprocess.run(cmd, check=True, capture_output=True)

        # Move images to final location
        frame_files = list(temp_dir.glob("frame_*.jpg")) + list(temp_dir.glob("frame_*.png"))

        for frame_file in frame_files:
            # Convert to PNG if needed (CVAT labels reference .PNG)
            dest_name = frame_file.stem + ".PNG"
            dest_path = images_dir / dest_name
            shutil.copy2(frame_file, dest_path)

        # Cleanup temp dir
        shutil.rmtree(temp_dir)

        print(f"  Copied {len(frame_files)} images")
        return len(frame_files)

    except subprocess.CalledProcessError as e:
        print(f"  Error copying from docker: {e.stderr.decode()}")
        return 0


def match_labels_with_images(
    labels_dir: Path,
    images_dir: Path,
    output_dir: Path,
    segment_name: str
) -> Dict[str, int]:
    """Match YOLO labels with extracted images."""
    print(f"\nMatching labels with images for {segment_name}...")

    labels_src = labels_dir / segment_name / "obj_train_data"
    if not labels_src.exists():
        print(f"  Warning: No labels found at {labels_src}")
        return {"matched": 0, "missing_images": 0, "missing_labels": 0}

    images_src = images_dir / segment_name / "images"
    if not images_src.exists():
        print(f"  Warning: No images found at {images_src}")
        return {"matched": 0, "missing_images": 0, "missing_labels": 0}

    # Create output structure
    final_images = output_dir / "images"
    final_labels = output_dir / "labels"
    final_images.mkdir(parents=True, exist_ok=True)
    final_labels.mkdir(parents=True, exist_ok=True)

    # Match and copy
    matched = 0
    missing_images = 0
    missing_labels = 0

    # Process all images
    for image_file in images_src.glob("*.PNG"):
        label_file = labels_src / f"{image_file.stem}.txt"

        if label_file.exists():
            # Copy both image and label
            dest_img = final_images / f"{segment_name}_{image_file.name}"
            dest_lbl = final_labels / f"{segment_name}_{image_file.stem}.txt"

            shutil.copy2(image_file, dest_img)
            shutil.copy2(label_file, dest_lbl)
            matched += 1
        else:
            missing_labels += 1

    # Check for labels without images
    for label_file in labels_src.glob("*.txt"):
        image_file = images_src / f"{label_file.stem}.PNG"
        if not image_file.exists():
            missing_images += 1

    stats = {
        "matched": matched,
        "missing_images": missing_images,
        "missing_labels": missing_labels
    }

    print(f"  Matched: {matched}")
    print(f"  Missing images: {missing_images}")
    print(f"  Missing labels: {missing_labels}")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract CVAT images and pair with labels.")
    parser.add_argument(
        "--cvat-labels-dir",
        type=Path,
        required=True,
        help="Directory with CVAT labels (cvat_rescued)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for paired data"
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_images_dir = args.output_dir / "temp_extracted_images"
    temp_images_dir.mkdir(parents=True, exist_ok=True)

    # Extract images from CVAT
    print("=" * 60)
    print("STEP 1: Extracting images from CVAT docker container")
    print("=" * 60)

    total_images = 0
    for task_id, segment_name in TASK_MAPPING.items():
        count = copy_images_from_cvat(task_id, segment_name, temp_images_dir)
        total_images += count

    print(f"\nTotal images extracted: {total_images}")

    # Match with labels
    print("\n" + "=" * 60)
    print("STEP 2: Matching images with labels")
    print("=" * 60)

    all_stats = []
    for task_id, segment_name in TASK_MAPPING.items():
        stats = match_labels_with_images(
            args.cvat_labels_dir,
            temp_images_dir,
            args.output_dir,
            segment_name
        )
        stats["segment"] = segment_name
        all_stats.append(stats)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_matched = sum(s["matched"] for s in all_stats)
    total_missing_images = sum(s["missing_images"] for s in all_stats)
    total_missing_labels = sum(s["missing_labels"] for s in all_stats)

    print(f"Total matched pairs: {total_matched}")
    print(f"Labels missing images: {total_missing_images}")
    print(f"Images missing labels: {total_missing_labels}")

    # Save summary
    summary = {
        "total_matched": total_matched,
        "total_missing_images": total_missing_images,
        "total_missing_labels": total_missing_labels,
        "segments": all_stats
    }

    summary_path = args.output_dir / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to: {summary_path}")

    # Cleanup temp
    shutil.rmtree(temp_images_dir)

    print(f"\nFinal dataset ready at: {args.output_dir}")
    print(f"  Images: {args.output_dir / 'images'}")
    print(f"  Labels: {args.output_dir / 'labels'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
