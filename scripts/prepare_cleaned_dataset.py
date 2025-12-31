#!/usr/bin/env python3
"""Prepare a cleaned training dataset without AprilTag labels.

This script creates a new dataset from wildlife_merged by:
1. Filtering out apriltag-only labels
2. Keeping only deer and person classes
3. Adding false positive frames as hard negatives
4. Creating proper train/val splits
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import random


def parse_yolo_label(label_path: Path) -> List[Tuple[int, List[float]]]:
    """Parse YOLO label file and return list of (class_id, bbox) tuples."""
    if not label_path.exists():
        return []
    
    labels = []
    for line in label_path.read_text().strip().split('\n'):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
            labels.append((class_id, bbox))
    
    return labels


def filter_labels(
    labels: List[Tuple[int, List[float]]],
    keep_classes: set
) -> List[Tuple[int, List[float]]]:
    """Filter labels to keep only specified classes."""
    return [(cls, bbox) for cls, bbox in labels if cls in keep_classes]


def write_yolo_label(label_path: Path, labels: List[Tuple[int, List[float]]]) -> None:
    """Write YOLO label file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    for class_id, bbox in labels:
        line = f"{class_id} {' '.join(str(x) for x in bbox)}"
        lines.append(line)
    
    label_path.write_text('\n'.join(lines) + '\n' if lines else '')


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-dataset",
        type=Path,
        default=Path("outdoor/deer-vision/data/datasets/wildlife_merged"),
        help="Input dataset directory (default: %(default)s)"
    )
    parser.add_argument(
        "--feedback-dir",
        type=Path,
        default=Path("outdoor/deer-vision/data/raw/feedback"),
        help="Feedback frames directory (default: %(default)s)"
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=Path("outdoor/deer-vision/data/datasets/wildlife_cleaned"),
        help="Output dataset directory (default: %(default)s)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split ratio (default: %(default)s)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s)"
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    input_img_dir = args.input_dataset / "images"
    input_lbl_dir = args.input_dataset / "labels"
    
    if not input_img_dir.exists():
        print(f"Input images directory not found: {input_img_dir}")
        return 1
    
    # Class mapping: 0=deer, 2=person (skip 1=unknown_animal, 3=apriltag)
    keep_classes = {0, 2}
    
    # Collect all valid image/label pairs
    print("Processing wildlife_merged dataset...")
    valid_pairs: List[Tuple[Path, Path]] = []
    stats = defaultdict(int)
    
    for img_path in input_img_dir.rglob("*.jpg"):
        # Find corresponding label
        rel_path = img_path.relative_to(input_img_dir)
        label_path = input_lbl_dir / rel_path.parent / (img_path.stem + ".txt")
        
        if not label_path.exists():
            stats["no_label"] += 1
            continue
        
        # Parse and filter labels
        labels = parse_yolo_label(label_path)
        filtered_labels = filter_labels(labels, keep_classes)
        
        if not filtered_labels:
            stats["apriltag_only"] += 1
            continue
        
        valid_pairs.append((img_path, label_path))
        stats["valid"] += 1
    
    print(f"  Valid images: {stats['valid']}")
    print(f"  Skipped (no label): {stats['no_label']}")
    print(f"  Skipped (apriltag only): {stats['apriltag_only']}")
    
    # Add false positive frames as hard negatives
    print("\nAdding false positive frames as hard negatives...")
    fp_dir = args.feedback_dir / "false_positive"
    fp_count = 0
    if fp_dir.exists():
        for img_path in fp_dir.glob("*.jpg"):
            # Create empty label (hard negative)
            valid_pairs.append((img_path, None))  # None = empty label
            fp_count += 1
    print(f"  Added {fp_count} hard negative frames")
    
    # TODO: Add person feedback frames with labels
    # For now, we'll need to manually label these or use autolabel
    
    # Shuffle and split
    random.shuffle(valid_pairs)
    val_size = int(len(valid_pairs) * args.val_split)
    train_pairs = valid_pairs[val_size:]
    val_pairs = valid_pairs[:val_size]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_pairs)} images")
    print(f"  Val: {len(val_pairs)} images")
    print(f"  Total: {len(valid_pairs)} images")
    
    # Create output directories
    train_img_dir = args.output_dataset / "images" / "train"
    train_lbl_dir = args.output_dataset / "labels" / "train"
    val_img_dir = args.output_dataset / "images" / "val"
    val_lbl_dir = args.output_dataset / "labels" / "val"
    
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy train set
    print("\nCopying train set...")
    for img_path, label_path in train_pairs:
        # Copy image
        dst_img = train_img_dir / img_path.name
        shutil.copy2(img_path, dst_img)
        
        # Copy/create label
        dst_lbl = train_lbl_dir / (img_path.stem + ".txt")
        if label_path is None:
            # Empty label for hard negative
            dst_lbl.write_text('')
        else:
            # Filter and copy labels
            labels = parse_yolo_label(label_path)
            filtered_labels = filter_labels(labels, keep_classes)
            write_yolo_label(dst_lbl, filtered_labels)
    
    # Copy val set
    print("Copying val set...")
    for img_path, label_path in val_pairs:
        # Copy image
        dst_img = val_img_dir / img_path.name
        shutil.copy2(img_path, dst_img)
        
        # Copy/create label
        dst_lbl = val_lbl_dir / (img_path.stem + ".txt")
        if label_path is None:
            # Empty label for hard negative
            dst_lbl.write_text('')
        else:
            # Filter and copy labels
            labels = parse_yolo_label(label_path)
            filtered_labels = filter_labels(labels, keep_classes)
            write_yolo_label(dst_lbl, filtered_labels)
    
    # Create data.yaml
    data_yaml_content = f"""# Wildlife Detection Dataset - Cleaned (No AprilTags)
# Created: {Path(__file__).stem}
# Total: {len(valid_pairs)} images ({len(train_pairs)} train, {len(val_pairs)} val)
# Includes {fp_count} hard negative frames from false positives

path: {args.output_dataset.resolve()}
train: images/train
val: images/val

names:
  0: deer
  2: person

# Note: Classes 1 (unknown_animal) and 3 (apriltag) have been filtered out
"""
    
    (args.output_dataset / "data.yaml").write_text(data_yaml_content)
    print(f"\nDataset created at: {args.output_dataset}")
    print("data.yaml written.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
