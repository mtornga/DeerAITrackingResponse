"""Split cleaned labels into train/val/test YOLO dataset versions."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List
import sys

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from utils import copy_image, ensure_relative, iter_image_files, load_yolo_file, save_yolo_file  # noqa: E402


@dataclass
class Sample:
    image: Path
    label: Path
    rel_path: Path
    primary_class: int


def load_samples(images_root: Path, labels_root: Path) -> List[Sample]:
    samples: List[Sample] = []
    for image_path in iter_image_files(images_root):
        rel = ensure_relative(image_path, images_root)
        label_path = labels_root / rel.with_suffix(".txt")
        annotations = load_yolo_file(label_path)
        primary_class = annotations[0].cls if annotations else -1
        samples.append(Sample(image=image_path, label=label_path, rel_path=rel, primary_class=primary_class))
    if not samples:
        raise RuntimeError(f"No samples found under {images_root}")
    return samples


def stratified_split(samples: List[Sample], val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[Sample]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[Sample]] = {}
    for sample in samples:
        by_class.setdefault(sample.primary_class, []).append(sample)
    splits = {"train": [], "val": [], "test": []}
    for cls_samples in by_class.values():
        rng.shuffle(cls_samples)
        total = len(cls_samples)
        val_count = int(total * val_ratio)
        test_count = int(total * test_ratio)
        train_count = total - val_count - test_count
        splits["train"].extend(cls_samples[:train_count])
        splits["val"].extend(cls_samples[train_count : train_count + val_count])
        splits["test"].extend(cls_samples[train_count + val_count :])
    return splits


def copy_split(samples: List[Sample], split_name: str, dst_root: Path) -> None:
    images_out = dst_root / "images" / split_name
    labels_out = dst_root / "labels" / split_name
    for sample in samples:
        copy_image(sample.image, images_out / sample.rel_path)
        if sample.label.exists():
            save_yolo_file(labels_out / sample.rel_path.with_suffix(".txt"), load_yolo_file(sample.label))


def write_dataset_yaml(dst_root: Path, out_path: Path, class_names: Dict[int, str]) -> None:
    dataset_yaml = {
        "path": str(dst_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test.")
    parser.add_argument("--src", type=Path, default=Path("data/interim/autolabel"), help="Source directory with images/ + labels/.")
    parser.add_argument("--dst", type=Path, default=Path("data/datasets/deer_det_v01"), help="Destination dataset directory.")
    parser.add_argument("--val", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--classes", type=str, nargs="*", default=["deer", "unknown_mammal"], help="Class names in order.")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/data_deer.yaml"),
        help="Path to write the YOLO dataset YAML.",
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/datasets/manifest.json"), help="Split manifest output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_root = args.src / "images"
    labels_root = args.src / "labels"
    samples = load_samples(images_root, labels_root)
    splits = stratified_split(samples, args.val, args.test, args.seed)
    args.dst.mkdir(parents=True, exist_ok=True)
    for split_name, split_samples in splits.items():
        copy_split(split_samples, split_name, args.dst)
    class_names = {idx: name for idx, name in enumerate(args.classes)}
    write_dataset_yaml(args.dst, args.dataset_config, class_names)
    manifest = {
        "dataset": str(args.dst),
        "splits": {split: len(items) for split, items in splits.items()},
        "class_names": class_names,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote dataset to {args.dst}")


if __name__ == "__main__":
    main()
