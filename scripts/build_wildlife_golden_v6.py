#!/usr/bin/env python3
"""Build day/night split datasets for wildlife_golden_v6 from golden_v5 frames."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np


def classify_image(image_path: Path) -> Tuple[str, float]:
    """Classify a single image as day or night using HSV heuristics."""
    img = cv2.imread(str(image_path))
    if img is None:
        return "day", 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hsv = hsv.reshape(-1, 3).mean(axis=0)
    mean_sat = float(mean_hsv[1])
    mean_val = float(mean_hsv[2])

    sat_score = float(np.clip((mean_sat - 15.0) / 70.0, 0.0, 1.0))
    val_score = float(np.clip((mean_val - 60.0) / 140.0, 0.0, 1.0))
    day_score = 0.6 * sat_score + 0.4 * val_score
    if mean_val >= 120 and mean_sat >= 5:
        label = "day"
    elif day_score >= 0.35:
        label = "day"
    else:
        label = "night"
    conf = abs(day_score - 0.5) * 2.0
    return label, conf


def build_manifest_lookup(manifest_path: Path) -> Dict[str, str]:
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    lookup: Dict[str, str] = {}
    for entry in manifest.get("clips", []):
        path = entry.get("path") or ""
        lighting = entry.get("lighting")
        if lighting not in {"day", "night"}:
            continue
        base = Path(path).name.lower()
        variants = {base, base.replace("nest_", ""), f"nest_{base}", base.replace("segment_", "")}
        for v in variants:
            lookup.setdefault(v, lighting)
    return lookup


def resolve_lighting(
    stem: str,
    manifest_lookup: Dict[str, str],
) -> Tuple[Optional[str], str]:
    """Try to resolve lighting from manifest keys; return (lighting, source)."""
    key = stem.lower()
    for candidate in (
        key,
        f"nest_{key}",
        key.replace("nest_", ""),
        key.replace("segment_", ""),
    ):
        if candidate in manifest_lookup:
            return manifest_lookup[candidate], "manifest"
    if "night" in key:
        return "night", "name"
    if "day" in key:
        return "day", "name"
    return None, "none"


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    src_ds = repo / "outdoor/deer-vision/data/datasets/wildlife_golden_v5"
    manifest_path = Path("/srv/deer-share/golden_clips/manifest.json")

    out_root = repo / "outdoor/deer-vision/data/datasets"
    day_ds = out_root / "wildlife_golden_v6_day"
    night_ds = out_root / "wildlife_golden_v6_night"

    for ds in (day_ds, night_ds):
        if ds.exists():
            shutil.rmtree(ds)
        (ds / "images/train").mkdir(parents=True, exist_ok=True)
        (ds / "images/val").mkdir(parents=True, exist_ok=True)
        (ds / "labels/train").mkdir(parents=True, exist_ok=True)
        (ds / "labels/val").mkdir(parents=True, exist_ok=True)

    manifest_lookup = build_manifest_lookup(manifest_path)
    stats = {"day": 0, "night": 0}
    source_counts = {"manifest": 0, "heuristic": 0}
    conf_accum = {"day": [], "night": []}

    for split in ["train", "val"]:
        img_dir = src_ds / "images" / split
        lbl_dir = src_ds / "labels" / split
        for img_path in img_dir.glob("*.jpg"):
            stem = img_path.stem
            lighting, conf_source = resolve_lighting(stem, manifest_lookup)
            if lighting is None:
                lighting, conf = classify_image(img_path)
                source_counts["heuristic"] += 1
            else:
                conf = 1.0
                source_counts["manifest"] += 1
            stats[lighting] += 1
            conf_accum[lighting].append(conf)
            target_ds = day_ds if lighting == "day" else night_ds
            shutil.copy2(img_path, target_ds / "images" / split / img_path.name)
            lbl_src = lbl_dir / f"{stem}.txt"
            if lbl_src.exists():
                shutil.copy2(lbl_src, target_ds / "labels" / split / lbl_src.name)

    print("Split counts:", stats)
    print("Source counts:", source_counts)
    for k in ["day", "night"]:
        if conf_accum[k]:
            print(f"{k} mean_conf={np.mean(conf_accum[k]):.3f}")

    data_tpl = """path: {path}
train: images/train
val: images/val

names:
  0: deer
  1: person
"""
    for ds in [day_ds, night_ds]:
        (ds / "data.yaml").write_text(data_tpl.format(path=ds), encoding="utf-8")
        print("Wrote", ds / "data.yaml")


if __name__ == "__main__":
    main()
