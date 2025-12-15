#!/usr/bin/env python3
"""
Continuous feedback cycle for improving deer detection models.

This orchestrator runs the full virtuous cycle:
1. INGEST: Pull reviewed clips from daily_review, extract frames
2. AUTOLABEL: Run MegaDetector/YOLOv8 on animal frames to generate labels
3. CURATE: Prepare hard negatives from false_positive/hard_negative frames
4. MERGE: Combine feedback data with existing training set
5. TRAIN: Train multiple model variants (YOLOv8n, RT-DETR)
6. EVAL: Evaluate all models against full eval corpus
7. COMPARE: Run bakeoff, select winner based on quality gates
8. PROMOTE: Deploy best model to live pipeline

Designed to run continuously on the Ubuntu server with GTX 3080.

Usage:
    # Single cycle (for testing)
    python scripts/feedback_cycle.py --single-run

    # Continuous mode (production)
    python scripts/feedback_cycle.py --continuous --poll-interval 3600

    # Skip certain stages
    python scripts/feedback_cycle.py --skip-train --eval-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


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
DEER_VISION_ROOT = REPO_ROOT / "outdoor" / "deer-vision"

try:
    from env_loader import load_env_file
    load_env_file()
except ImportError:
    pass


UTC = timezone.utc


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CycleConfig:
    """Configuration for a feedback cycle run."""
    # Paths
    share_root: Path = field(default_factory=lambda: Path.home() / "DeerShare")
    feedback_frames_dir: Path = field(default_factory=lambda: DEER_VISION_ROOT / "data" / "raw" / "feedback")
    interim_dir: Path = field(default_factory=lambda: DEER_VISION_ROOT / "data" / "interim")
    datasets_dir: Path = field(default_factory=lambda: DEER_VISION_ROOT / "data" / "datasets")
    models_dir: Path = field(default_factory=lambda: DEER_VISION_ROOT / "models")
    eval_packs_dir: Path = field(default_factory=lambda: DEER_VISION_ROOT / "data" / "eval")
    runs_dir: Path = field(default_factory=lambda: REPO_ROOT / "runs" / "feedback_cycles")

    # Pseudo-labels (auto-accepted high-confidence detections)
    pseudo_labels_dir: Path = field(default_factory=lambda: Path.home() / "DeerShare" / "runs" / "live" / "pseudo_labels")
    pseudo_label_ratio: float = 0.3  # Max 30% of training data from pseudo-labels
    use_pseudo_labels: bool = True

    # Models to train
    model_variants: List[str] = field(default_factory=lambda: ["yolov8n", "yolov8s", "rtdetr-l"])

    # Training parameters
    epochs: int = 80
    batch_size: int = 16
    imgsz: int = 960
    patience: int = 20

    # Quality thresholds (from eval_thresholds.yaml)
    map50_min: float = 0.80
    recall_min: float = 0.85
    fp_per_min_max: float = 0.25
    latency_ms_max: float = 40.0

    # Autolabeling (MegaDetector for wildlife)
    autolabel_conf: float = 0.2
    autolabel_model: Path = field(default_factory=lambda: REPO_ROOT / "models" / "md_v5a.0.0.pt")

    # Device
    device: str = ""  # Auto-detect


@dataclass
class CycleResult:
    """Result of a feedback cycle run."""
    cycle_id: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"

    # Stage results
    frames_ingested: int = 0
    frames_autolabeled: int = 0
    hard_negatives_added: int = 0
    pseudo_labels_added: int = 0  # Auto-accepted high-confidence clips
    models_trained: List[str] = field(default_factory=list)
    models_evaluated: Dict[str, Dict[str, float]] = field(default_factory=dict)
    winner: Optional[str] = None
    winner_metrics: Dict[str, float] = field(default_factory=dict)
    promoted: bool = False

    errors: List[str] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def detect_device() -> str:
    """Detect best available device for training."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# =============================================================================
# Stage 1: Ingest - Extract frames from reviewed clips
# =============================================================================

def stage_ingest(config: CycleConfig, result: CycleResult, logger: logging.Logger) -> bool:
    """Extract frames from newly reviewed clips."""
    logger.info("=== STAGE 1: INGEST ===")

    # Check for new reviewed clips
    index_path = config.share_root / "index" / "daily_review_index.json"
    if not index_path.exists():
        logger.warning(f"No review index found at {index_path}")
        return True  # Not an error, just nothing to do

    index_data = json.loads(index_path.read_text())
    clips = index_data.get("clips", {})

    # Find reviewed clips that haven't been processed yet
    processed_manifest = config.feedback_frames_dir / "processed_clips.json"
    if processed_manifest.exists():
        processed = set(json.loads(processed_manifest.read_text()).get("clips", []))
    else:
        processed = set()

    new_clips = []
    for clip_id, data in clips.items():
        if data.get("review_status") != "done":
            continue
        if clip_id in processed:
            continue
        new_clips.append((clip_id, data))

    if not new_clips:
        logger.info("No new reviewed clips to process")
        return True

    logger.info(f"Found {len(new_clips)} new reviewed clips to process")

    # Run extraction script
    extract_script = REPO_ROOT / "scripts" / "extract_feedback_frames.py"
    cmd = [
        sys.executable, str(extract_script),
        "--output-dir", str(config.feedback_frames_dir),
        "--stride", "5",
        "--max-frames", "50",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(proc.stdout)

        # Parse extraction results
        for line in proc.stdout.split("\n"):
            if "Frames extracted:" in line:
                result.frames_ingested = int(line.split(":")[1].strip())

        # Update processed manifest
        processed.update(clip_id for clip_id, _ in new_clips)
        processed_manifest.parent.mkdir(parents=True, exist_ok=True)
        processed_manifest.write_text(json.dumps({"clips": list(processed)}, indent=2))

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Extraction failed: {e.stderr}")
        result.errors.append(f"Ingest failed: {e.stderr}")
        return False


# =============================================================================
# Stage 2: Autolabel - Generate labels for animal frames
# =============================================================================

def stage_autolabel(config: CycleConfig, result: CycleResult, logger: logging.Logger) -> bool:
    """Run autolabeling on animal frames."""
    logger.info("=== STAGE 2: AUTOLABEL ===")

    animal_dir = config.feedback_frames_dir / "animal"
    if not animal_dir.exists() or not any(animal_dir.iterdir()):
        logger.info("No animal frames to autolabel")
        return True

    # Count frames
    frames = list(animal_dir.glob("*.jpg")) + list(animal_dir.glob("*.png"))
    logger.info(f"Found {len(frames)} animal frames to autolabel")

    if not frames:
        return True

    autolabel_out = config.interim_dir / "feedback_autolabel"
    autolabel_script = REPO_ROOT / "scripts" / "autolabel_megadetector.py"

    cmd = [
        sys.executable, str(autolabel_script),
        "--src", str(animal_dir),
        "--out", str(autolabel_out),
        "--model", str(config.autolabel_model),
        "--conf", str(config.autolabel_conf),
        "--imgsz", str(config.imgsz),
    ]

    if config.device:
        cmd.extend(["--device", config.device])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(REPO_ROOT))
        logger.info(proc.stdout)

        # Check autolabel summary
        summary_path = autolabel_out / "autolabel_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            result.frames_autolabeled = summary.get("image_count", 0)
            logger.info(f"Autolabeled {result.frames_autolabeled} frames with {summary.get('detections', 0)} detections")

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Autolabeling failed: {e.stderr}")
        result.errors.append(f"Autolabel failed: {e.stderr}")
        return False


# =============================================================================
# Stage 3: Curate - Prepare hard negatives
# =============================================================================

def stage_curate_hard_negatives(config: CycleConfig, result: CycleResult, logger: logging.Logger) -> bool:
    """Prepare hard negatives from false_positive and hard_negative frames."""
    logger.info("=== STAGE 3: CURATE HARD NEGATIVES ===")

    hard_neg_dirs = [
        config.feedback_frames_dir / "false_positive",
        config.feedback_frames_dir / "hard_negative",
    ]

    hard_neg_out = config.interim_dir / "feedback_hard_negatives"
    hard_neg_out.mkdir(parents=True, exist_ok=True)
    images_out = hard_neg_out / "images"
    labels_out = hard_neg_out / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for source_dir in hard_neg_dirs:
        if not source_dir.exists():
            continue

        for img_path in source_dir.glob("*.jpg"):
            # Copy image
            dest_img = images_out / img_path.name
            shutil.copy2(img_path, dest_img)

            # Create empty label file (no detections = hard negative)
            label_path = labels_out / img_path.with_suffix(".txt").name
            label_path.write_text("")

            count += 1

    result.hard_negatives_added = count
    logger.info(f"Prepared {count} hard negative frames")

    # Write manifest
    manifest = {
        "count": count,
        "sources": [str(d) for d in hard_neg_dirs if d.exists()],
        "created_at": _now_iso(),
    }
    (hard_neg_out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return True


# =============================================================================
# Stage 4: Merge - Combine feedback with training data
# =============================================================================

def stage_merge_datasets(config: CycleConfig, result: CycleResult, logger: logging.Logger) -> bool:
    """Merge feedback data into the training dataset."""
    logger.info("=== STAGE 4: MERGE DATASETS ===")

    # Create a new versioned dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"deer_det_feedback_{timestamp}"
    dataset_dir = config.datasets_dir / dataset_name

    # Copy base dataset structure if exists
    base_dataset = config.datasets_dir / "deer_det_v01"
    if base_dataset.exists() and any((base_dataset / "images").glob("*/*.jpg")):
        logger.info(f"Copying base dataset from {base_dataset}")
        shutil.copytree(base_dataset, dataset_dir)
    else:
        logger.info("Creating new dataset structure (base dataset empty or missing)")

    # Ensure directories exist regardless of whether we copied a base
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Add autolabeled frames to training set
    autolabel_dir = config.interim_dir / "feedback_autolabel"
    if autolabel_dir.exists():
        autolabel_images = autolabel_dir / "images"
        autolabel_labels = autolabel_dir / "labels"

        if autolabel_images.exists():
            for img_path in autolabel_images.glob("*.jpg"):
                # 80/20 split
                split = "train" if hash(img_path.name) % 5 != 0 else "val"
                shutil.copy2(img_path, dataset_dir / "images" / split / img_path.name)

                label_path = autolabel_labels / img_path.with_suffix(".txt").name
                if label_path.exists():
                    shutil.copy2(label_path, dataset_dir / "labels" / split / label_path.name)

    # Add hard negatives to training set
    hard_neg_dir = config.interim_dir / "feedback_hard_negatives"
    if hard_neg_dir.exists():
        hard_neg_images = hard_neg_dir / "images"
        hard_neg_labels = hard_neg_dir / "labels"

        if hard_neg_images.exists():
            for img_path in hard_neg_images.glob("*.jpg"):
                split = "train" if hash(img_path.name) % 5 != 0 else "val"
                shutil.copy2(img_path, dataset_dir / "images" / split / img_path.name)

                label_path = hard_neg_labels / img_path.with_suffix(".txt").name
                if label_path.exists():
                    shutil.copy2(label_path, dataset_dir / "labels" / split / label_path.name)

    # Add pseudo-labels (auto-accepted high-confidence detections)
    # Pseudo-labels use detection results from the model ensemble, not MegaDetector
    pseudo_count = 0
    if config.use_pseudo_labels and config.pseudo_labels_dir.exists():
        pseudo_index_path = config.pseudo_labels_dir / "pseudo_label_index.json"
        if pseudo_index_path.exists():
            pseudo_index = json.loads(pseudo_index_path.read_text())

            # Calculate how many pseudo-labels to use (capped by ratio)
            # Count current images BEFORE adding pseudo
            train_imgs = list((dataset_dir / "images" / "train").glob("*.jpg"))
            val_imgs = list((dataset_dir / "images" / "val").glob("*.jpg"))
            current_count = len(train_imgs) + len(val_imgs)
            max_pseudo = int(current_count * config.pseudo_label_ratio / (1 - config.pseudo_label_ratio)) if current_count > 0 else 100
            logger.info(f"Pseudo-label pool has {len(pseudo_index.get('clips', {}))} clips, using up to {max_pseudo}")

            # Category mapping for YOLO format
            category_to_class = {"animal": 0, "deer": 0, "unknown_animal": 1, "person": 2}

            # Process pseudo-labeled clips
            for clip_path, clip_info in pseudo_index.get("clips", {}).items():
                if pseudo_count >= max_pseudo:
                    break

                # Find the event directory for this clip
                event_base = config.share_root / "runs" / "live" / "events"
                clip_parts = Path(clip_path)
                event_dir = event_base / clip_parts.parent / clip_parts.stem

                if not event_dir.exists():
                    continue

                # Read meta.json to get detection boxes
                meta_path = event_dir / "meta.json"
                if not meta_path.exists():
                    continue

                meta = json.loads(meta_path.read_text())
                model_results = meta.get("models", {})

                # Find the video segment
                segment_path = None
                for ext in [".mkv", ".mp4"]:
                    candidate = event_dir / (clip_parts.stem + ext)
                    if candidate.exists():
                        segment_path = candidate
                        break

                if not segment_path:
                    continue

                # Extract frame and use existing detections as labels
                import cv2
                cap = cv2.VideoCapture(str(segment_path))
                if not cap.isOpened():
                    continue

                try:
                    # Get frame from middle of video (where detection likely occurred)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)

                    ret, frame = cap.read()
                    if not ret:
                        continue

                    # Get best detection from consensus model results
                    yolo_labels = []
                    for model_name, model_data in model_results.items():
                        hits = model_data.get("hits", [])
                        if not hits:
                            continue

                        # Use highest confidence hit from this model
                        best_hit = max(hits, key=lambda h: h.get("confidence", 0))
                        category = best_hit.get("category", "animal")
                        class_id = category_to_class.get(category, 0)
                        bbox = best_hit.get("bbox", [])

                        if bbox and len(bbox) == 4:
                            # YOLO format: class x_center y_center width height (normalized)
                            x, y, w, h = bbox
                            x_center = x + w / 2
                            y_center = y + h / 2
                            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                        break  # Use only first model's detection to avoid duplicates

                    if not yolo_labels:
                        continue

                    # Save frame and label
                    frame_name = f"pseudo_{clip_parts.stem}.jpg"
                    split = "train" if hash(frame_name) % 5 != 0 else "val"

                    frame_path = dataset_dir / "images" / split / frame_name
                    cv2.imwrite(str(frame_path), frame)

                    label_path = dataset_dir / "labels" / split / f"pseudo_{clip_parts.stem}.txt"
                    label_path.write_text("\n".join(yolo_labels))

                    pseudo_count += 1

                finally:
                    cap.release()

            result.pseudo_labels_added = pseudo_count
            logger.info(f"Added {pseudo_count} pseudo-labeled frames with YOLO labels")

    # Create dataset YAML
    data_yaml = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "deer",
            1: "unknown_animal",
            2: "person",
            3: "apriltag",
        },
    }
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(yaml.dump(data_yaml))

    # Count images
    train_count = len(list((dataset_dir / "images" / "train").glob("*.jpg")))
    val_count = len(list((dataset_dir / "images" / "val").glob("*.jpg")))
    logger.info(f"Dataset {dataset_name}: {train_count} train, {val_count} val images")

    # Store dataset name for training stage
    config._current_dataset = dataset_name
    config._dataset_yaml = yaml_path

    return True


# =============================================================================
# Stage 5: Train - Train multiple model variants
# =============================================================================

def stage_train_models(config: CycleConfig, result: CycleResult, logger: logging.Logger) -> bool:
    """Train multiple model variants."""
    logger.info("=== STAGE 5: TRAIN MODELS ===")

    if not hasattr(config, "_dataset_yaml"):
        logger.error("No dataset prepared for training")
        result.errors.append("No dataset prepared for training")
        return False

    dataset_yaml = config._dataset_yaml
    dataset_name = config._current_dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for variant in config.model_variants:
        logger.info(f"Training {variant}...")

        run_name = f"{variant}_{dataset_name}_{timestamp}"

        train_script = DEER_VISION_ROOT / "scripts" / "train.py"

        # Determine base model
        if "rtdetr" in variant.lower():
            base_model = f"{variant}.pt"
        else:
            base_model = f"{variant}.pt"

        cmd = [
            sys.executable, str(train_script),
            "--data", str(dataset_yaml),
            "--model", base_model,
            "--name", run_name,
            "--project", str(config.models_dir),
            "--epochs", str(config.epochs),
            "--imgsz", str(config.imgsz),
            "--batch", str(config.batch_size),
        ]

        if config.device:
            cmd.extend(["--device", config.device])

        try:
            logger.info(f"Running: {' '.join(cmd)}")
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(DEER_VISION_ROOT),
                timeout=7200,  # 2 hour timeout per model
            )
            logger.info(proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout)

            result.models_trained.append(run_name)
            logger.info(f"Successfully trained {run_name}")

        except subprocess.TimeoutExpired:
            logger.error(f"Training {variant} timed out after 2 hours")
            result.errors.append(f"Training {variant} timed out")
        except subprocess.CalledProcessError as e:
            logger.error(f"Training {variant} failed: {e.stderr[-1000:] if e.stderr else 'No stderr'}")
            result.errors.append(f"Training {variant} failed")

    return len(result.models_trained) > 0


# =============================================================================
# Stage 6: Evaluate - Test against eval corpus
# =============================================================================

def stage_evaluate_models(config: CycleConfig, result: CycleResult, logger: logging.Logger) -> bool:
    """Evaluate all trained models against the eval corpus."""
    logger.info("=== STAGE 6: EVALUATE MODELS ===")

    eval_script = DEER_VISION_ROOT / "scripts" / "evaluate.py"

    for run_name in result.models_trained:
        model_path = config.models_dir / run_name / "weights" / "best.pt"

        if not model_path.exists():
            logger.warning(f"Model weights not found: {model_path}")
            continue

        metrics_path = config.models_dir / run_name / "eval_metrics.json"

        cmd = [
            sys.executable, str(eval_script),
            "--model", str(model_path),
            "--eval-packs", str(config.eval_packs_dir),
            "--output", str(metrics_path),
            "--conf", "0.25",
            "--imgsz", str(config.imgsz),
        ]

        if config.device:
            cmd.extend(["--device", config.device])

        try:
            logger.info(f"Evaluating {run_name}...")
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(DEER_VISION_ROOT),
                timeout=1800,  # 30 min timeout
            )
            logger.info(proc.stdout)

            # Load metrics
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text())
                result.models_evaluated[run_name] = {
                    "map50": metrics.get("map50", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "latency_ms_p95": metrics.get("latency_ms_p95", 999.0),
                    "fp_per_min": max(metrics.get("fp_per_min", {}).values()) if metrics.get("fp_per_min") else 999.0,
                }
                logger.info(f"  {run_name}: mAP50={result.models_evaluated[run_name]['map50']:.3f}, "
                           f"recall={result.models_evaluated[run_name]['recall']:.3f}")

        except subprocess.TimeoutExpired:
            logger.error(f"Evaluation {run_name} timed out")
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation {run_name} failed: {e.stderr}")

    return len(result.models_evaluated) > 0


# =============================================================================
# Stage 7: Compare - Select best model based on quality gates
# =============================================================================

def stage_compare_models(config: CycleConfig, result: CycleResult, logger: logging.Logger) -> bool:
    """Compare models and select winner based on quality gates."""
    logger.info("=== STAGE 7: COMPARE MODELS ===")

    if not result.models_evaluated:
        logger.warning("No models evaluated, cannot compare")
        return True

    # Score each model
    scores: List[Tuple[str, float, Dict[str, float]]] = []

    for model_name, metrics in result.models_evaluated.items():
        # Check quality gates
        passes_gates = (
            metrics["map50"] >= config.map50_min and
            metrics["recall"] >= config.recall_min and
            metrics["latency_ms_p95"] <= config.latency_ms_max and
            metrics["fp_per_min"] <= config.fp_per_min_max
        )

        # Compute composite score (weighted)
        # Higher is better for map50, recall
        # Lower is better for latency, fp_per_min
        score = (
            metrics["map50"] * 0.35 +
            metrics["recall"] * 0.35 +
            (1.0 - min(metrics["latency_ms_p95"] / 100, 1.0)) * 0.15 +
            (1.0 - min(metrics["fp_per_min"] / 1.0, 1.0)) * 0.15
        )

        # Penalize models that don't pass quality gates
        if not passes_gates:
            score *= 0.5

        logger.info(f"{model_name}: score={score:.4f}, passes_gates={passes_gates}")
        logger.info(f"  mAP50={metrics['map50']:.3f} (min={config.map50_min}), "
                   f"recall={metrics['recall']:.3f} (min={config.recall_min})")
        logger.info(f"  latency={metrics['latency_ms_p95']:.1f}ms (max={config.latency_ms_max}), "
                   f"fp_per_min={metrics['fp_per_min']:.2f} (max={config.fp_per_min_max})")

        scores.append((model_name, score, metrics))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    if scores:
        winner_name, winner_score, winner_metrics = scores[0]
        result.winner = winner_name
        result.winner_metrics = winner_metrics
        logger.info(f"Winner: {winner_name} (score={winner_score:.4f})")

    return True


# =============================================================================
# Stage 8: Promote - Deploy best model to live pipeline
# =============================================================================

def stage_promote_model(config: CycleConfig, result: CycleResult, logger: logging.Logger) -> bool:
    """Promote the winning model to the live pipeline."""
    logger.info("=== STAGE 8: PROMOTE MODEL ===")

    if not result.winner:
        logger.info("No winner to promote")
        return True

    winner_weights = config.models_dir / result.winner / "weights" / "best.pt"
    if not winner_weights.exists():
        logger.error(f"Winner weights not found: {winner_weights}")
        return False

    # Create symlink or copy to models/live/
    live_models_dir = REPO_ROOT / "models"
    live_models_dir.mkdir(parents=True, exist_ok=True)

    # Determine model type from name
    if "yolov8n" in result.winner:
        target_name = "yolov8n_deer.pt"
    elif "yolov8s" in result.winner:
        target_name = "yolov8s_deer.pt"
    elif "rtdetr" in result.winner:
        target_name = "rtdetr_deer.pt"
    else:
        target_name = "best_deer.pt"

    target_path = live_models_dir / target_name

    # Backup existing model
    if target_path.exists():
        backup_path = target_path.with_suffix(f".pt.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(target_path, backup_path)
        logger.info(f"Backed up existing model to {backup_path}")

    # Copy new model
    shutil.copy2(winner_weights, target_path)
    logger.info(f"Promoted {result.winner} to {target_path}")

    # Write promotion manifest
    manifest = {
        "promoted_at": _now_iso(),
        "model": result.winner,
        "target": str(target_path),
        "metrics": result.winner_metrics,
    }
    (live_models_dir / f"{target_name}.manifest.json").write_text(json.dumps(manifest, indent=2))

    result.promoted = True
    return True


# =============================================================================
# Main orchestrator
# =============================================================================

def run_feedback_cycle(config: CycleConfig, logger: logging.Logger) -> CycleResult:
    """Run a single feedback cycle."""
    cycle_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = CycleResult(cycle_id=cycle_id, started_at=_now_iso())

    logger.info(f"Starting feedback cycle {cycle_id}")

    # Auto-detect device if not specified
    if not config.device:
        config.device = detect_device()
        logger.info(f"Detected device: {config.device}")

    stages = [
        ("ingest", stage_ingest),
        ("autolabel", stage_autolabel),
        ("curate", stage_curate_hard_negatives),
        ("merge", stage_merge_datasets),
        ("train", stage_train_models),
        ("evaluate", stage_evaluate_models),
        ("compare", stage_compare_models),
        ("promote", stage_promote_model),
    ]

    for stage_name, stage_fn in stages:
        try:
            success = stage_fn(config, result, logger)
            if not success:
                logger.error(f"Stage {stage_name} failed, aborting cycle")
                result.status = f"failed_at_{stage_name}"
                break
        except Exception as e:
            logger.exception(f"Stage {stage_name} raised exception: {e}")
            result.errors.append(f"{stage_name}: {str(e)}")
            result.status = f"error_at_{stage_name}"
            break
    else:
        result.status = "completed"

    result.completed_at = _now_iso()

    # Save cycle result
    results_dir = config.runs_dir / cycle_id
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "result.json").write_text(json.dumps(asdict(result), indent=2))

    logger.info(f"Cycle {cycle_id} finished with status: {result.status}")
    if result.winner:
        logger.info(f"Winner: {result.winner}")
    if result.errors:
        logger.warning(f"Errors: {result.errors}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--poll-interval", type=int, default=3600, help="Seconds between cycles in continuous mode")
    parser.add_argument("--single-run", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--device", type=str, default="", help="Force specific device (cuda:0, cpu, mps)")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--model-variants", type=str, default="yolov8n,yolov8s", help="Comma-separated model variants")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (eval only)")
    parser.add_argument("--log-file", type=Path, default=Path("logs/feedback_cycle.log"))
    args = parser.parse_args()

    # Setup logging
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    # Build config
    config = CycleConfig(
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_variants=args.model_variants.split(","),
    )

    # Resolve share root
    for candidate in [Path("/srv/deer-share"), Path.home() / "DeerShare"]:
        if candidate.exists():
            config.share_root = candidate
            break

    logger.info(f"Share root: {config.share_root}")
    logger.info(f"Device: {config.device or 'auto'}")
    logger.info(f"Model variants: {config.model_variants}")

    if args.continuous:
        logger.info(f"Running in continuous mode (poll interval: {args.poll_interval}s)")
        while True:
            try:
                run_feedback_cycle(config, logger)
            except Exception as e:
                logger.exception(f"Cycle failed with exception: {e}")

            logger.info(f"Sleeping for {args.poll_interval}s before next cycle...")
            time.sleep(args.poll_interval)
    else:
        result = run_feedback_cycle(config, logger)
        return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
