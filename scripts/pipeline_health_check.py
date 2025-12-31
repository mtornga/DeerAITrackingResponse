#!/usr/bin/env python3
"""
Pipeline health check script.

Runs test clips through the detection pipeline and validates results against
ground truth. Used to verify the pipeline is operational and models are working.

Supports --lighting day|night mode:
  - day: Tests both deer and person detection (full health check)
  - night: Tests only deer detection (skips person_test.mkv since night models
           are not trained on person data)

Usage:
    python scripts/pipeline_health_check.py
    python scripts/pipeline_health_check.py --verbose
    python scripts/pipeline_health_check.py --models yolov8n_wildlife.pt
    python scripts/pipeline_health_check.py --lighting night --models yolov8n_wildlife_v6_night.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2


def _ensure_repo_root_on_path() -> Path:
    """Add repo root to path for imports."""
    script_path = Path(__file__).resolve()
    for parent in (script_path.parent, *script_path.parents):
        candidate = parent / ".env"
        if candidate.exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    root = script_path.parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_path()
TEST_CLIPS_DIR = REPO_ROOT / "test_clips"
MODELS_DIR = REPO_ROOT / "models"


# Category mapping (matches live_detector_multimodel.py)
COCO_ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "deer", "rabbit", "squirrel", "raccoon", "fox"
}
COCO_PERSON_CLASSES = {"person"}


@dataclass
class HealthCheckResult:
    """Result of a single test clip check."""
    clip_name: str
    passed: bool
    detections: Dict[str, List[Dict]]  # model_name -> list of detections
    errors: List[str]
    warnings: List[str]
    inference_time_ms: float

    def to_dict(self) -> Dict:
        return {
            "clip": self.clip_name,
            "passed": self.passed,
            "detections": self.detections,
            "errors": self.errors,
            "warnings": self.warnings,
            "inference_time_ms": self.inference_time_ms,
        }


@dataclass
class PipelineStatus:
    """Overall pipeline health status."""
    healthy: bool
    timestamp: str
    clip_results: List[HealthCheckResult]
    system_checks: Dict[str, bool]
    summary: str

    def to_dict(self) -> Dict:
        return {
            "healthy": self.healthy,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "system_checks": self.system_checks,
            "clip_results": [r.to_dict() for r in self.clip_results],
        }


def check_system_components() -> Dict[str, bool]:
    """Check that system components are available."""
    checks = {}

    # Check test clips exist
    checks["test_clips_dir_exists"] = TEST_CLIPS_DIR.exists()
    checks["deer_test_clip_exists"] = (TEST_CLIPS_DIR / "deer_test.mkv").exists()
    checks["deer_test_night_clip_exists"] = (TEST_CLIPS_DIR / "deer_test_night.mkv").exists()
    checks["person_test_clip_exists"] = (TEST_CLIPS_DIR / "person_test.mkv").exists()
    checks["ground_truth_exists"] = (TEST_CLIPS_DIR / "ground_truth.json").exists()

    # Check models directory
    checks["models_dir_exists"] = MODELS_DIR.exists()

    # Check if ultralytics is available
    try:
        from ultralytics import YOLO
        checks["ultralytics_available"] = True
    except ImportError:
        checks["ultralytics_available"] = False

    # Check if CUDA is available
    try:
        import torch
        checks["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        checks["cuda_available"] = False

    return checks


def load_ground_truth() -> Dict:
    """Load ground truth annotations."""
    gt_path = TEST_CLIPS_DIR / "ground_truth.json"
    if not gt_path.exists():
        return {}
    with open(gt_path) as f:
        return json.load(f)


def map_detection_to_category(label: str) -> str:
    """Map a detection label to our category system."""
    label_lower = label.lower()
    if label_lower in COCO_PERSON_CLASSES or label_lower == "person":
        return "person"
    if label_lower in COCO_ANIMAL_CLASSES or label_lower in ("deer", "animal", "unknown_animal"):
        return "deer"
    if label_lower == "apriltag":
        return "apriltag"
    return label_lower


def run_detection_on_clip(
    clip_path: Path,
    model_paths: List[Path],
    sample_interval: int = 15,
    confidence_threshold: float = 0.2,
    device: str = "cuda:0",
    verbose: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Run detection models on a video clip.

    Returns dict mapping model_name -> list of detections
    """
    from ultralytics import YOLO

    results = {}

    for model_path in model_paths:
        model_name = model_path.stem
        if verbose:
            print(f"  Loading model: {model_name}")

        try:
            model = YOLO(str(model_path))
            # Move to device
            if device.startswith("cuda"):
                try:
                    import torch
                    if torch.cuda.is_available():
                        model.to(device)
                except Exception:
                    pass
        except Exception as e:
            if verbose:
                print(f"  ERROR loading model {model_name}: {e}")
            results[model_name] = []
            continue

        detections = []
        cap = cv2.VideoCapture(str(clip_path))
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames
            if frame_idx % sample_interval == 0:
                try:
                    preds = model(frame, verbose=False)
                    for pred in preds:
                        if pred.boxes is None:
                            continue
                        for box in pred.boxes:
                            conf = float(box.conf[0])
                            if conf < confidence_threshold:
                                continue
                            cls_id = int(box.cls[0])
                            label = model.names.get(cls_id, str(cls_id))
                            category = map_detection_to_category(label)

                            # Get normalized bbox
                            x1, y1, x2, y2 = box.xyxyn[0].tolist()
                            bbox = [
                                (x1 + x2) / 2,  # x_center
                                (y1 + y2) / 2,  # y_center
                                x2 - x1,         # width
                                y2 - y1,         # height
                            ]

                            detections.append({
                                "frame": frame_idx,
                                "category": category,
                                "label": label,
                                "confidence": conf,
                                "bbox": bbox,
                            })
                except Exception as e:
                    if verbose:
                        print(f"  ERROR on frame {frame_idx}: {e}")

            frame_idx += 1

        cap.release()
        results[model_name] = detections

        if verbose:
            # Summarize detections
            categories = {}
            for d in detections:
                cat = d["category"]
                categories[cat] = categories.get(cat, 0) + 1
            print(f"  {model_name}: {len(detections)} detections - {categories}")

    return results


def validate_clip_result(
    clip_name: str,
    detections: Dict[str, List[Dict]],
    ground_truth: Dict,
    verbose: bool = False,
) -> HealthCheckResult:
    """Validate detections against ground truth."""
    errors = []
    warnings = []

    gt_clip = ground_truth.get("clips", {}).get(clip_name, {})
    if not gt_clip:
        errors.append(f"No ground truth found for {clip_name}")
        return HealthCheckResult(
            clip_name=clip_name,
            passed=False,
            detections=detections,
            errors=errors,
            warnings=warnings,
            inference_time_ms=0,
        )

    expected = gt_clip.get("expected_detections", {})
    criteria = gt_clip.get("quality_criteria", {})

    # Aggregate detections across all models
    all_detections = []
    for model_dets in detections.values():
        all_detections.extend(model_dets)

    # Count by category
    category_counts = {}
    max_confidence = {}
    for d in all_detections:
        cat = d["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if cat not in max_confidence or d["confidence"] > max_confidence[cat]:
            max_confidence[cat] = d["confidence"]

    if verbose:
        print(f"  Category counts: {category_counts}")
        print(f"  Max confidence: {max_confidence}")

    # Check criteria
    if criteria.get("must_detect_deer"):
        if category_counts.get("deer", 0) == 0:
            errors.append("FAIL: No deer detected (expected at least 1)")
        else:
            min_conf = criteria.get("deer_confidence_min", 0.5)
            if max_confidence.get("deer", 0) < min_conf:
                errors.append(f"FAIL: Deer confidence {max_confidence.get('deer', 0):.2f} < {min_conf}")

    if criteria.get("must_detect_person"):
        if category_counts.get("person", 0) == 0:
            errors.append("FAIL: No person detected (expected at least 1)")
        else:
            min_conf = criteria.get("person_confidence_min", 0.4)
            if max_confidence.get("person", 0) < min_conf:
                errors.append(f"FAIL: Person confidence {max_confidence.get('person', 0):.2f} < {min_conf}")

    if criteria.get("must_not_detect_deer"):
        if category_counts.get("deer", 0) > 0:
            # Check if these might be AprilTag misclassifications
            deer_dets = [d for d in all_detections if d["category"] == "deer"]
            # Check if detections are at fixed positions (likely AprilTags)
            fixed_positions = []
            for d in deer_dets:
                x, y = d["bbox"][0], d["bbox"][1]
                # AprilTag positions are typically at edges of frame
                if x < 0.2 or x > 0.8 or y < 0.2 or y > 0.8:
                    fixed_positions.append(d)
            if len(fixed_positions) == len(deer_dets):
                warnings.append(f"WARNING: {len(deer_dets)} deer detections appear to be AprilTag misclassifications")
            else:
                errors.append(f"FAIL: {category_counts.get('deer', 0)} deer detected (expected 0)")

    if criteria.get("must_not_detect_person"):
        if category_counts.get("person", 0) > 0:
            errors.append(f"FAIL: {category_counts.get('person', 0)} person detected (expected 0)")

    # Check AprilTag misclassification (critical failure)
    apriltag_as_animal = ground_truth.get("pass_criteria", {}).get("apriltag_as_animal_is_failure", True)
    if apriltag_as_animal:
        # Check for high-confidence animal detections at AprilTag positions
        for d in all_detections:
            if d["category"] == "deer" and d["confidence"] > 0.85:
                x, y = d["bbox"][0], d["bbox"][1]
                # AprilTag positions are typically at edges
                if (x < 0.2 or x > 0.8) and (y < 0.3 or y > 0.7):
                    errors.append(f"FAIL: Likely AprilTag misclassified as deer at ({x:.2f}, {y:.2f}) with conf {d['confidence']:.2f}")

    passed = len(errors) == 0

    return HealthCheckResult(
        clip_name=clip_name,
        passed=passed,
        detections=detections,
        errors=errors,
        warnings=warnings,
        inference_time_ms=0,
    )


def run_health_check(
    model_names: Optional[List[str]] = None,
    device: str = "cuda:0",
    verbose: bool = False,
    lighting: str = "day",
) -> PipelineStatus:
    """Run full pipeline health check."""
    timestamp = datetime.now().astimezone().isoformat()

    # System checks
    system_checks = check_system_components()

    if verbose:
        print("System checks:")
        for check, passed in system_checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {check}: {status}")

    # Check if we can proceed
    if not system_checks.get("ultralytics_available"):
        return PipelineStatus(
            healthy=False,
            timestamp=timestamp,
            clip_results=[],
            system_checks=system_checks,
            summary="FAIL: ultralytics not available",
        )

    # Load ground truth
    ground_truth = load_ground_truth()
    if not ground_truth:
        return PipelineStatus(
            healthy=False,
            timestamp=timestamp,
            clip_results=[],
            system_checks=system_checks,
            summary="FAIL: ground_truth.json not found",
        )

    # Determine models to test
    if model_names is None:
        # Default: test wildlife models
        model_names = ["yolov8n_wildlife_v2.pt", "rtdetr_wildlife_v2.pt"]

    model_paths = []
    for name in model_names:
        path = MODELS_DIR / name
        if path.exists():
            model_paths.append(path)
        else:
            if verbose:
                print(f"WARNING: Model not found: {path}")

    if not model_paths:
        return PipelineStatus(
            healthy=False,
            timestamp=timestamp,
            clip_results=[],
            system_checks=system_checks,
            summary="FAIL: No models found",
        )

    if verbose:
        print(f"\nModels to test: {[p.name for p in model_paths]}")
        print(f"Lighting mode: {lighting}")

    # Run tests on each clip
    # Night mode: test deer_test_night.mkv only (night IR footage, no person test)
    # Day mode: test deer_test.mkv + person_test.mkv
    clip_results = []
    if lighting == "night":
        test_clips = ["deer_test_night.mkv"]
        if verbose:
            print("Night mode: testing deer_test_night.mkv (IR footage)")
            print("Night mode: skipping person test (night models not trained on person data)")
    else:
        test_clips = ["deer_test.mkv", "person_test.mkv"]

    for clip_name in test_clips:
        clip_path = TEST_CLIPS_DIR / clip_name
        if not clip_path.exists():
            clip_results.append(HealthCheckResult(
                clip_name=clip_name,
                passed=False,
                detections={},
                errors=[f"Test clip not found: {clip_path}"],
                warnings=[],
                inference_time_ms=0,
            ))
            continue

        if verbose:
            print(f"\nTesting: {clip_name}")

        start_time = time.time()
        detections = run_detection_on_clip(
            clip_path=clip_path,
            model_paths=model_paths,
            sample_interval=15,
            confidence_threshold=0.2,
            device=device,
            verbose=verbose,
        )
        inference_time = (time.time() - start_time) * 1000

        result = validate_clip_result(clip_name, detections, ground_truth, verbose)
        result.inference_time_ms = inference_time
        clip_results.append(result)

        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"  Result: {status}")
            for err in result.errors:
                print(f"    {err}")
            for warn in result.warnings:
                print(f"    {warn}")

    # Determine overall health
    all_passed = all(r.passed for r in clip_results)
    healthy = all_passed and all(system_checks.values())

    # Generate summary
    passed_count = sum(1 for r in clip_results if r.passed)
    total_count = len(clip_results)

    if healthy:
        summary = f"HEALTHY: All {total_count} test clips passed"
    else:
        failed_clips = [r.clip_name for r in clip_results if not r.passed]
        failed_checks = [k for k, v in system_checks.items() if not v]
        issues = []
        if failed_clips:
            issues.append(f"clips failed: {failed_clips}")
        if failed_checks:
            issues.append(f"system checks failed: {failed_checks}")
        summary = f"UNHEALTHY: {passed_count}/{total_count} clips passed. Issues: {'; '.join(issues)}"

    return PipelineStatus(
        healthy=healthy,
        timestamp=timestamp,
        clip_results=clip_results,
        system_checks=system_checks,
        summary=summary,
    )


def main():
    parser = argparse.ArgumentParser(description="Pipeline health check")
    parser.add_argument("--models", type=str, help="Comma-separated model names")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument(
        "--lighting",
        type=str,
        choices=["day", "night"],
        default="day",
        help="Lighting mode: 'day' tests deer+person, 'night' tests deer only",
    )
    args = parser.parse_args()

    model_names = None
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]

    print("=" * 60)
    print(f"Pipeline Health Check ({args.lighting} mode)")
    print("=" * 60)

    status = run_health_check(
        model_names=model_names,
        device=args.device,
        verbose=args.verbose,
        lighting=args.lighting,
    )

    if args.json:
        print(json.dumps(status.to_dict(), indent=2))
    else:
        print("\n" + "=" * 60)
        print(f"Status: {status.summary}")
        print("=" * 60)

        if not status.healthy:
            print("\nDetails:")
            for result in status.clip_results:
                if not result.passed:
                    print(f"\n  {result.clip_name}:")
                    for err in result.errors:
                        print(f"    - {err}")

    # Exit with appropriate code
    sys.exit(0 if status.healthy else 1)


if __name__ == "__main__":
    main()
