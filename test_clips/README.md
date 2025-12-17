# Test Clips for Pipeline Health Checks

This directory contains test clips and ground truth annotations for verifying the detection pipeline is operational.

## Purpose

These clips serve as regression tests to:
1. Verify the pipeline is online and processing clips
2. Detect model regressions (e.g., AprilTag misclassification as animals)
3. Ensure both deer and person detection are working

## Contents

- `deer_test.mkv` - Daytime clip with a deer in the backyard (source: 2025-12-14 21:56 UTC)
- `person_test.mkv` - Nighttime clip with a person walking by (source: 2025-12-16 00:30 UTC)
- `deer_test_frame.jpg` - Reference frame from deer clip (frame 165)
- `person_test_frame.jpg` - Reference frame from person clip (frame 15)
- `ground_truth.json` - Expected detections and pass/fail criteria

## Running the Health Check

```bash
# From repo root on Ubuntu server
cd ~/projects/DeerAITrackingResponse
source .venv/bin/activate

# Run with default models (yolov8n_wildlife_v2.pt, rtdetr_wildlife_v2.pt)
python scripts/pipeline_health_check.py --verbose

# Run with specific model(s)
python scripts/pipeline_health_check.py --verbose --models yolov8n.pt
python scripts/pipeline_health_check.py --verbose --models yolov8n_wildlife.pt,rtdetr_wildlife.pt

# Output JSON format
python scripts/pipeline_health_check.py --json
```

## Expected Results

A healthy pipeline should:
- Detect deer in `deer_test.mkv` with confidence >= 0.25
- NOT detect person in `deer_test.mkv` (no false positives)
- Detect person in `person_test.mkv` with confidence >= 0.4
- NOT detect deer in `person_test.mkv` (no false positives)
- NOT classify AprilTags as animals

## Known Issues

### AprilTag Misclassification
Models trained on datasets that included AprilTag labels may misclassify AprilTags as animals.
This is a critical failure because it causes false positives in every clip.

Symptoms:
- High-confidence (>0.85) "animal" detections at fixed positions in the frame
- Positions near edges of frame (x < 0.2 or x > 0.8)
- 100+ detections per clip instead of 1-5

### Model Comparison (as of 2025-12-17)

| Model | deer_test.mkv | person_test.mkv | Notes |
|-------|---------------|-----------------|-------|
| yolov8n.pt (base COCO) | PASS (deer 0.31) | PASS (person 0.54) | Reliable baseline |
| yolov8n_wildlife.pt | FAIL (AprilTag FPs) | FAIL (no person) | AprilTag misclass |
| yolov8n_wildlife_v2.pt | FAIL (no deer) | FAIL (AprilTag FPs) | Needs retraining |
| rtdetr_wildlife_v2.pt | FAIL (no deer) | FAIL (AprilTag FPs) | Needs retraining |

## Updating Test Clips

When adding new test clips:
1. Copy the clip to this directory
2. Extract a reference frame: `ffmpeg -i clip.mkv -vf 'select=eq(n\,FRAME)' -vframes 1 frame.jpg`
3. Update `ground_truth.json` with expected detections and criteria
4. Run health check to verify

## Integration with Pipeline

The health check can be run:
- Manually before/after model retraining
- As part of CI/CD (exit code 0 = healthy, 1 = unhealthy)
- Periodically via cron to detect pipeline failures
