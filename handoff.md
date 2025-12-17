# Developer Handoff Notes
**Date**: 2025-12-17
**Session**: Pipeline Recovery & Test Infrastructure
**Status**: Ingest Online, Detection Using Base `yolov8n`, Wildlife Models Need Retraining

## What Was Accomplished

### 1. Diagnosed and Fixed Pipeline Outage
**Problem**: Pipeline was offline - no new clips since morning walk-by.

**Root Cause**: Disk 100% full (492GB/492GB). RTSP capture was failing with "ffmpeg exit status 228" (no space left on device).

**Solution**:
- Deleted old segments: `/srv/deer-share/runs/live/segments/2025-12-14/` (54GB) and `2025-12-15/` (38GB)
- Freed 90GB, disk now at 82%
- Restarted pipeline - capture resumed successfully

**Files Modified**: None (operational fix)

### 2. Created Pipeline Health Check System
**Problem**: No automated way to verify models are working or detect regressions like AprilTag misclassification.

**Solution**: Created test clips infrastructure with ground truth annotations

**New Files**:
- `test_clips/deer_test.mkv` - Daytime clip with real deer (from 2025-12-14)
- `test_clips/person_test.mkv` - Nighttime person walk-by (from 2025-12-16)
- `test_clips/ground_truth.json` - Expected detections and pass/fail criteria
- `test_clips/README.md` - Documentation
- `scripts/pipeline_health_check.py` - Automated health check script

**Usage**:
```bash
cd ~/projects/DeerAITrackingResponse
source .venv/bin/activate
python scripts/pipeline_health_check.py --verbose
python scripts/pipeline_health_check.py --verbose --models yolov8n.pt
```

### 3. Discovered Critical Model Issues
**Problem**: The v2 wildlife models (deployed Dec 16) are broken.

**Findings from health check**:

| Model | deer_test.mkv | person_test.mkv | Status |
|-------|---------------|-----------------|--------|
| yolov8n.pt (base COCO) | PASS (deer 0.31) | PASS (person 0.54) | HEALTHY |
| yolov8n_wildlife.pt | FAIL (AprilTag FPs) | FAIL (no person) | BROKEN |
| yolov8n_wildlife_v2.pt | FAIL (0 detections) | FAIL (AprilTag FPs) | BROKEN |
| rtdetr_wildlife_v2.pt | FAIL (0 detections) | FAIL (AprilTag FPs) | BROKEN |

**Root Cause Theory**: The wildlife models were trained on a dataset that included AprilTag labels. The models learned to detect AprilTags as high-confidence "animals", causing:
1. False positives on every clip (AprilTags at fixed positions)
2. Missed actual animals (model confused)

---

## Current System State

### Pipeline Status
- **Ingest**: ONLINE - capturing 60s segments
- **Detection**: RUNNING with base `yolov8n.pt` (healthy on test clips)
- **Disk**: 82% (87GB free)

### Active Models
```
DETECTOR_MODELS: yolov8n.pt
```
- Base COCO model passes health check for deer/person.
- Wildlife v2 models remain BROKEN (kept out of rotation until retrained).

### Monitoring Commands
```bash
# Check pipeline status
ssh mtornga@192.168.68.71 "cd ~/projects/DeerAITrackingResponse && ./scripts/run_pipeline.sh status"

# Run health check
ssh mtornga@192.168.68.71 "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && python scripts/pipeline_health_check.py --verbose"

# Check disk space
ssh mtornga@192.168.68.71 "df -h /"

# View detector logs
ssh mtornga@192.168.68.71 "tail -30 /srv/deer-share/runs/live/logs/detector.log"
```

---

## Known Issues

### 1. Wildlife Models Are Broken (CRITICAL)
**Issue**: Both yolov8n_wildlife_v2.pt and rtdetr_wildlife_v2.pt fail the health check (0 detections on deer, AprilTag FPs on person).

**Impact**: Removed from pipeline; base `yolov8n.pt` temporarily in use.

**Fix Required**: Retrain models WITHOUT AprilTag labels in the training dataset.

### 2. AprilTag Misclassification History
**Issue**: Training data included AprilTag annotations, which corrupted model learning.

**Evidence**: High-confidence (>0.85) "animal" detections at fixed positions matching AprilTag locations:
- Position ~(0.85, 0.83) - right side of frame
- Position ~(0.29, 0.65) - center-left
- Position ~(0.16, 0.45) - left edge

**Prevention**: Always run health check after retraining. The test clips will catch this regression.

### 3. Disk Space Accumulation
**Issue**: Segments accumulate at ~150GB/day when pipeline is running.

**Mitigation Needed**:
- Implement retention policy (72h max)
- Add cron job to clean old segments
- Consider external storage

---

## Next Steps (Priority Order)

### Immediate (Model Fix)
1. **Switch to working model**: DONE. `run_pipeline.sh` default is now `yolov8n.pt` and pipeline restarted with this model.
2. **Retrain wildlife models** WITHOUT AprilTag labels:
   - Remove AprilTag class from dataset
   - Retrain yolov8n and rtdetr
   - Run health check to verify

### Short-term (Quality)
1. **Add more test clips** when CVAT is back online
2. **Tune confidence thresholds** based on health check results
3. **Add disk space monitoring** to prevent future outages

### Medium-term (Automation)
1. **Cron job for health checks** - daily smoke test
2. **Disk retention policy** - auto-delete segments older than 72h
3. **Alert on health check failures**

---

## Code Structure Reference

### New Files This Session
```
test_clips/
├── deer_test.mkv           # Deer test clip (137MB)
├── person_test.mkv         # Person test clip (137MB)
├── deer_test_frame.jpg     # Reference frame
├── person_test_frame.jpg   # Reference frame
├── ground_truth.json       # Expected results
└── README.md               # Documentation

scripts/
└── pipeline_health_check.py  # Health check script
```

### Key Paths
```
# Models
~/projects/DeerAITrackingResponse/models/
├── yolov8n.pt              # Base COCO - WORKING
├── yolov8n_wildlife.pt     # Dec 16 - BROKEN
├── yolov8n_wildlife_v2.pt  # Dec 16 - BROKEN
└── rtdetr_wildlife_v2.pt   # Dec 16 - BROKEN

# Live pipeline data
/srv/deer-share/runs/live/
├── segments/     # Raw captures (~150GB/day)
├── analysis/     # Copies for detector
├── detections/   # Detector JSON output
├── events/       # Promoted events
└── logs/         # Pipeline logs
```

---

## Environment & Credentials

### Ubuntu Server
- **Host**: `192.168.68.71`
- **User**: `mtornga`
- **Repo**: `~/projects/DeerAITrackingResponse`
- **Venv**: `.venv`

### Services
- **CVAT**: `http://192.168.68.71:8080/` (mtornga / TKE4life)
- **Streamlit**: `http://192.168.68.71:8502/`

---

## Session Summary

| Task | Status |
|------|--------|
| Diagnose pipeline outage | DONE - disk full |
| Free disk space | DONE - 90GB freed |
| Create test clips | DONE - deer + person |
| Create health check script | DONE |
| Identify model issues | DONE - AprilTag misclass |
| Fix models | NOT DONE - needs retraining |

**Critical Finding**: The v2 wildlife models are broken and need retraining without AprilTag labels. Use `yolov8n.pt` as temporary fallback.

**New Capability**: Run `python scripts/pipeline_health_check.py --verbose` to verify pipeline health at any time.

---

**Generated**: 2025-12-17 by Claude Code
**Handoff to**: Next development session
