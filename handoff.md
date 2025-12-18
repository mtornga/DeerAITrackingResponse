# Developer Handoff Notes
**Date**: 2025-12-18 (Late Night Session)
**Session**: Golden Clips, Model Training, Thumbnail UI
**Status**: Training in Progress, New Models ~Morning

## What Was Accomplished

### 1. Golden Clips Archive Created
**Problem**: Valuable buck clips and reviewed clips at risk of pruning.

**Solution**:
- Created `/srv/deer-share/golden_clips/` protected archive
- Preserved 3 buck clips from 12/17 evening (~340MB)
- Promoted 30 tagged clips from daily review (6 animal, 20 person, 5 false_positive)
- Added safety guard to `prune_events.py` to never touch golden_clips
- Created `scripts/promote_to_golden.py` helper utility

**Structure**:
```
/srv/deer-share/golden_clips/
├── buck/           # High-value deer clips
├── doe/            # Female deer clips
├── person/         # Person detection clips
├── mixed/          # Multiple subjects
├── manifest.json   # Metadata for all clips
└── README.md       # Usage documentation
```

### 2. Training Dataset Prepared (wildlife_golden_v3)
**Problem**: Previous wildlife models broken due to AprilTag labels.

**Solution**: Created clean dataset from golden clips:
- Base: `wildlife_cleaned` (1063 train, 187 val)
- Added: 180 golden animal frames, 600 person frames, 150 hard negatives
- **Final**: 1807 train, 373 val images
- Classes: `0: deer`, `1: person` (fixed non-consecutive index bug)

**Location**: `outdoor/deer-vision/data/datasets/wildlife_golden_v3/`

### 3. Model Training Started
**YOLOv8n** (running):
- Epoch ~52/100 at handoff
- mAP50: ~0.93, mAP50-95: ~0.77
- Config: batch=16, imgsz=960, 100 epochs
- Output: `models/yolov8n_wildlife_v3.pt`
- Log: `/tmp/yolov8n_golden_train.log`

**RT-DETR** (queued):
- Auto-starts after YOLOv8n completes
- Config: batch=4 (to avoid OOM), 80 epochs
- Output: `models/rtdetr_wildlife_v3.pt`
- Log: `/tmp/rtdetr_golden_train.log`

### 4. Thumbnail System for Review UI
**Problem**: Hard to quickly assess clips without watching full video.

**Solution**: Created thumbnail generation with bounding box overlay:
- Extracts frame with highest confidence detection
- Draws bbox with category label and confidence
- Saves as 640x360 JPEG (~70KB each)
- 92 thumbnails pre-generated for existing events

**New Files**:
- `scripts/generate_event_thumbnail.py` - Batch thumbnail generator
- Updated `scripts/daily_review_app.py` - Displays thumbnails in clip list

**Usage**:
```bash
# Generate thumbnails for all events
python scripts/generate_event_thumbnail.py --batch

# Single event
python scripts/generate_event_thumbnail.py --event-dir /path/to/event
```

---

## Current System State

### Training Status
```bash
# Check YOLOv8n progress
ssh mtornga@192.168.68.71 "grep -oP '\\d+/100' /tmp/yolov8n_golden_train.log | tail -1"

# Check RT-DETR progress (after YOLOv8n finishes)
ssh mtornga@192.168.68.71 "tail -20 /tmp/rtdetr_golden_train.log"

# Check if training processes running
ssh mtornga@192.168.68.71 "ps aux | grep -E 'train_yolov8n|train_rtdetr' | grep -v grep"
```

### Disk Status
- **85GB free** (82% of 492GB used)
- Events: 14GB
- Golden clips: 4.5GB
- Healthy headroom for training

### Pipeline Status
- **Ingest**: ONLINE
- **Detection**: Running with base `yolov8n.pt`
- **Review UI**: http://localhost:8501/ with thumbnails

---

## Files Changed This Session

### New Files
```
scripts/
├── generate_event_thumbnail.py   # Thumbnail generation utility
├── promote_to_golden.py          # Promote clips to golden archive
├── extract_feedback_frames.py    # Extract frames for training

outdoor/deer-vision/data/datasets/
└── wildlife_golden_v3/           # New training dataset
    ├── data.yaml
    ├── images/{train,val}/
    └── labels/{train,val}/

/srv/deer-share/golden_clips/     # Protected clip archive (on server)
```

### Modified Files
```
scripts/daily_review_app.py       # Added thumbnail display
scripts/prune_events.py           # Added golden_clips protection
```

### Training Scripts (on server)
```
~/projects/DeerAITrackingResponse/
├── train_yolov8n_golden.py       # YOLOv8n training script
└── train_rtdetr_golden.py        # RT-DETR training script
```

---

## Morning Checklist

### 1. Check Training Results
```bash
# See if training completed
ssh mtornga@192.168.68.71 "ls -la ~/projects/DeerAITrackingResponse/models/*_v3.pt"

# Check final metrics
ssh mtornga@192.168.68.71 "tail -50 /tmp/yolov8n_golden_train.log | grep -E 'all|Results'"
ssh mtornga@192.168.68.71 "tail -50 /tmp/rtdetr_golden_train.log | grep -E 'all|Results'"
```

### 2. Run Health Check
```bash
ssh mtornga@192.168.68.71 "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && python scripts/pipeline_health_check.py --verbose --models yolov8n_wildlife_v3.pt,rtdetr_wildlife_v3.pt"
```

### 3. Deploy New Models (if healthy)
```bash
# Update detector to use new models
# Edit scripts/live_detector_multimodel.py MODEL_CONFIGS
```

---

## Known Issues

### 1. Thumbnail Bbox Slightly Offset
**Status**: Minor visual issue, not blocking.
**Cause**: Bbox coordinates may be computed at different resolution than video frame.
**Impact**: Boxes visible but not pixel-perfect aligned.

### 2. Mac Review App Missing cv2
**Status**: Workaround in place.
**Impact**: Can't generate thumbnails on-the-fly on Mac, but pre-generated ones display fine.

---

## Next Steps (Priority Order)

### Immediate (Morning)
1. **Verify training completed** - Check for v3 models in models/
2. **Run health check** - Validate new models work
3. **Deploy if passing** - Update detector config

### Short-term
1. **Fix thumbnail bbox alignment** - Match resolution to detection
2. **Add model bakeoff metrics** - Side-by-side comparison in UI
3. **Commit and push** - Get changes to main

### Medium-term
1. **Continuous training loop** - Auto-retrain from reviewed clips
2. **Add more golden clips** - Build eval set over time
3. **RT-DETR vs YOLOv8 analysis** - Compare on real detections

---

## Session Summary

| Task | Status |
|------|--------|
| Create golden_clips archive | DONE |
| Promote 30 tagged clips | DONE |
| Prepare wildlife_golden_v3 dataset | DONE |
| Start YOLOv8n training | RUNNING |
| Queue RT-DETR training | QUEUED |
| Create thumbnail generator | DONE |
| Add thumbnails to review UI | DONE |
| Generate 92 thumbnails | DONE |

**Expected by Morning**: Two new trained models ready for health check.

---

**Generated**: 2025-12-18 ~3:40 AM CST by Claude Code
**Handoff to**: Morning session
