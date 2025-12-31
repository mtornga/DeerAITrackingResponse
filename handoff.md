# Developer Handoff Notes

**Date**: 2025-12-30 (Afternoon Session)
**Session**: Wildlife v7 Training - Label Fix & 2-Class Models
**Status**: RT-DETR Day training (77/100 epochs), Night queued

---

## What Was Accomplished

### 1. Fixed Critical Label Issue
**Problem**: Person folders were incorrectly labeled as class 0 (deer) instead of class 1.

**Solution**:
- Relabeled all person folders (`person_day`, `person_night`) to class 1
- Purged 260 corrupted apriltag labels from `golden_staging/` (identical bbox at 0.674, 0.117)
- Verified class distribution: deer=0, person=1

### 2. Trained YOLOv8n v7 Models (Complete)
| Model | mAP50 | mAP50-95 | Location |
|-------|-------|----------|----------|
| yolov8n_wildlife_v7_day | ~97% | ~70% | `/srv/deer-share/models/` |
| yolov8n_wildlife_v7_night | ~98% | ~70% | `/srv/deer-share/models/` |

### 3. RT-DETR v7 Training (In Progress)
| Model | Status | Current Metrics |
|-------|--------|-----------------|
| rtdetr_wildlife_v7_day | Epoch 77/100 | mAP50: 98%, mAP50-95: 84.4% |
| rtdetr_wildlife_v7_night | Queued | - |

### 4. Documentation Created
- `docs/wildlife_v7_training.md` - Full training details, data sources, results

## Training Data Summary

| Source | Deer | Person |
|--------|------|--------|
| nestcam_frames | 1,108 | 85 |
| golden_clips_frames | 724 | 2,013 |

## Commands to Complete Training

```bash
# Check RT-DETR progress on RunPod
ssh -p 10468 -i ~/.ssh/id_ed25519 root@213.173.108.200 \
  "grep -oP '\\d+/100' /workspace/train_rtdetr_v7_b2.log | tail -1"

# Transfer RT-DETR models when done (after epoch 100/100)
scp -P 10468 -i ~/.ssh/id_ed25519 \
  root@213.173.108.200:/workspace/runs/detect/rtdetr_v7_day/weights/best.pt \
  mtornga@192.168.68.71:/srv/deer-share/models/rtdetr_wildlife_v7_day.pt

# Start RT-DETR Night after Day completes
ssh -p 10468 -i ~/.ssh/id_ed25519 root@213.173.108.200 \
  "cd /workspace && sed 's/v7_day/v7_night/g' train_rtdetr_v7_b2.py > train_rtdetr_v7_night.py && nohup python train_rtdetr_v7_night.py > train_rtdetr_v7_night.log 2>&1 &"
```

## Key Learnings

1. **Label verification is critical** - Always check class distribution before training
2. **YOLO label format**: `class x_center y_center width height`
3. **RT-DETR memory**: Requires batch=2 on 20GB GPU (A4500)
4. **Corrupted labels**: AprilTag false positives had identical bbox positions

---

**Previous Sessions Below**

---

**Date**: 2025-12-18 (Morning/Afternoon Session)
**Session**: Model Retraining with Balanced Data, Golden Clips Expansion
**Status**: Training v4 models - YOLOv8n running, RT-DETR queued

---

**Date**: 2025-12-19 (Evening Session)
**Session**: Deploy v5.1 models, train RT-DETR v5.1, multi-model bakeoff, walk-by investigation

## What Was Accomplished
- Trained **RT-DETR v5.1** on `outdoor/deer-vision/data/datasets/wildlife_golden_v5/data.yaml` (batch=2, 80 epochs). Results: P 0.953 / R 0.941 / mAP50 0.970 / mAP50-95 0.828. Best saved to `models/rtdetr_wildlife_v5.1.pt`.
- Health check: `python scripts/pipeline_health_check.py --verbose --models rtdetr_wildlife_v5.1.pt` → **HEALTHY** (deer/person clips pass).
- Live pipeline updated to **multi-model**: `DETECTOR_MODELS=yolov8n_wildlife_v5.1.pt,rtdetr_wildlife_v5.1.pt`. Restarted tmux pipeline.
- Cleared backlog and fast-forwarded processing to today’s walk-by window by pruning older segments/detections/events (pre segment IDs < 143000) for 2025-12-19; pipeline now processes current-day clips.

## Findings / Issues
- Walk-by clip **segment_143801.mkv**: RT-DETR v5.1 detected a person (0.927) → routed to review; **YOLOv8n v5.1 missed entirely** (max_conf 0). Re-running offline with both models (including RT-DETR) produced no boxes, so auto-labeling failed.
- Continuation **segment_143904.mkv**: Both models detect person (YOLO max 0.835, RT-DETR 0.938) → review due to confidence < auto-accept threshold.
- Frames extracted from segment_143801 for manual labeling live at `/tmp/segment_frames/frame_*.jpg` on the Ubuntu host.

## Next Steps (recommended)
1. **Label segment_143801**: Use the extracted frames (`/tmp/segment_frames/`) in CVAT (image import to avoid high-res video limits). Mark person(s); add rain artifacts as needed for hard negatives.
2. Consider lowering YOLO live threshold to 0.20 if misses persist, or add light/contrast preprocessing for YOLO-only.
3. If labeling is done, fold into the next training set (feedback/golden) and retrain YOLOv8n to close this gap.

## Key Paths / Commands
- Models: `models/yolov8n_wildlife_v5.1.pt`, `models/rtdetr_wildlife_v5.1.pt`
- Training logs: `/tmp/rtdetr_golden_v5.1_train.log`
- Run outputs: `runs/train/rtdetr_golden_v5.1/weights/{best,last}.pt`
- Pipeline logs: `/srv/deer-share/runs/live/logs/{detector.log,ingest.log}`
- Segment of interest: `/srv/deer-share/runs/live/analysis/2025-12-19/segment_143801.mkv`

---

## What Was Accomplished

### 1. Diagnosed v3 Model Failure
**Problem**: YOLOv8n_wildlife_v3 was labeling deer as "person".

**Root Cause**: Severe class imbalance in training data:
- Deer: 266 annotations
- Person: 1123 annotations (4:1 ratio)

The model learned to call everything "person" due to the heavy bias.

### 2. Expanded Golden Clips Archive
Promoted 45 newly tagged clips to `/srv/deer-share/golden_clips/`:

| Category | Count | Notes |
|----------|-------|-------|
| Buck | 11 | Including 2 bucks, night deer |
| Person | 24 | Family groups, walk-bys |
| Predator | 3 | Fox/coyote clips |
| Other Wildlife | 4 | Skunk, small mammals |
| False Positive | 11 | Hard negatives for training |

**Total golden clips**: 54 (up from 9)

### 3. Freed Disk Space
Purged old runs to recover space:
- Deleted: `/srv/deer-share/runs/live/analysis/` (128GB)
- Deleted: `/srv/deer-share/runs/live/events/2025-12-17/` (14GB)
- Cleaned: `daily_review_index.json` (removed 1274 stale entries)

**Disk status**: 87% used, 62GB free (was 95%, 26GB free)

### 4. Created Balanced wildlife_golden_v4 Dataset
**Strategy**: Extract frames from golden clips with proper class assignment

| Split | Images | Deer Labels | Person Labels |
|-------|--------|-------------|---------------|
| Train | 1135 | 431 | 1225 |
| Val | 200 | 80 | 220 |

**Key improvements**:
- Class ratio now ~1:3 (was 1:4.2)
- Labels assigned by source folder (ground truth), not model prediction
- Added predator/other_wildlife as "deer" class (all animals)
- 500+ hard negative frames from false_positive clips

**Location**: `outdoor/deer-vision/data/datasets/wildlife_golden_v4/`

### 5. Started v4 Training
**YOLOv8n v4** (running):
- Epoch ~5/100 at handoff
- mAP50: 0.96 (already excellent!)
- Log: `/tmp/yolov8n_golden_v4_train.log`
- Output: `models/yolov8n_wildlife_v4.pt`

**RT-DETR v4** (queued):
- Auto-starts after YOLOv8n completes
- Log: `/tmp/rtdetr_golden_v4_train.log`
- Output: `models/rtdetr_wildlife_v4.pt`

---

## Current System State

### Training Status
```bash
# Check YOLOv8n v4 progress
ssh mtornga@192.168.68.71 "grep -oP '\\d+/100' /tmp/yolov8n_golden_v4_train.log | tail -1"
ssh mtornga@192.168.68.71 "tail -5 /tmp/yolov8n_golden_v4_train.log | grep 'all'"

# Check if RT-DETR v4 started (after YOLOv8n finishes)
ssh mtornga@192.168.68.71 "tail -20 /tmp/rtdetr_golden_v4_train.log 2>/dev/null"

# Check processes
ssh mtornga@192.168.68.71 "ps aux | grep -E 'train_yolov8n|train_rtdetr|run_rtdetr' | grep -v grep"
```

### Disk Status
- **62GB free** (87% of 492GB used)
- Golden clips: ~10.5GB
- Events: ~3GB (only 2025-12-18)

### Golden Clips Inventory
```
/srv/deer-share/golden_clips/
├── buck/           (11 clips, 1.4GB)
├── doe/            (empty)
├── person/         (24 clips, 6.6GB)
├── predator/       (3 clips, 279MB)
├── other_wildlife/ (4 clips, 380MB)
├── false_positive/ (11 clips, 1.2GB)
├── mixed/          (587MB)
└── manifest.json
```

---

## Morning Checklist

### 1. Check Training Results
```bash
# See if v4 models exist
ssh mtornga@192.168.68.71 "ls -la ~/projects/DeerAITrackingResponse/models/*_v4.pt 2>/dev/null"

# Check final metrics
ssh mtornga@192.168.68.71 "tail -50 /tmp/yolov8n_golden_v4_train.log | grep -E 'all|Results|best'"
ssh mtornga@192.168.68.71 "tail -50 /tmp/rtdetr_golden_v4_train.log | grep -E 'all|Results|best'"
```

### 2. Run Health Check
```bash
ssh mtornga@192.168.68.71 "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && python scripts/pipeline_health_check.py --verbose --models yolov8n_wildlife_v4.pt"
```

### 3. Deploy if Healthy
Update `scripts/live_detector_multimodel.py` to use v4 models.

---

## Files Changed This Session

### New Files
```
outdoor/deer-vision/data/datasets/wildlife_golden_v4/  # Balanced dataset
train_yolov8n_golden_v4.py                             # v4 training script
train_rtdetr_golden_v4.py                              # v4 RT-DETR script
/srv/deer-share/golden_clips/{predator,other_wildlife,false_positive}/  # New categories
```

### Modified Files
```
/srv/deer-share/golden_clips/manifest.json  # Added 45 new clips
/srv/deer-share/index/daily_review_index.json  # Cleared stale entries
```

---

## Key Learnings

### Class Imbalance Matters
The v3 model failure was a classic case of class imbalance. With 4:1 person:deer ratio, the model learned to predict the majority class.

### Ground Truth from Source
Using the source folder (buck/, person/, etc.) as ground truth for labels worked well. The v3 model detected objects correctly but mislabeled them, so we used its bboxes but assigned correct classes based on the known content.

### Balanced Training Recipe
1. Extract frames from known-class clips
2. Autolabel with existing model (for bbox locations)
3. Override class labels based on source folder
4. Add hard negatives from false_positive clips
5. Aim for ~1:2 or 1:3 class ratio max

---

## Next Steps

### Immediate
1. **Wait for training** - YOLOv8n ~2hrs, RT-DETR ~4hrs after
2. **Health check v4 models** - Should pass deer_test.mkv now
3. **Deploy to live detector** - If healthy

### Short-term
1. **Add more deer clips** - Still need more deer data
2. **Review new captures** - Streamlit app is fast now (empty index)
3. **Fix thumbnail bbox alignment** - Minor visual issue

### Medium-term
1. **Virtuous cycle** - Tag more clips -> retrain -> improve
2. **Species classification** - Separate deer/fox/skunk classes
3. **Night vs day models** - Consider separate models

---

## Session Summary

| Task | Status |
|------|--------|
| Diagnose v3 model failure | DONE - class imbalance |
| Promote 45 tagged clips | DONE |
| Free disk space (~140GB) | DONE |
| Create wildlife_golden_v4 | DONE |
| Train YOLOv8n v4 | RUNNING |
| Queue RT-DETR v4 | QUEUED |
| Clear review index | DONE |

**Expected**: v4 models ready in ~6 hours total.

---

**Generated**: 2025-12-18 ~1:40 PM CST by Claude Code
**Handoff to**: Evening session
