# Wildlife v7 Model Training

**Date**: 2025-12-30
**Purpose**: Train 2-class detection models (deer/person) with corrected labels

---

## Overview

Wildlife v7 addresses label quality issues discovered in previous versions:
- **Problem**: Person folders were incorrectly labeled as class 0 (deer) instead of class 1 (person)
- **Solution**: Relabeled all person folders to class 1, verified class distribution, retrained

## Class Configuration

```yaml
names:
  0: deer
  1: person
```

## Data Sources

### NestCam Frames (`/srv/deer-share/nestcam_frames/`)

| Folder | Class | Count |
|--------|-------|-------|
| deer_day | 0 (deer) | 790 |
| deer_night | 0 (deer) | 318 |
| person_day | 1 (person) | 73 |
| person_night | 1 (person) | 12 |

### Golden Clips Frames (`/srv/deer-share/golden_clips_frames/`)

| Folder | Class | Count |
|--------|-------|-------|
| deer_day | 0 (deer) | 475 |
| deer_night | 0 (deer) | 249 |
| person_day | 1 (person) | 2,013 |

**Note**: Golden staging labels were purged (260 corrupted apriltag false positives at identical bbox positions).

## Training Configuration

### Dataset Splits

| Split | Day Samples | Night Samples |
|-------|-------------|---------------|
| Train | 1,317 (85%) | 343 (85%) |
| Val | 232 (15%) | 60 (15%) |

### Models Trained

| Model | Base | Batch | Status | Location |
|-------|------|-------|--------|----------|
| yolov8n_wildlife_v7_day | yolov8n.pt | 16 | Complete | `/srv/deer-share/models/` |
| yolov8n_wildlife_v7_night | yolov8n.pt | 16 | Complete | `/srv/deer-share/models/` |
| rtdetr_wildlife_v7_day | rtdetr-l.pt | 2 | Training | RunPod |
| rtdetr_wildlife_v7_night | rtdetr-l.pt | 2 | Queued | RunPod |

### Hyperparameters

```python
epochs=100
imgsz=640
hsv_h=0.015, hsv_s=0.4, hsv_v=0.3
degrees=10.0, translate=0.1, scale=0.3, shear=2.0
flipud=0.0, fliplr=0.5, mosaic=0.8, mixup=0.1
lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005
warmup_epochs=3, dropout=0.1
patience=20
```

## Results

### YOLOv8n Models (Complete)

**Day Model**:
- mAP50: ~97%
- mAP50-95: ~70%

**Night Model**:
- mAP50: ~98%
- mAP50-95: ~70%

### RT-DETR Models (In Progress)

**Day Model** (Epoch 75/100):
- mAP50: 97.7%
- mAP50-95: 83.1%
- Precision: 97.2%
- Recall: 95.9%

## Training Infrastructure

- **Platform**: RunPod GPU Cloud
- **GPU**: NVIDIA RTX A4500 (20GB VRAM)
- **RT-DETR Batch Size**: Reduced to 2 due to VRAM constraints

## File Locations

### Models
```
/srv/deer-share/models/
├── yolov8n_wildlife_v7_day.pt    # Complete
├── yolov8n_wildlife_v7_night.pt  # Complete
├── rtdetr_wildlife_v7_day.pt     # Pending (on RunPod)
└── rtdetr_wildlife_v7_night.pt   # Pending (on RunPod)
```

### Training Scripts
```
/workspace/train_v7_2class.py        # YOLOv8n training
/workspace/train_rtdetr_v7_b2.py     # RT-DETR training (batch=2)
```

### Data on RunPod
```
/workspace/
├── nestcam_frames/
│   ├── deer_day/, deer_night/
│   ├── person_day/, person_night/
│   └── labels/
├── golden_clips_frames/
│   ├── deer_day/, deer_night/, person_day/
│   └── labels/
├── v7_day/data.yaml    # Day dataset config
└── v7_night/data.yaml  # Night dataset config
```

## Deployment

After RT-DETR training completes:

```bash
# Transfer RT-DETR models from RunPod
scp -P 10468 -i ~/.ssh/id_ed25519 \
  root@213.173.108.200:/workspace/rtdetr_wildlife_v7_day.pt \
  root@213.173.108.200:/workspace/rtdetr_wildlife_v7_night.pt \
  mtornga@192.168.68.71:/srv/deer-share/models/

# Update live detector config
# Edit scripts/live_detector_multimodel.py or .env:
DETECTOR_MODELS=yolov8n_wildlife_v7_day.pt,rtdetr_wildlife_v7_day.pt
```

## Key Learnings

1. **Label verification is critical**: Always verify class distribution before training
2. **YOLO label format**: `class x_center y_center width height` - class index matters
3. **RT-DETR memory**: Large transformer model requires batch=2 on 20GB GPU
4. **Day/night split**: Separate models for different lighting conditions improves accuracy

---

**Generated**: 2025-12-30 by Claude Code
