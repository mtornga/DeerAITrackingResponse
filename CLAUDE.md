# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a wildlife management platform for tracking and deterring deer and other animals on private property, inspired by the University of Minnesota's GUARD paper. The system implements a full spatial intelligence pipeline: Detection → Tracking → Localization → Behavior Analysis → Trajectory Forecasting.

The project has two parallel environments:
1. **Indoor tabletop simulation**: 24"×30" table with Wyze v2 + Reolink E1 cameras, test objects, and a CuteBot UGV for rapid iteration
2. **Outdoor production**: 4-5 Reolink cameras covering property, tracking real wildlife with coordinate mapping and deterrent control

## Environment Setup

```bash
# Create and activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install base dependencies (OpenCV, PyTorch, Ultralytics)
pip install --no-cache-dir --force-reinstall -r constraints.txt

# Install higher-level tooling (HuggingFace, timm, etc.)
pip install --no-cache-dir -r requirements.txt
```

**Critical**: Always install `constraints.txt` first to lock numpy==1.26.4, opencv-python==4.8.1.78, torch==2.2.2, and ultralytics 8.x before adding other packages.

## Configuration

Copy `.env.example` to `.env` and populate with actual RTSP credentials:

```bash
WYZE_TABLETOP_RTSP="rtsp://user:password@camera-ip/live"
REOLINK_DUO_RTSP="rtsp://user:password@camera-ip/h264Preview_01_main"
DEERVISION_UBUNTU_HOST="192.168.68.71"
DEERVISION_UBUNTU_USER="mtornga"
DEERVISION_UBUNTU_REPO_PATH="~/projects/DeerAITrackingResponse"
```

Scripts automatically load `.env` via `env_loader.py`. Override with CLI flags when needed.

## AWS S3 Access

Project artifacts stored in `s3://wildlife-ugv/` (us-east-2):

```bash
set -a; source .env; set +a
unset AWS_PROFILE AWS_SESSION_TOKEN
aws s3 ls s3://wildlife-ugv/models/yolo/
```

## Shared Storage (Ubuntu Server)

- Samba share: `smb://192.168.68.71/deer-share` (credentials: mtornga/mtornga)
- Server mount: `/srv/deer-share`
- Mac client:
  ```bash
  mkdir -p ~/DeerShare
  mount_smbfs //mtornga@192.168.68.71/deer-share ~/DeerShare
  ```

Used for exchanging captures and detector outputs not suitable for git.

## Common Development Commands

### Indoor Tabletop Work

```bash
# Capture RTSP footage
python scripts/capture_rtsp_clip.py --seconds 10  # or --interval for stills

# Run top-down tracking validation
python demo/topdown_tracker.py --rtsp $WYZE_TABLETOP_RTSP

# Manual calibration workflow
python scripts/calibrate_tabletop.py
```

### Outdoor Detector Pipeline (deer-vision/)

All commands run from `outdoor/deer-vision/`:

```bash
cd outdoor/deer-vision

# Data ingestion & preparation
make frames       # Extract frames from raw clips
make autolabel    # Generate YOLO draft labels
make qc           # Validate dataset integrity
make split        # Create train/val/test splits

# Small object boosters
make tiles        # Generate tiled crops
make negatives    # Curate hard negatives

# Training & evaluation
make train        # Train YOLOv8n with pinned hyperparameters
make eval         # Run metrics + pytest quality gates

# Export & review
make export       # Export ONNX/TorchScript for edge deployment
make ui           # Launch Streamlit review dashboard
```

**Retraining loop**: Drop new MP4s into `data/raw/clips/`, run `make frames autolabel`, correct labels in LabelImg/CVAT, then `make qc split train eval export`.

### Outdoor Tracking Pipeline (MegaDetector + MegaDescriptor)

```bash
# Stage 1: Detect animals with MegaDetector v5
PYTHONPATH=external/MegaDetector:external/ai4eutils:external/yolov5 \
python external/MegaDetector/detection/process_video.py \
  models/md_v5a.0.0.pt outdoor/my_clip.mp4 \
  --output_json_file runs/megadetector/my_clip.json \
  --render_output_video \
  --json_confidence_threshold 0.2 \
  --keep_extracted_frames

# Stage 2: Track with Kalman + appearance embeddings
python scripts/track_megadetector_kf.py \
  --frames-dir tmp/megadetector_frames_my_clip \
  --detections-json runs/megadetector/my_clip.json \
  --class-map 1=deer \
  --conf-threshold 0.2 \
  --max-tracks-per-class 4 \
  --output-json runs/megadetector/my_clip_tracks_kf.json \
  --output-video runs/megadetector/my_clip_tracked_kf.mp4
```

**Descriptor upgrade**: Swap `--descriptor-weights models/MegaDescriptor-L-384/pytorch_model.bin` for better ID stability (requires GPU, 384px crops).

### System Monitoring

Launch tmux dashboard from repo root (Mac):

```bash
# One-time: load SSH keys
ssh-add --apple-use-keychain ~/.ssh/id_ed25519

# Start/attach to dashboard
scripts/deervision_dashboard_tmux.sh
```

Spawns 4-pane layout: Mac status, Ubuntu status, CPU/GPU monitor, log tail.

## High-Level Architecture

### Spatial Intelligence Pipeline

1. **Detection**: MegaDetector v5 (outdoor) or YOLOv8n (custom deer detector)
2. **Tracking**: Kalman filter + MegaDescriptor appearance embeddings for ID persistence
3. **Localization**: AprilTag-based homography to convert pixel→yard coordinates (feet)
4. **Behavior**: Posture/speed/direction classification (future)
5. **Trajectory**: Path forecasting for deterrent routing (future)

### Key Modules

- **`cutebot/`**: Indoor UGV control (BLE serial, automation loop, dashboard)
- **`calibration/`**: Homography matrices (tabletop + outdoor AprilTag-based)
- **`outdoor/deer-vision/`**: End-to-end detector workspace (ingestion, training, evaluation, export)
- **`scripts/`**: One-off utilities (RTSP capture, tracking, MegaDetector processing)
- **`demo/`**: Tabletop validation demos
- **`perception/`**: Reolink GPT snapshot experiments

### Data Flow (Outdoor)

```
Raw clip (RTSP/MP4)
  → MegaDetector detection JSON
  → MegaDescriptor embedding extraction
  → Kalman+Hungarian tracking
  → AprilTag homography projection
  → World coordinates (x_ft, y_ft, range, bearing)
  → Behavior classification / trajectory forecast
  → UGV routing commands
```

### Calibration (Outdoor)

AprilTag IDs 0-5 (tag36h11 family) provide metric ground truth:
- `calibration/outdoor/apriltag_pixels.json`: Detected pixel corners
- `calibration/outdoor/apriltag_homography.json`: 3×3 transform matrix (image→yard feet)
- Validate with `BackyardMe*.jpg` human-distance stills (16 ft, 32 ft, 48 ft reference points)

**Coordinate system**: `(0, 0)` at camera tripod, +X right, +Y forward into yard.

### Testing & Quality Gates

- `outdoor/deer-vision/tests/test_detector_quality.py`: Enforces AP50, recall, latency thresholds from `configs/eval_thresholds.yaml`
- Run after training: `make eval` (auto-executes pytest)
- Frozen eval packs: `data/eval/{daytime_easy,night_hard,mixed_weather}`

## Project Conventions

- **Naming**: PEP 8, snake_case, 4-space indentation
- **Imports**: stdlib → third-party → local, sorted within groups
- **Config**: JSON alongside consumers or `.env` keys via `env_loader.require_env`
- **Exceptions**: Raise descriptive errors, no silent `print()` statements
- **Artifacts**: `runs/`, `logs/`, `tmp_shm/` are throwaway—never commit

## Commit Guidelines

- Present tense, concise subjects: `"iterations on cutebot movement"`, `"chore: ignore .DS_Store"`
- One behavior/dataset tweak per commit
- Mention calibration data sources when changed
- PRs: link issues, include testing commands, attach screenshots/clips for visual changes

## Additional Notes

- **Context7 MCP**: Automatically use Context7 tools for library docs/API lookups when generating code or configuration steps
- **Camera coordinates**: Tabletop origin is bottom-left from Wyze POV (0",0" to 30",24")
- **Model versioning**: DVC tracks `outdoor/deer-vision/data/datasets/`, push to S3 after each version bump
