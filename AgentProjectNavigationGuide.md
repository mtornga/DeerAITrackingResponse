# Agent Project Navigation Guide

## Quick Status
- Live pipeline (Ubuntu host 192.168.68.71) runs two models: `yolov8n_wildlife_v5.1.pt,rtdetr_wildlife_v5.1.pt`.
- RT-DETR v5.1 trained and healthy; YOLOv8n v5.1 already deployed.
- Current logs: `/srv/deer-share/runs/live/logs/detector.log` and `ingest.log`.
- Segments/events: `/srv/deer-share/runs/live/segments|analysis|detections|events/YYYY-MM-DD/`.

## Routine Commands
- Pipeline status: `./scripts/run_pipeline.sh status`
- Restart pipeline: `./scripts/run_pipeline.sh stop` then `./scripts/run_pipeline.sh start`
- Health check: `source .venv/bin/activate && python scripts/pipeline_health_check.py --verbose --models <model>.pt`
- Detector log tail: `tail -f /srv/deer-share/runs/live/logs/detector.log`

## Model Artifacts
- YOLO v5.1: `models/yolov8n_wildlife_v5.1.pt`
- RT-DETR v5.1: `models/rtdetr_wildlife_v5.1.pt`
- Latest RT-DETR training run: `runs/train/rtdetr_golden_v5.1/` (log `/tmp/rtdetr_golden_v5.1_train.log`)

## Data of Interest
- Problem clip (YOLO miss): `/srv/deer-share/runs/live/analysis/2025-12-19/segment_143801.mkv`
- Follow-on clip: `/srv/deer-share/runs/live/analysis/2025-12-19/segment_143904.mkv`
- Extracted frames for 143801 (for CVAT image import): `/tmp/segment_frames/frame_*.jpg` on the Ubuntu host.

## Datasets
- Current training data: `outdoor/deer-vision/data/datasets/wildlife_golden_v5/data.yaml`
- Older balanced set: `wildlife_golden_v4`

## Environment
- Host: `mtornga@192.168.68.71`, repo at `~/projects/DeerAITrackingResponse`
- Virtualenv: `source .venv/bin/activate`
- `.env` includes `DETECTOR_MODELS` and RTSP URLs.

## Next Suggested Tasks
- Label frames from segment_143801 in CVAT (use extracted images to avoid video import size limits) and add to feedback/golden dataset.
- Optionally lower YOLO confidence threshold in pipeline or add minor preprocessing for low-contrast walk-bys.
- Re-run health check after any model swap or threshold change.
