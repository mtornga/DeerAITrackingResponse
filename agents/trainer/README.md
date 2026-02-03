# Trainer Agent

Builds training datasets from `golden_clips` and trains YOLOv8n day/night models.
Outputs candidate models to `/srv/deer-share/models/staging/` and writes run logs.

Primary runtime: ubuntubasement.

## Usage (ubuntubasement)

Dry run:
```bash
ssh mtornga@192.168.68.71 \
  "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && \
   python agents/trainer/run.py --dry-run"
```

Train:
```bash
ssh mtornga@192.168.68.71 \
  "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && \
   python agents/trainer/run.py --execute"
```

## Outputs
- Datasets: `/srv/deer-share/training_datasets/<run_id>/{day,night}/`
- Models: `/srv/deer-share/models/staging/yolov8n_wildlife_<run_id>_{day|night}.pt`
- Logs: `runs/logs/trainer/TR-YYYYMMDD-HHMMSS.json` and `trainer-latest.json`

## Config
- `configs/training_loop.json` controls defaults for epochs, batch, imgsz, and queue thresholds.

