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

## RunPod Launcher
Use `scripts/runpod_launch_training.py` to spin up a GPU pod. It can list GPUs,
launch a pod, and optionally rsync a dataset.

Example:
```bash
ssh mtornga@192.168.68.71 \
  "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && \
   python scripts/runpod_launch_training.py --list-gpus"
```

```bash
ssh mtornga@192.168.68.71 \
  "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && \
   python scripts/runpod_launch_training.py \
     --gpu 'NVIDIA RTX 4090' \
     --name deer-train-$(date +%Y%m%d-%H%M%S) \
     --wait"
```

## Outputs
- Datasets: `/srv/deer-share/training_datasets/<run_id>/{day,night}/`
- Models: `/srv/deer-share/models/staging/yolov8n_wildlife_<run_id>_{day|night}.pt`
- Logs: `runs/logs/trainer/TR-YYYYMMDD-HHMMSS.json` and `trainer-latest.json`

## Config
- `configs/training_loop.json` controls defaults for epochs, batch, imgsz, and queue thresholds.
