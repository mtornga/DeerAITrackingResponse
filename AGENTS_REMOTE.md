# AGENTS_REMOTE.md

Guidelines for autonomous agents executing tasks on the Ubuntu GPU server.

## Connection Details

- **Host**: `192.168.68.71` (or `$DEERVISION_UBUNTU_HOST`)
- **User**: `mtornga` (or `$DEERVISION_UBUNTU_USER`)
- **Auth**: SSH key-based (no password prompts)
- **Repo path**: `/home/mtornga/projects/DeerAITrackingResponse`

## Environment Activation

Always activate the project virtualenv before running Python:

```bash
cd ~/projects/DeerAITrackingResponse
source .venv/bin/activate
```

The `.venv` contains:
- `torch==2.2.2+cu121` (CUDA 12.1 support)
- `opencv-python==4.8.1.78`
- `ultralytics` (YOLOv8)
- Full stack from `constraints.txt` + `requirements.txt`

**Important**: Do NOT use the legacy `~/.local/share/mamba/envs/deer` environment. It is deprecated and will be removed.

## Remote Execution Patterns

### From Mac: Trigger a Remote Job

Use `scripts/run_remote_inference.sh` as the template:

```bash
# Quick GPU smoke test
scripts/run_remote_inference.sh

# Custom command
ssh mtornga@192.168.68.71 "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && python scripts/your_script.py"
```

### Long-Running Jobs

For jobs that may take hours (training, bulk inference):

```bash
# Use nohup to persist after SSH disconnect
ssh mtornga@192.168.68.71 "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && nohup python scripts/train.py > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &"

# Or use tmux for interactive monitoring
ssh -t mtornga@192.168.68.71 "tmux new-session -s training 'cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && python scripts/train.py'"
```

## Key Paths

| Purpose | Path |
|---------|------|
| Project root | `~/projects/DeerAITrackingResponse` |
| Python env | `~/projects/DeerAITrackingResponse/.venv` |
| External tools | `~/projects/DeerAITrackingResponse/external/` |
| Shared storage | `/srv/deer-share` |
| Live runs | `/srv/deer-share/runs/live/` |
| Daily review index | `/srv/deer-share/index/daily_review_index.json` |
| Logs | `~/projects/DeerAITrackingResponse/logs/` |

## External Tools (MegaDetector Stack)

Located in `external/` (not tracked by git):

```bash
external/
├── MegaDetector/   # microsoft/CameraTraps
├── ai4eutils/      # microsoft/ai4eutils
└── yolov5/         # ultralytics/yolov5
```

When running MegaDetector scripts, set PYTHONPATH:

```bash
PYTHONPATH=external/MegaDetector:external/ai4eutils:external/yolov5 python scripts/run_md_on_segment.py ...
```

## Common Agent Tasks

### 1. Run MegaDetector on a Clip

```bash
ssh mtornga@192.168.68.71 << 'EOF'
cd ~/projects/DeerAITrackingResponse
source .venv/bin/activate
PYTHONPATH=external/MegaDetector:external/ai4eutils:external/yolov5 \
python scripts/run_md_on_segment.py \
  --segment /srv/deer-share/runs/live/analysis/segment_20251201_080000.mp4 \
  --output-dir /srv/deer-share/runs/live/detections/
EOF
```

### 2. Check GPU Status

```bash
ssh mtornga@192.168.68.71 "nvidia-smi"
```

### 3. Verify Environment

```bash
ssh mtornga@192.168.68.71 "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && python scripts/hello_gpu.py"
```

### 4. Sync Code from GitHub

```bash
ssh mtornga@192.168.68.71 "cd ~/projects/DeerAITrackingResponse && git pull"
```

### 5. Check Disk Space

```bash
ssh mtornga@192.168.68.71 "df -h / /srv/deer-share"
```

## Safety Constraints

**DO**:
- Always activate `.venv` before running Python
- Write outputs to `/srv/deer-share/` or `logs/` directories
- Use `nohup` or `tmux` for long jobs
- Check disk space before large operations

**DO NOT**:
- Modify files in `external/` (treat as read-only clones)
- Run `rm -rf` on shared directories without explicit user approval
- Install packages system-wide (`pip install --user` or in venv only)
- Start multiple GPU-intensive jobs simultaneously (single RTX 3080)
- Store large datasets under `~/` (use `/srv/deer-share` instead)

## Disk Space Management

Root disk is constrained (~93% used). Keep heavy data on the Samba share:

```bash
# Good: write to shared storage
/srv/deer-share/runs/
/srv/deer-share/clips/
/srv/deer-share/exports/

# Avoid: filling root disk
~/projects/*/data/
~/projects/*/runs/
```

To reclaim space, the legacy mamba env can be removed (after confirming nothing depends on it):

```bash
# ~6.5GB reclaimable
rm -rf ~/.local/share/mamba/envs/deer
```

## Monitoring

The tmux dashboard (run from Mac) provides live status:

```bash
# From Mac repo root
scripts/deervision_dashboard_tmux.sh
```

Panes show: Mac status, Ubuntu status, htop, log tails.
