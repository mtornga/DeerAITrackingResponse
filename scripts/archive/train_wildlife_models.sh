#!/bin/bash
set -e

# Train both YOLOv8n and RT-DETR models on cleaned wildlife dataset
# Run in tmux sessions to monitor progress

REPO_ROOT="$HOME/projects/DeerAITrackingResponse"
VENV="$REPO_ROOT/.venv"
DATASET="$REPO_ROOT/outdoor/deer-vision/data/datasets/wildlife_merged/data.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Wildlife Model Training ==="
echo "Dataset: $DATASET"
echo "Timestamp: $TIMESTAMP"
echo ""

# Activate venv
source "$VENV/bin/activate"

# Training hyperparameters
EPOCHS=100
IMGSZ=960
BATCH=16
DEVICE="cuda:0"

echo "Starting YOLO training in tmux session 'train-yolo'..."
tmux new-session -d -s train-yolo "
cd $REPO_ROOT/outdoor/deer-vision && \
source $VENV/bin/activate && \
python scripts/train.py \
  --data $DATASET \
  --model yolov8n.pt \
  --name yolov8n_wildlife_v2_$TIMESTAMP \
  --project models \
  --epochs $EPOCHS \
  --imgsz $IMGSZ \
  --batch $BATCH \
  --device $DEVICE \
  2>&1 | tee logs/train_yolov8n_wildlife_v2_$TIMESTAMP.log
"

echo "YOLO training started. Attach with: tmux attach -t train-yolo"
echo ""

# Wait a bit to avoid GPU conflicts
sleep 10

echo "Starting RT-DETR training in tmux session 'train-rtdetr'..."
tmux new-session -d -s train-rtdetr "
cd $REPO_ROOT/outdoor/deer-vision && \
source $VENV/bin/activate && \
python scripts/train.py \
  --data $DATASET \
  --model rtdetr-l.pt \
  --name rtdetr_wildlife_v2_$TIMESTAMP \
  --project models \
  --epochs $EPOCHS \
  --imgsz $IMGSZ \
  --batch 8 \
  --device $DEVICE \
  2>&1 | tee logs/train_rtdetr_wildlife_v2_$TIMESTAMP.log
"

echo "RT-DETR training started. Attach with: tmux attach -t train-rtdetr"
echo ""

echo "=== Training Sessions Active ==="
echo "YOLO:    tmux attach -t train-yolo"
echo "RT-DETR: tmux attach -t train-rtdetr"
echo ""
echo "Monitor with: tmux ls"
echo "Logs in: outdoor/deer-vision/logs/"
