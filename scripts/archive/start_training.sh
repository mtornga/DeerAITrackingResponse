#!/bin/bash
cd ~/projects/DeerAITrackingResponse/outdoor/deer-vision
source ../../.venv/bin/activate

mkdir -p logs

# Start YOLO training
echo "Starting YOLO training..."
nohup python scripts/train.py \
  --data data/datasets/wildlife_merged/data.yaml \
  --model yolov8n.pt \
  --name yolov8n_wildlife_v2 \
  --project models \
  --epochs 100 \
  --imgsz 960 \
  --batch 16 \
  --device cuda:0 \
  > logs/train_yolov8n_v2.log 2>&1 &

YOLO_PID=$!
echo "YOLO training started with PID: $YOLO_PID"

# Wait a bit before starting RT-DETR to avoid GPU conflicts
sleep 30

# Start RT-DETR training
echo "Starting RT-DETR training..."
nohup python scripts/train.py \
  --data data/datasets/wildlife_merged/data.yaml \
  --model rtdetr-l.pt \
  --name rtdetr_wildlife_v2 \
  --project models \
  --epochs 100 \
  --imgsz 960 \
  --batch 8 \
  --device cuda:0 \
  > logs/train_rtdetr_v2.log 2>&1 &

RTDETR_PID=$!
echo "RT-DETR training started with PID: $RTDETR_PID"

echo ""
echo "Training processes started:"
echo "  YOLO: PID $YOLO_PID (logs/train_yolov8n_v2.log)"
echo "  RT-DETR: PID $RTDETR_PID (logs/train_rtdetr_v2.log)"
echo ""
echo "Monitor with:"
echo "  tail -f logs/train_yolov8n_v2.log"
echo "  tail -f logs/train_rtdetr_v2.log"
