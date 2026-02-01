#!/usr/bin/env python3
"""Incremental fine-tuning from v7 to v8.

Uses low learning rate to avoid catastrophic forgetting.
"""
from ultralytics import YOLO
from pathlib import Path
import shutil

# Paths
DATA_YAML = Path('outdoor/deer-vision/data/datasets/incremental_v8/data.yaml')
MODELS_DIR = Path('/srv/deer-share/models')

# v7 base models
YOLO_V7 = MODELS_DIR / 'yolov8n_wildlife_v7_night.pt'
RTDETR_V7 = MODELS_DIR / 'rtdetr_wildlife_v7_night.pt'

# Output names
YOLO_V8 = 'yolov8n_wildlife_v8_night'
RTDETR_V8 = 'rtdetr_wildlife_v8_night'

def finetune_yolo():
    print('=' * 60)
    print('Fine-tuning YOLOv8 v7 -> v8')
    print('=' * 60)
    
    model = YOLO(str(YOLO_V7))
    
    results = model.train(
        data=str(DATA_YAML),
        epochs=10,
        imgsz=640,
        batch=16,
        lr0=0.001,       # 1/10th normal LR
        lrf=0.1,         # Final LR factor
        warmup_epochs=1,
        patience=5,
        project='runs/finetune',
        name=YOLO_V8,
        exist_ok=True,
        verbose=True,
    )
    
    # Copy best weights to models dir
    best = Path(f'runs/finetune/{YOLO_V8}/weights/best.pt')
    if best.exists():
        dest = MODELS_DIR / f'{YOLO_V8}.pt'
        shutil.copy(best, dest)
        print(f'Saved: {dest}')
    
    return results

def finetune_rtdetr():
    print('=' * 60)
    print('Fine-tuning RT-DETR v7 -> v8')
    print('=' * 60)
    
    model = YOLO(str(RTDETR_V7))
    
    results = model.train(
        data=str(DATA_YAML),
        epochs=10,
        imgsz=640,
        batch=8,         # Smaller batch for RT-DETR memory
        lr0=0.0001,      # Even lower for transformer
        lrf=0.1,
        warmup_epochs=1,
        patience=5,
        project='runs/finetune',
        name=RTDETR_V8,
        exist_ok=True,
        verbose=True,
    )
    
    best = Path(f'runs/finetune/{RTDETR_V8}/weights/best.pt')
    if best.exists():
        dest = MODELS_DIR / f'{RTDETR_V8}.pt'
        shutil.copy(best, dest)
        print(f'Saved: {dest}')
    
    return results

if __name__ == '__main__':
    finetune_yolo()
    print()
    finetune_rtdetr()
    print()
    print('Done! v8 models saved to:', MODELS_DIR)
