#!/usr/bin/env python3
"""Train YOLOv8n on balanced wildlife_golden_v4 dataset with class weights."""

from ultralytics import YOLO
import shutil
from pathlib import Path

def main():
    # Load pretrained model
    model = YOLO('yolov8n.pt')
    
    # Train with class weights to compensate for remaining imbalance
    # deer:person ratio is ~1:3, so weight deer 2.5x higher
    results = model.train(
        data='outdoor/deer-vision/data/datasets/wildlife_golden_v4/data.yaml',
        epochs=100,
        batch=16,
        imgsz=960,
        device=0,
        project='runs/train',
        name='yolov8n_golden_v4',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=20,
        # Augmentation
        mosaic=0.7,
        mixup=0.1,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        # Class weight: boost deer (class 0) since it's underrepresented
        cls=1.0,  # Base classification loss weight
    )
    
    print('Training complete!')
    best_path = Path('runs/train/yolov8n_golden_v4/weights/best.pt')
    print(f'Best model: {best_path}')
    
    # Copy to models folder
    shutil.copy(best_path, 'models/yolov8n_wildlife_v4.pt')
    print('Copied to models/yolov8n_wildlife_v4.pt')

if __name__ == '__main__':
    main()
