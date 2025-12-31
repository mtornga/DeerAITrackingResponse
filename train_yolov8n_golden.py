#!/usr/bin/env python3
"""Train YOLOv8n on wildlife_golden_v3 dataset."""

from ultralytics import YOLO
import shutil

def main():
    print('Loading YOLOv8n base model...')
    model = YOLO('yolov8n.pt')
    
    print('Starting training on wildlife_golden_v3...')
    results = model.train(
        data='outdoor/deer-vision/data/datasets/wildlife_golden_v3/data.yaml',
        epochs=100,
        batch=16,
        imgsz=960,
        patience=20,
        device=0,
        project='runs/train',
        name='yolov8n_golden_v3',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        augment=True,
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
    )
    
    print('Training complete!')
    best_path = 'runs/train/yolov8n_golden_v3/weights/best.pt'
    print(f'Best model saved to: {best_path}')
    
    shutil.copy(best_path, 'models/yolov8n_wildlife_v3.pt')
    print('Copied to models/yolov8n_wildlife_v3.pt')

if __name__ == '__main__':
    main()
