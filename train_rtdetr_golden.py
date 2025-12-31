#!/usr/bin/env python3
"""Train RT-DETR on wildlife_golden_v3 dataset.

Uses batch=4 to avoid OOM on RTX 3080 (10GB VRAM).
"""

from ultralytics import RTDETR
import shutil

def main():
    print('Loading RT-DETR-L base model...')
    model = RTDETR('rtdetr-l.pt')
    
    print('Starting training on wildlife_golden_v3 (batch=4 for OOM avoidance)...')
    results = model.train(
        data='outdoor/deer-vision/data/datasets/wildlife_golden_v3/data.yaml',
        epochs=80,
        batch=4,
        imgsz=960,
        patience=15,
        device=0,
        project='runs/train',
        name='rtdetr_golden_v3',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.0001,
        lrf=0.01,
        warmup_epochs=5,
    )
    
    print('Training complete!')
    best_path = 'runs/train/rtdetr_golden_v3/weights/best.pt'
    print(f'Best model saved to: {best_path}')
    
    shutil.copy(best_path, 'models/rtdetr_wildlife_v3.pt')
    print('Copied to models/rtdetr_wildlife_v3.pt')

if __name__ == '__main__':
    main()
