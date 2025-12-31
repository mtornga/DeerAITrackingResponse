#!/usr/bin/env python3
"""Train RT-DETR on balanced wildlife_golden_v4 dataset."""

from ultralytics import RTDETR
import shutil
from pathlib import Path

def main():
    model = RTDETR('rtdetr-l.pt')
    
    results = model.train(
        data='outdoor/deer-vision/data/datasets/wildlife_golden_v4/data.yaml',
        epochs=80,
        batch=4,  # Small batch to avoid OOM
        imgsz=960,
        device=0,
        project='runs/train',
        name='rtdetr_golden_v4',
        exist_ok=True,
        optimizer='AdamW',
        lr0=0.0001,
        patience=20,
    )
    
    print('Training complete!')
    best_path = Path('runs/train/rtdetr_golden_v4/weights/best.pt')
    print(f'Best model: {best_path}')
    
    shutil.copy(best_path, 'models/rtdetr_wildlife_v4.pt')
    print('Copied to models/rtdetr_wildlife_v4.pt')

if __name__ == '__main__':
    main()
