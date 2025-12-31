#!/usr/bin/env python3
import argparse
import cv2
import shutil
import os
from pathlib import Path
from ultralytics import YOLO

def extract_and_label(video_path, output_img_dir, output_lbl_dir, model, category_id=0, stride=30):
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Could not open video {video_path}")
        return 0
        
    frame_idx = 0
    extracted_count = 0
    video_name = video_path.stem
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % stride == 0:
            # Use v4 model to get initial bounding boxes
            results = model(frame, verbose=False)
            
            labels = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    # Normalize xywh
                    xywh = box.xywhn[0].tolist()
                    conf = float(box.conf[0])
                    
                    if conf > 0.3: 
                        # Force label to 0 (deer) for known deer clips
                        labels.append(f"{category_id} {' '.join(map(str, xywh))}")
            
            if labels:
                img_name = f"{video_name}_frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(output_img_dir / img_name), frame)
                lbl_name = f"{video_name}_frame_{frame_idx:04d}.txt"
                (output_lbl_dir / lbl_name).write_text('\n'.join(labels))
                extracted_count += 1
                
        frame_idx += 1
    
    cap.release()
    return extracted_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/yolov8n_wildlife_v4.pt')
    parser.add_argument('--output-dir', default='outdoor/deer-vision/data/datasets/wildlife_golden_v5')
    parser.add_argument('--v4-dir', default='outdoor/deer-vision/data/datasets/wildlife_golden_v4')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).resolve()
    v4_dir = Path(args.v4_dir).resolve()
    
    # Create structure
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    print(f"Copying v4 dataset base from {v4_dir}...")
    for split in ['train', 'val']:
        if not (v4_dir / 'images' / split).exists():
            continue
        for f in (v4_dir / 'images' / split).glob('*.jpg'):
            shutil.copy2(f, output_dir / 'images' / split / f.name)
        for f in (v4_dir / 'labels' / split).glob('*.txt'):
            shutil.copy2(f, output_dir / 'labels' / split / f.name)
            
    model = YOLO(args.model)
    
    # Clips are relative to repo root
    root = Path(__file__).resolve().parent.parent
    new_clips = []
    
    # Attempt to find deer_test.mkv
    deer_test = root / "test_clips/deer_test.mkv"
    if deer_test.exists():
        new_clips.append(deer_test)
    
    # Nest clips
    nest_dir = root / "nest_clips"
    if nest_dir.exists():
        new_clips.extend(list(nest_dir.glob("*.mp4")))
    
    print(f"Extracting frames from {len(new_clips)} daytime clips...")
    
    total_new = 0
    for i, clip_path in enumerate(new_clips):
        print(f"Processing {clip_path.name}...")
        split = 'val' if i % 5 == 0 else 'train'
        count = extract_and_label(
            clip_path, 
            output_dir / 'images' / split, 
            output_dir / 'labels' / split, 
            model,
            stride=15
        )
        total_new += count
        print(f"  Extracted {count} frames to {split}")
        
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(f"""
path: {output_dir}
train: images/train
val: images/val

names:
  0: deer
  1: person
""")
    
    print(f"Done! Total new daytime frames added: {total_new}")
    print(f"V5 dataset ready at {output_dir}")

if __name__ == '__main__':
    main()
