Here's what GPT-5 said to do about the fp/fn issues: 

Got it—those overlays tell a pretty clear story:

Lowering the match IoU in the QC tool didn’t change counts → the remaining FPs/FNs are true mismatches (duplicates, class mismatch, or unlabeled deer), not just “tight vs loose box” issues.

The “double-bounded deer (cyan FP stacked on blue TP)” screams NMS/post-proc tuning.

Frames with two TPs and two FNs in the same scene usually point to label gaps (track interpolation ended early) or recall limits on small/far deer.

Here’s a tight plan of attack with concrete knobs to turn—start at the top and work down.

1) Fix label hygiene (highest leverage)

Run a focused FN/FP audit:
Use your summary.json to open only frames still flagged FN/FP and check:

FN: deer present but unlabeled? If yes, extend the CVAT track/interpolation through the tail end.

FP-on-top-of-TP: look for class mismatch (e.g., model says deer, GT says something else) or slightly shifted GT that’s actually a different policy.

Unify your boxing policy: always full body (ears→hooves) even if occluded; no “head-only” boxes. Mixed policies create “duplicate” detections that can’t match a single GT.

Negative frames: include some frames with zero deer (windy leaves, IR glare, empty yard). This reins in spurious FPs.

Balance day/night & distance: add more labeled, small/far deer (like the top of your screenshot). YOLO often misses tiny targets unless they’re well represented.

2) Tune post-processing to kill duplicates (fast win)

You’re seeing a TP + a second overlapping FP on the same animal. Raise the NMS IoU (and optionally lower confidence a hair) during eval/predict:

# Example: higher NMS IoU and class-agnostic merge to suppress near-duplicates
yolo predict model=PATH/TO/best.pt source=VAL_IMAGES \
  conf=0.2 iou=0.75 agnostic_nms=True max_det=50


If duplicates persist, try Soft-NMS or WBF (Weighted Box Fusion) in your visualizer/eval stage. Ultralytics doesn’t ship WBF by default, but it’s ~20 lines to add in your visualize_errors.py for frames flagged with duplicate clusters.

3) Improve recall on small/far deer (training knobs)

Bigger images: tiny deer need pixels. Re-train at imgsz=1280 (or 1536 if VRAM allows).

Model size: yolov8s (or m) is a big bump in recall vs n, with modest compute hit on your 3080.

Warm-start from your last best: you’re currently fine-tuning from COCO each time—resume from your prior best.pt to keep domain features.

# good baseline
yolo train model=yolov8s.pt data=deer.yaml imgsz=1280 epochs=150 batch=16 \
  workers=8 lr0=0.005 cos_lr=True close_mosaic=15 \
  patience=50 mosaic=0.2 mixup=0.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=5 translate=0.08 scale=0.5 shear=0.0 perspective=0.0


Tips:

close_mosaic keeps last N epochs mosaic-free → cleaner boxes on small objects.

Mild color/geo augments help night/day generalization; avoid extreme blur/noise that hides small deer.

If you have heavy class imbalance, set fl_gamma=1.5 (focal loss) to focus on hard examples.

4) Pick the right confidence threshold (don’t eyeball it)

Plot F1 vs confidence (or PR curves) on your val set and adopt the per-class optimal conf for deer (often ~0.25–0.35). Hard-coding 0.5 frequently inflates FN without reducing meaningful FP.

5) Make evaluation reflect your use-case (less “paper”, more “yard”)

Your QC currently uses a single IoU match rule. For small/far deer, an IoU of 0.35–0.4 is reasonable; for large/near deer, 0.5+. Consider size-aware matching in your visualizer: smaller GT boxes accept a slightly lower IoU.

Add area-bucketed metrics (small/medium/large deer) so you can see where recall is failing and target new labels/augments accordingly.

6) Use tracking to de-dupe over time (optional but practical)

At runtime (not for mAP), run a lightweight tracker (ByteTrack/OC-SORT) to merge jittery duplicates and carry detections across a missed frame:

yolo track model=PATH/TO/best.pt source=VIDEO.mp4 tracker=bytetrack.yaml \
  conf=0.25 iou=0.7 agnostic_nms=True


This won’t change ground-truth eval, but it cleans the overlays and your downstream path-history.

7) “Why are those two in the same frame FN while two are TP?”

Common causes + fixes:

Interpolation stopped 2–10 frames early → extend those tracks in CVAT.

Occlusion policy mismatch (e.g., you skipped labeling partially hidden deer) → label them anyway with the visible extent.

Tiny targets at the top/right suffer from down-scaling → see #3 (img size/model) and add more “tiny deer” labels.

8) Quick experiment matrix (2–3 hours total to learn a lot)

Re-eval current weights with iou=0.75 agnostic_nms=True at predict time. Check how many cyan FPs disappear.

Train yolov8s @ imgsz=1280, resume from your best (--model path/to/prev/best.pt). Compare FN on small/far deer.

Label pass: fix 25–50 FN frames you find are unlabeled; re-run QC—FN should drop immediately, proving label gaps were a driver.

Sweep confidence 0.15→0.5 in 0.05 steps, log F1 and FN. Adopt the best conf for yard use.