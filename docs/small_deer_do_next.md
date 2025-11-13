Nice—frame 171 looks clean now, and the new summary (FP=6 / FN=164 / TP=1929) tells us what’s left: almost all the misses are the first ~160 frames where the smallest deer is far away in low light. That’s a “tiny, low-SNR object” problem more than a post-proc problem.

Here’s how I’d tackle it, in order of ROI.

0) Lock in your runtime settings (they worked)

    Notes 2025-11-12: visualizer/predict now defaults to `conf=0.35 iou=0.45 agnostic_nms=True max_det=15`; this combo cut FP to 6/300 frames.

Keep for overlays/eval unless noted:

conf=0.35 iou=0.45 agnostic_nms=True max_det=15

1) Fix the dataset signal for tiny deer (fastest lift)

    Done: imported `092708tighter.zip`, QC’d, and pushed to GPU host so every frame has tight deer/apriltag boxes.

Label pass on those first 160 frames. Make sure the tiny deer are actually labeled on every frame (interpolation often stops early). Each added tiny box teaches the model exactly what you want.

Oversample “tiny” frames. Create a hard_examples.txt list (those first 160 + any other far deer) and sample them 2–4× in training. (Ultralytics: duplicate their entries in the train txt.)

Zoom-crop copies. For each tiny-deer image, auto-crop a 1.5–2.0× box around the GT (with some margin) and save as additional training images with scaled labels. This synthetically turns tiny deer into medium deer the model can learn from.

2) Make the target bigger for the model

    TODO: next burst of training should use yolov8s6 @ imgsz 1280 with warm-start from yolov8n_det_v02 weights.

Bump resolution. Train at imgsz=1280 (or 1536 if VRAM allows) and use a P6 model:

yolo train model=yolov8s6.pt data=deer.yaml imgsz=1280 epochs=150 batch=16 \
  lr0=0.005 cos_lr=True close_mosaic=15 mosaic=0.2 mixup=0.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 degrees=5 translate=0.08 scale=0.6 \
  fl_gamma=1.5


yolov8s6 (P6) + 1280 helps small objects substantially vs yolov8n @ 640.

fl_gamma=1.5 (focal loss) nudges learning toward harder (tiny) examples.

Warm-start from your best. Use --model path/to/last_best.pt rather than COCO every time.

3) Inference tricks to catch tiny/far deer (no retrain needed)

    Pending: need to prototype 2×2 tiled inference in visualize_errors.py before wiring into live pipeline.

Tiled inference (2×2 with 15–20% overlap), then merge with NMS. Effective 2× resolution on the far deer without resizing the whole frame. Add this in your visualizer/pipeline:

Split image into tiles with overlap.

Run predict on tiles.

Map tile boxes back to full image coords.

Global NMS (use same conf/iou as above).

Multi-scale predict: run once at 1.0× and once at 1.5× (short side), fuse with NMS. It’s slower than tiles but dead simple.

4) Improve low-light SNR (camera-side)

    Blocked on hardware: waiting for 1 TB flash + IR illuminators; no action yet.

Add one or two IR illuminators aimed toward the far corner those early frames come from; use narrow beam (15°–30°) so the distant ground gets photons without blasting the near foreground.

In Reolink: prefer H.264, raise bitrate, shorten exposure (reduce motion blur), and keep a short GOP so frames are cleaner on reconnect.

5) Evaluation that matches reality

    Open: match-IoU override hooked up (0.35 for tiny). Still need size bucket stats in evaluate.py.

Area-bucket metrics (S/M/L) so you can see the small-object trend move as you try steps 1–3.

Keep match IoU at 0.35–0.4 for tiny GT boxes; 0.5 for medium/large. (Or implement size-aware matching in the QC script.)

6) Small experiment matrix (do these next)

    Next actions lined up: (a) tiled predict experiment, (b) oversampled yolov8s6 run, (c) zoom-crop augmentation script.

Tiled predict (2×2, 20% overlap) on the current weights; report FN on the first 160 frames.

Train yolov8s6 @ 1280 warm-started from your best, with oversampled “tiny” frames.

Zoom-crop augmentation for tiny GT in training (keep originals too).

If still missing far deer, push to imgsz=1536 or try yolov8m6.pt.
