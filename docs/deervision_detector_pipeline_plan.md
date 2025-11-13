Wildlife UGV — Detector Pipeline Plan (Test-Driven, Fast, and Repeatable)

Goal: Absolutely nail the detector step (boxes only) on deer (and friends) with a workflow that is:

-Test-driven (clear acceptance criteria + quick evals on fixed clips),

-Fast to resume after a break (one-command scripts + a tiny local UI),

-Label-light (auto-label first, correct second),

-Versioned (datasets + models tracked),

-Edge-ready (exports for realtime inference).

Below is a complete, end-to-end plan you can drop into your repo and hand to your Codex agents to scaffold.

0) Tech Choices (optimized for “no Docker, simple UI”)

-Detector base: yolov8n (boxes only) for speed → graduate to yolov8s if needed.

-Auto-label assist: Run yolov8n-pose/yolov8n OOTB to generate first-pass boxes, then correct.

-Label editor: LabelImg (lightweight, local, no Docker) for bounding boxes.

-Dataset versioning: DVC (remote = S3) so you can branch/merge data like code.

-Evaluation & review UI: Minimal Streamlit dashboard (local) + pytest gates to enforce “quality bars”.

-Orchestration: make + scripts/*.py so it’s muscle-memory to resume.

deer-vision/
├── data/
│   ├── raw/                 # unmodified clips + frames (never edited)
│   │   ├── clips/
│   │   └── frames/
│   ├── interim/             # auto-label outputs, tiling, drafts
│   ├── datasets/
│   │   ├── deer_det_v01/    # YOLO format: images/train|val|test + labels/...
│   │   └── deer_det_v02/
│   └── eval/                # frozen eval sets (never touched)
│       ├── daytime_easy/
│       ├── night_hard/
│       └── mixed_weather/
├── models/
│   ├── yolov8n_det_v01/
│   └── yolov8n_det_v02/
├── scripts/
│   ├── extract_frames.py
│   ├── auto_label.py
│   ├── curate_hard_negatives.py
│   ├── split_dataset.py
│   ├── tile_images.py
│   ├── train.py             # wraps Ultralytics train with locked hyperparams
│   ├── evaluate.py          # metrics + PR curves + per-clip FP/min
│   ├── export_edge.py       # export to onnx/torchscript/ncnn
│   ├── visualize_errors.py  # draws FN/FP panels
│   └── qc_dataset.py        # sanity checks on labels
├── ui/
│   └── app.py               # Streamlit: pick dataset, train, evaluate, review
├── configs/
│   ├── data_deer.yaml       # YOLO dataset YAML (paths auto-generated)
│   ├── hyp_det.yaml         # training hyperparams
│   └── eval_thresholds.yaml # acceptance bars
├── tests/
│   └── test_detector_quality.py  # pytest gates (fail CI if below bar)
├── Makefile
├── dvc.yaml                 # DVC pipeline (frames→labels→train→eval)
└── README.md

S3 (DVC remote):

- s3://wildlife-ugv/raw/ → mirrors data/raw/

- s3://wildlife-ugv/datasets/ → mirrors data/datasets/

- s3://wildlife-ugv/models/ → mirrors models/

2) Golden Evals (freeze these first)

Create 3–5 small “frozen” eval packs (~100–300 frames each), reflecting reality:

- daytime_easy: close-range deer, high contrast.

- night_hard: IR, small deer, partial occlusion.

- mixed_weather: rain/fog, glare.

- (Optional) multi_deer_overlap: crossings, occlusions.

- (Optional) non_targets: dogs, people, skunks, raccoons (to quantify FP).

Label these once with LabelImg (YOLO txt). Store under data/eval/<pack>/images|labels.
Never change these. They’re your unit tests.

3) Data Ingestion & Auto-Label

Frame extraction: (every N frames, plus motion-triggered sampling)

- scripts/extract_frames.py

    - Inputs: clips in data/raw/clips/

    - Outputs: frames to data/raw/frames/SEGMENT_ID/frame_XXXXX.jpg

    - Options: --stride 5, --max-per-clip 300, --keep-metadata

Auto-label pass (proposal generation):

- scripts/auto_label.py

    - Run pretrained yolov8n (or yolov8n-pose) on frames

    - Output YOLO format label drafts to data/interim/autolabel/

    - Confidence threshold low (0.15–0.25) to favor recall

You’ll then correct these in LabelImg (fast).

4) Labeling Workflow (fast, no Docker)

- Launch LabelImg: labelImg data/interim/autolabel/images data/interim/autolabel/labels

- Fix boxes; ensure class set is tight (e.g., deer, unknown_mammal for early stages).

- Run scripts/qc_dataset.py to:

    - Check class names (no typos), empty/missing labels, off-image coords, near-zero width/height, duplicates.

When a batch is cleaned, run:

- scripts/split_dataset.py --src data/interim/autolabel --dst data/datasets/deer_det_v01 --val 0.15 --test 0.15 --stratify class

- This writes a dataset version in YOLO format and a configs/data_deer.yaml.

Version it: dvc add data/datasets/deer_det_v01 && dvc push

5) Small-Object Boosters (before training)

- Tiling (scripts/tile_images.py): split 1920×1080 into 2×2 or 3×3 tiles for night/small-deer; update labels accordingly.

- Hard negatives (scripts/curate_hard_negatives.py): gather empty frames that fooled the detector → labeled as background (or no label files).

- Augmentations tuned for night (configs/hyp_det.yaml):

    - low HSV jitter,

    - slight motion blur,

    - brightness/contrast,

    - random perspective ≤ 0.0005 (tiny),

    - copy-paste off initially for realism.

6) Training (pinned hyperparams, reproducible)

configs/hyp_det.yaml (starter):

lr0: 0.01
lrf: 0.05
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
warmup_bias_lr: 0.05
box: 7.5
cls: 0.5
hsv_h: 0.005
hsv_s: 0.5
hsv_v: 0.3
degrees: 0.0
translate: 0.02
scale: 0.2
shear: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 0.7
mixup: 0.05
copy_paste: 0.0

Command (wrapped by scripts/train.py):

yolo detect train \
  model=yolov8n.pt \
  data=configs/data_deer.yaml \
  imgsz=960 \
  epochs=80 \
  batch=16 \
  lr0=0.01 \
  patience=20 \
  hyp=configs/hyp_det.yaml \
  project=models \
  name=yolov8n_det_v01 \
  pretrained=True \
  workers=8

- imgsz: start at 960; move to 1280 if small-object recall is low.

- mAP/Recall target: enforced via tests (below).

7) Evaluation (test-driven gates)

Acceptance thresholds (configs/eval_thresholds.yaml):

map50_min: 0.80
recall_min: 0.85
fp_per_min_max_day: 0.10
fp_per_min_max_night: 0.25
latency_ms_p95_max: 40


scripts/evaluate.py should:

- Evaluate on frozen packs in data/eval/*.

- Report: mAP@.5, Recall@.5, PR curves, confusion matrix.

- Compute FP/min by pack (day/night separately).

- Log inference latency (p95) on your Mac mini.

Pytest gate (tests/test_detector_quality.py):

def test_quality_bars():
    metrics = json.load(open("models/yolov8n_det_v01/metrics.json"))
    assert metrics["map50"] >= cfg["map50_min"]
    assert metrics["recall"] >= cfg["recall_min"]
    assert metrics["fp_per_min"]["daytime_easy"] <= cfg["fp_per_min_max_day"]
    assert metrics["fp_per_min"]["night_hard"]   <= cfg["fp_per_min_max_night"]
    assert metrics["latency_ms_p95"] <= cfg["latency_ms_p95_max"]


Run anytime:

make eval   # runs evaluate.py + pytest


If it fails, the model does not ship.

8) Review UI (1-file Streamlit)

ui/app.py provides:

- Dataset picker (any data/datasets/*),

- One-click train/eval buttons,

- Metric dashboards (pack-wise, class-wise),

- Error galleries: false negatives first (by pack), then false positives,

- “Send to relabel” button → copies errored frames to data/interim/to_fix/.

Launch:

make ui  # streamlit run ui/app.py

9) Retraining Loop (weekly flow)

1. Collect new clips, make frames to sample → data/raw/frames/.

2. Auto-label (make autolabel), then correct in LabelImg.

3. QC dataset (make qc).

4. Split into new dataset version (deer_det_v02), dvc add && dvc push.

5. Train (make train), Eval (make eval).

6. If tests pass, export edge (make export) to ONNX/TorchScript for your realtime node.

7. Tag release (models/yolov8n_det_v02/), update symlink models/current/.

Everything is a make target so it’s one-liners to resume.

10) Make Targets (muscle memory)
.PHONY: frames autolabel qc split train eval ui export

frames:          ## Extract frames from new clips
	python scripts/extract_frames.py --src data/raw/clips --dst data/raw/frames --stride 5

autolabel:       ## Run OOTB YOLO to propose boxes
	python scripts/auto_label.py --src data/raw/frames --out data/interim/autolabel --conf 0.2

qc:              ## Validate labels
	python scripts/qc_dataset.py --path data/interim/autolabel

split:           ## Create dataset version
	python scripts/split_dataset.py --src data/interim/autolabel --dst data/datasets/deer_det_v$V --val 0.15 --test 0.15

train:           ## Train detector (uses configs/hyp_det.yaml)
	python scripts/train.py --data configs/data_deer.yaml --name yolov8n_det_v$V

eval:            ## Evaluate on frozen packs + pytest gates
	python scripts/evaluate.py --model models/yolov8n_det_v$V --eval-packs data/eval
	pytest -q

ui:
	streamlit run ui/app.py

export:
	python scripts/export_edge.py --model models/yolov8n_det_v$V --formats onnx torchscript

    
11) Hard-Mode Enhancements (when needed)

- Tiling-on-the-fly during eval to boost night-recall on tiny deer.

- Curriculum training: start at imgsz=1280 for 20 epochs → drop to 960 for speed.

- Class balancing: oversample rare night frames.

- Error mining: automatically promote top-K false negatives to to_fix/ each run.

- Confidence calibration: Platt scaling/temperature to stabilize thresholds across packs.

12) Definition of “Done” (Detector)

- On frozen eval sets:

    - mAP@.5 ≥ 0.80, Recall@.5 ≥ 0.85.

    - FP/min ≤ 0.10 (day) and ≤ 0.25 (night).

    - p95 latency ≤ 40 ms on your Mac mini at 960 px.

- Reproducible: make train && make eval passes from a clean checkout.

- Usable: make export produces an artifact that your realtime node consumes.

13) Next Steps After Detector (preview)

- Add ByteTrack for IDs, log (track_id, x, y, t) → SQLite.

- Build path replays and occupancy heatmaps from the logged tracks.

- Introduce posture classifier (grazing/alert/running) from detector crops.

- Begin short-term forecasting (Kalman) → graduate to sequence models once paths accumulate.