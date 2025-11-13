# Deer Vision Detector Workspace

This directory contains the end-to-end workflow for the outdoor detector pipeline described in `docs/deervision_detector_pipeline_plan.md`. The focus here is on a fast, test-driven loop that gets from raw clips to trained/exported YOLOv8 detectors with reproducible gates.

## Layout

```
deer-vision/
├── data/
│   ├── raw/               # untouched clips + extracted frames
│   ├── interim/           # auto-label drafts, tiles, relabel queues
│   ├── datasets/          # versioned YOLO-format datasets (DVC tracked)
│   └── eval/              # frozen evaluation packs (images + labels)
├── models/                # yolov8n_det_vXX runs + metrics
├── scripts/               # ingestion, labeling, training, evaluation helpers
├── ui/                    # Streamlit review dashboard
├── configs/               # dataset YAML, hyp + eval thresholds
├── tests/                 # pytest gates (quality bars)
└── Makefile               # muscle-memory entrypoints
```

## Plan Progress (Steps 0–12)

- **0 – Tech Choices**: YOLOv8n detectors with Ultralytics CLI, LabelImg for corrections, DVC + S3 for dataset/model versioning, Streamlit UI, pytest gates, `make` orchestration. Config placeholders live under `configs/`.
- **2 – Golden Eval Packs**: Establish `data/eval/{daytime_easy,night_hard,mixed_weather}`. `night_hard` is seeded with the first clip (`outdoor/eval_clips/nighthard1.mp4` → `data/eval/night_hard/clips/`), with empty `images/` + `labels/` ready for frozen frames.
- **3 – Data Ingestion & Auto-Label**: Scripts under `scripts/` (see docstrings) cover frame extraction and auto-labeling using pretrained YOLO weights.
- **4 – Labeling Workflow**: CVAT is the default for outdoor clips (export YOLO → unzip → copy txt files into `data/eval/.../labels/*`), while `LabelImg` remains a fallback for quick edits; run `qc_dataset.py` afterward to freeze the pack.
- **5 – Small-Object Boosters**: `tile_images.py` and `curate_hard_negatives.py` manage tiling + hard-negative staging in `data/interim/`.
- **6 – Training**: `train.py` wraps Ultralytics commands with pinned defaults from `configs/hyp_det.yaml`.
- **7 – Evaluation**: `evaluate.py` computes the metrics/latency JSON enforced by `tests/test_detector_quality.py` using `configs/eval_thresholds.yaml`.
- **8 – Review UI**: Streamlit stub in `ui/app.py` wires up dataset/model selection and status summaries.
- **9 – Retraining Loop**: Documented in `docs/retraining_loop.md` (auto-generated summary) and mirrored as `make` targets.
- **10 – Make Targets**: Top-level `Makefile` in this directory covers `frames`, `autolabel`, `qc`, `split`, `train`, `eval`, `ui`, and `export`.
- **11 – Hard-Mode Enhancements**: Captured as TODO comments inside the relevant scripts for future iteration.
- **12 – Definition of Done**: Stored in `docs/detector_definition_of_done.md` (see `docs/`) and enforced via pytest.

## Usage

1. Activate the repo virtual environment (`python3 -m venv .venv && source .venv/bin/activate && pip install -r constraints.txt`).
2. `cd outdoor/deer-vision`.
3. Run `make frames` / `make autolabel` / `make qc` / `make split` to build the dataset, then `make train` and `make eval`.
4. `make ui` to launch the review dashboard; `make export` to emit ONNX/TorchScript artifacts under `models/`.

See `docs/deervision_detector_pipeline_plan.md` for the authoritative spec and acceptance thresholds.
