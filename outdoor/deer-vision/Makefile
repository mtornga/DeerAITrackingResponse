PYTHON ?= python
VENV ?= .venv
CONF ?= 0.4

.PHONY: frames autolabel qc split tiles negatives train eval export ui clean

frames:
	$(PYTHON) scripts/extract_frames.py --src data/raw/clips --dst data/raw/frames --stride 5

autolabel:
	$(PYTHON) scripts/auto_label.py --src data/raw/frames --out data/interim/autolabel --conf 0.2

qc:
	$(PYTHON) scripts/qc_dataset.py --labels data/interim/autolabel/labels --images data/interim/autolabel/images

split:
	$(PYTHON) scripts/split_dataset.py --src data/interim/autolabel --dst data/datasets/deer_det_v01

tiles:
	$(PYTHON) scripts/tile_images.py --src data/interim/autolabel/images --labels data/interim/autolabel/labels

negatives:
	$(PYTHON) scripts/curate_hard_negatives.py --src data/interim/autolabel/images --labels data/interim/autolabel/labels

train:
	$(PYTHON) scripts/train.py --data configs/data_deer.yaml --name yolov8n_det_v01

eval:
	$(PYTHON) scripts/evaluate.py --model models/yolov8n_det_v01/weights/best.pt --eval-packs data/eval --output models/yolov8n_det_v01/metrics.json --conf $(CONF)
	pytest -q tests/test_detector_quality.py

export:
	$(PYTHON) scripts/export_edge.py --model models/yolov8n_det_v01/weights/best.pt

ui:
	streamlit run ui/app.py

clean:
	rm -rf data/interim/autolabel data/interim/tiles data/interim/hard_negatives
