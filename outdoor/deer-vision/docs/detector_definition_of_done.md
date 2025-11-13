## Detector Definition of Done (Steps 11–12)

1. **Metrics**  
   - `mAP@0.5 ≥ 0.80`, `Recall@0.5 ≥ 0.85`.  
   - `FP/min ≤ 0.10` on `daytime_easy`, `≤ 0.25` on `night_hard`.  
   - `p95 latency ≤ 40 ms` on the Mac mini @ 960 px.

2. **Repeatability**  
   - Fresh checkout + `python -m venv .venv && pip install -r constraints.txt`.  
   - `make train && make eval` completes without manual tweaks.  
   - `pytest -q` passes using `tests/test_detector_quality.py`.

3. **Artifacts**  
   - `make export` generates ONNX + TorchScript bundles stored under `models/yolov8n_det_vXX/`.  
   - `models/yolov8n_det_vXX/metrics.json` committed with the evaluation report consumed by pytest + UI.

4. **Data Hygiene**  
   - `data/eval/*` frozen with immutable labels.  
   - `data/datasets/deer_det_vXX` versioned through DVC + pushed to `s3://wildlife-ugv/datasets/`.

5. **Operational Notes**  
   - Night/difficult clips referenced in PR description with observed behavior deltas.  
   - Any new S3 paths, credentials, or calibration data called out for downstream operators.
