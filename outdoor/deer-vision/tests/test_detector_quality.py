"""Quality gates for detector exports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

METRICS_PATH = Path("models/yolov8n_det_v01/metrics.json")
THRESHOLDS_PATH = Path("configs/eval_thresholds.yaml")


@pytest.mark.skipif(not METRICS_PATH.exists(), reason="metrics.json not generated yet")
def test_quality_bars() -> None:
    metrics = json.loads(METRICS_PATH.read_text())
    thresholds = yaml.safe_load(THRESHOLDS_PATH.read_text())
    assert metrics["map50"] >= thresholds["map50_min"], "mAP@0.5 below threshold"
    assert metrics["recall"] >= thresholds["recall_min"], "Recall below threshold"
    fp_day = metrics.get("fp_per_min", {}).get("daytime_easy", 0.0)
    fp_night = metrics.get("fp_per_min", {}).get("night_hard", 0.0)
    assert fp_day <= thresholds["fp_per_min_max_day"], "Daytime FP/min too high"
    assert fp_night <= thresholds["fp_per_min_max_night"], "Night FP/min too high"
    assert metrics.get("latency_ms_p95", 0.0) <= thresholds["latency_ms_p95_max"], "Latency exceeds cap"
