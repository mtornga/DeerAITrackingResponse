from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def load_snow_scoring():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "agents" / "queue-enricher" / "snow_scoring.py"
    spec = importlib.util.spec_from_file_location("snow_scoring", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_meta(hits):
    return {
        "models": {
            "demo": {
                "hits": hits,
            }
        }
    }


def test_metrics_empty_hits():
    snow_scoring = load_snow_scoring()
    metrics = snow_scoring.compute_snow_metrics(make_meta([]))
    assert metrics["small_box_ratio"] == 0.0
    assert metrics["hit_persistence"] == 0.0
    assert metrics["spatial_dispersion"] == 0.0
    assert metrics["burstiness"] == 0.0


def test_metrics_basic_distribution():
    snow_scoring = load_snow_scoring()
    hits = [
        {"bbox": [0.1, 0.1, 0.02, 0.02], "frame": "frame_000001"},
        {"bbox": [0.2, 0.2, 0.02, 0.02], "frame": "frame_000002"},
        {"bbox": [0.3, 0.3, 0.2, 0.2], "frame": "frame_000003"},
        {"bbox": [0.8, 0.8, 0.02, 0.02], "frame": "frame_000010"},
    ]
    metrics = snow_scoring.compute_snow_metrics(make_meta(hits))

    assert metrics["small_box_ratio"] == pytest.approx(0.75)
    assert metrics["hit_persistence"] == pytest.approx(2.0)
    assert metrics["spatial_dispersion"] > 0.0
    assert metrics["burstiness"] >= 0.0


def test_burstiness_multiple_hits_same_frame():
    snow_scoring = load_snow_scoring()
    hits = [
        {"bbox": [0.1, 0.1, 0.02, 0.02], "frame": "frame_000001"},
        {"bbox": [0.12, 0.12, 0.02, 0.02], "frame": "frame_000001"},
        {"bbox": [0.15, 0.15, 0.02, 0.02], "frame": "frame_000001"},
        {"bbox": [0.5, 0.5, 0.02, 0.02], "frame": "frame_000002"},
    ]
    metrics = snow_scoring.compute_snow_metrics(make_meta(hits))
    assert metrics["burstiness"] == pytest.approx(0.5)


def test_score_and_reason():
    snow_scoring = load_snow_scoring()
    hits = [
        {"bbox": [0.1, 0.1, 0.01, 0.01], "frame": "frame_000001"},
        {"bbox": [0.2, 0.2, 0.01, 0.01], "frame": "frame_000010"},
        {"bbox": [0.3, 0.3, 0.01, 0.01], "frame": "frame_000020"},
    ]
    score, metrics, reason = snow_scoring.compute_snow_score(make_meta(hits))
    assert 0.0 <= score <= 1.0
    assert "small-box" in reason or "small" in reason
    assert metrics["small_box_ratio"] >= 0.9


def test_models_list_input():
    snow_scoring = load_snow_scoring()
    meta = {
        "models": [
            {
                "hits": [
                    {"bbox": [0.1, 0.1, 0.02, 0.02], "frame": 1},
                    {"bbox": [0.12, 0.12, 0.02, 0.02], "frame": 2},
                ]
            }
        ]
    }
    metrics = snow_scoring.compute_snow_metrics(meta)
    assert metrics["small_box_ratio"] == pytest.approx(1.0)
    assert metrics["hit_persistence"] == pytest.approx(2.0)


def test_frame_parse_non_numeric():
    snow_scoring = load_snow_scoring()
    hits = [
        {"bbox": [0.1, 0.1, 0.02, 0.02], "frame": "frame_000001"},
        {"bbox": [0.2, 0.2, 0.02, 0.02], "frame": "2"},
        {"bbox": [0.3, 0.3, 0.02, 0.02], "frame": "2.0"},
        {"bbox": [0.4, 0.4, 0.02, 0.02], "frame": "frame_bad"},
    ]
    metrics = snow_scoring.compute_snow_metrics(make_meta(hits))
    assert metrics["hit_persistence"] >= 1.0


def test_missing_models_key_defaults():
    snow_scoring = load_snow_scoring()
    metrics = snow_scoring.compute_snow_metrics({})
    assert metrics["small_box_ratio"] == 0.0
    assert metrics["hit_persistence"] == 0.0
    assert metrics["spatial_dispersion"] == 0.0
    assert metrics["burstiness"] == 0.0


def test_invalid_bbox_ignored():
    snow_scoring = load_snow_scoring()
    hits = [
        {"bbox": [0.1, 0.1, 0.02], "frame": 1},
        {"bbox": "bad", "frame": 2},
        {"bbox": [0.2, 0.2, -0.1, 0.1], "frame": 3},
    ]
    metrics = snow_scoring.compute_snow_metrics(make_meta(hits))
    assert metrics["small_box_ratio"] == 0.0
    assert metrics["spatial_dispersion"] == 0.0
