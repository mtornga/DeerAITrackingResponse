import importlib.util
from pathlib import Path


import sys
def _load_preview_frames_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "agents" / "queue-enricher" / "preview_frames.py"
    spec = importlib.util.spec_from_file_location("preview_frames", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_extract_peak_frame_prefers_timeline():
    preview_frames = _load_preview_frames_module()
    meta = {"detection_timeline": {"peak_frame": 42}, "peak_frame": 7}
    assert preview_frames.extract_peak_frame(meta) == 42


def test_extract_peak_frame_handles_missing():
    preview_frames = _load_preview_frames_module()
    assert preview_frames.extract_peak_frame({}) is None


def test_select_preview_frame_indices_with_total_frames():
    preview_frames = _load_preview_frames_module()
    indices = preview_frames.select_preview_frame_indices(total_frames=300, peak_frame=120)
    assert indices[0] == 120
    assert len(indices) == 3
    assert len(set(indices)) == 3
    assert all(0 <= idx < 300 for idx in indices)


def test_select_preview_frame_indices_small_total_frames():
    preview_frames = _load_preview_frames_module()
    indices = preview_frames.select_preview_frame_indices(total_frames=2, peak_frame=1)
    assert len(indices) == 3
    assert all(0 <= idx < 2 for idx in indices)
