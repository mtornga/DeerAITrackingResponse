import json
import os
import time
from importlib import util
from pathlib import Path


import sys
def _load_idempotency_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "agents" / "queue-enricher" / "idempotency.py"
    spec = util.spec_from_file_location("idempotency", module_path)
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_has_enrichment():
    mod = _load_idempotency_module()
    assert mod.has_enrichment({"enrichment": {"enriched_at": "2026-02-01T00:00:00Z"}})
    assert not mod.has_enrichment({"enrichment": {}})
    assert not mod.has_enrichment({})


def test_preview_frames_exist(tmp_path: Path):
    mod = _load_idempotency_module()
    assert not mod.preview_frames_exist(tmp_path, count=3)
    (tmp_path / "preview_001.jpg").write_text("x")
    (tmp_path / "preview_002.jpg").write_text("x")
    assert not mod.preview_frames_exist(tmp_path, count=3)
    (tmp_path / "preview_003.jpg").write_text("x")
    assert mod.preview_frames_exist(tmp_path, count=3)


def test_lock_is_recent(tmp_path: Path):
    mod = _load_idempotency_module()
    lock_path = tmp_path / ".enriching"
    assert not mod.lock_is_recent(lock_path, max_age_seconds=10)
    lock_path.write_text("")
    assert mod.lock_is_recent(lock_path, max_age_seconds=10)

    old_time = time.time() - 100
    os.utime(lock_path, (old_time, old_time))
    assert not mod.lock_is_recent(lock_path, max_age_seconds=10)


def test_should_skip_clip(tmp_path: Path):
    mod = _load_idempotency_module()
    queue_meta = tmp_path / "queue_meta.json"
    queue_meta.write_text(json.dumps({"enrichment": {"enriched_at": "2026-02-01T00:00:00Z"}}))
    result = mod.should_skip_clip(
        queue_dir=tmp_path,
        queue_meta_path=queue_meta,
        lock_path=tmp_path / ".enriching",
        lock_max_age_seconds=10,
        preview_count=3,
    )
    assert result.should_skip
    assert "enrichment_complete" in result.reasons
