import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path


import sys
def _load_logging_utils():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "agents" / "queue-enricher" / "logging_utils.py"
    spec = importlib.util.spec_from_file_location("queue_enricher_logging_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


logging_utils = _load_logging_utils()


def test_append_run_log_writes_daily_and_latest(tmp_path):
    logs_dir = tmp_path / "logs"
    started = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
    finished = datetime(2026, 2, 1, 12, 5, tzinfo=UTC)
    run_log = logging_utils.QueueEnricherRunLog(
        started_at=logging_utils.format_utc(started),
        finished_at=logging_utils.format_utc(finished),
        status="complete",
        summary="ok",
        stats={"queued": 2},
        clips=[{"clip": "segment_001"}],
    )

    daily_path = logging_utils.append_run_log(
        logs_dir=logs_dir,
        run_log=run_log,
        when=started,
        write_latest=True,
    )

    assert daily_path.exists()
    daily_payload = json.loads(daily_path.read_text())
    assert isinstance(daily_payload, list)
    assert daily_payload[-1]["status"] == "complete"

    latest_path = logs_dir / "queue-enricher-latest.json"
    assert latest_path.exists()
    latest_payload = json.loads(latest_path.read_text())
    assert latest_payload["summary"] == "ok"


def test_load_daily_log_handles_non_list(tmp_path):
    daily_path = tmp_path / "queue-enricher-2026-02-01.json"
    daily_path.write_text(json.dumps({"status": "noop"}))

    runs = logging_utils.load_daily_log(daily_path)
    assert runs == [{"status": "noop"}]


def test_load_daily_log_handles_invalid_json(tmp_path):
    daily_path = tmp_path / "queue-enricher-2026-02-02.json"
    daily_path.write_text("{not-json")

    runs = logging_utils.load_daily_log(daily_path)
    assert runs == []


def test_append_run_log_without_latest(tmp_path):
    logs_dir = tmp_path / "logs"
    started = datetime(2026, 2, 2, 8, 0, tzinfo=UTC)
    finished = datetime(2026, 2, 2, 8, 2, tzinfo=UTC)
    run_log = logging_utils.QueueEnricherRunLog(
        started_at=logging_utils.format_utc(started),
        finished_at=logging_utils.format_utc(finished),
        status="noop",
        summary="no changes",
        stats={},
        clips=[],
    )

    daily_path = logging_utils.append_run_log(
        logs_dir=logs_dir,
        run_log=run_log,
        when=started,
        write_latest=False,
    )

    assert daily_path.exists()
    latest_path = logs_dir / "queue-enricher-latest.json"
    assert not latest_path.exists()
