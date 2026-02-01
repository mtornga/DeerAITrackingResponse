#!/usr/bin/env python3
"""
Day Watchman patrol agent.

Runs deterministic patrol tasks and reports patrol start/completion via MCP Agent Mail.
Designed for cron execution at 3pm Central time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo
import urllib.error
import urllib.request

TIMEZONE = ZoneInfo("America/Chicago")
VERBOSE = False

DEFAULT_AGENT_NAME = os.environ.get("DAYWATCHMAN_AGENT", "BrightHeron")
DEFAULT_THREAD_ID = os.environ.get("DAYWATCHMAN_THREAD", "DAY-WATCHMAN")
DEFAULT_MCP_URL = os.environ.get("DAYWATCHMAN_MCP_URL", "http://127.0.0.1:8765/mcp/")
DEFAULT_HEALTH_TIMEOUT = int(os.environ.get("DAYWATCHMAN_HEALTH_TIMEOUT", "600"))
DEFAULT_MAX_QUEUE = int(os.environ.get("DAYWATCHMAN_MAX_QUEUE", "6"))
DEFAULT_LOGS_DIRNAME = "runs/logs/watchman"
DEFAULT_DETECTOR_LOG = Path(
    os.environ.get("DAYWATCHMAN_DETECTOR_LOG", "/srv/deer-share/runs/live/logs/detector.log")
)
DEFAULT_PUSHOVER_STALE_HOURS = int(os.environ.get("DAYWATCHMAN_PUSHOVER_STALE_HOURS", "24"))
DEFAULT_PUSHOVER_LOG_LINES = int(os.environ.get("DAYWATCHMAN_PUSHOVER_LOG_LINES", "5000"))


def env_truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


DEFAULT_PROGRESS_MAIL = env_truthy("DAYWATCHMAN_PROGRESS_MAIL", default=False)

ANALYSIS_DIR = Path(os.environ.get("DAYWATCHMAN_ANALYSIS_DIR", "/srv/deer-share/runs/live/analysis"))
DETECTIONS_DIR = Path(os.environ.get("DAYWATCHMAN_DETECTIONS_DIR", "/srv/deer-share/runs/live/detections"))
EVENTS_DIR = Path(os.environ.get("DAYWATCHMAN_EVENTS_DIR", "/srv/deer-share/runs/live/events"))
QUEUE_DIR = Path(os.environ.get("DAYWATCHMAN_QUEUE_DIR", "/srv/deer-share/training_queue"))


@dataclass
class DiskReport:
    path: str
    used_pct: float
    free_gb: float
    status: str


@dataclass
class PipelineReport:
    healthy: bool
    summary: str
    duration_s: float
    output_tail: str


@dataclass
class BacklogDiagnosis:
    detector_status: str
    oldest_clip: str
    analysis_count: int
    detection_count: int
    unprocessed_count: int
    last_prune: str
    proposed_fixes: List[str]


@dataclass
class SegmentInfo:
    date: str
    segment: str
    path: Path
    meta_path: Path
    confidence: float
    animal: int
    person: int
    meta: Dict[str, Any]


@dataclass
class FrameExam:
    segment: str
    confidence_pct: float
    timeline: str
    movement: str
    assessment: str


@dataclass
class QueueReport:
    queued: List[str]
    queue_size: int


@dataclass
class PushoverReport:
    status: str
    last_sent_utc: Optional[str]
    last_sent_local: Optional[str]
    last_segment: Optional[str]
    segment_capture_utc: Optional[str]
    lag_seconds: Optional[float]
    lag_human: Optional[str]
    last_error: Optional[str]
    log_path: str


LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(?P<ms>\d+) "
    r"\[(?P<level>[A-Z]+)\] (?P<msg>.*)$"
)
PROMOTED_RE = re.compile(
    r"Promoted (?P<segment>\d{4}-\d{2}-\d{2}/segment_[0-9_]+\.(?:mkv|mp4))"
)
SEGMENT_TS_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})/segment_(?P<time>\d{6})"
)


def count_training_queue(queue_dir: Path, errors: List[str]) -> int:
    if not queue_dir.exists():
        return 0
    try:
        return sum(1 for entry in queue_dir.iterdir() if entry.is_dir())
    except Exception as exc:  # pragma: no cover - filesystem issues
        errors.append(f"Failed to read training queue {queue_dir}: {exc}")
        return 0


class MCPClient:
    def __init__(self, base_url: str, timeout: int = 10, token: Optional[str] = None) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.token = token
        self._next_id = 1

    def ping(self) -> bool:
        try:
            self._request({"jsonrpc": "2.0", "id": self._alloc_id(), "method": "ping"})
            return True
        except Exception:
            return False

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._alloc_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        response = self._request(payload)
        if "error" in response:
            raise RuntimeError(f"MCP tool error: {response['error']}")
        return response.get("result", {})

    def _alloc_id(self) -> int:
        current = self._next_id
        self._next_id += 1
        return current

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            self.base_url,
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)


def log(message: str, level: str = "INFO") -> None:
    timestamp = datetime.now(TIMEZONE).strftime("%H:%M:%S")
    print(f"[{level} {timestamp}] {message}", flush=True)


def log_verbose(message: str) -> None:
    if VERBOSE:
        log(message, "DEBUG")


def parse_recipients(value: Optional[str], fallback: str) -> List[str]:
    if value:
        recipients = [item.strip() for item in value.split(",") if item.strip()]
        if recipients:
            return recipients
    return [fallback]


def get_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    return script_path.parents[2]


def get_logs_dir(repo_root: Path) -> Path:
    return repo_root / DEFAULT_LOGS_DIRNAME


def classify_disk(used_pct: float) -> str:
    if used_pct >= 92.0:
        return "CRITICAL"
    if used_pct >= 85.0:
        return "WARNING"
    return "OK"


def disk_report(path: str) -> DiskReport:
    usage = shutil.disk_usage(path)
    used_pct = (usage.used / usage.total) * 100.0
    free_gb = usage.free / (1024 ** 3)
    return DiskReport(
        path=path,
        used_pct=used_pct,
        free_gb=free_gb,
        status=classify_disk(used_pct),
    )


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
) -> Tuple[int, str, float]:
    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.time() - start
    output = (result.stdout or "") + (result.stderr or "")
    log(f"Command finished in {elapsed:.1f}s: {' '.join(cmd)}")
    if VERBOSE and output.strip():
        tail = "\n".join(output.splitlines()[-8:])
        log_verbose(f"Command output tail:\n{tail}")
    return result.returncode, output, elapsed


def run_pipeline_health_check(
    repo_root: Path,
    lighting: str,
    timeout: int,
) -> PipelineReport:
    cmd = [sys.executable, "scripts/pipeline_health_check.py", "--lighting", lighting]
    code, output, elapsed = run_command(cmd, cwd=repo_root, timeout=timeout)
    summary = "UNKNOWN"
    for line in output.splitlines():
        if line.strip().startswith("Status:"):
            summary = line.split("Status:", 1)[1].strip()
            break
    healthy = code == 0 and summary.startswith("HEALTHY")
    tail_lines = output.splitlines()[-12:]
    output_tail = "\n".join(tail_lines).strip()
    return PipelineReport(
        healthy=healthy,
        summary=summary,
        duration_s=elapsed,
        output_tail=output_tail,
    )


def classify_backlog(count: int) -> str:
    if count <= 50:
        return "OK"
    if count <= 100:
        return "WARNING"
    if count <= 500:
        return "HIGH"
    return "CRITICAL"


def scan_analysis_backlog(
    analysis_dir: Path,
    detections_dir: Path,
) -> Tuple[int, int, Optional[Path]]:
    analysis_count = 0
    pending_count = 0
    oldest_pending: Optional[Path] = None
    oldest_mtime: Optional[float] = None

    for mkv in analysis_dir.rglob("segment_*.mkv"):
        if not mkv.is_file():
            continue
        analysis_count += 1
        date = mkv.parent.name
        seg = mkv.stem
        meta = detections_dir / date / seg / "meta.json"
        if meta.exists():
            continue
        pending_count += 1
        mtime = mkv.stat().st_mtime
        if oldest_mtime is None or mtime < oldest_mtime:
            oldest_mtime = mtime
            oldest_pending = mkv

    return analysis_count, pending_count, oldest_pending


def count_files(root: Path, pattern: str) -> Tuple[int, Optional[Path]]:
    count = 0
    oldest_path = None
    oldest_mtime = None
    for path in root.rglob(pattern):
        if not path.is_file():
            continue
        count += 1
        mtime = path.stat().st_mtime
        if oldest_mtime is None or mtime < oldest_mtime:
            oldest_mtime = mtime
            oldest_path = path
    return count, oldest_path


def check_detector_running() -> str:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "live_detector"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "UNKNOWN (pgrep missing)"
    if result.returncode == 0:
        pid = result.stdout.strip().splitlines()[0]
        return f"RUNNING (PID {pid})"
    return "NOT RUNNING"


def get_last_modified(path: Path) -> str:
    if not path.exists():
        return "NOT FOUND"
    timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=TIMEZONE)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S %Z")


def run_backlog_diagnosis(
    analysis_count: int,
    detection_count: int,
    oldest_path: Optional[Path],
) -> BacklogDiagnosis:
    detector_status = check_detector_running()
    oldest_clip = oldest_path.as_posix() if oldest_path else "none found"
    unprocessed = max(analysis_count - detection_count, 0)
    prune_log = Path("/srv/deer-share/logs/prune_events.log")
    last_prune = get_last_modified(prune_log)
    proposed_fixes = []
    if detector_status.startswith("NOT"):
        proposed_fixes.append(
            "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate "
            "&& python scripts/live_detector_multimodel.py --daemon"
        )
    if analysis_count > 100:
        proposed_fixes.append(
            "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate "
            "&& python scripts/prune_segments_without_events.py --dry-run"
        )
    proposed_fixes.append("Verify cron is scheduling prune_events.py")
    return BacklogDiagnosis(
        detector_status=detector_status,
        oldest_clip=oldest_clip,
        analysis_count=analysis_count,
        detection_count=detection_count,
        unprocessed_count=unprocessed,
        last_prune=last_prune,
        proposed_fixes=proposed_fixes,
    )


def list_segments(dates: Iterable[str], errors: List[str]) -> List[SegmentInfo]:
    segments: List[SegmentInfo] = []
    for date in dates:
        day_dir = EVENTS_DIR / date
        if not day_dir.exists():
            continue
        for entry in sorted(day_dir.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("segment_"):
                continue
            meta_path = entry / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except Exception as exc:  # pragma: no cover - file system issues
                errors.append(f"Failed to read {meta_path}: {exc}")
                continue
            confidence = float(meta.get("max_confidence") or 0.0)
            counts = meta.get("counts") or {}
            animal = int(counts.get("animal") or 0)
            person = int(counts.get("person") or 0)
            segments.append(SegmentInfo(
                date=date,
                segment=entry.name,
                path=entry,
                meta_path=meta_path,
                confidence=confidence,
                animal=animal,
                person=person,
                meta=meta,
            ))
    return segments


def select_best_segment(segments: List[SegmentInfo]) -> Optional[SegmentInfo]:
    if not segments:
        return None
    return max(segments, key=lambda seg: seg.confidence)


def analyze_frame(segment: SegmentInfo) -> FrameExam:
    meta = segment.meta
    timeline = meta.get("detection_timeline") or {}
    first_frame = timeline.get("first_frame", "?")
    last_frame = timeline.get("last_frame", "?")
    duration = float(timeline.get("duration") or 0.0)

    hits: List[List[float]] = []
    for model in meta.get("models", []) or []:
        for hit in model.get("hits", []) or []:
            bbox = hit.get("bbox")
            if isinstance(bbox, list) and len(bbox) >= 4:
                hits.append(bbox)

    if not hits:
        return FrameExam(
            segment=segment.segment,
            confidence_pct=segment.confidence * 100,
            timeline=f"frames {first_frame}-{last_frame} ({duration:.2f}s duration)",
            movement="no bbox hits found",
            assessment="No detection hits to analyze",
        )

    xs = [bbox[0] for bbox in hits]
    avg_x = sum(xs) / len(xs)
    x_std = math.sqrt(sum((x - avg_x) ** 2 for x in xs) / len(xs))
    avg_size = sum(bbox[2] * bbox[3] for bbox in hits) / len(hits)
    avg_x_pct = avg_x * 100.0
    avg_size_pct = avg_size * 100.0

    if x_std < 0.02 and (avg_x < 0.15 or avg_x > 0.85):
        assessment = "Possible AprilTag false positive - static at edge"
    elif avg_size < 0.03 and x_std < 0.02:
        assessment = "Possible false positive - small static object"
    elif x_std > 0.05 or duration > 2.0:
        assessment = "Likely real animal - shows movement"
    else:
        assessment = "Inconclusive - moderate movement"

    movement = f"x_std={x_std:.3f}, avg_x={avg_x_pct:.0f}%, avg_size={avg_size_pct:.2f}%"
    timeline_str = f"frames {first_frame}-{last_frame} ({duration:.2f}s duration)"
    return FrameExam(
        segment=segment.segment,
        confidence_pct=segment.confidence * 100,
        timeline=timeline_str,
        movement=movement,
        assessment=assessment,
    )


def queue_for_training(
    segments: List[SegmentInfo],
    max_queue: int,
    dry_run: bool,
    errors: List[str],
) -> QueueReport:
    queued: List[str] = []
    if not dry_run:
        QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        for segment in segments:
            if len(queued) >= max_queue:
                break
            if segment.animal <= 0 and segment.person <= 0:
                continue
            queue_name = f"{segment.date}_{segment.segment}"
            dest = QUEUE_DIR / queue_name
            if dest.exists():
                continue
            shutil.copytree(segment.path, dest)
            queued.append(queue_name)
    queue_size = count_training_queue(QUEUE_DIR, errors)
    return QueueReport(queued=queued, queue_size=queue_size)


def write_digest(
    repo_root: Path,
    agent_name: str,
    start_time: datetime,
    disk_reports: List[DiskReport],
    pipeline_report: PipelineReport,
    pushover_report: PushoverReport,
    backlog_count: int,
    backlog_status: str,
    analysis_count: int,
    backlog_diagnosis: Optional[BacklogDiagnosis],
    segments: List[SegmentInfo],
    best_segment: Optional[SegmentInfo],
    frame_exam: Optional[FrameExam],
    queue_report: QueueReport,
    errors: List[str],
    dry_run: bool,
) -> Path:
    logs_dir = get_logs_dir(repo_root)
    logs_dir.mkdir(parents=True, exist_ok=True)
    digest_path = logs_dir / f"day-watchman-{start_time.strftime('%Y-%m-%d')}.md"

    if dry_run:
        return digest_path

    total_animals = sum(seg.animal for seg in segments)
    total_person = sum(seg.person for seg in segments)

    lines: List[str] = []
    lines.append(f"# Day Watchman Patrol - {start_time.strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append(f"Agent: {agent_name} | Time: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append("")
    lines.append("## Health")
    for report in disk_reports:
        lines.append(
            f"- Disk {report.path}: {report.status} ({report.used_pct:.1f}% used, "
            f"{report.free_gb:.1f} GB free)"
        )
    lines.append(f"- Pipeline: {pipeline_report.summary}")
    if not pipeline_report.healthy and pipeline_report.output_tail:
        lines.append("### Pipeline Output (tail)")
        lines.append("```")
        lines.append(pipeline_report.output_tail)
        lines.append("```")
    lines.append(
        f"- Backlog: {backlog_count} pending ({backlog_status}) of {analysis_count} analysis clips"
    )
    lines.append("")

    lines.append("## Pushover")
    lines.append(f"- Status: {pushover_report.status}")
    if pushover_report.last_sent_local:
        lines.append(f"- Last notification: {pushover_report.last_sent_local}")
    if pushover_report.last_segment:
        lines.append(f"- Last segment: {pushover_report.last_segment}")
    if pushover_report.lag_human:
        lines.append(f"- Capture -> notify lag: {pushover_report.lag_human}")
    if pushover_report.last_error:
        lines.append(f"- Last error: {pushover_report.last_error}")
    lines.append("")

    if backlog_diagnosis:
        lines.append("## Backlog Diagnosis")
        lines.append(f"- Detector: {backlog_diagnosis.detector_status}")
        lines.append(f"- Oldest clip: {backlog_diagnosis.oldest_clip}")
        lines.append(
            f"- Analysis clips: {backlog_diagnosis.analysis_count} | "
            f"Detection JSONs: {backlog_diagnosis.detection_count} | "
            f"Unprocessed: {backlog_diagnosis.unprocessed_count}"
        )
        lines.append(f"- Last prune: {backlog_diagnosis.last_prune}")
        lines.append("- Proposed fixes:")
        for fix in backlog_diagnosis.proposed_fixes:
            lines.append(f"  - {fix}")
        lines.append("")

    lines.append("## Detections")
    lines.append(
        f"- Total segments: {len(segments)} | Deer: {total_animals} | Person: {total_person}"
    )
    if best_segment:
        lines.append(
            f"- Highest confidence: {best_segment.segment} "
            f"({best_segment.confidence * 100:.1f}%) on {best_segment.date}"
        )
    if segments:
        lines.append("- Segments:")
        for seg in segments:
            lines.append(
                f"  - {seg.date}/{seg.segment}: {seg.confidence * 100:.1f}% "
                f"animal={seg.animal} person={seg.person}"
            )
    lines.append("")

    lines.append("## Frame Examination")
    if frame_exam:
        lines.append(
            f"- Segment: {frame_exam.segment} ({frame_exam.confidence_pct:.1f}%)"
        )
        lines.append(f"- Timeline: {frame_exam.timeline}")
        lines.append(f"- Movement: {frame_exam.movement}")
        lines.append(f"- Assessment: {frame_exam.assessment}")
    else:
        lines.append("- No events available for frame examination")
    lines.append("")

    lines.append("## Training Queue")
    lines.append(f"- Queue size: {queue_report.queue_size} clips")
    if queue_report.queued:
        lines.append(f"- Added this run: {len(queue_report.queued)} clips")
        for name in queue_report.queued:
            lines.append(f"  - {name}")
    lines.append("")

    if errors:
        lines.append("## Errors")
        for err in errors:
            lines.append(f"- {err}")
        lines.append("")

    digest_path.write_text("\n".join(lines))
    return digest_path


def build_snapshot(
    agent_name: str,
    start_time: datetime,
    disk_reports: List[DiskReport],
    pipeline_report: PipelineReport,
    pushover_report: PushoverReport,
    backlog_count: int,
    backlog_status: str,
    analysis_count: int,
    detection_count: int,
    oldest_backlog: Optional[Path],
    backlog_diagnosis: Optional[BacklogDiagnosis],
    segments: List[SegmentInfo],
    best_segment: Optional[SegmentInfo],
    frame_exam: Optional[FrameExam],
    queue_report: QueueReport,
    digest_path: Path,
    errors: List[str],
) -> Dict[str, Any]:
    oldest_clip = oldest_backlog.as_posix() if oldest_backlog else ""
    return {
        "agent": agent_name,
        "timestamp": start_time.isoformat(),
        "date": start_time.strftime("%Y-%m-%d"),
        "timezone": str(TIMEZONE),
        "digest_path": str(digest_path),
        "disk": [
            {
                "path": report.path,
                "used_pct": report.used_pct,
                "free_gb": report.free_gb,
                "status": report.status,
            }
            for report in disk_reports
        ],
        "pipeline": {
            "healthy": pipeline_report.healthy,
            "summary": pipeline_report.summary,
            "duration_s": pipeline_report.duration_s,
            "output_tail": pipeline_report.output_tail,
        },
        "pushover": {
            "status": pushover_report.status,
            "last_sent_utc": pushover_report.last_sent_utc,
            "last_sent_local": pushover_report.last_sent_local,
            "last_segment": pushover_report.last_segment,
            "segment_capture_utc": pushover_report.segment_capture_utc,
            "lag_seconds": pushover_report.lag_seconds,
            "lag_human": pushover_report.lag_human,
            "last_error": pushover_report.last_error,
            "log_path": pushover_report.log_path,
        },
        "backlog": {
            "count": backlog_count,
            "pending_count": backlog_count,
            "status": backlog_status,
            "analysis_count": analysis_count,
            "detection_count": detection_count,
            "unprocessed_count": max(analysis_count - detection_count, 0),
            "oldest_clip": backlog_diagnosis.oldest_clip if backlog_diagnosis else oldest_clip,
            "diagnosis": (
                {
                    "detector_status": backlog_diagnosis.detector_status,
                    "last_prune": backlog_diagnosis.last_prune,
                    "proposed_fixes": backlog_diagnosis.proposed_fixes,
                }
                if backlog_diagnosis
                else None
            ),
        },
        "detections": {
            "total_segments": len(segments),
            "total_animals": sum(seg.animal for seg in segments),
            "total_person": sum(seg.person for seg in segments),
            "segments": [
                {
                    "date": seg.date,
                    "segment": seg.segment,
                    "confidence": seg.confidence,
                    "animal": seg.animal,
                    "person": seg.person,
                    "meta_path": str(seg.meta_path),
                }
                for seg in segments
            ],
            "best_segment": (
                {
                    "date": best_segment.date,
                    "segment": best_segment.segment,
                    "confidence": best_segment.confidence,
                    "animal": best_segment.animal,
                    "person": best_segment.person,
                    "meta_path": str(best_segment.meta_path),
                }
                if best_segment
                else None
            ),
        },
        "frame_exam": (
            {
                "segment": frame_exam.segment,
                "confidence_pct": frame_exam.confidence_pct,
                "timeline": frame_exam.timeline,
                "movement": frame_exam.movement,
                "assessment": frame_exam.assessment,
            }
            if frame_exam
            else None
        ),
        "training_queue": {
            "queued": queue_report.queued,
            "queue_size": queue_report.queue_size,
        },
        "errors": errors,
    }


def write_snapshot(
    repo_root: Path,
    snapshot: Dict[str, Any],
    start_time: datetime,
    dry_run: bool,
) -> Path:
    logs_dir = get_logs_dir(repo_root)
    logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = logs_dir / f"day-watchman-{start_time.strftime('%Y-%m-%d')}.json"
    if dry_run:
        return snapshot_path
    snapshot_with_path = dict(snapshot)
    snapshot_with_path["snapshot_path"] = str(snapshot_path)
    snapshot_path.write_text(json.dumps(snapshot_with_path, indent=2))
    latest_path = logs_dir / "day-watchman-latest.json"
    latest_path.write_text(json.dumps(snapshot_with_path, indent=2))
    return snapshot_path


def format_disk_summary(reports: List[DiskReport]) -> str:
    return "; ".join(
        f"{report.path} {report.status} ({report.used_pct:.1f}% used)"
        for report in reports
    )


def format_frame_exam_summary(frame_exam: Optional[FrameExam]) -> str:
    if not frame_exam:
        return "No events to examine"
    return f"{frame_exam.segment}: {frame_exam.assessment} ({frame_exam.movement})"


def format_utc(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def format_local(timestamp: datetime) -> str:
    return timestamp.astimezone(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")


def format_duration(seconds: float) -> str:
    if seconds < 0:
        return "n/a"
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def tail_lines(path: Path, max_lines: int, errors: List[str]) -> List[str]:
    if max_lines <= 0:
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return list(deque(handle, maxlen=max_lines))
    except FileNotFoundError:
        return []
    except Exception as exc:  # pragma: no cover - filesystem issues
        errors.append(f"Failed to read log {path}: {exc}")
        return []


def parse_log_line(line: str) -> Optional[Tuple[datetime, str]]:
    match = LOG_LINE_RE.match(line.strip())
    if not match:
        return None
    timestamp = match.group("ts")
    message = match.group("msg")
    try:
        parsed = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return parsed, message


def parse_segment_capture_time(segment: str) -> Optional[datetime]:
    match = SEGMENT_TS_RE.search(segment)
    if not match:
        return None
    stamp = f"{match.group('date')} {match.group('time')}"
    try:
        parsed = datetime.strptime(stamp, "%Y-%m-%d %H%M%S")
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC)


def collect_pushover_report(
    log_path: Path,
    now: datetime,
    errors: List[str],
) -> PushoverReport:
    if not log_path.exists():
        return PushoverReport(
            status="MISSING",
            last_sent_utc=None,
            last_sent_local=None,
            last_segment=None,
            segment_capture_utc=None,
            lag_seconds=None,
            lag_human=None,
            last_error=None,
            log_path=str(log_path),
        )

    lines = tail_lines(log_path, DEFAULT_PUSHOVER_LOG_LINES, errors)
    last_sent_idx = None
    last_sent_ts = None
    last_error_ts = None
    last_error_msg = None

    for idx in range(len(lines) - 1, -1, -1):
        parsed = parse_log_line(lines[idx])
        if not parsed:
            continue
        ts, msg = parsed
        if last_sent_idx is None and "Pushover notification sent" in msg:
            last_sent_idx = idx
            last_sent_ts = ts
        if last_error_ts is None and (
            "Pushover API error" in msg
            or "Pushover request failed" in msg
            or "Pushover not configured or disabled" in msg
        ):
            last_error_ts = ts
            last_error_msg = msg
        if last_sent_idx is not None and last_error_ts is not None:
            break

    last_segment = None
    if last_sent_idx is not None:
        for idx in range(last_sent_idx - 1, -1, -1):
            parsed = parse_log_line(lines[idx])
            if not parsed:
                continue
            _, msg = parsed
            promoted = PROMOTED_RE.search(msg)
            if promoted:
                last_segment = promoted.group("segment")
                break

    capture_time = parse_segment_capture_time(last_segment) if last_segment else None
    last_sent_utc = None
    last_sent_local = None
    lag_seconds = None
    lag_human = None

    if last_sent_ts is not None:
        last_sent_utc = last_sent_ts.replace(tzinfo=TIMEZONE).astimezone(UTC)
        if capture_time is not None:
            lag_seconds = (last_sent_utc - capture_time).total_seconds()
            if lag_seconds < 0:
                alt_sent_utc = last_sent_ts.replace(tzinfo=UTC)
                alt_lag = (alt_sent_utc - capture_time).total_seconds()
                if alt_lag >= 0:
                    last_sent_utc = alt_sent_utc
                    lag_seconds = alt_lag
            lag_human = format_duration(lag_seconds)
        last_sent_local = format_local(last_sent_utc)

    status = "UNKNOWN"
    now_utc = now.astimezone(UTC)
    if last_sent_utc is None:
        status = "ERROR" if last_error_ts else "NO_DATA"
    else:
        status = "OK"
        if last_error_ts is not None:
            last_error_utc = last_error_ts.replace(tzinfo=TIMEZONE).astimezone(UTC)
            if last_error_utc > last_sent_utc:
                status = "ERROR"
        if status == "OK":
            stale_after = timedelta(hours=DEFAULT_PUSHOVER_STALE_HOURS)
            if now_utc - last_sent_utc > stale_after:
                status = "STALE"

    return PushoverReport(
        status=status,
        last_sent_utc=format_utc(last_sent_utc) if last_sent_utc else None,
        last_sent_local=last_sent_local,
        last_segment=last_segment,
        segment_capture_utc=format_utc(capture_time) if capture_time else None,
        lag_seconds=lag_seconds,
        lag_human=lag_human,
        last_error=last_error_msg,
        log_path=str(log_path),
    )


def send_message_safe(
    client: Optional[MCPClient],
    project_key: str,
    sender: str,
    recipients: List[str],
    subject: str,
    body: str,
    thread_id: str,
    skip_mail: bool,
    errors: List[str],
) -> None:
    if skip_mail or client is None:
        log(f"Mail skipped: {subject}")
        return
    try:
        client.call_tool(
            "send_message",
            {
                "project_key": project_key,
                "sender_name": sender,
                "to": recipients,
                "subject": subject,
                "body_md": body,
                "thread_id": thread_id,
                "importance": "normal",
                "ack_required": False,
            },
        )
    except Exception as exc:
        errors.append(f"Failed to send mail '{subject}': {exc}")
        log(f"Mail error: {exc}", "WARN")


def register_agent_safe(
    client: Optional[MCPClient],
    project_key: str,
    agent_name: str,
    skip_mail: bool,
    errors: List[str],
) -> None:
    if skip_mail or client is None:
        return
    try:
        client.call_tool("ensure_project", {"human_key": project_key})
        client.call_tool(
            "register_agent",
            {
                "project_key": project_key,
                "program": "codex-cli",
                "model": "gpt-5",
                "name": agent_name,
                "task_description": "Day Watchman patrol agent",
            },
        )
    except Exception as exc:
        errors.append(f"Failed to register agent: {exc}")
        log(f"Agent register error: {exc}", "WARN")


def fetch_inbox_safe(
    client: Optional[MCPClient],
    project_key: str,
    agent_name: str,
    skip_mail: bool,
    errors: List[str],
) -> None:
    if skip_mail or client is None:
        return
    try:
        raw_messages = client.call_tool(
            "fetch_inbox",
            {
                "project_key": project_key,
                "agent_name": agent_name,
                "include_bodies": True,
                "limit": 5,
            },
        )
        messages = normalize_inbox_messages(raw_messages, errors)
        for message in messages:
            message_id = message.get("id")
            if message.get("ack_required") and message_id is not None:
                client.call_tool(
                    "acknowledge_message",
                    {
                        "project_key": project_key,
                        "agent_name": agent_name,
                        "message_id": message_id,
                    },
                )
    except Exception as exc:
        errors.append(f"Failed to fetch/ack inbox: {exc}")
        log(f"Inbox error: {exc}", "WARN")


def normalize_inbox_messages(raw: Any, errors: List[str]) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            errors.append("Inbox payload was a non-JSON string")
            return []
    if isinstance(raw, dict):
        for key in ("messages", "items", "result"):
            if key in raw:
                raw = raw[key]
                break
    if not isinstance(raw, list):
        errors.append(f"Inbox payload had unexpected type: {type(raw).__name__}")
        return []
    messages = [item for item in raw if isinstance(item, dict)]
    if len(messages) < len(raw):
        errors.append("Inbox payload contained non-dict entries")
    return messages


def run_patrol(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    start_time = datetime.now(TIMEZONE)
    errors: List[str] = []
    skip_mail = args.skip_mail or args.dry_run
    collect_only = args.collect_only
    args.progress_mail = args.progress_mail or DEFAULT_PROGRESS_MAIL

    recipients = parse_recipients(args.recipients, args.agent_name)
    token = os.environ.get("DAYWATCHMAN_MCP_TOKEN")
    client = MCPClient(args.mcp_url, timeout=10, token=token)

    mcp_ok = client.ping()
    if not mcp_ok:
        log("MCP Agent Mail server not responding", "WARN")
        client = None

    register_agent_safe(client, str(repo_root), args.agent_name, skip_mail, errors)

    if args.progress_mail:
        send_message_safe(
            client,
            str(repo_root),
            args.agent_name,
            recipients,
            "Patrol Starting",
            (
                f"Starting day patrol at {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}. "
                "Steps: health, backlog, detections, frame exam, training queue, pushover, digest, report, inbox."
            ),
            args.thread_id,
            skip_mail,
            errors,
        )

    disk_reports = [
        disk_report("/"),
        disk_report("/srv/deer-share"),
    ]
    pipeline_report = run_pipeline_health_check(
        repo_root,
        lighting=args.lighting,
        timeout=args.health_timeout,
    )
    if pipeline_report.summary == "UNKNOWN":
        errors.append("Pipeline health check did not emit a Status line")

    analysis_count, pending_count, oldest_backlog = scan_analysis_backlog(
        ANALYSIS_DIR,
        DETECTIONS_DIR,
    )
    backlog_count = pending_count
    backlog_status = classify_backlog(backlog_count)
    detection_count = max(analysis_count - pending_count, 0)
    backlog_diagnosis = None
    if backlog_status in {"HIGH", "CRITICAL"}:
        backlog_diagnosis = run_backlog_diagnosis(
            analysis_count=analysis_count,
            detection_count=detection_count,
            oldest_path=oldest_backlog,
        )

    today = start_time.strftime("%Y-%m-%d")
    yesterday = (start_time - timedelta(days=1)).strftime("%Y-%m-%d")
    segments = list_segments([today, yesterday], errors)
    best_segment = select_best_segment(segments)

    frame_exam = analyze_frame(best_segment) if best_segment else None

    queue_report = queue_for_training(segments, args.max_queue, args.dry_run, errors)

    pushover_report = collect_pushover_report(DEFAULT_DETECTOR_LOG, start_time, errors)

    digest_path = write_digest(
        repo_root=repo_root,
        agent_name=args.agent_name,
        start_time=start_time,
        disk_reports=disk_reports,
        pipeline_report=pipeline_report,
        pushover_report=pushover_report,
        backlog_count=backlog_count,
        backlog_status=backlog_status,
        analysis_count=analysis_count,
        backlog_diagnosis=backlog_diagnosis,
        segments=segments,
        best_segment=best_segment,
        frame_exam=frame_exam,
        queue_report=queue_report,
        errors=errors,
        dry_run=args.dry_run,
    )

    snapshot = build_snapshot(
        agent_name=args.agent_name,
        start_time=start_time,
        disk_reports=disk_reports,
        pipeline_report=pipeline_report,
        pushover_report=pushover_report,
        backlog_count=backlog_count,
        backlog_status=backlog_status,
        analysis_count=analysis_count,
        detection_count=detection_count,
        oldest_backlog=oldest_backlog,
        backlog_diagnosis=backlog_diagnosis,
        segments=segments,
        best_segment=best_segment,
        frame_exam=frame_exam,
        queue_report=queue_report,
        digest_path=digest_path,
        errors=errors,
    )
    snapshot_path = write_snapshot(
        repo_root=repo_root,
        snapshot=snapshot,
        start_time=start_time,
        dry_run=args.dry_run,
    )
    log(f"Snapshot: {snapshot_path}")

    if collect_only:
        log("Collect-only mode: skipping final report and inbox check")

    if not collect_only:
        send_message_safe(
            client,
            str(repo_root),
            args.agent_name,
            recipients,
            "Patrol Complete",
            (
                "## Day Watchman Patrol Complete\n\n"
                f"Health: {format_disk_summary(disk_reports)}; "
                f"Pipeline {pipeline_report.summary}\n"
                f"Pushover: {pushover_report.status} "
                f"(lag {pushover_report.lag_human or 'n/a'})\n"
                f"Backlog: {backlog_count} pending of {analysis_count} ({backlog_status})\n"
                f"Detections: {len(segments)} segments\n"
                f"Frame Exam: {format_frame_exam_summary(frame_exam)}\n"
                f"Training Queue: {queue_report.queue_size} clips queued\n\n"
                f"Digest: {digest_path}\n\n"
                f"-- {args.agent_name} @ {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            ),
            args.thread_id,
            skip_mail,
            errors,
        )

        fetch_inbox_safe(client, str(repo_root), args.agent_name, skip_mail, errors)

    log("PATROL COMPLETE")
    log(
        f"Health: {format_disk_summary(disk_reports)}; "
        f"Pipeline {pipeline_report.summary}"
    )
    log(f"Pushover: {pushover_report.status} (lag {pushover_report.lag_human or 'n/a'})")
    log(f"Backlog: {backlog_count} pending of {analysis_count} ({backlog_status})")
    log(f"Detections: {len(segments)} segments")
    log(f"Frame Exam: {format_frame_exam_summary(frame_exam)}")
    log(f"Training Queue: {queue_report.queue_size} queued")
    if errors:
        log(f"Errors: {len(errors)} (see digest)", "WARN")

    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day Watchman patrol agent")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing or mail")
    parser.add_argument("--skip-mail", action="store_true", help="Skip Agent Mail reporting")
    parser.add_argument(
        "--progress-mail",
        action="store_true",
        help="Send a patrol starting notice",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect data and write snapshot/digest, skip final report + inbox",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--lighting", default="day", help="Lighting mode for health check")
    parser.add_argument("--health-timeout", type=int, default=DEFAULT_HEALTH_TIMEOUT)
    parser.add_argument("--max-queue", type=int, default=DEFAULT_MAX_QUEUE)
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--thread-id", default=DEFAULT_THREAD_ID)
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL)
    parser.add_argument("--recipients", default=os.environ.get("DAYWATCHMAN_TO"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    global VERBOSE
    VERBOSE = args.verbose
    sys.exit(run_patrol(args))


if __name__ == "__main__":
    main()
