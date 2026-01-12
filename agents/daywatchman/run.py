#!/usr/bin/env python3
"""
Day Watchman patrol agent.

Runs deterministic patrol tasks and reports progress through MCP Agent Mail.
Designed for cron execution at 3pm Central time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
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
                "Steps: health, backlog, detections, frame exam, training queue, digest, report, inbox."
            ),
            args.thread_id,
            skip_mail,
            errors,
        )

    if args.progress_mail:
        send_message_safe(
            client,
            str(repo_root),
            args.agent_name,
            recipients,
            "Step 1: Health Check",
            "Checking disk space and pipeline health.",
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

    if args.progress_mail:
        send_message_safe(
            client,
            str(repo_root),
            args.agent_name,
            recipients,
            "Step 2: Backlog Check",
            "Checking analysis backlog and running diagnosis if needed.",
            args.thread_id,
            skip_mail,
            errors,
        )
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

    if args.progress_mail:
        send_message_safe(
            client,
            str(repo_root),
            args.agent_name,
            recipients,
            "Step 3: Detection Summary",
            "Summarizing detections from today and yesterday.",
            args.thread_id,
            skip_mail,
            errors,
        )
    today = start_time.strftime("%Y-%m-%d")
    yesterday = (start_time - timedelta(days=1)).strftime("%Y-%m-%d")
    segments = list_segments([today, yesterday], errors)
    best_segment = select_best_segment(segments)

    if args.progress_mail:
        send_message_safe(
            client,
            str(repo_root),
            args.agent_name,
            recipients,
            "Step 4: Frame Examination",
            "Analyzing highest-confidence detection metadata.",
            args.thread_id,
            skip_mail,
            errors,
        )
    frame_exam = analyze_frame(best_segment) if best_segment else None

    if args.progress_mail:
        send_message_safe(
            client,
            str(repo_root),
            args.agent_name,
            recipients,
            "Step 5: Training Queue",
            "Queueing deer/person clips for training.",
            args.thread_id,
            skip_mail,
            errors,
        )
    queue_report = queue_for_training(segments, args.max_queue, args.dry_run, errors)

    if args.progress_mail:
        send_message_safe(
            client,
            str(repo_root),
            args.agent_name,
            recipients,
            "Step 6: Writing Digest",
            "Writing patrol report to disk.",
            args.thread_id,
            skip_mail,
            errors,
        )
    digest_path = write_digest(
        repo_root=repo_root,
        agent_name=args.agent_name,
        start_time=start_time,
        disk_reports=disk_reports,
        pipeline_report=pipeline_report,
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

        if args.progress_mail:
            send_message_safe(
                client,
                str(repo_root),
                args.agent_name,
                recipients,
                "Step 7: Inbox Check",
                "Checking inbox for new messages.",
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
        help="Send progress updates for each step",
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
