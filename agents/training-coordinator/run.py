#!/usr/bin/env python3
"""
Training Coordinator agent.

Scans the training queue, reads queue_meta.json enrichment output, decides which
clips to promote into golden_clips, and writes a TC run log + manifest updates.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import urllib.error
import urllib.request


def _ensure_repo_root_on_path() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / ".env").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    fallback = script_path.parents[2]
    if str(fallback) not in sys.path:
        sys.path.insert(0, str(fallback))
    return fallback


REPO_ROOT = _ensure_repo_root_on_path()

try:
    from env_loader import load_env_file

    load_env_file()
except Exception:
    pass

DEFAULT_AGENT_NAME = os.environ.get("TRAINING_COORDINATOR_AGENT", "AmberField")
DEFAULT_THREAD_ID = os.environ.get("TRAINING_COORDINATOR_THREAD", "TRAINING-COORD")
DEFAULT_MCP_URL = os.environ.get("TRAINING_COORDINATOR_MCP_URL", "http://127.0.0.1:8765/mcp/")
DEFAULT_QUEUE_DIR = os.environ.get("TRAINING_COORDINATOR_QUEUE_DIR", "/srv/deer-share/training_queue")
DEFAULT_NEGATIVE_CAP = int(os.environ.get("TRAINING_COORDINATOR_NEGATIVE_CAP", "15"))
DEFAULT_LOGS_DIRNAME = "runs/logs/training_coordinator"

ALLOWED_CATEGORIES = {"deer", "person", "other_wildlife"}


@dataclass
class ClipDecision:
    clip_id: str
    queue_dir: Path
    triage: Optional[str]
    category: Optional[str]
    lighting: Optional[str]
    decision: str
    reason: str
    dest_dir: Optional[Path] = None
    tags: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    queue_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    run_id: str
    status: str
    started_at: str
    finished_at: str
    queue_dir: str
    share_root: str
    golden_root: str
    counts: Dict[str, int]
    errors: List[str]
    decisions: List[Dict[str, Any]]
    log_path: Optional[str] = None


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


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[TrainingCoordinator {timestamp}] {message}", flush=True)


def format_utc(ts: datetime) -> str:
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return cleaned if cleaned else None
    return None


def normalize_tags(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    tags = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip().lower()
            if cleaned:
                tags.append(cleaned)
    return sorted(set(tags))


def coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def get_repo_root() -> Path:
    return REPO_ROOT


def resolve_share_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()

    candidates: List[Path] = []
    env_server = os.environ.get("DEER_SHARE_SERVER_PATH")
    env_local = os.environ.get("DEER_SHARE_LOCAL_MOUNT")

    if env_server:
        candidates.append(Path(env_server).expanduser())
    if env_local:
        candidates.append(Path(env_local).expanduser())

    candidates.append(Path("/srv/deer-share"))
    candidates.append(Path.home() / "DeerShare")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise SystemExit(
        f"Unable to locate shared storage root; checked: {', '.join(str(c) for c in candidates)}"
    )


def resolve_queue_dir(queue_root: Path, clip: str) -> Optional[Path]:
    candidate = Path(clip)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    dest = queue_root / clip
    return dest if dest.exists() else None


def iter_queue_dirs(queue_root: Path) -> List[Path]:
    if not queue_root.exists():
        return []
    return sorted([p for p in queue_root.iterdir() if p.is_dir()])


def load_queue_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def build_decision(
    clip_id: str,
    queue_dir: Path,
    queue_meta: Dict[str, Any],
    negative_cap: int,
    negative_used: int,
) -> ClipDecision:
    enrichment = queue_meta.get("enrichment")
    if not isinstance(enrichment, dict):
        return ClipDecision(
            clip_id=clip_id,
            queue_dir=queue_dir,
            triage=None,
            category=None,
            lighting=None,
            decision="hold",
            reason="missing_enrichment",
        )

    status = normalize_text(enrichment.get("status"))
    if status != "complete":
        return ClipDecision(
            clip_id=clip_id,
            queue_dir=queue_dir,
            triage=normalize_text(enrichment.get("triage")),
            category=normalize_text(enrichment.get("category")),
            lighting=normalize_text(enrichment.get("lighting")),
            decision="hold",
            reason=f"enrichment_status_{status or 'missing'}",
        )

    triage = normalize_text(enrichment.get("triage"))
    category = normalize_text(enrichment.get("category"))
    lighting = normalize_text(enrichment.get("lighting"))
    tags = normalize_tags(enrichment.get("tags"))

    if triage not in {"keep", "maybe", "reject"}:
        return ClipDecision(
            clip_id=clip_id,
            queue_dir=queue_dir,
            triage=triage,
            category=category,
            lighting=lighting,
            decision="hold",
            reason="unknown_triage",
            tags=tags,
        )

    if lighting not in {"day", "night"}:
        return ClipDecision(
            clip_id=clip_id,
            queue_dir=queue_dir,
            triage=triage,
            category=category,
            lighting=lighting,
            decision="hold",
            reason="missing_lighting",
            tags=tags,
        )

    if triage == "keep":
        if category in ALLOWED_CATEGORIES:
            return ClipDecision(
                clip_id=clip_id,
                queue_dir=queue_dir,
                triage=triage,
                category=category,
                lighting=lighting,
                decision="promote",
                reason="triage_keep",
                tags=tags,
            )
        return ClipDecision(
            clip_id=clip_id,
            queue_dir=queue_dir,
            triage=triage,
            category=category,
            lighting=lighting,
            decision="hold",
            reason="category_not_promotable",
            tags=tags,
        )

    if triage == "reject":
        if category == "false_positive" and any(tag in {"snow", "rain", "flies"} for tag in tags):
            if negative_used < negative_cap:
                return ClipDecision(
                    clip_id=clip_id,
                    queue_dir=queue_dir,
                    triage=triage,
                    category=category,
                    lighting=lighting,
                    decision="negative_mine",
                    reason="snow_precip_negative",
                    tags=tags,
                )
            return ClipDecision(
                clip_id=clip_id,
                queue_dir=queue_dir,
                triage=triage,
                category=category,
                lighting=lighting,
                decision="reject",
                reason="negative_cap_reached",
                tags=tags,
            )
        return ClipDecision(
            clip_id=clip_id,
            queue_dir=queue_dir,
            triage=triage,
            category=category,
            lighting=lighting,
            decision="reject",
            reason="triage_reject",
            tags=tags,
        )

    return ClipDecision(
        clip_id=clip_id,
        queue_dir=queue_dir,
        triage=triage,
        category=category,
        lighting=lighting,
        decision="hold",
        reason="triage_maybe",
        tags=tags,
    )


def gather_segments(queue_dir: Path) -> List[str]:
    segments: List[str] = []
    for item in queue_dir.iterdir():
        if item.is_file() and item.suffix.lower() in {".mkv", ".mp4"}:
            segments.append(item.name)
    return sorted(segments)


def copy_queue_dir(queue_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=False)
    try:
        for item in queue_dir.iterdir():
            if item.name == ".enriching":
                continue
            target = dest_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
    except Exception:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise


def load_manifest(golden_root: Path) -> Dict[str, Any]:
    manifest_path = golden_root / "manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            return payload
    return {
        "version": 1,
        "description": "Protected archive of high-value wildlife clips",
        "created": datetime.now(UTC).strftime("%Y-%m-%d"),
        "clips": [],
    }


def save_manifest(golden_root: Path, manifest: Dict[str, Any]) -> None:
    manifest_path = golden_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def build_manifest_entry(
    clip: ClipDecision,
    queue_meta: Dict[str, Any],
    dest_dir: Path,
    golden_root: Path,
    run_id: str,
    agent_name: str,
) -> Dict[str, Any]:
    queue_info = queue_meta.get("queue") if isinstance(queue_meta.get("queue"), dict) else {}
    enrichment = queue_meta.get("enrichment") if isinstance(queue_meta.get("enrichment"), dict) else {}
    meta_summary = queue_meta.get("meta_summary") if isinstance(queue_meta.get("meta_summary"), dict) else {}

    entry: Dict[str, Any] = {
        "id": clip.clip_id,
        "category": clip.category,
        "lighting": clip.lighting,
        "decision": clip.decision,
        "path": str(dest_dir.relative_to(golden_root)),
        "dest_dir": str(dest_dir),
        "segments": gather_segments(clip.queue_dir),
        "source_path": str(clip.queue_dir),
        "queued_from": queue_info.get("source_path"),
        "queued_by": queue_info.get("queued_by"),
        "queued_at": queue_info.get("queued_at"),
        "queued_note": queue_info.get("note_suffix"),
        "tc_run_id": run_id,
        "tc_agent": agent_name,
        "tags": clip.tags,
        "review_confidence": coerce_float(enrichment.get("review_confidence")),
        "review_reason": enrichment.get("review_reason"),
        "snow_score": enrichment.get("snow_score"),
        "snow_reason": enrichment.get("snow_reason"),
        "meta_summary": meta_summary,
        "added_by": os.environ.get("USER", "unknown"),
        "added_date": datetime.now(UTC).strftime("%Y-%m-%d"),
    }
    return entry


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
                "task_description": "Training Coordinator",
            },
        )
    except Exception as exc:
        errors.append(f"Failed to register agent: {exc}")


def parse_recipients(value: Optional[str], fallback: str) -> List[str]:
    if value:
        recipients = [item.strip() for item in value.split(",") if item.strip()]
        if recipients:
            return recipients
    return [fallback]


def send_summary_safe(
    client: Optional[MCPClient],
    project_key: str,
    agent_name: str,
    recipients: List[str],
    thread_id: str,
    summary: RunSummary,
    skip_mail: bool,
    errors: List[str],
) -> None:
    if skip_mail or client is None:
        return

    body_lines = [
        f"Run: {summary.run_id}",
        f"Status: {summary.status}",
        f"Started: {summary.started_at}",
        f"Finished: {summary.finished_at}",
        f"Queue dir: {summary.queue_dir}",
        f"Golden dir: {summary.golden_root}",
        "",
        "Counts:",
    ]
    for key, value in summary.counts.items():
        body_lines.append(f"- {key}: {value}")
    if summary.errors:
        body_lines.append("")
        body_lines.append("Errors:")
        for err in summary.errors[:10]:
            body_lines.append(f"- {err}")
        if len(summary.errors) > 10:
            body_lines.append(f"- ... {len(summary.errors) - 10} more")
    if summary.log_path:
        body_lines.append("")
        body_lines.append(f"Run log: {summary.log_path}")

    try:
        client.call_tool(
            "send_message",
            {
                "project_key": project_key,
                "sender_name": agent_name,
                "to": recipients,
                "subject": f"Training Coordinator run {summary.run_id}",
                "body_md": "\n".join(body_lines),
                "thread_id": thread_id,
            },
        )
    except Exception as exc:
        errors.append(f"Failed to send summary: {exc}")


def run_coordinator(args: argparse.Namespace) -> RunSummary:
    started_at = datetime.now(UTC)
    repo_root = get_repo_root()
    share_root = resolve_share_root(args.share_root)
    queue_root = Path(args.queue_dir)
    golden_root = share_root / "golden_clips"

    if not golden_root.exists():
        raise SystemExit(f"Golden clips directory not found: {golden_root}")

    run_id = args.run_id or datetime.now(UTC).strftime("TC-%Y%m%d-%H%M%S")
    errors: List[str] = []
    decisions: List[ClipDecision] = []

    negative_used = 0

    if args.only_clip:
        queue_dirs = []
        resolved = resolve_queue_dir(queue_root, args.only_clip)
        if resolved:
            queue_dirs = [resolved]
        else:
            errors.append(f"Queue clip not found: {args.only_clip}")
    else:
        queue_dirs = iter_queue_dirs(queue_root)

    if args.limit is not None:
        queue_dirs = queue_dirs[: args.limit]

    manifest: Optional[Dict[str, Any]] = None
    manifest_changed = False
    if not args.dry_run:
        manifest = load_manifest(golden_root)

    for queue_dir in queue_dirs:
        clip_id = queue_dir.name
        meta_path = queue_dir / "queue_meta.json"
        meta = load_queue_meta(meta_path)
        if meta is None:
            decisions.append(
                ClipDecision(
                    clip_id=clip_id,
                    queue_dir=queue_dir,
                    triage=None,
                    category=None,
                    lighting=None,
                    decision="hold",
                    reason="missing_or_invalid_queue_meta",
                )
            )
            continue

        clip_decision = build_decision(clip_id, queue_dir, meta, args.negative_cap, negative_used)
        clip_decision.queue_meta = meta

        dest_dir: Optional[Path] = None
        if clip_decision.decision == "promote":
            dest_dir = golden_root / clip_decision.lighting / clip_decision.category / clip_id
        elif clip_decision.decision == "negative_mine":
            dest_dir = golden_root / "negatives" / "snow" / clip_id

        clip_decision.dest_dir = dest_dir

        if dest_dir and dest_dir.exists():
            clip_decision.decision = "hold"
            clip_decision.reason = "destination_exists"
            clip_decision.errors.append(str(dest_dir))
            decisions.append(clip_decision)
            continue

        if dest_dir and not args.dry_run:
            try:
                copy_queue_dir(queue_dir, dest_dir)
            except Exception as exc:
                clip_decision.decision = "hold"
                clip_decision.reason = "copy_failed"
                clip_decision.errors.append(str(exc))
                decisions.append(clip_decision)
                continue

            if manifest is not None:
                entry = build_manifest_entry(
                    clip_decision,
                    meta,
                    dest_dir,
                    golden_root,
                    run_id,
                    args.agent_name,
                )
                manifest.setdefault("clips", []).append(entry)
                manifest_changed = True

        if clip_decision.decision == "negative_mine":
            if args.dry_run or dest_dir is None or dest_dir.exists():
                negative_used += 1

        if clip_decision.decision in {"promote", "negative_mine", "reject"}:
            if not args.no_delete and not args.dry_run:
                try:
                    shutil.rmtree(queue_dir)
                except Exception as exc:
                    clip_decision.errors.append(f"Failed to delete queue dir: {exc}")

        decisions.append(clip_decision)

    if manifest_changed and manifest is not None and not args.dry_run:
        manifest["updated"] = format_utc(datetime.now(UTC))
        save_manifest(golden_root, manifest)

    finished_at = datetime.now(UTC)

    counts = {
        "total": len(decisions),
        "promoted": sum(1 for d in decisions if d.decision == "promote"),
        "negatives": sum(1 for d in decisions if d.decision == "negative_mine"),
        "rejected": sum(1 for d in decisions if d.decision == "reject"),
        "held": sum(1 for d in decisions if d.decision == "hold"),
    }

    summary_errors = list(errors)
    for decision in decisions:
        for err in decision.errors:
            summary_errors.append(f"{decision.clip_id}: {err}")

    summary = RunSummary(
        run_id=run_id,
        status="complete",
        started_at=format_utc(started_at),
        finished_at=format_utc(finished_at),
        queue_dir=str(queue_root),
        share_root=str(share_root),
        golden_root=str(golden_root),
        counts=counts,
        errors=summary_errors,
        decisions=[
            {
                "clip_id": d.clip_id,
                "queue_dir": str(d.queue_dir),
                "triage": d.triage,
                "category": d.category,
                "lighting": d.lighting,
                "decision": d.decision,
                "reason": d.reason,
                "dest_dir": str(d.dest_dir) if d.dest_dir else None,
                "tags": d.tags,
                "errors": d.errors,
            }
            for d in decisions
        ],
    )

    if not args.dry_run:
        logs_dir = repo_root / DEFAULT_LOGS_DIRNAME
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{run_id}.json"
        log_path.write_text(json.dumps(summary.__dict__, indent=2))
        latest_path = logs_dir / "training-coordinator-latest.json"
        latest_path.write_text(json.dumps(summary.__dict__, indent=2))
        summary.log_path = str(log_path)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", default=DEFAULT_QUEUE_DIR, help="Training queue directory")
    parser.add_argument("--share-root", help="Override share root (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without modifying files")
    parser.add_argument("--limit", type=int, help="Limit number of queue clips to process")
    parser.add_argument("--only-clip", help="Process only a specific queue clip directory")
    parser.add_argument("--no-delete", action="store_true", help="Do not delete queue dirs after decisions")
    parser.add_argument("--run-id", help="Override run id (default: TC-YYYYMMDD-HHMMSS)")
    parser.add_argument("--negative-cap", type=int, default=DEFAULT_NEGATIVE_CAP, help="Negative mining cap")

    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME, help="MCP agent name")
    parser.add_argument("--thread-id", default=DEFAULT_THREAD_ID, help="MCP thread id")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="MCP server URL")
    parser.add_argument("--recipients", help="Comma-separated recipients for MCP summary")
    parser.add_argument("--skip-mail", action="store_true", help="Skip MCP mail reporting")

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    token = os.environ.get("TRAINING_COORDINATOR_MCP_TOKEN")
    client: Optional[MCPClient] = MCPClient(args.mcp_url, timeout=10, token=token)
    if not client.ping():
        client = None

    errors: List[str] = []
    register_agent_safe(client, str(get_repo_root()), args.agent_name, args.skip_mail, errors)

    summary = run_coordinator(args)

    recipients = parse_recipients(
        args.recipients or os.environ.get("TRAINING_COORDINATOR_TO"),
        args.agent_name,
    )
    send_summary_safe(
        client=client,
        project_key=str(get_repo_root()),
        agent_name=args.agent_name,
        recipients=recipients,
        thread_id=args.thread_id,
        summary=summary,
        skip_mail=args.skip_mail,
        errors=errors,
    )

    if errors:
        for err in errors:
            log(f"WARN: {err}")

    log(
        "Finished TC run "
        f"{summary.run_id} (promoted={summary.counts['promoted']}, "
        f"negatives={summary.counts['negatives']}, rejected={summary.counts['rejected']}, "
        f"held={summary.counts['held']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
