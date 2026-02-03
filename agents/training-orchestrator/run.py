#!/usr/bin/env python3
"""
Training loop orchestrator.

Runs Training Coordinator -> Trainer -> Evaluator -> Deployer when queue threshold is met.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
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
AGENTS_DIR = Path(__file__).resolve().parents[1]
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

try:
    from env_loader import load_env_file

    load_env_file()
except Exception:
    pass

from training_loop_utils import load_json, resolve_share_root


DEFAULT_AGENT_NAME = os.environ.get("TRAINING_LOOP_AGENT", "BrightCedar")
DEFAULT_THREAD_ID = os.environ.get("TRAINING_LOOP_THREAD", "TRAINING-LOOP")
DEFAULT_MCP_URL = os.environ.get("TRAINING_LOOP_MCP_URL", "http://127.0.0.1:8765/mcp/")
DEFAULT_LOGS_DIRNAME = "runs/logs/training_loop"


@dataclass
class StepResult:
    name: str
    status: str
    return_code: int
    log_path: Optional[str] = None


@dataclass
class LoopSummary:
    run_id: str
    status: str
    started_at: str
    finished_at: str
    queue_dir: str
    queue_count: int
    threshold: int
    steps: List[Dict[str, Any]]
    errors: List[str] = field(default_factory=list)
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
    print(f"[Orchestrator {timestamp}] {message}", flush=True)


def format_utc(ts: datetime) -> str:
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_training_config() -> Dict[str, Any]:
    config_path = REPO_ROOT / "configs" / "training_loop.json"
    payload = load_json(config_path)
    return payload or {}


def hash_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.md5(path.read_bytes()).hexdigest()


def golden_frames_stale(dataset_root: Path, manifest_path: Path, golden_manifest: Path) -> bool:
    if not dataset_root.exists():
        return True
    if not (dataset_root / "day.yaml").exists() or not (dataset_root / "night.yaml").exists():
        return True
    meta_path = dataset_root / ".build_meta.json"
    meta = load_json(meta_path)
    if not meta:
        return True
    if meta.get("manifest_hash") != hash_file(manifest_path):
        return True
    if golden_manifest.exists():
        meta_mtime = meta.get("golden_clips_manifest_mtime")
        try:
            meta_mtime_val = float(meta_mtime) if meta_mtime is not None else 0.0
        except (TypeError, ValueError):
            meta_mtime_val = 0.0
        if meta_mtime_val < golden_manifest.stat().st_mtime:
            return True
    return False


def count_queue(queue_dir: Path) -> int:
    if not queue_dir.exists():
        return 0
    return sum(1 for child in queue_dir.iterdir() if child.is_dir())


def run_step(command: List[str]) -> int:
    result = subprocess.run(command)
    return result.returncode


def find_latest_log(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return str(path)


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
                "task_description": "Training Loop Orchestrator",
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
    summary: LoopSummary,
    skip_mail: bool,
    errors: List[str],
) -> None:
    if skip_mail or client is None:
        return

    body_lines = [
        f"Run: {summary.run_id}",
        f"Status: {summary.status}",
        f"Queue count: {summary.queue_count} (threshold {summary.threshold})",
        "",
        "Steps:",
    ]
    for step in summary.steps:
        body_lines.append(f"- {step.get('name')}: {step.get('status')} (rc={step.get('return_code')})")
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
                "subject": f"Training loop {summary.run_id}",
                "body_md": "\n".join(body_lines),
                "thread_id": thread_id,
            },
        )
    except Exception as exc:
        errors.append(f"Failed to send summary: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share-root", help="Override share root (default: auto-detect)")
    parser.add_argument("--queue-dir", default="/srv/deer-share/training_queue", help="Queue directory")
    parser.add_argument("--threshold", type=int, help="Queue threshold to trigger loop")
    parser.add_argument("--dry-run", action="store_true", help="Run all steps in dry-run mode")
    parser.add_argument("--force", action="store_true", help="Force run regardless of queue size")
    parser.add_argument("--execute-train", action="store_true", help="Force training execution in trainer step")

    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME, help="MCP agent name")
    parser.add_argument("--thread-id", default=DEFAULT_THREAD_ID, help="MCP thread id")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="MCP server URL")
    parser.add_argument("--recipients", help="Comma-separated recipients for MCP summary")
    parser.add_argument("--skip-mail", action="store_true", help="Skip MCP mail reporting")

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    config = load_training_config()
    threshold = args.threshold if args.threshold is not None else int(config.get("queue_threshold", 50))

    queue_dir = Path(args.queue_dir)
    queue_count = count_queue(queue_dir)

    share_root = resolve_share_root(args.share_root)
    golden_frames_root = config.get("golden_frames_root", "golden_frames_v8")
    dataset_root = share_root / str(golden_frames_root)
    manifest_path = REPO_ROOT / "configs" / "golden_v8_manifest.json"
    golden_manifest = share_root / "golden_clips" / "manifest.json"

    run_id = datetime.now(UTC).strftime("LOOP-%Y%m%d-%H%M%S")
    started_at = datetime.now(UTC)
    errors: List[str] = []
    steps: List[StepResult] = []

    if queue_count < threshold and not args.force:
        status = "skipped"
    else:
        status = "running"

    execute_train = args.execute_train or bool(config.get("auto_train", True))

    if status == "running":
        tc_cmd = [sys.executable, "agents/training-coordinator/run.py"]
        if args.dry_run:
            tc_cmd.append("--dry-run")
        rc = run_step(tc_cmd)
        steps.append(
            StepResult(
                name="training-coordinator",
                status="ok" if rc == 0 else "failed",
                return_code=rc,
                log_path=find_latest_log(REPO_ROOT / "runs/logs/training_coordinator/training-coordinator-latest.json"),
            )
        )
        if rc != 0:
            status = "failed"

    if status == "running":
        if golden_frames_stale(dataset_root, manifest_path, golden_manifest):
            manifest_cmd = [sys.executable, "scripts/build_golden_v8_manifest.py"]
            frames_cmd = [sys.executable, "scripts/build_golden_v8_frames.py"]
            if args.dry_run:
                manifest_cmd.append("--dry-run")
                frames_cmd.append("--dry-run")
            rc = run_step(manifest_cmd)
            steps.append(
                StepResult(
                    name="golden-manifest",
                    status="ok" if rc == 0 else "failed",
                    return_code=rc,
                    log_path=None,
                )
            )
            if rc != 0:
                status = "failed"
            if status == "running":
                rc = run_step(frames_cmd)
                steps.append(
                    StepResult(
                        name="golden-frames",
                        status="ok" if rc == 0 else "failed",
                        return_code=rc,
                        log_path=None,
                    )
                )
                if rc != 0:
                    status = "failed"
        else:
            steps.append(
                StepResult(
                    name="golden-manifest",
                    status="skipped",
                    return_code=0,
                    log_path=None,
                )
            )
            steps.append(
                StepResult(
                    name="golden-frames",
                    status="skipped",
                    return_code=0,
                    log_path=None,
                )
            )

    if status == "running":
        trainer_cmd = [sys.executable, "agents/trainer/run.py"]
        if args.dry_run:
            trainer_cmd.append("--dry-run")
        elif execute_train:
            trainer_cmd.append("--execute")
        rc = run_step(trainer_cmd)
        steps.append(
            StepResult(
                name="trainer",
                status="ok" if rc == 0 else "failed",
                return_code=rc,
                log_path=find_latest_log(REPO_ROOT / "runs/logs/trainer/trainer-latest.json"),
            )
        )
        if rc != 0:
            status = "failed"

    if status == "running":
        eval_cmd = [sys.executable, "agents/evaluator/run.py"]
        if args.dry_run:
            eval_cmd.append("--dry-run")
        rc = run_step(eval_cmd)
        steps.append(
            StepResult(
                name="evaluator",
                status="ok" if rc == 0 else "failed",
                return_code=rc,
                log_path=find_latest_log(REPO_ROOT / "runs/logs/evaluator/evaluator-latest.json"),
            )
        )
        if rc != 0:
            status = "failed"

    if status == "running":
        deploy_cmd = [sys.executable, "agents/deployer/run.py"]
        if args.dry_run:
            deploy_cmd.append("--dry-run")
        rc = run_step(deploy_cmd)
        steps.append(
            StepResult(
                name="deployer",
                status="ok" if rc == 0 else "failed",
                return_code=rc,
                log_path=find_latest_log(REPO_ROOT / "runs/logs/deployer/deployer-latest.json"),
            )
        )
        if rc != 0:
            status = "failed"

    if status == "running":
        status = "complete"

    finished_at = datetime.now(UTC)

    summary = LoopSummary(
        run_id=run_id,
        status=status,
        started_at=format_utc(started_at),
        finished_at=format_utc(finished_at),
        queue_dir=str(queue_dir),
        queue_count=queue_count,
        threshold=threshold,
        steps=[step.__dict__ for step in steps],
        errors=errors,
    )

    logs_dir = REPO_ROOT / DEFAULT_LOGS_DIRNAME
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_id}.json"
    log_path.write_text(json.dumps(summary.__dict__, indent=2))
    latest_path = logs_dir / "loop-latest.json"
    latest_path.write_text(json.dumps(summary.__dict__, indent=2))
    summary.log_path = str(log_path)

    token = os.environ.get("TRAINING_LOOP_MCP_TOKEN")
    client: Optional[MCPClient] = MCPClient(args.mcp_url, timeout=10, token=token)
    if not client.ping():
        client = None

    register_agent_safe(client, str(REPO_ROOT), args.agent_name, args.skip_mail, errors)

    recipients = parse_recipients(args.recipients or os.environ.get("TRAINING_LOOP_TO"), args.agent_name)
    send_summary_safe(
        client=client,
        project_key=str(REPO_ROOT),
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

    log(f"Finished loop {summary.run_id} (status={summary.status})")
    return 0 if summary.status in {"complete", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
