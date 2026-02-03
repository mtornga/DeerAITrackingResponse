#!/usr/bin/env python3
"""
Deployer agent.

Promotes candidate models to live, updates DETECTOR_MODELS in .env, and runs
pipeline health checks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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


DEFAULT_AGENT_NAME = os.environ.get("DEPLOYER_AGENT", "WinterStone")
DEFAULT_THREAD_ID = os.environ.get("TRAINING_LOOP_THREAD", "TRAINING-LOOP")
DEFAULT_MCP_URL = os.environ.get("DEPLOYER_MCP_URL", "http://127.0.0.1:8765/mcp/")
DEFAULT_LOGS_DIRNAME = "runs/logs/deployer"


@dataclass
class DeploySummary:
    run_id: str
    status: str
    started_at: str
    finished_at: str
    candidate_models: Dict[str, str]
    target_models: Dict[str, str]
    env_updates: Dict[str, str]
    health_checks: Dict[str, Any]
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
    print(f"[Deployer {timestamp}] {message}", flush=True)


def format_utc(ts: datetime) -> str:
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_latest_evaluator_log() -> Optional[Dict[str, Any]]:
    latest = REPO_ROOT / "runs" / "logs" / "evaluator" / "evaluator-latest.json"
    return load_json(latest)


def update_env_file(path: Path, updates: Dict[str, str]) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    new_lines: List[str] = []
    seen: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue
        key, _ = line.split("=", 1)
        key = key.strip()
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            new_lines.append(line)

    for key, value in updates.items():
        if key not in seen:
            new_lines.append(f"{key}={value}")

    path.write_text("\n".join(new_lines) + "\n")


def extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    idx = text.find("{")
    if idx == -1:
        return None
    try:
        return json.loads(text[idx:])
    except json.JSONDecodeError:
        return None


def run_health_check(model_name: str, lighting: str, device: str) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/pipeline_health_check.py",
        "--json",
        "--models",
        model_name,
        "--lighting",
        lighting,
        "--device",
        device,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    payload = extract_json_payload(result.stdout + result.stderr) or {}
    payload["exit_code"] = result.returncode
    return payload


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
                "task_description": "Training Loop Deployer",
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
    summary: DeploySummary,
    skip_mail: bool,
    errors: List[str],
) -> None:
    if skip_mail or client is None:
        return

    body_lines = [
        f"Run: {summary.run_id}",
        f"Status: {summary.status}",
        "",
        "Candidate models:",
    ]
    for lighting, path in summary.candidate_models.items():
        body_lines.append(f"- {lighting}: {path}")
    body_lines.append("")
    body_lines.append("Target models:")
    for lighting, path in summary.target_models.items():
        body_lines.append(f"- {lighting}: {path}")

    try:
        client.call_tool(
            "send_message",
            {
                "project_key": project_key,
                "sender_name": agent_name,
                "to": recipients,
                "subject": f"Deployer run {summary.run_id}",
                "body_md": "\n".join(body_lines),
                "thread_id": thread_id,
            },
        )
    except Exception as exc:
        errors.append(f"Failed to send summary: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share-root", help="Override share root (default: auto-detect)")
    parser.add_argument("--run-id", help="Override run id (default: DP-YYYYMMDD-HHMMSS)")
    parser.add_argument("--evaluator-log", help="Path to evaluator log JSON")
    parser.add_argument("--device", default="cuda:0", help="Device for health checks")
    parser.add_argument("--dry-run", action="store_true", help="Skip file writes and health checks")

    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME, help="MCP agent name")
    parser.add_argument("--thread-id", default=DEFAULT_THREAD_ID, help="MCP thread id")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="MCP server URL")
    parser.add_argument("--recipients", help="Comma-separated recipients for MCP summary")
    parser.add_argument("--skip-mail", action="store_true", help="Skip MCP mail reporting")

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    share_root = resolve_share_root(args.share_root)
    models_root = share_root / "models"
    staging_root = models_root / "staging"

    run_id = args.run_id or datetime.now(UTC).strftime("DP-%Y%m%d-%H%M%S")
    started_at = datetime.now(UTC)
    errors: List[str] = []

    eval_log = load_json(Path(args.evaluator_log)) if args.evaluator_log else load_latest_evaluator_log()
    if not eval_log:
        raise SystemExit("Evaluator log not found")

    if not eval_log.get("gate_passed"):
        raise SystemExit("Evaluator gate not passed; deployment aborted")

    candidate_models = eval_log.get("candidate_models") if isinstance(eval_log.get("candidate_models"), dict) else {}
    if "day" not in candidate_models or "night" not in candidate_models:
        raise SystemExit("Candidate model paths missing from evaluator log")

    target_models: Dict[str, str] = {}
    env_updates: Dict[str, str] = {}
    health_checks: Dict[str, Any] = {}

    day_path = Path(candidate_models["day"])
    night_path = Path(candidate_models["night"])

    if not day_path.exists() or not night_path.exists():
        raise SystemExit("Candidate model files not found")

    if not args.dry_run:
        models_root.mkdir(parents=True, exist_ok=True)
        target_day = models_root / day_path.name
        target_night = models_root / night_path.name
        shutil.copy2(day_path, target_day)
        shutil.copy2(night_path, target_night)
        target_models = {"day": str(target_day), "night": str(target_night)}

        repo_models = REPO_ROOT / "models"
        repo_models.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_day, repo_models / target_day.name)
        shutil.copy2(target_night, repo_models / target_night.name)

        env_updates = {
            "DETECTOR_MODELS_DAY": target_day.name,
            "DETECTOR_MODELS_NIGHT": target_night.name,
            "DETECTOR_MODELS": f"{target_day.name},{target_night.name}",
        }

        env_path = REPO_ROOT / ".env"
        backup_path = env_path.with_suffix(f".bak-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}")
        if env_path.exists():
            shutil.copy2(env_path, backup_path)
        update_env_file(env_path, env_updates)

        health_checks["day"] = run_health_check(target_day.name, "day", args.device)
        health_checks["night"] = run_health_check(target_night.name, "night", args.device)

        if health_checks["day"].get("exit_code", 1) != 0 or health_checks["night"].get("exit_code", 1) != 0:
            if backup_path.exists():
                shutil.copy2(backup_path, env_path)
            errors.append("Health check failed; restored .env backup")

    status = "complete" if not errors else "failed"
    if args.dry_run:
        status = "dry_run"

    finished_at = datetime.now(UTC)

    summary = DeploySummary(
        run_id=run_id,
        status=status,
        started_at=format_utc(started_at),
        finished_at=format_utc(finished_at),
        candidate_models=candidate_models,
        target_models=target_models,
        env_updates=env_updates,
        health_checks=health_checks,
        errors=errors,
    )

    if not args.dry_run:
        logs_dir = REPO_ROOT / DEFAULT_LOGS_DIRNAME
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{run_id}.json"
        log_path.write_text(json.dumps(summary.__dict__, indent=2))
        latest_path = logs_dir / "deployer-latest.json"
        latest_path.write_text(json.dumps(summary.__dict__, indent=2))
        summary.log_path = str(log_path)

    token = os.environ.get("DEPLOYER_MCP_TOKEN")
    client: Optional[MCPClient] = MCPClient(args.mcp_url, timeout=10, token=token)
    if not client.ping():
        client = None

    register_agent_safe(client, str(REPO_ROOT), args.agent_name, args.skip_mail, errors)

    recipients = parse_recipients(args.recipients or os.environ.get("DEPLOYER_TO"), args.agent_name)
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

    log(f"Finished deployer run {summary.run_id} (status={summary.status})")
    return 0 if summary.status in {"complete", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
