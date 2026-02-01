#!/usr/bin/env python3
"""
Day Watchman reporter.

Uses headless Codex to analyze the latest snapshot + digest, then sends the final
report via MCP Agent Mail. Falls back to a deterministic report if Codex fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import urllib.request

TIMEZONE = ZoneInfo("America/Chicago")

DEFAULT_AGENT_NAME = os.environ.get("DAYWATCHMAN_AGENT", "BrightHeron")
DEFAULT_THREAD_ID = os.environ.get("DAYWATCHMAN_THREAD", "DAY-WATCHMAN")
DEFAULT_MCP_URL = os.environ.get("DAYWATCHMAN_MCP_URL", "http://127.0.0.1:8765/mcp/")
DEFAULT_TIMEOUT = int(os.environ.get("DAYWATCHMAN_REPORT_TIMEOUT", "420"))


class MCPClient:
    def __init__(self, base_url: str, timeout: int = 10, token: Optional[str] = None) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.token = token
        self._next_id = 1

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


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_codex_binary() -> Optional[str]:
    candidates = []
    env_candidate = os.environ.get("CODEX_BIN")
    if env_candidate:
        candidates.append(env_candidate)
    npm_global = Path.home() / ".npm-global" / "bin" / "codex"
    if npm_global.exists():
        candidates.append(str(npm_global))
    local_bin = Path.home() / ".local" / "bin" / "codex"
    if local_bin.exists():
        candidates.append(str(local_bin))
    candidates.extend(["codex", "codex-cli"])

    for candidate in candidates:
        if not candidate:
            continue
        expanded = os.path.expanduser(candidate)
        path = Path(expanded)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        resolved = shutil.which(expanded)
        if resolved:
            return resolved
    return None


def parse_recipients(value: Optional[str], fallback: str) -> List[str]:
    if value:
        recipients = [item.strip() for item in value.split(",") if item.strip()]
        if recipients:
            return recipients
    return [fallback]


def load_snapshot(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def build_analysis_path(snapshot: Dict[str, Any], logs_dir: Path) -> Path:
    date_str = snapshot.get("date")
    if not date_str:
        date_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
    return logs_dir / f"day-watchman-analysis-{date_str}.md"


def build_report_path(snapshot: Dict[str, Any], logs_dir: Path) -> Path:
    date_str = snapshot.get("date")
    if not date_str:
        date_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
    return logs_dir / f"day-watchman-report-{date_str}.md"


def slugify_project_key(project_key: str) -> str:
    lowered = project_key.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug


def mailbox_messages_dir(repo_root: Path) -> Path:
    slug = slugify_project_key(str(repo_root))
    return Path.home() / ".mcp_agent_mail_git_mailbox_repo" / "projects" / slug / "messages"


def parse_message_header(path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = path.read_text()
    except Exception:
        return None
    if not text.startswith("---json"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    header = parts[1].lstrip("json").strip()
    try:
        return json.loads(header)
    except json.JSONDecodeError:
        return None


def was_patrol_message_sent(
    repo_root: Path,
    since_iso: str,
    agent_name: str,
    thread_id: str,
    subject: str = "Patrol Complete",
) -> bool:
    messages_dir = mailbox_messages_dir(repo_root)
    if not messages_dir.exists():
        return False
    try:
        since_dt = datetime.fromisoformat(since_iso)
    except ValueError:
        since_dt = datetime.now(TIMEZONE) - timedelta(hours=2)

    month_dir = messages_dir / since_dt.strftime("%Y") / since_dt.strftime("%m")
    candidates = []
    if month_dir.exists():
        candidates.extend(month_dir.glob("*__patrol-complete__*.md"))
    if not candidates:
        candidates.extend(messages_dir.rglob("*__patrol-complete__*.md"))

    for path in sorted(candidates, reverse=True):
        header = parse_message_header(path)
        if not header:
            continue
        if header.get("subject") != subject:
            continue
        if header.get("from") != agent_name:
            continue
        if header.get("thread_id") != thread_id:
            continue
        created = header.get("created")
        if not created:
            continue
        try:
            created_dt = datetime.fromisoformat(created)
        except ValueError:
            continue
        if created_dt >= since_dt:
            return True
    return False


def build_codex_prompt(
    snapshot_path: Path,
    digest_path: Path,
    analysis_path: Path,
    report_path: Path,
    repo_root: Path,
    agent_name: str,
    recipients: List[str],
    thread_id: str,
) -> str:
    return (
        "You are the Day Watchman reporter. Read the snapshot JSON and digest, "
        "investigate anything concerning, and then send the final report via MCP Agent Mail.\n\n"
        f"Snapshot JSON: {snapshot_path}\n"
        f"Digest: {digest_path}\n"
        f"Analysis output: {analysis_path}\n\n"
        f"Report body output: {report_path}\n\n"
        "Steps:\n"
        "1) Read the snapshot JSON and digest markdown.\n"
        "2) Write a short analysis to the analysis output path. Call out anomalies "
        "(pipeline unhealthy, backlog HIGH/CRITICAL, errors, or unusual counts) and "
        "suggest next actions if needed.\n"
        "3) Write the final report body to the report body output path.\n"
        "4) Send a final report with subject 'Patrol Complete' using "
        "mcp__mcp_agent_mail__send_message.\n"
        "   - project_key: {repo_root}\n"
        "   - sender_name: {agent_name}\n"
        "   - to: {recipients}\n"
        "   - thread_id: {thread_id}\n"
        "   - body_md: include health, backlog, pushover status + capture lag "
        "(use pushover.status and pushover.lag_human when present), "
        "detections, frame exam, training queue contribution and total "
        "(use training_queue.queued_count and training_queue.queue_size from the "
        "snapshot when available, e.g., 'Training Queue: +3 this run, 45 total'), "
        "and the digest path.\n\n"
        "5) Fetch inbox for {agent_name} with mcp__mcp_agent_mail__fetch_inbox and "
        "acknowledge any messages with ack_required=true using "
        "mcp__mcp_agent_mail__acknowledge_message.\n\n"
        "You may use shell commands, but only to read the snapshot/digest and write the "
        "analysis/report files. Do not run long or destructive commands.\n"
    ).format(
        repo_root=repo_root,
        agent_name=agent_name,
        recipients=recipients,
        thread_id=thread_id,
    )


def run_codex_report(
    codex_binary: str,
    prompt: str,
    repo_root: Path,
    timeout: int,
    analysis_path: Path,
) -> int:
    cmd = [
        codex_binary,
        "-a",
        "never",
        "exec",
        "-s",
        "workspace-write",
        "--skip-git-repo-check",
        "-",
    ]
    env = os.environ.copy()
    env["AGENT_MAIL_PROJECT"] = str(repo_root)
    env["AGENT_MAIL_AGENT"] = env.get("DAYWATCHMAN_AGENT", DEFAULT_AGENT_NAME)
    start = time.time()
    log(f"Running Codex reporter (timeout {timeout}s)")
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            input=prompt,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        log("Codex reporter timed out", "WARN")
        return 124

    elapsed = time.time() - start
    log(f"Codex reporter finished in {elapsed:.1f}s with exit code {result.returncode}")

    if result.returncode != 0:
        combined = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        combined = combined.strip()
        if combined:
            log("Codex reporter output (tail):", "WARN")
            for line in combined.splitlines()[-8:]:
                log(line, "WARN")

    if not analysis_path.exists() and result.stdout:
        analysis_path.write_text(result.stdout)

    return result.returncode


def format_disk_summary(snapshot: Dict[str, Any]) -> str:
    parts = []
    for report in snapshot.get("disk", []):
        path = report.get("path", "?")
        status = report.get("status", "UNKNOWN")
        used_pct = report.get("used_pct", 0.0)
        parts.append(f"{path} {status} ({used_pct:.1f}% used)")
    return "; ".join(parts) if parts else "no disk data"


def build_fallback_body(snapshot: Dict[str, Any], agent_name: str, digest_path: Path) -> str:
    backlog = snapshot.get("backlog", {})
    analysis_total = backlog.get("analysis_count", 0)
    pending_count = backlog.get("pending_count", backlog.get("count", 0))
    detections = snapshot.get("detections", {})
    frame_exam = snapshot.get("frame_exam") or {}
    pipeline = snapshot.get("pipeline", {})
    pushover = snapshot.get("pushover", {})
    training_queue = snapshot.get("training_queue", {})
    queue_size = training_queue.get("queue_size")
    if queue_size is None:
        queue_size = len(training_queue.get("queued", []))
    queued_count = training_queue.get("queued_count")
    if queued_count is None:
        queued_count = len(training_queue.get("queued", []))
    errors = snapshot.get("errors", [])
    pushover_status = pushover.get("status", "UNKNOWN")
    pushover_sent = pushover.get("last_sent_local") or pushover.get("last_sent_utc")
    pushover_lag = pushover.get("lag_human")
    if pushover_lag is None and pushover.get("lag_seconds") is not None:
        pushover_lag = f"{pushover['lag_seconds']:.0f}s"
    if not pushover_sent:
        pushover_sent = "unknown"
    if not pushover_lag:
        pushover_lag = "n/a"

    frame_summary = "No events to examine"
    if frame_exam:
        frame_summary = (
            f"{frame_exam.get('segment')}: {frame_exam.get('assessment')} "
            f"({frame_exam.get('movement')})"
        )

    body_lines = [
        "## Day Watchman Patrol Complete (Deterministic Fallback)",
        "",
        f"Health: {format_disk_summary(snapshot)}; Pipeline {pipeline.get('summary', 'UNKNOWN')}",
        f"Pushover: {pushover_status} (last {pushover_sent}, lag {pushover_lag})",
        (
            f"Backlog: {pending_count} pending of {analysis_total} "
            f"({backlog.get('status', 'UNKNOWN')})"
        ),
        f"Detections: {detections.get('total_segments', 0)} segments",
        f"Frame Exam: {frame_summary}",
        f"Training Queue: +{queued_count} this run, {queue_size} total",
        "",
        f"Digest: {digest_path}",
        "",
        f"-- {agent_name} @ {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}",
    ]

    if errors:
        body_lines.insert(8, f"Errors: {len(errors)} (see digest)")

    if pushover_status == "ERROR" and pushover.get("last_error"):
        body_lines.insert(4, f"Pushover error: {pushover.get('last_error')}")

    diagnosis = backlog.get("diagnosis")
    if diagnosis:
        body_lines.append("")
        body_lines.append("Backlog diagnosis:")
        body_lines.append(f"- Detector: {diagnosis.get('detector_status')}")
        body_lines.append(f"- Last prune: {diagnosis.get('last_prune')}")
        for fix in diagnosis.get("proposed_fixes", []):
            body_lines.append(f"- Fix: {fix}")

    return "\n".join(body_lines)


def write_fallback_analysis(
    analysis_path: Path,
    snapshot: Dict[str, Any],
    digest_path: Path,
) -> None:
    backlog = snapshot.get("backlog", {})
    analysis_total = backlog.get("analysis_count", 0)
    pending_count = backlog.get("pending_count", backlog.get("count", 0))
    lines = [
        "# Day Watchman Analysis (Fallback)",
        "",
        "Codex reporter was unavailable. Using deterministic fallback summary.",
        "",
        f"Digest: {digest_path}",
        "",
        "Snapshot summary:",
        f"- Health: {format_disk_summary(snapshot)}",
        f"- Pipeline: {snapshot.get('pipeline', {}).get('summary', 'UNKNOWN')}",
        f"- Pushover: {snapshot.get('pushover', {}).get('status', 'UNKNOWN')}",
        f"- Backlog: {pending_count} pending of {analysis_total} "
        f"({backlog.get('status', 'UNKNOWN')})",
        f"- Detections: {snapshot.get('detections', {}).get('total_segments', 0)}",
    ]
    analysis_path.write_text("\n".join(lines))


def load_report_body(report_path: Path) -> Optional[str]:
    if not report_path.exists():
        return None
    try:
        text = report_path.read_text().strip()
    except Exception:
        return None
    return text or None


def send_fallback_report(
    repo_root: Path,
    snapshot: Dict[str, Any],
    digest_path: Path,
    agent_name: str,
    recipients: List[str],
    thread_id: str,
    mcp_url: str,
    mcp_token: Optional[str],
    body_override: Optional[str],
    report_path: Optional[Path],
    skip_mail: bool,
) -> int:
    if skip_mail:
        log("Skip-mail enabled: fallback report not sent")
        return 0
    client = MCPClient(mcp_url, timeout=10, token=mcp_token)
    body = body_override or build_fallback_body(snapshot, agent_name, digest_path)
    if report_path:
        try:
            report_path.write_text(body)
        except Exception:
            pass
    client.call_tool(
        "send_message",
        {
            "project_key": str(repo_root),
            "sender_name": agent_name,
            "to": recipients,
            "subject": "Patrol Complete",
            "body_md": body,
            "thread_id": thread_id,
            "importance": "normal",
            "ack_required": False,
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day Watchman reporter")
    parser.add_argument("--snapshot", type=str, help="Path to snapshot JSON")
    parser.add_argument("--digest", type=str, help="Path to digest markdown")
    parser.add_argument("--analysis", type=str, help="Path to analysis markdown")
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--thread-id", default=DEFAULT_THREAD_ID)
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL)
    parser.add_argument("--recipients", default=os.environ.get("DAYWATCHMAN_TO"))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--skip-mail", action="store_true", help="Skip Agent Mail reporting")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = get_repo_root()
    logs_dir = repo_root / "runs" / "logs" / "watchman"

    snapshot_path = Path(args.snapshot) if args.snapshot else logs_dir / "day-watchman-latest.json"
    if not snapshot_path.exists():
        log(f"Snapshot not found: {snapshot_path}", "ERROR")
        sys.exit(1)

    snapshot = load_snapshot(snapshot_path)
    digest_path = Path(args.digest) if args.digest else Path(snapshot.get("digest_path", ""))
    if not digest_path.is_absolute():
        digest_path = repo_root / digest_path

    analysis_path = Path(args.analysis) if args.analysis else build_analysis_path(snapshot, logs_dir)
    report_path = build_report_path(snapshot, logs_dir)

    recipients = parse_recipients(args.recipients, args.agent_name)
    prompt = build_codex_prompt(
        snapshot_path=snapshot_path,
        digest_path=digest_path,
        analysis_path=analysis_path,
        report_path=report_path,
        repo_root=repo_root,
        agent_name=args.agent_name,
        recipients=recipients,
        thread_id=args.thread_id,
    )

    codex_binary = discover_codex_binary()
    if not codex_binary:
        log("Codex CLI not found; using fallback report", "WARN")
        write_fallback_analysis(analysis_path, snapshot, digest_path)
        token = os.environ.get("DAYWATCHMAN_MCP_TOKEN")
        sys.exit(
            send_fallback_report(
                repo_root,
                snapshot,
                digest_path,
                args.agent_name,
                recipients,
                args.thread_id,
                args.mcp_url,
                token,
                body_override=None,
                report_path=report_path,
                skip_mail=args.skip_mail,
            )
        )

    analysis_before = analysis_path.stat().st_mtime if analysis_path.exists() else None
    report_before = report_path.stat().st_mtime if report_path.exists() else None
    exit_code = run_codex_report(
        codex_binary=codex_binary,
        prompt=prompt,
        repo_root=repo_root,
        timeout=args.timeout,
        analysis_path=analysis_path,
    )
    if exit_code == 0:
        analysis_updated = analysis_path.exists() and (
            analysis_before is None or analysis_path.stat().st_mtime > analysis_before
        )
        report_updated = report_path.exists() and (
            report_before is None or report_path.stat().st_mtime > report_before
        )
        message_sent = was_patrol_message_sent(
            repo_root=repo_root,
            since_iso=snapshot.get("timestamp", ""),
            agent_name=args.agent_name,
            thread_id=args.thread_id,
        )
        if analysis_updated and report_updated and message_sent:
            sys.exit(0)

        log("Codex report incomplete; sending deterministic report", "WARN")
        body_override = load_report_body(report_path)
        token = os.environ.get("DAYWATCHMAN_MCP_TOKEN")
        sys.exit(
            send_fallback_report(
                repo_root,
                snapshot,
                digest_path,
                args.agent_name,
                recipients,
                args.thread_id,
                args.mcp_url,
                token,
                body_override=body_override,
                report_path=report_path,
                skip_mail=args.skip_mail,
            )
        )

    log("Codex reporter failed; sending fallback report", "WARN")
    write_fallback_analysis(analysis_path, snapshot, digest_path)
    token = os.environ.get("DAYWATCHMAN_MCP_TOKEN")
    body_override = load_report_body(report_path)
    sys.exit(
        send_fallback_report(
            repo_root,
            snapshot,
            digest_path,
            args.agent_name,
            recipients,
            args.thread_id,
            args.mcp_url,
            token,
            body_override=body_override,
            report_path=report_path,
            skip_mail=args.skip_mail,
        )
    )


if __name__ == "__main__":
    main()
