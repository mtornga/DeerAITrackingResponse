#!/usr/bin/env python3
"""
Queue Enricher agent (skeleton).

Initial scaffold to register with MCP Agent Mail and emit a single
run summary. Clip enrichment logic will be layered in subsequent tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request


DEFAULT_AGENT_NAME = os.environ.get("QUEUE_ENRICHER_AGENT", "QuietHarbor")
DEFAULT_THREAD_ID = os.environ.get("QUEUE_ENRICHER_THREAD", "QUEUE-ENRICHER")
DEFAULT_MCP_URL = os.environ.get("QUEUE_ENRICHER_MCP_URL", "http://127.0.0.1:8765/mcp/")
DEFAULT_LOGS_DIRNAME = "runs/logs/queue_enricher"
DEFAULT_QUEUE_DIR = Path(os.environ.get("QUEUE_ENRICHER_QUEUE_DIR", "/srv/deer-share/training_queue"))


@dataclass
class RunSummary:
    status: str
    started_at: str
    finished_at: str
    message: str
    log_path: str
    inbox_messages: int = 0
    queued_clips: int = 0
    acknowledged: int = 0
    batch_results: int = 0
    batch_updated: int = 0
    batch_missing: int = 0
    batch_parse_errors: int = 0


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
    print(f"[QueueEnricher {timestamp}] {message}", flush=True)


def parse_recipients(value: Optional[str], fallback: str) -> List[str]:
    if value:
        recipients = [item.strip() for item in value.split(",") if item.strip()]
        if recipients:
            return recipients
    return [fallback]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_logs_dir(repo_root: Path) -> Path:
    return repo_root / DEFAULT_LOGS_DIRNAME


def format_utc(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
                "task_description": "Training Queue Enricher",
            },
        )
    except Exception as exc:
        errors.append(f"Failed to register agent: {exc}")


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


def fetch_inbox_safe(
    client: Optional[MCPClient],
    project_key: str,
    agent_name: str,
    skip_mail: bool,
    errors: List[str],
) -> List[Dict[str, Any]]:
    if skip_mail or client is None:
        return []
    try:
        raw = client.call_tool(
            "fetch_inbox",
            {
                "project_key": project_key,
                "agent_name": agent_name,
                "include_bodies": True,
                "limit": 10,
            },
        )
        return normalize_inbox_messages(raw, errors)
    except Exception as exc:
        errors.append(f"Failed to fetch inbox: {exc}")
        return []


def acknowledge_message_safe(
    client: Optional[MCPClient],
    project_key: str,
    agent_name: str,
    message_id: Optional[int],
    skip_mail: bool,
    errors: List[str],
) -> bool:
    if skip_mail or client is None or message_id is None:
        return False
    try:
        client.call_tool(
            "acknowledge_message",
            {
                "project_key": project_key,
                "agent_name": agent_name,
                "message_id": message_id,
            },
        )
        return True
    except Exception as exc:
        errors.append(f"Failed to acknowledge message {message_id}: {exc}")
        return False


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


def extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    fence = "```json"
    if fence in text:
        start = text.find(fence) + len(fence)
        end = text.find("```", start)
        if end != -1:
            snippet = text[start:end].strip()
            if snippet:
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_queue_payload(message: Dict[str, Any]) -> Tuple[List[str], Optional[str]]:
    body = message.get("body_md") or message.get("body") or ""
    payload = extract_json_payload(body)
    if not isinstance(payload, dict):
        return [], None
    clips = payload.get("clips") if isinstance(payload.get("clips"), list) else []
    clip_names = [clip for clip in clips if isinstance(clip, str)]
    queued_at = payload.get("queued_at") if isinstance(payload.get("queued_at"), str) else None
    return clip_names, queued_at


def load_json_payload(path: Path, errors: List[str]) -> Optional[Any]:
    if not path.exists():
        errors.append(f"Batch results file not found: {path}")
        return None
    try:
        text = path.read_text()
    except Exception as exc:  # pragma: no cover - filesystem issues
        errors.append(f"Failed to read batch results {path}: {exc}")
        return None
    if not text.strip():
        return []
    if path.suffix.lower() == ".jsonl":
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                errors.append("Batch results JSONL contained invalid JSON line")
        return items
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        errors.append("Batch results file contained invalid JSON")
        return None


def normalize_batch_items(raw: Any, errors: List[str]) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        for key in ("data", "items", "results"):
            if key in raw:
                raw = raw[key]
                break
    if isinstance(raw, list):
        items = [item for item in raw if isinstance(item, dict)]
        if len(items) < len(raw):
            errors.append("Batch results contained non-dict entries")
        return items
    errors.append(f"Batch results had unexpected type: {type(raw).__name__}")
    return []


def resolve_queue_dir(queue_root: Path, custom_id: Optional[str]) -> Optional[Path]:
    if not custom_id:
        return None
    candidate = Path(custom_id)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    return (queue_root / custom_id) if (queue_root / custom_id).exists() else None


def extract_response_body(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    response = item.get("response")
    if isinstance(response, dict) and "body" in response:
        body = response.get("body")
    else:
        body = item.get("response_body")
    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return None
    if isinstance(body, dict):
        return body
    return None


def extract_completion_content(response_body: Dict[str, Any]) -> Optional[Any]:
    choices = response_body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                return message.get("content")
            return first.get("text")
    return response_body.get("output_text")


def coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def update_queue_meta(
    queue_dir: Path,
    review_payload: Optional[Dict[str, Any]],
    response_body: Optional[Dict[str, Any]],
    batch_id: Optional[str],
    errors: List[str],
) -> bool:
    meta_path = queue_dir / "queue_meta.json"
    if not meta_path.exists():
        errors.append(f"queue_meta.json missing for {queue_dir}")
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        errors.append(f"queue_meta.json invalid JSON for {queue_dir}")
        return False
    if not isinstance(meta, dict):
        errors.append(f"queue_meta.json invalid payload for {queue_dir}")
        return False

    enrichment = meta.get("enrichment")
    if not isinstance(enrichment, dict):
        enrichment = {}
    if batch_id and not enrichment.get("batch_job_id"):
        enrichment["batch_job_id"] = batch_id

    if review_payload is None:
        review_payload = {"triage": "unknown"}

    enrichment["status"] = "complete"
    enrichment["enriched_at"] = format_utc(datetime.now(UTC))

    if response_body and isinstance(response_body.get("model"), str):
        enrichment["review_model"] = response_body["model"]

    for key in ("triage", "category", "lighting"):
        value = review_payload.get(key)
        if isinstance(value, str) and value:
            enrichment[key] = value

    tags = review_payload.get("tags")
    if isinstance(tags, list):
        enrichment["tags"] = [tag for tag in tags if isinstance(tag, str)]

    confidence = coerce_float(review_payload.get("confidence"))
    if confidence is not None:
        enrichment["review_confidence"] = confidence

    reason = review_payload.get("reason")
    if isinstance(reason, str) and reason:
        enrichment["review_reason"] = reason

    meta["enrichment"] = enrichment
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return True


def ingest_batch_results(
    batch_results: Path,
    queue_root: Path,
    batch_id: Optional[str],
    errors: List[str],
) -> Tuple[int, int, int, int]:
    raw = load_json_payload(batch_results, errors)
    items = normalize_batch_items(raw, errors)
    total = len(items)
    updated = 0
    missing = 0
    parse_errors = 0

    for item in items:
        custom_id = item.get("custom_id") if isinstance(item.get("custom_id"), str) else None
        queue_dir = resolve_queue_dir(queue_root, custom_id)
        if not queue_dir:
            missing += 1
            continue

        response_body = extract_response_body(item)
        review_payload: Optional[Dict[str, Any]] = None
        if response_body:
            content = extract_completion_content(response_body)
            if isinstance(content, dict):
                review_payload = content
            elif isinstance(content, str):
                review_payload = extract_json_payload(content)
        if review_payload is None:
            parse_errors += 1

        if update_queue_meta(queue_dir, review_payload, response_body, batch_id, errors):
            updated += 1

    return total, updated, missing, parse_errors


def write_run_log(
    logs_dir: Path,
    summary: RunSummary,
    dry_run: bool,
) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "queue-enricher-latest.json"
    if dry_run:
        return log_path
    log_path.write_text(json.dumps(summary.__dict__, indent=2))
    return log_path


def run_enricher(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    logs_dir = get_logs_dir(repo_root)
    started_at = datetime.now(UTC)
    errors: List[str] = []

    recipients = parse_recipients(args.recipients, args.agent_name)
    token = os.environ.get("QUEUE_ENRICHER_MCP_TOKEN")
    client: Optional[MCPClient] = MCPClient(args.mcp_url, timeout=10, token=token)

    if not client.ping():
        log("MCP Agent Mail server not responding")
        client = None

    register_agent_safe(client, str(repo_root), args.agent_name, args.skip_mail, errors)

    inbox_messages = fetch_inbox_safe(
        client=client,
        project_key=str(repo_root),
        agent_name=args.agent_name,
        skip_mail=args.skip_mail,
        errors=errors,
    )
    clip_requests: List[str] = []
    acked = 0
    for message in inbox_messages:
        clips, _queued_at = parse_queue_payload(message)
        if clips:
            clip_requests.extend(clips)
        if message.get("ack_required"):
            if acknowledge_message_safe(
                client=client,
                project_key=str(repo_root),
                agent_name=args.agent_name,
                message_id=message.get("id"),
                skip_mail=args.skip_mail,
                errors=errors,
            ):
                acked += 1

    batch_total = 0
    batch_updated = 0
    batch_missing = 0
    batch_parse_errors = 0
    if args.batch_results:
        batch_total, batch_updated, batch_missing, batch_parse_errors = ingest_batch_results(
            batch_results=Path(args.batch_results),
            queue_root=Path(args.queue_dir),
            batch_id=args.batch_id,
            errors=errors,
        )

    log("Inbox parsed; skeleton run started")

    summary_message = (
        "Queue Enricher parsed inbox; no batch results ingested."
        if batch_total == 0
        else "Queue Enricher parsed inbox and ingested batch results."
    )
    summary = RunSummary(
        status="inbox_only" if batch_total == 0 else "batch_ingested",
        started_at=format_utc(started_at),
        finished_at=format_utc(datetime.now(UTC)),
        message=summary_message,
        log_path=str(logs_dir / "queue-enricher-latest.json"),
        inbox_messages=len(inbox_messages),
        queued_clips=len(clip_requests),
        acknowledged=acked,
        batch_results=batch_total,
        batch_updated=batch_updated,
        batch_missing=batch_missing,
        batch_parse_errors=batch_parse_errors,
    )
    write_run_log(logs_dir, summary, args.dry_run)

    send_message_safe(
        client,
        str(repo_root),
        args.agent_name,
        recipients,
        "Queue Enricher inbox parse",
        (
            f"{summary.message}\n\n"
            f"Status: {summary.status}\n"
            f"Started: {summary.started_at}\n"
            f"Finished: {summary.finished_at}\n"
            f"Inbox messages: {summary.inbox_messages}\n"
            f"Queued clips: {summary.queued_clips}\n"
            f"Acked: {summary.acknowledged}\n"
            f"Batch results: {summary.batch_results}\n"
            f"Batch updated: {summary.batch_updated}\n"
            f"Batch missing: {summary.batch_missing}\n"
            f"Batch parse errors: {summary.batch_parse_errors}\n"
            f"Log: {summary.log_path}\n"
            f"Errors: {len(errors)}"
        ),
        args.thread_id,
        args.skip_mail,
        errors,
    )

    if errors:
        for error in errors:
            log(f"WARN: {error}")

    log("Skeleton run complete")
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Queue Enricher agent (skeleton)")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing logs")
    parser.add_argument("--skip-mail", action="store_true", help="Skip Agent Mail reporting")
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--thread-id", default=DEFAULT_THREAD_ID)
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL)
    parser.add_argument("--recipients", default=os.environ.get("QUEUE_ENRICHER_TO"))
    parser.add_argument("--batch-results", help="Path to OpenAI batch results (.jsonl or .json)")
    parser.add_argument("--batch-id", help="Batch job id to record in queue_meta.json")
    parser.add_argument("--queue-dir", default=str(DEFAULT_QUEUE_DIR))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run_enricher(args))


if __name__ == "__main__":
    main()
