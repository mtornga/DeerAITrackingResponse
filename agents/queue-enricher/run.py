#!/usr/bin/env python3
"""
Queue Enricher agent.

Processes queued clips, extracts preview frames, computes snow heuristics,
submits GPT-4o mini vision triage via Batch API, and updates queue_meta.json.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import urllib.request
import urllib.error
import uuid


def _ensure_repo_root_on_path() -> None:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / ".env").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
    fallback = script_path.parents[2]
    if str(fallback) not in sys.path:
        sys.path.insert(0, str(fallback))


_ensure_repo_root_on_path()

try:
    from env_loader import load_env_file
    load_env_file()
except Exception:
    pass

QUEUE_ENRICHER_DIR = Path(__file__).resolve().parent
if str(QUEUE_ENRICHER_DIR) not in sys.path:
    sys.path.insert(0, str(QUEUE_ENRICHER_DIR))

import batch_requests  # type: ignore
import idempotency  # type: ignore
import preview_frames  # type: ignore
import snow_scoring  # type: ignore


DEFAULT_AGENT_NAME = os.environ.get("QUEUE_ENRICHER_AGENT", "QuietHarbor")
DEFAULT_THREAD_ID = os.environ.get("QUEUE_ENRICHER_THREAD", "QUEUE-ENRICHER")
DEFAULT_MCP_URL = os.environ.get("QUEUE_ENRICHER_MCP_URL", "http://127.0.0.1:8765/mcp/")
DEFAULT_LOGS_DIRNAME = "runs/logs/queue_enricher"
DEFAULT_QUEUE_DIR = Path(os.environ.get("QUEUE_ENRICHER_QUEUE_DIR", "/srv/deer-share/training_queue"))
DEFAULT_MODEL = os.environ.get("QUEUE_ENRICHER_MODEL", "gpt-4o-mini")
DEFAULT_IMAGE_DETAIL = os.environ.get("QUEUE_ENRICHER_IMAGE_DETAIL", "low")
DEFAULT_PREVIEW_COUNT = int(os.environ.get("QUEUE_ENRICHER_PREVIEW_COUNT", "5"))
DEFAULT_SUBMIT_BATCH = os.environ.get("QUEUE_ENRICHER_SUBMIT_BATCH", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
DEFAULT_SCAN_QUEUE = os.environ.get("QUEUE_ENRICHER_SCAN_QUEUE", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
DEFAULT_BATCH_COMPLETION_WINDOW = os.environ.get("QUEUE_ENRICHER_BATCH_WINDOW", "24h")
DEFAULT_OPENAI_ENDPOINT = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
FRAME_REVIEW_ENV = "FrameReview_OPEN_AI_API_KEY"


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
    clips_processed: int = 0
    clips_skipped: int = 0
    batch_requests: int = 0
    batch_file: Optional[str] = None
    batch_id: Optional[str] = None
    batch_status: Optional[str] = None
    batch_results: int = 0
    batch_updated: int = 0
    batch_missing: int = 0
    batch_parse_errors: int = 0


@dataclass
class ClipTask:
    queue_dir: Path
    preview_paths: List[Path]
    meta_path: Path


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


def load_openai_key() -> Optional[str]:
    value = os.environ.get(FRAME_REVIEW_ENV)
    if value:
        return value.strip()
    return None


def encode_image_data_uri(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_triage_prompt() -> str:
    return (
        "You are reviewing detector preview images for deer/person/other wildlife. "
        "You will see 1 full-frame image plus several crops around detections from the same clip. "
        "Return a SINGLE JSON object with:\n"
        "triage: keep|maybe|reject\n"
        "category: deer|person|other_wildlife|false_positive|unknown\n"
        "lighting: day|night|unknown\n"
        "tags: array of short lowercase tags (e.g., buck, doe, sprint, snow, rain, flies)\n"
        "confidence: number 0-1\n"
        "reason: short string\n"
        "If unsure, use triage=maybe and category=unknown.\n"
    )


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
        if isinstance(raw, dict):
            return []
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


def parse_queue_payload(message: Dict[str, Any]) -> Tuple[List[str], Optional[str], Optional[str]]:
    body = message.get("body_md") or message.get("body") or ""
    payload = extract_json_payload(body)
    if not isinstance(payload, dict):
        return [], None, None
    clips = payload.get("clips") if isinstance(payload.get("clips"), list) else []
    clip_names = [clip for clip in clips if isinstance(clip, str)]
    queued_at = payload.get("queued_at") if isinstance(payload.get("queued_at"), str) else None
    queue_dir = payload.get("queue_dir") if isinstance(payload.get("queue_dir"), str) else None
    return clip_names, queued_at, queue_dir


def parse_queue_dir_name(name: str) -> Tuple[Optional[str], Optional[str]]:
    match = re.match(r"^(\d{4}-\d{2}-\d{2})_(segment_\d{6})(.*)$", name)
    if not match:
        return None, None
    date_part, segment_part, suffix = match.groups()
    segment_id = f"{date_part}_{segment_part}"
    suffix = suffix.strip()
    return segment_id, (suffix or None)


def select_model_name(meta: Dict[str, Any]) -> Optional[str]:
    routing = meta.get("routing")
    if isinstance(routing, dict):
        votes = routing.get("votes")
        if isinstance(votes, list) and votes:
            first_vote = votes[0]
            if isinstance(first_vote, dict):
                model = first_vote.get("model")
                if isinstance(model, str) and model:
                    return model
    models = meta.get("models")
    if isinstance(models, dict) and models:
        return next(iter(models.keys()))
    return None


def build_queue_meta_from_meta(queue_dir: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    segment_id, note_suffix = parse_queue_dir_name(queue_dir.name)
    if not segment_id:
        segment_id = queue_dir.name
    queue_meta: Dict[str, Any] = {
        "schema_version": 1,
        "queue": {
            "segment_id": segment_id,
            "source_path": str(queue_dir),
            "queued_by": "unknown",
            "queued_at": format_utc(datetime.now(UTC)),
        },
        "meta_summary": {
            "model_name": select_model_name(meta),
            "max_confidence": meta.get("max_confidence"),
            "counts": meta.get("counts") or {},
            "routing": (
                {
                    "decision": meta.get("routing", {}).get("decision"),
                    "confidence_score": meta.get("routing", {}).get("confidence_score"),
                }
                if isinstance(meta.get("routing"), dict)
                else None
            ),
        },
    }
    if note_suffix:
        queue_meta["queue"]["note_suffix"] = note_suffix
    return queue_meta


def ensure_queue_meta(queue_dir: Path, errors: List[str]) -> Path:
    meta_path = queue_dir / "queue_meta.json"
    if meta_path.exists():
        return meta_path
    meta_json = preview_frames.load_meta_json(queue_dir / "meta.json")
    if not meta_json:
        errors.append(f"meta.json missing or invalid for {queue_dir}")
        return meta_path
    payload = build_queue_meta_from_meta(queue_dir, meta_json)
    try:
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        errors.append(f"Failed to write {meta_path}: {exc}")
    return meta_path


def is_pending(queue_meta: Dict[str, Any]) -> bool:
    enrichment = queue_meta.get("enrichment")
    if not isinstance(enrichment, dict):
        return False
    return enrichment.get("status") == "pending"


def mark_enrichment_pending(
    queue_dir: Path,
    preview_paths: Sequence[Path],
    snow_score: float,
    snow_reason: str,
    motion_stats: Dict[str, float],
    model_name: str,
    batch_id: Optional[str],
    status: str,
    errors: List[str],
) -> None:
    meta_path = queue_dir / "queue_meta.json"
    if not meta_path.exists():
        errors.append(f"queue_meta.json missing for {queue_dir}")
        return
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        errors.append(f"queue_meta.json invalid JSON for {queue_dir}")
        return
    if not isinstance(meta, dict):
        errors.append(f"queue_meta.json invalid payload for {queue_dir}")
        return
    enrichment = meta.get("enrichment")
    if not isinstance(enrichment, dict):
        enrichment = {}

    enrichment["status"] = status
    enrichment["pending_at"] = format_utc(datetime.now(UTC))
    enrichment["review_model"] = model_name
    if batch_id:
        enrichment["batch_job_id"] = batch_id
    enrichment["preview_frames"] = [path.name for path in preview_paths]
    enrichment["snow_score"] = snow_score
    enrichment["snow_reason"] = snow_reason
    enrichment["motion_stats"] = motion_stats
    meta["enrichment"] = enrichment
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_queue_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def collect_queue_dirs(
    queue_root: Path,
    requested_clips: Sequence[str],
    scan_queue: bool,
) -> List[Path]:
    paths: Dict[Path, None] = {}
    for clip in requested_clips:
        candidate = Path(clip)
        if candidate.is_absolute():
            if candidate.exists():
                paths[candidate] = None
        else:
            dest = queue_root / clip
            if dest.exists():
                paths[dest] = None
    if scan_queue and queue_root.exists():
        for child in queue_root.iterdir():
            if child.is_dir():
                paths.setdefault(child, None)
    return list(paths.keys())


def build_clip_tasks(
    queue_dirs: Iterable[Path],
    preview_count: int,
    errors: List[str],
    force: bool = False,
) -> Tuple[List[ClipTask], List[Tuple[Path, str]]]:
    tasks: List[ClipTask] = []
    skipped: List[Tuple[Path, str]] = []
    for queue_dir in queue_dirs:
        queue_meta_path = ensure_queue_meta(queue_dir, errors)
        queue_meta = load_queue_meta(queue_meta_path)
        if not force:
            if idempotency.has_enrichment(queue_meta):
                skipped.append((queue_dir, "enrichment_complete"))
                continue
            if is_pending(queue_meta):
                skipped.append((queue_dir, "batch_pending"))
                continue
            lock_path = queue_dir / ".enriching"
            if idempotency.lock_is_recent(lock_path, 1800):
                skipped.append((queue_dir, "lock_active"))
                continue

        meta_path = queue_dir / "meta.json"
        if not meta_path.exists():
            errors.append(f"meta.json missing for {queue_dir}")
            continue

        video_path = preview_frames.find_queue_video(queue_dir)
        if not video_path:
            errors.append(f"No segment video found in {queue_dir}")
            continue

        meta_json = preview_frames.load_meta_json(meta_path)
        existing_previews = sorted(queue_dir.glob("preview_*.jpg"))
        if len(existing_previews) >= preview_count and not force:
            tasks.append(
                ClipTask(
                    queue_dir=queue_dir,
                    preview_paths=existing_previews[:preview_count],
                    meta_path=meta_path,
                )
            )
            continue

        preferred_category = None
        counts = meta_json.get("counts") if isinstance(meta_json, dict) else None
        if isinstance(counts, dict) and counts.get("person", 0):
            preferred_category = "person"
        preview = preview_frames.write_preview_bundle(
            meta_path=meta_path,
            video_path=video_path,
            output_dir=queue_dir,
            full_count=2,
            crop_count=preview_count,
            preferred_category=preferred_category,
        )
        if preview.errors:
            errors.extend([f"{queue_dir}: {err}" for err in preview.errors])
        if not preview.output_paths:
            errors.append(f"Failed to write preview frames for {queue_dir}")
            continue

        tasks.append(ClipTask(queue_dir=queue_dir, preview_paths=preview.output_paths, meta_path=meta_path))
    return tasks, skipped


def build_batch_requests_for_tasks(
    tasks: Sequence[ClipTask],
    model: str,
    prompt: str,
    image_detail: str,
    errors: List[str],
) -> List[batch_requests.BatchRequest]:
    requests: List[batch_requests.BatchRequest] = []
    for task in tasks:
        image_urls: List[str] = []
        for path in task.preview_paths:
            uri = encode_image_data_uri(path)
            if not uri:
                errors.append(f"Failed to encode preview frame {path}")
                continue
            image_urls.append(uri)
        if not image_urls:
            errors.append(f"No preview frames encoded for {task.queue_dir}")
            continue
        body = batch_requests.build_vision_chat_body(
            model=model,
            prompt=prompt,
            image_urls=image_urls,
            image_detail=image_detail,
        )
        custom_id = task.queue_dir.name
        requests.append(batch_requests.build_batch_request(custom_id=custom_id, body=body))
    return requests


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


def _openai_headers(api_key: str, content_type: Optional[str] = None) -> Dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _openai_request(
    api_key: str,
    method: str,
    url: str,
    data: Optional[bytes] = None,
    content_type: Optional[str] = None,
) -> Dict[str, Any]:
    headers = _openai_headers(api_key, content_type=content_type)
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = "<no body>"
        log(f"OpenAI HTTP error {exc.code} for {method} {url}: {body}")
        raise
    except urllib.error.URLError as exc:
        log(f"OpenAI request failed for {method} {url}: {exc}")
        raise


def create_chat_completion(
    api_key: str,
    model: str,
    prompt: str,
    image_urls: Sequence[str],
    image_detail: str,
    max_retries: int = 5,
) -> Optional[Dict[str, Any]]:
    body = batch_requests.build_vision_chat_body(
        model=model,
        prompt=prompt,
        image_urls=image_urls,
        image_detail=image_detail,
    )
    url = f"{DEFAULT_OPENAI_ENDPOINT}/chat/completions"
    data = json.dumps(body).encode("utf-8")
    for attempt in range(max_retries):
        try:
            return _openai_request(api_key, "POST", url, data=data, content_type="application/json")
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                return None
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            wait_seconds: Optional[float] = None
            if retry_after:
                try:
                    wait_seconds = float(retry_after)
                except ValueError:
                    wait_seconds = None
            if wait_seconds is None:
                try:
                    body = exc.read().decode("utf-8")
                except Exception:
                    body = ""
                match = re.search(r"try again in ([0-9.]+)s", body)
                if match:
                    try:
                        wait_seconds = float(match.group(1))
                    except ValueError:
                        wait_seconds = None
            if wait_seconds is None:
                wait_seconds = min(2 ** attempt, 8)
            time.sleep(wait_seconds)
            continue
        except urllib.error.URLError:
            return None
    return None


def _encode_multipart_formdata(fields: Dict[str, str], file_field: str, file_path: Path) -> Tuple[bytes, str]:
    boundary = f"----queue-enricher-{uuid.uuid4().hex}"
    lines: List[bytes] = []
    for key, value in fields.items():
        lines.append(f"--{boundary}".encode("utf-8"))
        lines.append(f'Content-Disposition: form-data; name="{key}"'.encode("utf-8"))
        lines.append(b"")
        lines.append(value.encode("utf-8"))
    lines.append(f"--{boundary}".encode("utf-8"))
    lines.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"'.encode("utf-8")
    )
    lines.append(b"Content-Type: application/jsonl")
    lines.append(b"")
    lines.append(file_path.read_bytes())
    lines.append(f"--{boundary}--".encode("utf-8"))
    lines.append(b"")
    body = b"\r\n".join(lines)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def upload_batch_file(api_key: str, batch_path: Path) -> Optional[str]:
    body, content_type = _encode_multipart_formdata({"purpose": "batch"}, "file", batch_path)
    url = f"{DEFAULT_OPENAI_ENDPOINT}/files"
    try:
        response = _openai_request(api_key, "POST", url, data=body, content_type=content_type)
    except urllib.error.URLError:
        return None
    file_id = response.get("id")
    return file_id if isinstance(file_id, str) else None


def create_batch_job(api_key: str, file_id: str, completion_window: str) -> Optional[str]:
    url = f"{DEFAULT_OPENAI_ENDPOINT}/batches"
    payload = {
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": completion_window,
        "metadata": {"source": "queue-enricher"},
    }
    data = json.dumps(payload).encode("utf-8")
    try:
        response = _openai_request(api_key, "POST", url, data=data, content_type="application/json")
    except urllib.error.URLError:
        return None
    batch_id = response.get("id")
    return batch_id if isinstance(batch_id, str) else None


def fetch_batch_status(api_key: str, batch_id: str) -> Optional[Dict[str, Any]]:
    url = f"{DEFAULT_OPENAI_ENDPOINT}/batches/{batch_id}"
    try:
        return _openai_request(api_key, "GET", url)
    except urllib.error.URLError:
        return None


def download_batch_output(api_key: str, file_id: str, dest: Path) -> Optional[Path]:
    url = f"{DEFAULT_OPENAI_ENDPOINT}/files/{file_id}/content"
    headers = _openai_headers(api_key, content_type="application/json")
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
    except urllib.error.URLError:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return dest


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


def _choose_most_common(values: List[str], confidences: List[float]) -> Optional[str]:
    if not values:
        return None
    counts: Dict[str, int] = {}
    conf_sums: Dict[str, float] = {}
    for value, conf in zip(values, confidences):
        counts[value] = counts.get(value, 0) + 1
        conf_sums[value] = conf_sums.get(value, 0.0) + conf
    best = sorted(counts.items(), key=lambda item: (-item[1], -conf_sums.get(item[0], 0.0)))[0][0]
    return best


def normalize_review_payload(review_payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(review_payload, dict):
        return None
    if "frames" not in review_payload:
        return review_payload
    frames = review_payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return review_payload

    triage_vals: List[str] = []
    category_vals: List[str] = []
    lighting_vals: List[str] = []
    confidences: List[float] = []
    tags: List[str] = []
    best_reason = None
    best_conf = -1.0

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        triage = frame.get("triage")
        category = frame.get("category")
        lighting = frame.get("lighting")
        conf = coerce_float(frame.get("confidence")) or 0.0
        if isinstance(triage, str):
            triage_vals.append(triage)
        if isinstance(category, str):
            category_vals.append(category)
        if isinstance(lighting, str):
            lighting_vals.append(lighting)
        confidences.append(conf)
        frame_tags = frame.get("tags")
        if isinstance(frame_tags, list):
            tags.extend([t for t in frame_tags if isinstance(t, str)])
        reason = frame.get("reason")
        if isinstance(reason, str) and conf >= best_conf:
            best_conf = conf
            best_reason = reason

    if not confidences:
        confidences = [0.0] * max(len(triage_vals), len(category_vals), len(lighting_vals), 1)

    triage = _choose_most_common(triage_vals, confidences)
    category = _choose_most_common(category_vals, confidences)
    lighting = _choose_most_common(lighting_vals, confidences)

    normalized: Dict[str, Any] = {}
    if triage:
        normalized["triage"] = triage
    if category:
        normalized["category"] = category
    if lighting:
        normalized["lighting"] = lighting
    if tags:
        normalized["tags"] = sorted(set(tags))
    if confidences:
        normalized["confidence"] = max(confidences)
    if best_reason:
        normalized["reason"] = best_reason
    return normalized if normalized else None


def run_live_review(
    task: ClipTask,
    model: str,
    image_detail: str,
    prompt: str,
    errors: List[str],
    overwrite: bool = False,
) -> Optional[Dict[str, Any]]:
    api_key = load_openai_key()
    if not api_key:
        errors.append(f"Missing {FRAME_REVIEW_ENV}; live review skipped")
        return None
    image_urls: List[str] = []
    for path in task.preview_paths:
        uri = encode_image_data_uri(path)
        if not uri:
            errors.append(f"Failed to encode preview frame {path}")
            continue
        image_urls.append(uri)
    if not image_urls:
        return None
    response_body = create_chat_completion(api_key, model, prompt, image_urls, image_detail)
    if not response_body:
        errors.append(f"Live review failed for {task.queue_dir}")
        return None
    content = extract_completion_content(response_body)
    if isinstance(content, dict):
        payload = content
    elif isinstance(content, str):
        payload = extract_json_payload(content)
    else:
        payload = None
    payload = normalize_review_payload(payload)
    if payload is None:
        errors.append(f"Live review response could not be parsed for {task.queue_dir}")
        payload = {"triage": "unknown"}
    if update_queue_meta(task.queue_dir, payload, response_body, None, errors, overwrite=overwrite):
        return payload
    return None


def update_queue_meta(
    queue_dir: Path,
    review_payload: Optional[Dict[str, Any]],
    response_body: Optional[Dict[str, Any]],
    batch_id: Optional[str],
    errors: List[str],
    overwrite: bool = False,
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
    if overwrite:
        enrichment.pop("batch_job_id", None)

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
        review_payload = normalize_review_payload(review_payload)
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
    queue_dir_override: Optional[str] = None
    for message in inbox_messages:
        clips, _queued_at, queue_dir = parse_queue_payload(message)
        if clips:
            clip_requests.extend(clips)
        if queue_dir and not queue_dir_override:
            queue_dir_override = queue_dir
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

    queue_root = Path(queue_dir_override) if queue_dir_override else Path(args.queue_dir)
    if args.single_clip:
        queue_dirs = collect_queue_dirs(queue_root, [args.single_clip], scan_queue=False)
    else:
        queue_dirs = collect_queue_dirs(queue_root, clip_requests, args.scan_queue)
    tasks, skipped = build_clip_tasks(queue_dirs, args.preview_count, errors, force=args.force)

    prompt = build_triage_prompt()
    batch_id: Optional[str] = None
    batch_file: Optional[Path] = None
    batch_requests_count = 0
    batch_status: Optional[str] = None
    batch_results_path: Optional[Path] = None

    if args.single_clip and tasks:
        prompt = build_triage_prompt()
        task = tasks[0]
        meta = preview_frames.load_meta_json(task.meta_path)
        snow_score, motion_stats, snow_reason = snow_scoring.compute_snow_score(meta)
        mark_enrichment_pending(
            queue_dir=task.queue_dir,
            preview_paths=task.preview_paths,
            snow_score=snow_score,
            snow_reason=snow_reason,
            motion_stats=motion_stats,
            model_name=args.model,
            batch_id=None,
            status="prepared",
            errors=errors,
        )
        run_live_review(
            task,
            model=args.model,
            image_detail=args.image_detail,
            prompt=prompt,
            errors=errors,
            overwrite=args.force,
        )
    elif args.live_review_all and tasks:
        prompt = build_triage_prompt()
        for task in tasks:
            meta = preview_frames.load_meta_json(task.meta_path)
            snow_score, motion_stats, snow_reason = snow_scoring.compute_snow_score(meta)
            mark_enrichment_pending(
                queue_dir=task.queue_dir,
                preview_paths=task.preview_paths,
                snow_score=snow_score,
                snow_reason=snow_reason,
                motion_stats=motion_stats,
                model_name=args.model,
                batch_id=None,
                status="prepared",
                errors=errors,
            )
            run_live_review(
                task,
                model=args.model,
                image_detail=args.image_detail,
                prompt=prompt,
                errors=errors,
                overwrite=True,
            )
            if args.live_review_delay > 0:
                time.sleep(args.live_review_delay)
    elif tasks:
        batch_reqs = build_batch_requests_for_tasks(
            tasks=tasks,
            model=args.model,
            prompt=prompt,
            image_detail=args.image_detail,
            errors=errors,
        )
        batch_requests_count = len(batch_reqs)
        if batch_reqs:
            batch_file = batch_requests.write_jsonl(
                logs_dir / f"batch-{started_at.strftime('%Y%m%d-%H%M%S')}.jsonl",
                batch_reqs,
            )

    if batch_file and args.submit_batch:
        api_key = load_openai_key()
        if not api_key:
            errors.append(f"Missing {FRAME_REVIEW_ENV}; batch submission skipped")
            batch_status = "missing_key"
        elif args.dry_run:
            batch_status = "dry_run"
        else:
            file_id = upload_batch_file(api_key, batch_file)
            if not file_id:
                errors.append("Failed to upload batch file to OpenAI")
                batch_status = "upload_failed"
            else:
                batch_id = create_batch_job(api_key, file_id, args.batch_window)
                if not batch_id:
                    errors.append("Failed to create OpenAI batch job")
                    batch_status = "batch_create_failed"
                else:
                    batch_status = "pending"

    if args.poll_batch and args.batch_id:
        api_key = load_openai_key()
        if not api_key:
            errors.append(f"Missing {FRAME_REVIEW_ENV}; batch status fetch skipped")
        else:
            status_payload = fetch_batch_status(api_key, args.batch_id)
            if not status_payload:
                errors.append(f"Failed to fetch batch status for {args.batch_id}")
            else:
                batch_status = status_payload.get("status") or batch_status
                output_file_id = status_payload.get("output_file_id")
                if isinstance(output_file_id, str) and output_file_id:
                    dest = (
                        Path(args.batch_output)
                        if args.batch_output
                        else logs_dir / f"batch-output-{args.batch_id}.jsonl"
                    )
                    downloaded = download_batch_output(api_key, output_file_id, dest)
                    if downloaded:
                        batch_results_path = downloaded
                    else:
                        errors.append(f"Failed to download batch output file {output_file_id}")

    if not args.single_clip and not args.live_review_all:
        for task in tasks:
            meta = preview_frames.load_meta_json(task.meta_path)
            snow_score, motion_stats, snow_reason = snow_scoring.compute_snow_score(meta)
            status = "pending" if batch_id else "prepared"
            try:
                mark_enrichment_pending(
                    queue_dir=task.queue_dir,
                    preview_paths=task.preview_paths,
                    snow_score=snow_score,
                    snow_reason=snow_reason,
                    motion_stats=motion_stats,
                    model_name=args.model,
                    batch_id=batch_id,
                    status=status,
                    errors=errors,
                )
            except Exception as exc:
                errors.append(f"Failed to mark pending for {task.queue_dir}: {exc}")

    batch_total = 0
    batch_updated = 0
    batch_missing = 0
    batch_parse_errors = 0
    results_path = Path(args.batch_results) if args.batch_results else batch_results_path
    if results_path:
        batch_total, batch_updated, batch_missing, batch_parse_errors = ingest_batch_results(
            batch_results=results_path,
            queue_root=Path(args.queue_dir),
            batch_id=args.batch_id,
            errors=errors,
        )

    log("Queue Enricher run completed")

    summary_message = (
        "Queue Enricher processed inbox and prepared batch requests."
        if tasks
        else "Queue Enricher parsed inbox; no clips to enrich."
    )
    summary = RunSummary(
        status="batch_ingested" if batch_total else ("batch_pending" if tasks else "idle"),
        started_at=format_utc(started_at),
        finished_at=format_utc(datetime.now(UTC)),
        message=summary_message,
        log_path=str(logs_dir / "queue-enricher-latest.json"),
        inbox_messages=len(inbox_messages),
        queued_clips=len(clip_requests),
        acknowledged=acked,
        clips_processed=len(tasks),
        clips_skipped=len(skipped),
        batch_requests=batch_requests_count,
        batch_file=str(batch_file) if batch_file else None,
        batch_id=batch_id,
        batch_status=batch_status,
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
            f"Clips processed: {summary.clips_processed}\n"
            f"Clips skipped: {summary.clips_skipped}\n"
            f"Batch requests: {summary.batch_requests}\n"
            f"Batch file: {summary.batch_file or 'n/a'}\n"
            f"Batch id: {summary.batch_id or 'n/a'}\n"
            f"Batch status: {summary.batch_status or 'n/a'}\n"
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

    log("Queue Enricher run complete")
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Queue Enricher agent")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing logs")
    parser.add_argument("--skip-mail", action="store_true", help="Skip Agent Mail reporting")
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--thread-id", default=DEFAULT_THREAD_ID)
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL)
    parser.add_argument("--recipients", default=os.environ.get("QUEUE_ENRICHER_TO"))
    parser.add_argument("--batch-results", help="Path to OpenAI batch results (.jsonl or .json)")
    parser.add_argument("--batch-id", help="Batch job id to record in queue_meta.json")
    parser.add_argument(
        "--poll-batch",
        action="store_true",
        help="Fetch batch status (and download results when ready)",
    )
    parser.add_argument(
        "--batch-output",
        help="Path to write downloaded batch output JSONL (defaults to logs dir)",
    )
    parser.add_argument("--queue-dir", default=str(DEFAULT_QUEUE_DIR))
    parser.add_argument("--preview-count", type=int, default=DEFAULT_PREVIEW_COUNT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image-detail", default=DEFAULT_IMAGE_DETAIL)
    parser.add_argument("--batch-window", default=DEFAULT_BATCH_COMPLETION_WINDOW)
    parser.add_argument(
        "--scan-queue",
        action="store_true",
        default=DEFAULT_SCAN_QUEUE,
        help="Scan the full queue for unenriched clips",
    )
    parser.add_argument(
        "--no-scan-queue",
        action="store_false",
        dest="scan_queue",
        help="Skip scanning full queue (use inbox clips only)",
    )
    parser.add_argument(
        "--submit-batch",
        action="store_true",
        default=DEFAULT_SUBMIT_BATCH,
        help="Submit batch requests to OpenAI",
    )
    parser.add_argument(
        "--no-submit-batch",
        action="store_false",
        dest="submit_batch",
        help="Only write batch JSONL file without submitting",
    )
    parser.add_argument(
        "--single-clip",
        help="Queue clip directory name or absolute path to run live review",
    )
    parser.add_argument(
        "--live-review-all",
        action="store_true",
        help="Run live (non-batch) review for all clips in this run",
    )
    parser.add_argument(
        "--live-review-delay",
        type=float,
        default=2.0,
        help="Seconds to sleep between live review requests",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of previews/enrichment even if already complete",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run_enricher(args))


if __name__ == "__main__":
    main()
