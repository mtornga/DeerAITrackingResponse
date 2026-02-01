"""Helpers for building OpenAI Batch request payloads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


@dataclass
class BatchRequest:
    custom_id: str
    method: str
    url: str
    body: Dict[str, Any]


def build_vision_chat_body(
    model: str,
    prompt: str,
    image_urls: Sequence[str],
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }


def build_batch_request(
    custom_id: str,
    body: Dict[str, Any],
    url: str = "/v1/chat/completions",
    method: str = "POST",
) -> BatchRequest:
    return BatchRequest(custom_id=custom_id, method=method, url=url, body=body)


def batch_request_to_jsonl_line(request: BatchRequest) -> str:
    payload = {
        "custom_id": request.custom_id,
        "method": request.method,
        "url": request.url,
        "body": request.body,
    }
    return json.dumps(payload, separators=(",", ":"))


def write_jsonl(path: Path, requests: Iterable[BatchRequest]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [batch_request_to_jsonl_line(req) for req in requests]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path
