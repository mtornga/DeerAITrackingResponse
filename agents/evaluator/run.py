#!/usr/bin/env python3
"""
Evaluator agent.

Evaluates candidate day/night models against baseline using a fixed golden eval set.
Computes mAP50 and recall deltas and gates promotion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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

from training_loop_utils import (
    extract_best_bbox,
    extract_frames,
    find_video_file,
    load_json,
    resolve_share_root,
    write_label_files,
)


DEFAULT_AGENT_NAME = os.environ.get("EVALUATOR_AGENT", "SilverField")
DEFAULT_THREAD_ID = os.environ.get("TRAINING_LOOP_THREAD", "TRAINING-LOOP")
DEFAULT_MCP_URL = os.environ.get("EVALUATOR_MCP_URL", "http://127.0.0.1:8765/mcp/")
DEFAULT_LOGS_DIRNAME = "runs/logs/evaluator"


@dataclass
class Metrics:
    map50: Optional[float] = None
    recall: Optional[float] = None


@dataclass
class EvalSummary:
    run_id: str
    status: str
    started_at: str
    finished_at: str
    candidate_models: Dict[str, str]
    baseline_models: Dict[str, str]
    metrics: Dict[str, Dict[str, Dict[str, Optional[float]]]]
    gate_passed: bool
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
    print(f"[Evaluator {timestamp}] {message}", flush=True)


def format_utc(ts: datetime) -> str:
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_training_config() -> Dict[str, Any]:
    config_path = REPO_ROOT / "configs" / "training_loop.json"
    payload = load_json(config_path)
    return payload or {}


def load_eval_sets() -> Dict[str, Any]:
    config_path = REPO_ROOT / "configs" / "eval_sets.json"
    payload = load_json(config_path)
    return payload or {}


def load_env_vars() -> Dict[str, str]:
    env_path = REPO_ROOT / ".env"
    env_vars: Dict[str, str] = {}
    if not env_path.exists():
        return env_vars
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env_vars[key.strip()] = value.strip()
    return env_vars


def resolve_model_paths(model_list: List[str], share_root: Path) -> List[str]:
    resolved: List[str] = []
    for name in model_list:
        if not name:
            continue
        candidate = Path(name)
        if candidate.is_absolute() and candidate.exists():
            resolved.append(str(candidate))
            continue
        share_path = share_root / "models" / name
        if share_path.exists():
            resolved.append(str(share_path))
            continue
        repo_path = REPO_ROOT / "models" / name
        if repo_path.exists():
            resolved.append(str(repo_path))
            continue
    return resolved


def resolve_baseline_models(share_root: Path) -> Dict[str, str]:
    env_vars = load_env_vars()
    models_day = env_vars.get("DETECTOR_MODELS_DAY", "")
    models_night = env_vars.get("DETECTOR_MODELS_NIGHT", "")
    fallback = env_vars.get("DETECTOR_MODELS", "")

    def pick_first(value: str) -> Optional[str]:
        if not value:
            return None
        return value.split(",")[0].strip()

    baseline_day = pick_first(models_day) or pick_first(fallback)
    baseline_night = pick_first(models_night) or pick_first(fallback)

    paths: Dict[str, str] = {}
    if baseline_day:
        resolved = resolve_model_paths([baseline_day], share_root)
        if resolved:
            paths["day"] = resolved[0]
    if baseline_night:
        resolved = resolve_model_paths([baseline_night], share_root)
        if resolved:
            paths["night"] = resolved[0]
    return paths


def resolve_candidate_models(trainer_log: Optional[Path]) -> Dict[str, str]:
    if trainer_log and trainer_log.exists():
        payload = load_json(trainer_log)
        if payload and isinstance(payload.get("model_paths"), dict):
            return {k: v for k, v in payload["model_paths"].items() if isinstance(v, str)}
    latest = REPO_ROOT / "runs" / "logs" / "trainer" / "trainer-latest.json"
    payload = load_json(latest)
    if payload and isinstance(payload.get("model_paths"), dict):
        return {k: v for k, v in payload["model_paths"].items() if isinstance(v, str)}
    return {}


def iter_eval_items(eval_cfg: Dict[str, Any], lighting: str) -> List[Dict[str, Any]]:
    clips = eval_cfg.get(lighting, {}).get("clips", []) if isinstance(eval_cfg.get(lighting), dict) else []
    items: List[Dict[str, Any]] = []
    for clip in clips:
        if isinstance(clip, str):
            items.append({"path": clip})
        elif isinstance(clip, dict):
            path = clip.get("path")
            if isinstance(path, str) and path.strip():
                items.append(clip)
    return items


def resolve_clip_path(clip: Path, golden_root: Path, lighting: str) -> Optional[Path]:
    if clip.is_absolute() and clip.exists():
        return clip
    candidate = golden_root / lighting / clip
    if candidate.exists():
        return candidate
    candidate = golden_root / clip
    if candidate.exists():
        return candidate
    return None


def infer_category_from_path(clip_path: Path, lighting: str) -> Optional[str]:
    parts = clip_path.parts
    if lighting in parts:
        idx = parts.index(lighting)
        if idx + 1 < len(parts) - 0 and idx + 1 < len(parts) - 0:
            if idx + 1 < len(parts) - 1:
                return parts[idx + 1]
        if idx - 1 >= 0:
            return parts[idx - 1]
    return clip_path.parent.name


def build_eval_dataset(
    golden_root: Path,
    eval_cfg: Dict[str, Any],
    dataset_root: Path,
    lighting: str,
    dry_run: bool,
) -> Path:
    items = iter_eval_items(eval_cfg, lighting)
    if not items:
        raise SystemExit(f"No eval clips configured for {lighting}")

    lighting_cfg = eval_cfg.get(lighting) if isinstance(eval_cfg.get(lighting), dict) else {}
    frame_interval = int(lighting_cfg.get("frame_interval", 15))
    max_frames = int(lighting_cfg.get("max_frames", 10))

    images_dir = dataset_root / lighting / "images" / "val"
    labels_dir = dataset_root / lighting / "labels" / "val"
    if not dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        raw_path = item.get("path")
        if not isinstance(raw_path, str):
            continue
        normalized = raw_path.replace("/Users/marktornga/DeerShare", "/srv/deer-share")
        clip = Path(normalized)
        clip_path = resolve_clip_path(clip, golden_root, lighting)
        if not clip_path:
            continue
        meta = load_json(clip_path / "meta.json")
        if not meta:
            continue
        video_path = find_video_file(clip_path)
        if not video_path:
            continue
        bbox = extract_best_bbox(meta)
        if bbox is None:
            continue
        category = item.get("category")
        if not isinstance(category, str):
            category = infer_category_from_path(clip_path, lighting)
        if category in {"deer", "buck", "doe", "animal"}:
            class_id = 0
        elif category == "person":
            class_id = 1
        else:
            class_id = None
        if class_id is None:
            continue
        if dry_run:
            continue
        frames = extract_frames(
            video_path=video_path,
            output_dir=images_dir,
            prefix=clip_path.name,
            frame_interval=frame_interval,
            max_frames=max_frames,
        )
        write_label_files(frames, labels_dir, class_id, bbox)

    data_yaml = dataset_root / lighting / "data.yaml"
    if not dry_run:
        data_yaml.write_text(
            f"path: {dataset_root / lighting}\n"
            "train: images/val\n"
            "val: images/val\n\n"
            "names:\n"
            "  0: deer\n"
            "  1: person\n"
        )
    return data_yaml


def eval_model(model_path: str, data_yaml: Path, device: str, dry_run: bool) -> Metrics:
    if dry_run:
        return Metrics(map50=None, recall=None)
    from ultralytics import YOLO  # type: ignore

    model = YOLO(model_path)
    results = model.val(data=str(data_yaml), device=device)
    metrics = Metrics()
    if hasattr(results, "box"):
        metrics.map50 = getattr(results.box, "map50", None)
        metrics.recall = getattr(results.box, "mr", None)
    return metrics


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
                "task_description": "Training Loop Evaluator",
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
    summary: EvalSummary,
    skip_mail: bool,
    errors: List[str],
) -> None:
    if skip_mail or client is None:
        return

    body_lines = [
        f"Run: {summary.run_id}",
        f"Status: {summary.status}",
        f"Gate passed: {summary.gate_passed}",
        "",
        "Candidate models:",
    ]
    for lighting, path in summary.candidate_models.items():
        body_lines.append(f"- {lighting}: {path}")
    body_lines.append("")
    body_lines.append("Baseline models:")
    for lighting, path in summary.baseline_models.items():
        body_lines.append(f"- {lighting}: {path}")

    try:
        client.call_tool(
            "send_message",
            {
                "project_key": project_key,
                "sender_name": agent_name,
                "to": recipients,
                "subject": f"Evaluator run {summary.run_id}",
                "body_md": "\n".join(body_lines),
                "thread_id": thread_id,
            },
        )
    except Exception as exc:
        errors.append(f"Failed to send summary: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share-root", help="Override share root (default: auto-detect)")
    parser.add_argument("--run-id", help="Override run id (default: EV-YYYYMMDD-HHMMSS)")
    parser.add_argument("--trainer-log", help="Path to trainer log JSON")
    parser.add_argument("--device", default="cuda:0", help="Device for evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Skip evaluation and file writes")

    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME, help="MCP agent name")
    parser.add_argument("--thread-id", default=DEFAULT_THREAD_ID, help="MCP thread id")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="MCP server URL")
    parser.add_argument("--recipients", help="Comma-separated recipients for MCP summary")
    parser.add_argument("--skip-mail", action="store_true", help="Skip MCP mail reporting")

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    config = load_training_config()
    eval_cfg = load_eval_sets()
    min_map_gain = float(config.get("min_map50_gain", 0.01))
    min_recall_delta = float(config.get("min_recall_delta", 0.0))

    share_root = resolve_share_root(args.share_root)
    golden_root = share_root / "golden_clips"
    if not golden_root.exists():
        raise SystemExit(f"Golden clips directory not found: {golden_root}")

    run_id = args.run_id or datetime.now(UTC).strftime("EV-%Y%m%d-%H%M%S")
    eval_root = share_root / "eval_datasets" / run_id

    started_at = datetime.now(UTC)
    errors: List[str] = []

    trainer_log = Path(args.trainer_log) if args.trainer_log else None
    candidate_models = resolve_candidate_models(trainer_log)
    if "day" not in candidate_models or "night" not in candidate_models:
        errors.append("Candidate models missing day/night paths")

    baseline_models = resolve_baseline_models(share_root)
    if "day" not in baseline_models or "night" not in baseline_models:
        errors.append("Baseline models missing day/night paths")

    metrics: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    gate_passed = False

    if not errors:
        data_day = build_eval_dataset(golden_root, eval_cfg, eval_root, "day", args.dry_run)
        data_night = build_eval_dataset(golden_root, eval_cfg, eval_root, "night", args.dry_run)

        cand_day = eval_model(candidate_models["day"], data_day, args.device, args.dry_run)
        base_day = eval_model(baseline_models["day"], data_day, args.device, args.dry_run)
        cand_night = eval_model(candidate_models["night"], data_night, args.device, args.dry_run)
        base_night = eval_model(baseline_models["night"], data_night, args.device, args.dry_run)

        metrics = {
            "day": {
                "candidate": {"map50": cand_day.map50, "recall": cand_day.recall},
                "baseline": {"map50": base_day.map50, "recall": base_day.recall},
            },
            "night": {
                "candidate": {"map50": cand_night.map50, "recall": cand_night.recall},
                "baseline": {"map50": base_night.map50, "recall": base_night.recall},
            },
        }

        def passes(cand: Metrics, base: Metrics) -> bool:
            if cand.map50 is None or cand.recall is None or base.map50 is None or base.recall is None:
                return False
            return cand.recall >= base.recall + min_recall_delta and cand.map50 >= base.map50 + min_map_gain

        gate_passed = passes(cand_day, base_day) and passes(cand_night, base_night)

    status = "complete" if gate_passed and not errors else "failed"
    if args.dry_run:
        status = "dry_run"

    finished_at = datetime.now(UTC)

    summary = EvalSummary(
        run_id=run_id,
        status=status,
        started_at=format_utc(started_at),
        finished_at=format_utc(finished_at),
        candidate_models=candidate_models,
        baseline_models=baseline_models,
        metrics=metrics,
        gate_passed=gate_passed,
        errors=errors,
    )

    if not args.dry_run:
        logs_dir = REPO_ROOT / DEFAULT_LOGS_DIRNAME
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{run_id}.json"
        log_path.write_text(json.dumps(summary.__dict__, indent=2))
        latest_path = logs_dir / "evaluator-latest.json"
        latest_path.write_text(json.dumps(summary.__dict__, indent=2))
        summary.log_path = str(log_path)

    token = os.environ.get("EVALUATOR_MCP_TOKEN")
    client: Optional[MCPClient] = MCPClient(args.mcp_url, timeout=10, token=token)
    if not client.ping():
        client = None

    register_agent_safe(client, str(REPO_ROOT), args.agent_name, args.skip_mail, errors)

    recipients = parse_recipients(args.recipients or os.environ.get("EVALUATOR_TO"), args.agent_name)
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

    log(
        f"Finished evaluator run {summary.run_id} (gate_passed={summary.gate_passed})"
    )
    return 0 if summary.gate_passed or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
