#!/usr/bin/env python3
"""
Trainer agent.

Builds day/night datasets from golden_clips and trains YOLOv8n models.
Outputs candidate models into /srv/deer-share/models/staging/.
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
AGENTS_DIR = Path(__file__).resolve().parents[1]
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

try:
    from env_loader import load_env_file

    load_env_file()
except Exception:
    pass

from training_loop_utils import (
    choose_split,
    extract_best_bbox,
    extract_frames,
    find_video_file,
    load_json,
    resolve_share_root,
    write_label_files,
)


DEFAULT_AGENT_NAME = os.environ.get("TRAINER_AGENT", "SwiftRiver")
DEFAULT_THREAD_ID = os.environ.get("TRAINING_LOOP_THREAD", "TRAINING-LOOP")
DEFAULT_MCP_URL = os.environ.get("TRAINER_MCP_URL", "http://127.0.0.1:8765/mcp/")
DEFAULT_LOGS_DIRNAME = "runs/logs/trainer"


@dataclass
class DatasetStats:
    lighting: str
    clips_total: int = 0
    clips_processed: int = 0
    frames_written: int = 0
    labels_written: int = 0
    skipped_missing_meta: int = 0
    skipped_missing_bbox: int = 0
    skipped_missing_video: int = 0
    negative_frames: int = 0


@dataclass
class RunSummary:
    run_id: str
    status: str
    started_at: str
    finished_at: str
    golden_root: str
    dataset_root: str
    staging_dir: str
    model_paths: Dict[str, str]
    train_commands: Dict[str, List[str]]
    dataset_stats: Dict[str, Dict[str, int]]
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
    print(f"[Trainer {timestamp}] {message}", flush=True)


def format_utc(ts: datetime) -> str:
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_training_config() -> Dict[str, Any]:
    config_path = REPO_ROOT / "configs" / "training_loop.json"
    payload = load_json(config_path)
    return payload or {}


def resolve_base_model(path_str: str, share_root: Path) -> str:
    candidate = Path(path_str)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    if candidate.exists():
        return str(candidate.resolve())
    share_path = share_root / "models" / path_str
    if share_path.exists():
        return str(share_path)
    repo_path = REPO_ROOT / "models" / path_str
    if repo_path.exists():
        return str(repo_path)
    return path_str


def iter_clip_dirs(golden_root: Path, lighting: str, categories: Sequence[str]) -> List[Path]:
    clip_dirs: List[Path] = []
    for category in categories:
        category_dir = golden_root / lighting / category
        if not category_dir.exists():
            continue
        for clip_dir in sorted(category_dir.iterdir()):
            if clip_dir.is_dir():
                clip_dirs.append(clip_dir)
    return clip_dirs


def include_negative_for_lighting(clip_dir: Path, lighting: str) -> bool:
    queue_meta = load_json(clip_dir / "queue_meta.json")
    if not queue_meta:
        return False
    enrichment = queue_meta.get("enrichment") if isinstance(queue_meta.get("enrichment"), dict) else {}
    return enrichment.get("lighting") == lighting


def build_dataset(
    golden_root: Path,
    dataset_root: Path,
    lighting: str,
    categories: Sequence[str],
    max_clips: Optional[int],
    val_split: float,
    frame_interval: int,
    max_frames: int,
    include_negatives: bool,
    dry_run: bool,
) -> DatasetStats:
    stats = DatasetStats(lighting=lighting)

    images_train = dataset_root / lighting / "images" / "train"
    images_val = dataset_root / lighting / "images" / "val"
    labels_train = dataset_root / lighting / "labels" / "train"
    labels_val = dataset_root / lighting / "labels" / "val"

    clip_dirs = iter_clip_dirs(golden_root, lighting, categories)
    if max_clips is not None:
        clip_dirs = clip_dirs[:max_clips]
    stats.clips_total = len(clip_dirs)

    for clip_dir in clip_dirs:
        clip_id = clip_dir.name
        stats.clips_processed += 1
        meta = load_json(clip_dir / "meta.json")
        if not meta:
            stats.skipped_missing_meta += 1
            continue
        video_path = find_video_file(clip_dir)
        if not video_path:
            stats.skipped_missing_video += 1
            continue

        bbox = extract_best_bbox(meta)
        if bbox is None:
            stats.skipped_missing_bbox += 1
            continue

        category = clip_dir.parent.name
        class_id = 0 if category == "deer" else 1 if category == "person" else None
        if class_id is None:
            continue

        split = choose_split(clip_id, val_split)
        images_dir = images_val if split == "val" else images_train
        labels_dir = labels_val if split == "val" else labels_train

        if dry_run:
            stats.frames_written += max_frames
            stats.labels_written += max_frames
            continue

        frames = extract_frames(
            video_path=video_path,
            output_dir=images_dir,
            prefix=clip_id,
            frame_interval=frame_interval,
            max_frames=max_frames,
        )
        if not frames:
            stats.skipped_missing_video += 1
            continue
        stats.frames_written += len(frames)
        stats.labels_written += write_label_files(frames, labels_dir, class_id, bbox)

    if include_negatives:
        negatives_root = golden_root / "negatives" / "snow"
        if negatives_root.exists():
            for clip_dir in sorted(negatives_root.iterdir()):
                if not clip_dir.is_dir():
                    continue
                if not include_negative_for_lighting(clip_dir, lighting):
                    continue
                video_path = find_video_file(clip_dir)
                if not video_path:
                    continue
                split = choose_split(clip_dir.name, val_split)
                images_dir = images_val if split == "val" else images_train
                if dry_run:
                    stats.negative_frames += max_frames
                    continue
                frames = extract_frames(
                    video_path=video_path,
                    output_dir=images_dir,
                    prefix=f"neg_{clip_dir.name}",
                    frame_interval=frame_interval,
                    max_frames=max_frames,
                )
                stats.negative_frames += len(frames)

    if not dry_run:
        (dataset_root / lighting / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_root / lighting / "labels" / "val").mkdir(parents=True, exist_ok=True)

    return stats


def write_data_yaml(dataset_root: Path, lighting: str) -> Path:
    data_path = dataset_root / lighting / "data.yaml"
    data = (
        f"path: {dataset_root / lighting}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: deer\n"
        "  1: person\n"
    )
    data_path.write_text(data)
    return data_path


def build_train_command(
    data_yaml: Path,
    base_model: str,
    project_dir: Path,
    name: str,
    epochs: int,
    batch: int,
    imgsz: int,
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "ultralytics",
        "detect",
        "train",
        f"data={data_yaml}",
        f"model={base_model}",
        f"epochs={epochs}",
        f"batch={batch}",
        f"imgsz={imgsz}",
        f"project={project_dir}",
        f"name={name}",
        "exist_ok=True",
    ]


def run_training(command: List[str], dry_run: bool) -> Tuple[int, str]:
    if dry_run:
        return 0, ""
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


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
                "task_description": "Training Loop Trainer",
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
        f"Dataset root: {summary.dataset_root}",
        f"Staging: {summary.staging_dir}",
        "",
        "Model outputs:",
    ]
    for lighting, path in summary.model_paths.items():
        body_lines.append(f"- {lighting}: {path}")
    if summary.errors:
        body_lines.append("")
        body_lines.append("Errors:")
        for err in summary.errors[:10]:
            body_lines.append(f"- {err}")

    try:
        client.call_tool(
            "send_message",
            {
                "project_key": project_key,
                "sender_name": agent_name,
                "to": recipients,
                "subject": f"Trainer run {summary.run_id}",
                "body_md": "\n".join(body_lines),
                "thread_id": thread_id,
            },
        )
    except Exception as exc:
        errors.append(f"Failed to send summary: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share-root", help="Override share root (default: auto-detect)")
    parser.add_argument("--run-id", help="Override run id (default: TR-YYYYMMDD-HHMMSS)")
    parser.add_argument("--dry-run", action="store_true", help="Plan without writing files or training")
    parser.add_argument("--execute", action="store_true", help="Execute training locally")
    parser.add_argument("--max-clips", type=int, help="Limit clips per lighting")
    parser.add_argument("--frame-interval", type=int, help="Frame interval for extraction")
    parser.add_argument("--max-frames", type=int, help="Max frames per clip")
    parser.add_argument("--val-split", type=float, help="Validation split ratio")
    parser.add_argument("--include-negatives", action="store_true", help="Include negatives as background")
    parser.add_argument("--base-model", help="Base model path/name (default from config)")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--batch", type=int, help="Training batch size")
    parser.add_argument("--imgsz", type=int, help="Training image size")
    parser.add_argument("--project-dir", help="Training runs directory (default from config)")

    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME, help="MCP agent name")
    parser.add_argument("--thread-id", default=DEFAULT_THREAD_ID, help="MCP thread id")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="MCP server URL")
    parser.add_argument("--recipients", help="Comma-separated recipients for MCP summary")
    parser.add_argument("--skip-mail", action="store_true", help="Skip MCP mail reporting")

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    config = load_training_config()
    trainer_cfg = config.get("trainer", {}) if isinstance(config.get("trainer"), dict) else {}

    share_root = resolve_share_root(args.share_root)
    golden_root = share_root / "golden_clips"
    if not golden_root.exists():
        raise SystemExit(f"Golden clips directory not found: {golden_root}")

    run_id = args.run_id or datetime.now(UTC).strftime("TR-%Y%m%d-%H%M%S")
    dataset_root = share_root / "training_datasets" / run_id
    staging_dir = share_root / "models" / "staging"

    val_split = args.val_split if args.val_split is not None else float(config.get("val_split", 0.15))
    frame_interval = args.frame_interval if args.frame_interval is not None else int(config.get("frame_interval", 15))
    max_frames = args.max_frames if args.max_frames is not None else int(config.get("max_frames", 20))
    include_negatives = args.include_negatives or bool(config.get("include_negatives", True))

    base_model = args.base_model or trainer_cfg.get("base_model", "yolov8n.pt")
    base_model = resolve_base_model(str(base_model), share_root)
    epochs = args.epochs if args.epochs is not None else int(trainer_cfg.get("epochs", 80))
    batch = args.batch if args.batch is not None else int(trainer_cfg.get("batch", 8))
    imgsz = args.imgsz if args.imgsz is not None else int(trainer_cfg.get("imgsz", 960))
    project_dir = Path(args.project_dir or trainer_cfg.get("project_dir", share_root / "runs/train"))

    errors: List[str] = []
    started_at = datetime.now(UTC)

    stats_day = build_dataset(
        golden_root=golden_root,
        dataset_root=dataset_root,
        lighting="day",
        categories=["deer", "person"],
        max_clips=args.max_clips,
        val_split=val_split,
        frame_interval=frame_interval,
        max_frames=max_frames,
        include_negatives=include_negatives,
        dry_run=args.dry_run,
    )
    stats_night = build_dataset(
        golden_root=golden_root,
        dataset_root=dataset_root,
        lighting="night",
        categories=["deer", "person"],
        max_clips=args.max_clips,
        val_split=val_split,
        frame_interval=frame_interval,
        max_frames=max_frames,
        include_negatives=include_negatives,
        dry_run=args.dry_run,
    )

    model_paths: Dict[str, str] = {}
    train_commands: Dict[str, List[str]] = {}

    status = "prepared"
    if not args.dry_run:
        data_day = write_data_yaml(dataset_root, "day")
        data_night = write_data_yaml(dataset_root, "night")

        name_day = f"yolov8n_wildlife_{run_id}_day"
        name_night = f"yolov8n_wildlife_{run_id}_night"

        cmd_day = build_train_command(data_day, base_model, project_dir, name_day, epochs, batch, imgsz)
        cmd_night = build_train_command(data_night, base_model, project_dir, name_night, epochs, batch, imgsz)
        train_commands = {"day": cmd_day, "night": cmd_night}

        if args.execute:
            ret, output = run_training(cmd_day, dry_run=False)
            if ret != 0:
                errors.append(f"Day training failed (exit {ret})")
                errors.append(output[-2000:])
            ret, output = run_training(cmd_night, dry_run=False)
            if ret != 0:
                errors.append(f"Night training failed (exit {ret})")
                errors.append(output[-2000:])

        if args.execute and not errors:
            best_day = project_dir / name_day / "weights" / "best.pt"
            best_night = project_dir / name_night / "weights" / "best.pt"
            staging_dir.mkdir(parents=True, exist_ok=True)
            if best_day.exists():
                target_day = staging_dir / f"{name_day}.pt"
                shutil.copy2(best_day, target_day)
                model_paths["day"] = str(target_day)
            else:
                errors.append(f"Day best.pt not found: {best_day}")
            if best_night.exists():
                target_night = staging_dir / f"{name_night}.pt"
                shutil.copy2(best_night, target_night)
                model_paths["night"] = str(target_night)
            else:
                errors.append(f"Night best.pt not found: {best_night}")
            status = "complete" if not errors else "failed"
        else:
            status = "prepared"
    else:
        status = "dry_run"

    finished_at = datetime.now(UTC)

    summary = RunSummary(
        run_id=run_id,
        status=status,
        started_at=format_utc(started_at),
        finished_at=format_utc(finished_at),
        golden_root=str(golden_root),
        dataset_root=str(dataset_root),
        staging_dir=str(staging_dir),
        model_paths=model_paths,
        train_commands={k: [str(x) for x in v] for k, v in train_commands.items()},
        dataset_stats={
            "day": stats_day.__dict__,
            "night": stats_night.__dict__,
        },
        errors=errors,
    )

    if not args.dry_run:
        logs_dir = REPO_ROOT / DEFAULT_LOGS_DIRNAME
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{run_id}.json"
        log_path.write_text(json.dumps(summary.__dict__, indent=2))
        latest_path = logs_dir / "trainer-latest.json"
        latest_path.write_text(json.dumps(summary.__dict__, indent=2))
        summary.log_path = str(log_path)

    token = os.environ.get("TRAINER_MCP_TOKEN")
    client: Optional[MCPClient] = MCPClient(args.mcp_url, timeout=10, token=token)
    if not client.ping():
        client = None

    register_agent_safe(client, str(REPO_ROOT), args.agent_name, args.skip_mail, errors)

    recipients = parse_recipients(args.recipients or os.environ.get("TRAINER_TO"), args.agent_name)
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
        f"Finished trainer run {summary.run_id} (status={summary.status}, "
        f"day_frames={stats_day.frames_written}, night_frames={stats_night.frames_written})"
    )
    return 0 if summary.status in {"complete", "prepared", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
