#!/usr/bin/env python3
"""
Launch a RunPod GPU pod for training and (optionally) sync a dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_repo_root_on_path() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / ".env").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    fallback = script_path.parents[1]
    if str(fallback) not in sys.path:
        sys.path.insert(0, str(fallback))
    return fallback


REPO_ROOT = _ensure_repo_root_on_path()

try:
    from env_loader import load_env_file  # type: ignore

    load_env_file()
except Exception:
    pass


def format_utc(ts: datetime) -> str:
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_runpod() -> Any:
    try:
        import runpod  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "RunPod SDK not installed. Run: pip install runpod"
        ) from exc
    return runpod


def parse_env_pairs(values: Optional[List[str]]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not values:
        return env
    for item in values:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        env[key] = value.strip()
    return env


def print_gpus(runpod: Any) -> None:
    gpus = runpod.get_gpus()
    for gpu in gpus:
        name = gpu.get("displayName") or gpu.get("id")
        memory = gpu.get("memoryInGb")
        gpu_id = gpu.get("id")
        print(f"{name} | {memory}GB | {gpu_id}")


def wait_for_pod(runpod: Any, pod_id: str, timeout_s: int, poll_s: int) -> Dict[str, Any]:
    start = datetime.now(UTC)
    while (datetime.now(UTC) - start).total_seconds() < timeout_s:
        pod = runpod.get_pod(pod_id)
        if not isinstance(pod, dict):
            status = None
        else:
            runtime = pod.get("runtime")
            if not isinstance(runtime, dict):
                runtime = {}
            status = runtime.get("status") or pod.get("status")
        if status and str(status).lower() in {"running", "ready"}:
            return pod
        time_left = timeout_s - int((datetime.now(UTC) - start).total_seconds())
        print(f"[runpod] waiting... status={status} ({time_left}s left)")
        import time

        time.sleep(poll_s)
    return runpod.get_pod(pod_id)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-gpus", action="store_true", help="List available GPU types")
    parser.add_argument("--name", help="Pod name (default: deer-train-YYYYMMDD-HHMMSS)")
    parser.add_argument("--gpu", help="GPU type (display name or id)")
    parser.add_argument("--gpu-count", type=int, default=1, help="GPU count")
    parser.add_argument(
        "--image",
        default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        help="Docker image name",
    )
    parser.add_argument("--container-gb", type=int, default=50, help="Container disk size in GB")
    parser.add_argument("--volume-gb", type=int, default=200, help="Volume size in GB")
    parser.add_argument(
        "--ports",
        default="8888/http,22/tcp",
        help="Ports to expose (e.g. 22/tcp,8888/http)",
    )
    parser.add_argument("--env", action="append", help="Env var (KEY=VALUE). Repeatable.")
    parser.add_argument("--jupyter-password", help="Jupyter password to set (JUPYTER_PASSWORD)")
    parser.add_argument(
        "--docker-args",
        default="jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser",
        help="Docker args / start command",
    )
    parser.add_argument("--wait", action="store_true", help="Wait for pod to be running")
    parser.add_argument("--wait-timeout", type=int, default=900, help="Wait timeout in seconds")
    parser.add_argument("--wait-poll", type=int, default=10, help="Polling interval in seconds")

    parser.add_argument("--sync-dataset", help="Local dataset path to sync to pod")
    parser.add_argument("--remote-path", default="/workspace/golden_frames_v8", help="Remote path for dataset")
    parser.add_argument("--ssh-target", help="SSH target (e.g. root@host -p 12345)")
    parser.add_argument("--execute-sync", action="store_true", help="Run rsync after launch")
    parser.add_argument("--json-out", help="Write pod details to JSON file")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    runpod = ensure_runpod()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is not set in the environment.")
    runpod.api_key = api_key

    if args.list_gpus:
        print_gpus(runpod)
        return 0

    if not args.gpu:
        raise SystemExit("--gpu is required unless --list-gpus is used.")

    name = args.name or datetime.now(UTC).strftime("deer-train-%Y%m%d-%H%M%S")
    env_vars = parse_env_pairs(args.env)
    if args.jupyter_password and "JUPYTER_PASSWORD" not in env_vars:
        env_vars["JUPYTER_PASSWORD"] = args.jupyter_password

    pod = runpod.create_pod(
        name=name,
        image_name=args.image,
        gpu_type_id=args.gpu,
        gpu_count=args.gpu_count,
        volume_in_gb=args.volume_gb,
        container_disk_in_gb=args.container_gb,
        ports=args.ports,
        env=env_vars,
        docker_args=args.docker_args,
    )

    pod_id = pod.get("id") if isinstance(pod, dict) else None
    if not pod_id:
        raise SystemExit("Failed to create pod (missing id).")

    if args.wait:
        pod = wait_for_pod(runpod, pod_id, args.wait_timeout, args.wait_poll)

    pod_host = None
    if isinstance(pod, dict):
        pod_host = pod.get("machine", {}).get("podHostId") or pod.get("runtime", {}).get("podHostId")

    summary = {
        "created_at": format_utc(datetime.now(UTC)),
        "pod_id": pod_id,
        "name": name,
        "image": args.image,
        "gpu": args.gpu,
        "gpu_count": args.gpu_count,
        "ports": args.ports,
        "pod_host": pod_host,
    }
    print(json.dumps(summary, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2))

    if args.sync_dataset:
        sync_src = Path(args.sync_dataset).expanduser()
        if not sync_src.exists():
            raise SystemExit(f"Dataset path not found: {sync_src}")
        ssh_target = args.ssh_target or pod_host
        if not ssh_target:
            print("[runpod] No SSH target available; skipping sync.")
            return 0
        rsync_cmd = [
            "rsync",
            "-az",
            "--info=progress2",
            f"{sync_src}/",
            f"{ssh_target}:{args.remote_path}/",
        ]
        print("[runpod] rsync command:")
        print(" ".join(rsync_cmd))
        if args.execute_sync:
            subprocess.run(rsync_cmd, check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
