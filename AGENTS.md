# Repository Guidelines

## Project Structure & Module Organization
Scripts for data capture and training live in `scripts/` (calibration, RTSP capture, YOLO dataset prep). Indoor tracking demos and assets are in `demo/`, while calibration matrices and geometry helpers live under `calibration/`. Place CVAT exports inside `cvat/` and trained weights under `models/` (Ultralytics outputs flow into `runs/`). Shared helpers such as `env_loader.py` stay at the root; prefer colocating domain code with the hardware or workflow it serves.

## Build, Test, and Development Commands
Start every session inside a virtualenv and install the pinned stack so OpenCV and Ultralytics load cleanly:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir --force-reinstall -r constraints.txt
```

Capture footage via `python scripts/capture_rtsp_clip.py --seconds 10` or stills with `--interval`. Top-down validation runs through `python demo/topdown_tracker.py --rtsp $WYZE_TABLETOP_RTSP`. Train detection models using the command logged in `trainingruncommand.txt`, updating the `project`/`name` flags as needed.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and snake_case naming (see `env_loader.py`). Keep modules import-sorted (stdlib, third-party, local) and add type hints on public helpers. Configuration belongs in JSON alongside consumers (`calibration/tabletop_homography.json`) or `.env` keys loaded via `env_loader.require_env`. Raise descriptive exceptions instead of silent `print` statements so scripts fail fast on bad camera or file paths.

## Testing Guidelines
Automated coverage is light; add `pytest` modules under a new `tests/` folder when extending calibration math or tracker logic. Reproduce high-value scenarios with the recorded assets in `demo/` and compare generated `detections_world.csv` rows before and after. When introducing model changes, capture at least one tabletop clip and note confidence deltas in the PR. Treat `runs/`, `logs/`, and `tmp_shm/` as throwaway artifacts—never commit them.

## Commit & Pull Request Guidelines
Git history favors concise, present-tense subjects (e.g., `iterations on cutebot movement`, `chore: ignore .DS_Store`). Keep each commit scoped to one behavior or dataset tweak, and mention calibration data sources when they change. Pull requests should link issues, summarize testing commands, and attach screenshots or short clips whenever output imagery shifts. Call out any new S3 paths or credentials needed so operators can refresh the deployment docs.

## Additional
Always use context7 when I need code generation, setup or configuration steps, or
library/API documentation. This means you should automatically use the Context7 MCP
tools to resolve library id and get library docs without me having to explicitly ask.

For AWS this command seems to work for auth and accessing s3: set -a; source .env; set +a; unset AWS_PROFILE AWS_SESSION_TOKEN; aws s3 ls s3://wildlife-ugv/models/yolo/