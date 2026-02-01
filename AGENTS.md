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

## Pipeline Health Check

When asked "Is the pipeline operational?" run:

```bash
ssh mtornga@192.168.68.71 "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && python scripts/pipeline_health_check.py --verbose"
```

**Interpret output**:
- `HEALTHY: All 2 test clips passed` = Pipeline working correctly
- `UNHEALTHY: ...` = Pipeline has issues, read error details

**If unhealthy**, also check:
```bash
# Pipeline process status
ssh mtornga@192.168.68.71 "cd ~/projects/DeerAITrackingResponse && ./scripts/run_pipeline.sh status"

# Disk space (common failure cause)
ssh mtornga@192.168.68.71 "df -h /"

# Recent detector logs
ssh mtornga@192.168.68.71 "tail -30 /srv/deer-share/runs/live/logs/detector.log"
```

**Test specific models**:
```bash
python scripts/pipeline_health_check.py --verbose --models yolov8n.pt
python scripts/pipeline_health_check.py --verbose --models yolov8n_wildlife.pt,rtdetr_wildlife.pt
```

**After model retraining**: Always run health check to catch regressions like AprilTag misclassification.

## MCP Agent Mail: coordination for multi-agent workflows

### What it is
- A mail-like layer that lets coding agents coordinate asynchronously via MCP tools and resources.
- Provides identities, inbox/outbox, searchable threads, and advisory file reservations, with human-auditable artifacts in Git.

### Why it's useful
- Prevents agents from stepping on each other with explicit file reservations (leases) for files/globs.
- Keeps communication out of your token budget by storing messages in a per-project archive.
- Offers quick reads (`resource://inbox/...`, `resource://thread/...`) and macros that bundle common flows.

### How to use effectively
1) Same repository
   - Register an identity: call `ensure_project`, then `register_agent` using this repo's absolute path as `project_key`.
   - Reserve files before you edit: `file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true)` to signal intent and avoid conflict.
   - Communicate with threads: use `send_message(..., thread_id="FEAT-123")`; check inbox with `fetch_inbox` and acknowledge with `acknowledge_message`.
   - Read fast: `resource://inbox/{Agent}?project=<abs-path>&limit=20` or `resource://thread/{id}?project=<abs-path>&include_bodies=true`.
   - Tip: set `AGENT_NAME` in your environment so the pre-commit guard can block commits that conflict with others' active exclusive file reservations.

2) Across different repos in one project (e.g., Next.js frontend + FastAPI backend)
   - Option A (single project bus): register both sides under the same `project_key` (shared key/path). Keep reservation patterns specific (e.g., `frontend/**` vs `backend/**`).
   - Option B (separate projects): each repo has its own `project_key`; use `macro_contact_handshake` or `request_contact`/`respond_contact` to link agents, then message directly. Keep a shared `thread_id` (e.g., ticket key) across repos for clean summaries/audits.

### Macros vs granular tools
- Prefer macros when you want speed or are on a smaller model: `macro_start_session`, `macro_prepare_thread`, `macro_file_reservation_cycle`, `macro_contact_handshake`.
- Use granular tools when you need control: `register_agent`, `file_reservation_paths`, `send_message`, `fetch_inbox`, `acknowledge_message`.

### Common pitfalls
- "from_agent not registered": always `register_agent` in the correct `project_key` first.
- "FILE_RESERVATION_CONFLICT": adjust patterns, wait for expiry, or use a non-exclusive reservation when appropriate.
- Auth errors: if JWT+JWKS is enabled, include a bearer token with a `kid` that matches server JWKS; static bearer is used only when JWT is disabled.

## Additional
Always use context7 when I need code generation, setup or configuration steps, or
library/API documentation. This means you should automatically use the Context7 MCP
tools to resolve library id and get library docs without me having to explicitly ask.

For AWS this command seems to work for auth and accessing s3: set -a; source .env; set +a; unset AWS_PROFILE AWS_SESSION_TOKEN; aws s3 ls s3://wildlife-ugv/models/yolo/

<!-- bv-agent-instructions-v1 -->

---

## Beads Workflow Integration

This project uses [beads_viewer](https://github.com/Dicklesworthstone/beads_viewer) for issue tracking. Issues are stored in `.beads/` and tracked in git.

### Essential Commands

```bash
# View issues (launches TUI - avoid in automated sessions)
bv

# CLI commands for agents (use these instead)
bd ready              # Show issues ready to work (no blockers)
bd list --status=open # All open issues
bd show <id>          # Full issue details with dependencies
bd create --title="..." --type=task --priority=2
bd update <id> --status=in_progress
bd close <id> --reason="Completed"
bd close <id1> <id2>  # Close multiple issues at once
bd sync               # Commit and push changes
```

### Workflow Pattern

1. **Start**: Run `bd ready` to find actionable work
2. **Claim**: Use `bd update <id> --status=in_progress`
3. **Work**: Implement the task
4. **Complete**: Use `bd close <id>`
5. **Sync**: Always run `bd sync` at session end

### Key Concepts

- **Dependencies**: Issues can block other issues. `bd ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers, not words)
- **Types**: task, bug, feature, epic, question, docs
- **Blocking**: `bd dep add <issue> <depends-on>` to add dependencies

### Session Protocol

**Before ending any session, run this checklist:**

```bash
git status              # Check what changed
git add <files>         # Stage code changes
bd sync                 # Commit beads changes
git commit -m "..."     # Commit code
bd sync                 # Commit any new beads changes
git push                # Push to remote
```

### Best Practices

- Check `bd ready` at session start to find available work
- Update status as you work (in_progress → closed)
- Create new issues with `bd create` when you discover tasks
- Use descriptive titles and set appropriate priority/type
- Always `bd sync` before ending session

<!-- end-bv-agent-instructions -->
