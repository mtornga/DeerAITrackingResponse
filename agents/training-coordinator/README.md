# Training Coordinator Agent

The Training Coordinator (TC) scans the training queue, uses `queue_meta.json` enrichment to decide keep/reject, promotes accepted clips into `golden_clips`, and writes a TC run log plus manifest updates.

Primary runtime is the Ubuntu basement host (ubuntubasement).

**Golden layout for new clips**  
`/srv/deer-share/golden_clips/<day|night>/<category>/<clip_id>/`

Negative mining goes to:  
`/srv/deer-share/golden_clips/negatives/snow/<clip_id>/` (cap 15 per run)

## Usage (ubuntubasement)

Dry run:
```bash
ssh mtornga@192.168.68.71 \
  "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && \
   python agents/training-coordinator/run.py --dry-run"
```

Real run:
```bash
ssh mtornga@192.168.68.71 \
  "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && \
   python agents/training-coordinator/run.py"
```

Limit to 10 clips:
```bash
ssh mtornga@192.168.68.71 \
  "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && \
   python agents/training-coordinator/run.py --limit 10"
```

Single clip:
```bash
ssh mtornga@192.168.68.71 \
  "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && \
   python agents/training-coordinator/run.py --only-clip 2026-02-01_segment_120001NightDeer"
```

## CLI Flags
- `--queue-dir`: override training queue path (default: `/srv/deer-share/training_queue`)
- `--share-root`: override share root (default: auto-detect)
- `--dry-run`: do not copy/delete files or write logs/manifest
- `--limit`: max queue dirs to process
- `--only-clip`: process a single queue dir
- `--no-delete`: keep queue dirs even after decisions
- `--run-id`: override run id (default: `TC-YYYYMMDD-HHMMSS`)
- `--negative-cap`: cap for snow/precip negative mining (default 15)
- `--skip-mail`: skip MCP summary message

## Environment Variables
- `TRAINING_COORDINATOR_AGENT`: MCP agent name (default `AmberField`)
- `TRAINING_COORDINATOR_THREAD`: MCP thread id (default `TRAINING-COORD`)
- `TRAINING_COORDINATOR_MCP_URL`: MCP endpoint (default `http://127.0.0.1:8765/mcp/`)
- `TRAINING_COORDINATOR_MCP_TOKEN`: optional bearer token
- `TRAINING_COORDINATOR_TO`: comma-separated recipients for summary
- `TRAINING_COORDINATOR_QUEUE_DIR`: default queue dir override
- `TRAINING_COORDINATOR_NEGATIVE_CAP`: negative mining cap (default 15)

## Decision Rules
- `enrichment.status` must be `complete` or the clip is **held**.
- `triage=keep` and `category in {deer, person, other_wildlife}` → **promote**.
- `triage=reject` and `category=false_positive` and tags include `snow` or `rain` or `flies` → **negative_mine** (cap 15).
- `triage=reject` otherwise → **reject**.
- `triage=maybe` or missing lighting → **hold**.

## Outputs
- Run logs: `runs/logs/training_coordinator/TC-YYYYMMDD-HHMMSS.json`
- Latest pointer: `runs/logs/training_coordinator/training-coordinator-latest.json`
- Manifest: `/srv/deer-share/golden_clips/manifest.json` updated with TC metadata

## Rollback
Use the TC run log to remove directories listed in `dest_dir` and delete their manifest entries. The run log is the canonical rollback handle.
