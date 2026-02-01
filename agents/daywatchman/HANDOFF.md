# Day Watchman Handoff

## Overview
Day Watchman is a two-phase patrol:
- **Collector (deterministic)**: `agents/daywatchman/run.py`
- **Reporter (Codex)**: `agents/daywatchman/report.py`

The collector gathers health/backlog/detection info and writes logs. The reporter reads
the snapshot + digest, writes analysis/report notes, and sends the final MCP Agent Mail
message (with a deterministic fallback if Codex fails or doesn't send).

## Schedule (Ubuntu)
Cron (installed on the Ubuntu server):
```
CRON_TZ=America/Chicago
0 15 * * * /home/mtornga/projects/DeerAITrackingResponse/agents/daywatchman/run.sh >> /var/log/day-watchman.log 2>&1
```

## How it Runs
`agents/daywatchman/run.sh`:
1. Activates `.venv`
2. Runs the collector with `--collect-only`
3. Runs the Codex reporter (unless `--dry-run` or `--collect-only` passed)

## Key Outputs
All files live in `runs/logs/watchman/`:
- `day-watchman-YYYY-MM-DD.md` (digest)
- `day-watchman-YYYY-MM-DD.json` (snapshot)
- `day-watchman-latest.json` (latest snapshot)
- `day-watchman-analysis-YYYY-MM-DD.md` (reporter analysis)
- `day-watchman-report-YYYY-MM-DD.md` (final report body)

Agent Mail archive:
`~/.mcp_agent_mail_git_mailbox_repo/projects/home-mtornga-projects-deeraitrackingresponse/messages/`

## Backlog Definition (Important)
Backlog count is **pending detections**, not total analysis clips:
- **analysis_count** = all `/srv/deer-share/runs/live/analysis/**/segment_*.mkv`
- **pending_count** = clips missing `/srv/deer-share/runs/live/detections/<date>/<segment>/meta.json`

This avoids a misleading "stuck at 1371" backlog. The report now prints:
`Backlog: <pending> pending of <analysis_count>`.

## Progress Messages
Progress mail is **off by default** to avoid extra chatter.
When enabled, only a single Patrol Starting notice is sent (no step-by-step mail).
Enable via:
- env: `DAYWATCHMAN_PROGRESS_MAIL=true`
- flag: `--progress-mail`

## Codex Reporter Notes
- Codex CLI is installed via npm at `~/.npm-global/bin/codex`.
- `run.sh` sets `CODEX_BIN` and adds `~/.npm-global/bin` to `PATH`.
- Reporter uses `codex exec -s workspace-write --skip-git-repo-check -`.

If Codex runs but no report is sent, `report.py` verifies:
- analysis file updated
- report body updated
- patrol message exists in Agent Mail archive

If any are missing, it sends a deterministic fallback report instead.

## Quick Commands
Collector only:
```
./agents/daywatchman/run.py --collect-only --verbose
```

Full run (collector + reporter):
```
./agents/daywatchman/run.sh --verbose
```

Reporter only:
```
./agents/daywatchman/report.py --verbose
```

## Environment Overrides
- `DAYWATCHMAN_AGENT` (default BrightHeron)
- `DAYWATCHMAN_THREAD` (default DAY-WATCHMAN)
- `DAYWATCHMAN_MCP_URL` (default http://127.0.0.1:8765/mcp/)
- `DAYWATCHMAN_MCP_TOKEN` (optional bearer)
- `DAYWATCHMAN_TO` (recipients, comma-separated)
- `DAYWATCHMAN_HEALTH_TIMEOUT` (default 600s)
- `DAYWATCHMAN_REPORT_TIMEOUT` (default 420s)
- `DAYWATCHMAN_MAX_QUEUE` (default 6)
- `DAYWATCHMAN_PROGRESS_MAIL` (false by default)
- `DAYWATCHMAN_ANALYSIS_DIR` (default /srv/deer-share/runs/live/analysis)
- `DAYWATCHMAN_DETECTIONS_DIR` (default /srv/deer-share/runs/live/detections)
- `DAYWATCHMAN_EVENTS_DIR` (default /srv/deer-share/runs/live/events)
- `DAYWATCHMAN_QUEUE_DIR` (default /srv/deer-share/training_queue)
- `DAYWATCHMAN_DETECTOR_LOG` (default /srv/deer-share/runs/live/logs/detector.log)
- `DAYWATCHMAN_PUSHOVER_STALE_HOURS` (default 24)
- `DAYWATCHMAN_PUSHOVER_LOG_LINES` (default 5000)

## Troubleshooting
- If backlog stays HIGH/CRITICAL, check detector status and prune logs:
  - `pgrep -f live_detector`
  - `/srv/deer-share/logs/prune_events.log`
- If Codex gets slow, consider disabling Playwright MCP in `~/.codex/config.toml`.
- If final reports are empty, check `day-watchman-report-YYYY-MM-DD.md`
  and the Agent Mail archive for the last `Patrol Complete` message.
