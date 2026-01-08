# Day Watchman Tasks

Deterministic patrol tasks that mirror the Night Watchman flow. A Codex reporter
reads the snapshot + digest and sends the final report via MCP Agent Mail.

## Steps (8)
Collector (deterministic):
1. Register agent and announce patrol start.
2. Health check: disk usage and `scripts/pipeline_health_check.py --lighting day`.
3. Backlog check: count pending detections (analysis segments missing detection meta),
   classify status, run diagnosis if HIGH/CRITICAL.
4. Detection summary: scan today + yesterday event segments for counts and confidence.
5. Frame examination: analyze the highest-confidence segment's metadata for movement.
6. Training queue: copy up to 6 deer/person segments into `/srv/deer-share/training_queue/`.
7. Write digest log and snapshot JSON.

Reporter (Codex):
8. Read snapshot + digest, write analysis notes, send final report via MCP Agent Mail, then check inbox.

## Outputs
- Digest log: `runs/logs/watchman/day-watchman-YYYY-MM-DD.md`
- Snapshot JSON: `runs/logs/watchman/day-watchman-YYYY-MM-DD.json`
- Latest snapshot: `runs/logs/watchman/day-watchman-latest.json`
- Analysis notes: `runs/logs/watchman/day-watchman-analysis-YYYY-MM-DD.md`
- Report body: `runs/logs/watchman/day-watchman-report-YYYY-MM-DD.md`
- Training queue: `/srv/deer-share/training_queue/DATE_segment_XXXXXX/`
- Agent Mail thread: `DAY-WATCHMAN`

## Configuration (env overrides)
- `DAYWATCHMAN_AGENT` (default: BrightHeron)
- `DAYWATCHMAN_THREAD` (default: DAY-WATCHMAN)
- `DAYWATCHMAN_MCP_URL` (default: http://127.0.0.1:8765/mcp/)
- `DAYWATCHMAN_MCP_TOKEN` (optional bearer token)
- `DAYWATCHMAN_TO` (comma-separated recipients; defaults to agent itself)
- `DAYWATCHMAN_PROGRESS_MAIL` (set to true to send step-by-step mail)
- `DAYWATCHMAN_HEALTH_TIMEOUT` (default: 600 seconds)
- `DAYWATCHMAN_MAX_QUEUE` (default: 6)
- `DAYWATCHMAN_ANALYSIS_DIR` (default: /srv/deer-share/runs/live/analysis)
- `DAYWATCHMAN_DETECTIONS_DIR` (default: /srv/deer-share/runs/live/detections)
- `DAYWATCHMAN_EVENTS_DIR` (default: /srv/deer-share/runs/live/events)
- `DAYWATCHMAN_QUEUE_DIR` (default: /srv/deer-share/training_queue)

## Cron
Use `agents/daywatchman/cron.example` for a 3pm Central schedule.
`agents/daywatchman/run.sh` runs the collector and then the Codex reporter.
