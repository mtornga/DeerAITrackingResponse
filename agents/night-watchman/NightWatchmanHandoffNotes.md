# Night Watchman Handoff Notes

**Date**: 2026-01-05
**Status**: Operational (v2.8)

## What Was Done

### Original Request
Increment the Night Watchman agent to:
1. Fix detection summary (was showing "High-confidence count unavailable")
2. Add frame examination to validate detections
3. Add clip tagging via Review UI
4. Queue clips for training (to avoid 24h pruning)
5. Add backlog monitoring

### Implementation Journey

| Version | Changes | Outcome |
|---------|---------|---------|
| v2.4 | Starting point | 8 steps, basic patrol |
| v2.5 | Added Puppeteer for Review UI tagging, 12 steps | Timeout at 10 min |
| v2.6 | Removed Puppeteer, simplified to 10 steps | Still timeout at 7 min |
| v2.7 | Batched operations with bash/jq | Still timeout at 7 min |
| v2.8 | Added progress messages, removed playwright MCP | **Success** - 2 min runtime |

### Root Cause of Timeouts
The **playwright MCP server** was causing massive initialization overhead, even when not being used. Removing it with `claude mcp remove playwright` fixed the issue.

### What's Implemented (v2.8)

**6-Step Workflow:**
1. **Step 0**: Register agent + announce patrol starting
2. **Step 1**: Health check (disk + pipeline) - single bash command
3. **Step 2**: Detection summary - batched jq extraction of all meta.json
4. **Step 3**: Queue for training - copy up to 6 deer/person clips
5. **Step 4**: Write digest to `runs/logs/watchman/YYYY-MM-DD.md`
6. **Step 5**: Send final report via Agent Mail
7. **Step 6**: Check inbox and acknowledge messages

**Key Features:**
- Progress messages via Agent Mail at each step (visibility)
- Batched bash/jq operations (speed)
- Training queue at `/srv/deer-share/training_queue/` (persists beyond 24h pruning)
- Properly reads `meta.json` for confidence data

### What's NOT Implemented (Deferred)

1. **Clip tagging via Review UI** - Removed due to Puppeteer timeout issues. Human does tagging manually at `http://192.168.68.71:8501/`

2. **Frame examination with screenshots** - Simplified to metadata-only analysis. Agent looks at bbox positions/variance but doesn't take visual screenshots.

3. **Detailed false positive heuristics** - The AprilTag detection logic is documented but not deeply implemented. Agent provides basic assessment.

## Current State

### Files
- `SKILL_v2.md` - Main skill definition (v2.8)
- `run.py` - Python launcher (finds claude binary, sets timeout)
- `run.sh` - Shell wrapper for cron
- `cron.example` - Cron schedule (0 7 * * * = 1am CT)

### Configuration
- **Timeout**: 420s (7 minutes) - patrol completes in ~2 min
- **Allowed tools**: `Read,Write,Bash,mcp__mcp-agent-mail__*`
- **Agent name**: CalmEagle
- **Thread**: NIGHT-WATCHMAN

### Dependencies
- MCP Agent Mail server at `http://127.0.0.1:8765/mcp/`
- `jq` for JSON parsing
- Pipeline health check script
- Shared storage at `/srv/deer-share/`

## Known Issues

1. **Backlog**: Last patrol showed 1,371 pending segments (CRITICAL threshold is >100). The detection pipeline may need attention.

2. **No visual verification**: Agent can't actually see frames, only analyzes metadata. May miss some false positives that a human would catch.

3. **Playwright removed**: If you need browser automation in the future, you'll need to add it back carefully (maybe as a separate agent to avoid initialization overhead).

## Multi-Agent Vision

The Night Watchman is part of a planned multi-agent system:

| Agent | Schedule | Status |
|-------|----------|--------|
| Night Watchman | 1am daily | **Operational** |
| Day Watchman | ~3pm daily | Future |
| Model Training Master | Weekly | Future |

The training queue (`/srv/deer-share/training_queue/`) is ready for the future Model Training Master to consume.

## Testing

```bash
# From Ubuntu server
cd ~/projects/DeerAITrackingResponse
source .venv/bin/activate
python agents/night-watchman/run.py --verbose

# Dry run
python agents/night-watchman/run.py --dry-run
```

## Maintenance

- **Logs**: `runs/logs/watchman/YYYY-MM-DD.md`
- **Agent Mail thread**: NIGHT-WATCHMAN
- **Training queue**: `/srv/deer-share/training_queue/`

If timeouts return, check:
1. `claude mcp list` - are new MCP servers added?
2. Network latency to Claude API
3. Complexity of skill file
