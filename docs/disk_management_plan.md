# Disk Space Management Plan

**Problem**: Disk regularly exceeds 80% capacity, causing pipeline failures.

**Root Cause Analysis** (Dec 17, 2025):
- Current video: 7680x2160 H.264 @ ~170MB per 60s segment
- Daily ingest at 24/7: ~250GB/day (if running continuously)
- `analysis/` and `events/` duplicate clips unnecessarily
- `events/` and `remote/` directories never pruned

## Current Directory Structure

| Directory | Purpose | Current Size | Pruning |
|-----------|---------|--------------|---------|
| `segments/` | Raw RTSP captures | ~288MB | Yes (72h) |
| `analysis/` | Copy for detector | ~22GB | Yes (72h) |
| `events/` | Promoted detections | ~9.5GB | **NO** |
| `remote/` | Unknown/legacy | ~22GB | **NO** |

## Proposed Changes (Priority Order)

### Phase 1: Immediate Fixes (Today)

1. **Delete `remote/` directory** - appears unused, 22GB savings
2. **Add retention to `events/`** - 7 days max for reviewed events
3. **Reduce retention window** - 24h instead of 72h for segments/analysis

### Phase 2: Reduce File Sizes (This Week)

4. **Transcode to lower resolution for storage**:
   - Keep original 7680x2160 for detection (better accuracy)
   - Transcode promoted events to 1920x540 (quarter res) for review
   - Estimated savings: 75% on event storage

5. **Delete analysis clips after detection**:
   - Current: keep all clips in analysis for 72h
   - Proposed: delete from analysis immediately after detection runs
   - Only keep in `events/` if something interesting found

### Phase 3: Smarter Retention (Next Week)

6. **Tiered retention policy**:
   - `segments/`: 6 hours (raw buffer for re-detection if needed)
   - `analysis/`: Delete immediately after detection
   - `events/` unreviewed: 48 hours, then auto-delete
   - `events/` reviewed (tagged): 14 days
   - `events/` with "keep" tag: permanent (move to archive)

7. **Automatic disk monitoring**:
   - Cron job to check disk usage every hour
   - If >75%: aggressive pruning (12h retention)
   - If >85%: emergency mode (delete all but last 6h)
   - Alert via log file

## Implementation Tasks

### Immediate (can do now)

```bash
# 1. Delete remote directory
rm -rf /srv/deer-share/runs/live/remote/

# 2. Reduce retention to 24h in run_pipeline.sh
RETENTION_HOURS=24
```

### Script Changes Needed

1. **`scripts/reolink_stream_ingest.py`**:
   - Add `--delete-after-detection` flag
   - When detector marks a segment as processed, delete from analysis/

2. **`scripts/live_detector_multimodel.py`**:
   - After processing, write a `.processed` marker
   - Ingest script watches for markers and deletes

3. **New script: `scripts/prune_events.py`**:
   - Delete events older than N days
   - Respect "keep" tag in daily_review_index.json
   - Optionally transcode before archiving

4. **New script: `scripts/disk_watchdog.py`**:
   - Cron job to monitor disk
   - Escalating retention based on usage

## Target Disk Budget

| Directory | Max Size | Retention |
|-----------|----------|-----------|
| `segments/` | 15GB | 6h rolling |
| `analysis/` | 0GB | Delete after detection |
| `events/` | 50GB | Tiered (48h/14d/perm) |
| `detections/` | 1GB | 30 days |
| **Total** | ~70GB | - |

With 492GB disk and 70GB for pipeline, leaves 85% for other uses.

## Quick Win: Resolution Reduction

If we must keep videos longer, transcode on promotion:

```bash
# Transcode to 1920x540 (1/4 pixels), CRF 28
ffmpeg -i input.mkv -vf "scale=1920:540" -c:v libx264 -crf 28 -preset fast output.mkv
# Typical savings: 170MB -> 15-25MB
```

## Recommended Immediate Actions

1. ✅ Delete `/srv/deer-share/runs/live/remote/` (22GB)
2. Change `RETENTION_HOURS=24` in run_pipeline.sh
3. Create cron job: `0 * * * * /path/to/prune_events.py --max-age-days 7`
