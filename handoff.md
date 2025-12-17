# Developer Handoff Notes
**Date**: 2025-12-17 (Evening Session)
**Session**: Disk Space Management & Git Sync
**Status**: Pipeline Online, Disk at 65%, Automated Pruning Active

## What Was Accomplished

### 1. Cleaned Up Old Clips and Freed Disk Space
**Problem**: Disk at 87% with old clips from 12/14-12/16 still present.

**Solution**:
- Deleted `/srv/deer-share/runs/live/analysis/2025-12-16/`
- Deleted `/srv/deer-share/runs/live/segments/2025-12-16/`
- Deleted `/srv/deer-share/runs/live/detections/2025-12-{14,15,16}/`
- Deleted `/srv/deer-share/runs/live/events/2025-12-{14,15,16}/`
- Deleted `/srv/deer-share/runs/live/remote/` (22GB - was being recreated by ingest)

**Result**: Disk now at 65% (168GB free)

### 2. Implemented Disk Space Management System
**Problem**: Disk space emergencies kept recurring. No automated pruning of events directory.

**Solution**: Created comprehensive disk management:

1. **Reduced retention from 72h to 24h** in `run_pipeline.sh`
2. **Disabled remote mirror** - was duplicating all clips unnecessarily
3. **Switched analysis to hardlinks** - shares disk blocks with segments instead of copying
4. **Created `scripts/prune_events.py`** with tiered retention:
   - Events with "keep" tag: kept forever
   - Reviewed with notes: 14 days
   - Reviewed without notes: 7 days
   - Pending review: 48 hours
   - Unindexed: 24 hours
5. **Added cron job** - runs every 4 hours to prune old events

**New Files**:
- `scripts/prune_events.py` - Event pruning script with tiered retention
- `docs/disk_management_plan.md` - Full analysis and roadmap

### 3. Synced Git Between Mac and Ubuntu Server
**Problem**: Server was 4 commits behind main with local uncommitted changes.

**Solution**:
- Stashed server changes, pulled latest from origin/main
- Merged server's useful changes (IGNORED_CLASSES filter) into main
- Server now in sync with main

**Committed Changes**:
- `d39f7b3`: Add IGNORED_CLASSES filter to detector (filters apriltag, unknown_animal, etc.)
- `ba9f80a`: Archive and gitignore TableTopSimulation/ and cutebot/
- `d744f35`: Add disk space management and event pruning
- `3483dfb`: Disable remote mirror, use hardlinks for analysis

### 4. Archived Inactive Code
**Problem**: TableTopSimulation/ and cutebot/ cluttering repo, not needed for outdoor pipeline work.

**Solution**:
- Created `TableTopSimulation.zip` and `cutebot.zip` archives
- Added both directories to `.gitignore`
- Removed from git tracking

### 5. Added AprilTag Filtering to Detector
**Problem**: Wildlife models were detecting AprilTags as animals.

**Solution**: Added `IGNORED_CLASSES` set to `scripts/live_detector_multimodel.py`:
- Filters out: apriltag, april_tag, tag, unknown_animal
- Only returns known COCO animal classes instead of defaulting everything to "animal"

---

## Current System State

### Pipeline Status
- **Ingest**: ONLINE - capturing 60s segments at 7680x2160
- **Detection**: RUNNING with base `yolov8n.pt`
- **Disk**: 65% (168GB free)
- **Retention**: 24 hours (reduced from 72h)

### Disk Usage Breakdown
```
/srv/deer-share/runs/live/
├── segments/     23GB  (24h retention, auto-pruned)
├── analysis/     23GB  (hardlinks to segments, 24h retention)
├── events/       9.5GB (tiered retention, cron every 4h)
├── detections/   636KB (30 days)
└── logs/         4.5MB
```

### Cron Jobs Active
```bash
# Event pruning - every 4 hours
0 */4 * * * cd ~/projects/DeerAITrackingResponse && .venv/bin/python scripts/prune_events.py

# Segment pruning (existing) - every 15 minutes
*/15 * * * * cd ~/projects/deer-vision && .venv/bin/python scripts/prune_segments_without_events.py
```

### Monitoring Commands
```bash
# Check disk space
ssh mtornga@192.168.68.71 "df -h / && du -sh /srv/deer-share/runs/live/*/"

# Check pipeline status
ssh mtornga@192.168.68.71 "cd ~/projects/DeerAITrackingResponse && ./scripts/run_pipeline.sh status"

# Run event pruning manually (dry run)
ssh mtornga@192.168.68.71 "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && python scripts/prune_events.py --dry-run -v"

# Check pruning logs
ssh mtornga@192.168.68.71 "tail -50 /srv/deer-share/runs/live/logs/prune_events.log"
```

---

## Known Issues

### 1. Wildlife Models Still Broken (CRITICAL)
**Status**: Unchanged from previous session.

Models `yolov8n_wildlife.pt`, `yolov8n_wildlife_v2.pt`, and `rtdetr_wildlife_v2.pt` fail health check.

**Fix Required**: Retrain without AprilTag labels in training dataset.

### 2. High Video Resolution
**Issue**: Camera captures at 7680x2160 (~170MB per 60s segment).

**Impact**: ~10GB/hour if running continuously.

**Future Fix**: Could transcode to lower resolution for storage, but keeping full res for now for detection accuracy.

---

## Next Steps (Priority Order)

### Immediate
1. ~~Disk space management~~ DONE
2. **Retrain wildlife models** without AprilTag labels

### Short-term
1. **Monitor disk usage** - verify 24h retention keeps disk under 80%
2. **Add disk watchdog** - escalating retention based on usage
3. **Implement delete-after-detection** - remove from analysis/ immediately after processing

### Medium-term
1. **Transcode events to lower resolution** for long-term storage
2. **Daily health check cron job**
3. **Alert on disk >80%**

---

## Code Structure Reference

### New Files This Session
```
scripts/
├── prune_events.py           # Event pruning with tiered retention

docs/
└── disk_management_plan.md   # Full disk management analysis and roadmap
```

### Key Configuration Changes
```bash
# run_pipeline.sh
RETENTION_HOURS=24                    # Was 72
--analysis-mirror-mode hardlink       # Was copy
--remote-mirror-mode none             # Was hardlink (disabled)
```

### Archived (in .gitignore)
```
TableTopSimulation/    # Zipped as TableTopSimulation.zip
cutebot/               # Zipped as cutebot.zip
```

---

## Streamlit Daily Review

- **URL**: http://192.168.68.71:8502/
- **Notes for segment_182217.mkv**: "Two adults two children" (tagged: person, hard_eval)
- Index stored at: `/srv/deer-share/index/daily_review_index.json`

---

## Session Summary

| Task | Status |
|------|--------|
| Delete old clips (12/16 and before) | DONE |
| Sync git between Mac and server | DONE |
| Add AprilTag filtering to detector | DONE |
| Create event pruning script | DONE |
| Set up pruning cron job | DONE |
| Reduce retention to 24h | DONE |
| Disable remote mirror | DONE |
| Switch analysis to hardlinks | DONE |
| Archive TableTopSimulation/cutebot | DONE |

**Key Improvement**: Disk space is now automatically managed with tiered retention policies. The system should stay under 80% usage going forward.

---

**Generated**: 2025-12-17 (Evening) by Claude Code
**Handoff to**: Next development session
