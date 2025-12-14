# Weekly Plan: Daily Workflow Activation

**Created**: 2024-12-14
**Goal**: Get the morning review workflow operational so mtornga sees deer clips daily

## Vision Reminder (from NewNotes.txt)

> "In the morning, mtornga should be presented with clips that need user input. False positives, false negatives, quality issues, unknown animals, and so on. The fidelity of detection, tracking, classification should improve based on this user feedback."

## Current State Assessment

| Component | Status | Notes |
|-----------|--------|-------|
| Infrastructure | ✅ Done | Mac↔Ubuntu SSH, Samba share, GPU env |
| Ingest script | ✅ Exists | `reolink_stream_ingest.py` - not running |
| Detector script | ✅ Exists | `live_megadetector.py` - not running |
| Daily Review UI | ✅ Exists | `daily_review_app.py` - 57 old clips indexed |
| **Live Pipeline** | ❌ Stopped | No processes running, no new clips since Nov 30 |

**The scripts exist but nothing is running. The morning review isn't happening.**

---

## This Week's Plan

### Phase 1: Smoke Test the Pipeline (Day 1)

**Goal**: Manually run each component end-to-end to verify everything still works.

#### 1.1 Verify Camera Access
- Test RTSP stream from Reolink camera
- Confirm credentials in `.env` are current
- Quick capture: `python scripts/capture_rtsp_clip.py --seconds 10`

#### 1.2 Test Stream Ingest
- Run ingest manually for 5 minutes on Ubuntu
- Verify segments appear in `/srv/deer-share/runs/live/segments/`
- Command: `python scripts/reolink_stream_ingest.py --duration 300`

#### 1.3 Test MegaDetector on a Segment
- Run detector on one captured segment
- Verify detection JSON and marked video output
- Command: `python scripts/live_megadetector.py --once`

#### 1.4 Test Daily Review UI
- Start Streamlit: `streamlit run scripts/daily_review_app.py`
- Verify new clip appears in the review queue
- Tag one clip to confirm persistence works

**Exit Criteria**: One clip flows through ingest → detect → review UI

---

### Phase 2: Continuous Pipeline (Day 2-3)

**Goal**: Pipeline runs unattended, survives disconnects.

#### 2.1 Create Pipeline Runner Script
- New script: `scripts/run_pipeline.sh`
- Starts both ingest and detector in supervised mode
- Uses `tmux` or `systemd` for persistence
- Logs to `/srv/deer-share/runs/live/logs/`

#### 2.2 Configure Auto-Restart
- Add watchdog to restart on crash
- Log rotation to prevent disk fill
- Health check endpoint or file

#### 2.3 Test Overnight Run
- Start pipeline before bed
- Check morning for:
  - New clips in `/srv/deer-share/runs/live/events/`
  - Detector ran on all segments
  - No crashes or disk issues

**Exit Criteria**: Pipeline runs overnight, produces clips, no manual intervention

---

### Phase 3: Morning Review Ritual (Day 4-5)

**Goal**: mtornga does first real morning review session.

#### 3.1 Review Session #1
- Open `daily_review_app.py`
- Review all overnight clips
- Apply tags: `deer`, `false_positive`, `hard_case`, `easy_eval`, etc.
- Note any issues (UI bugs, missing features, detection failures)

#### 3.2 Capture Feedback
- Document: What works well?
- Document: What's painful?
- Document: What edge cases appeared? (rain, spiderweb, far animal, etc.)

#### 3.3 Prioritize Improvements
- File issues or TODOs based on review experience
- Identify top 3 friction points

**Exit Criteria**: First real morning review completed, feedback documented

---

### Phase 4: Quality Feedback Loop (Day 6-7)

**Goal**: Human labels start improving the system.

#### 4.1 Define Evaluation Sets
- Create `eval/easy/` - clips that should always work
- Create `eval/hard/` - known difficult cases
- Move tagged clips to appropriate eval sets

#### 4.2 Baseline Metrics
- Run current detector on eval sets
- Record: precision, recall, confidence distribution
- Save as `eval/baseline_2024-12-XX.json`

#### 4.3 Plan Model Improvement
- Identify: Is MegaDetector good enough? Need fine-tuning?
- Identify: Which edge cases are most common?
- Draft next week's plan based on findings

**Exit Criteria**: Eval sets defined, baseline metrics recorded, next steps clear

---

## Success Metrics for This Week

1. ✅ Pipeline runs continuously without crashing
2. ✅ New clips appear in daily review each morning
3. ✅ mtornga completes at least 2 morning review sessions
4. ✅ At least 20 clips tagged with human labels
5. ✅ Baseline evaluation metrics documented

---

## Quick Reference Commands

```bash
# Mac: Mount share
mount_smbfs //mtornga@192.168.68.71/deer-share ~/DeerShare

# Mac: Start daily review UI
cd ~/Documents/repos/DeerAITrackingResponse
streamlit run scripts/daily_review_app.py

# Ubuntu: Test camera capture
source .venv/bin/activate
python scripts/capture_rtsp_clip.py --seconds 10

# Ubuntu: Run ingest (5 min test)
python scripts/reolink_stream_ingest.py --duration 300

# Ubuntu: Run detector once
python scripts/live_megadetector.py --once

# Ubuntu: Check pipeline logs
tail -f /srv/deer-share/runs/live/logs/*.log
```

---

## Notes

- Keep this plan focused on **getting the workflow running**, not perfecting it
- Resist urge to add features before the basic loop works
- Document edge cases for future work but don't fix them yet
- The goal is daily rhythm, not perfect detection
