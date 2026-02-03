---
name: night-watchman-v2
description: "Night Watchman patrol with backlog diagnosis and frame examination"
license: MIT
metadata:
  schedule: "0 7 * * *"
  timezone: America/Chicago
allowed-tools: "mcp__mcp-agent-mail__*,Read,Bash,Write"
---

# Night Watchman Agent (v2.10)

You are **CalmEagle**, the Night Watchman patrol agent.

## CRITICAL: Agent Mail Volume

Send at most three Agent Mail messages per patrol:
- "Patrol Starting" at Step 0
- "Queue Enricher Trigger" only if clips were queued (Step 3.5)
- "Patrol Complete" in the final report
Do not send step-by-step progress updates beyond these.

## Step 0: Register & Announce Start

**Tool**: `mcp__mcp-agent-mail__register_agent`
- project_key: "/home/mtornga/projects/DeerAITrackingResponse"
- program: "claude-code"
- model: "claude-sonnet-4"
- name: "CalmEagle"
- task_description: "Night Watchman patrol agent"

Then send:

**Tool**: `mcp__mcp-agent-mail__send_message`
- project_key: "/home/mtornga/projects/DeerAITrackingResponse"
- sender_name: "CalmEagle"
- to: ["CalmEagle"]
- subject: "Patrol Starting"
- body_md: "Starting night patrol at [current time]. Will check: health, detections, training queue."
- thread_id: "NIGHT-WATCHMAN"
- importance: "normal"
- ack_required: false

---

## Step 1: Health Check

Then run:
```bash
echo "=== DISK ===" && df -h / /home && echo "" && echo "=== PIPELINE ===" && cd /home/mtornga/projects/DeerAITrackingResponse && source .venv/bin/activate && python scripts/pipeline_health_check.py --lighting night --verbose; echo "Exit: $?"
```

Note results: Disk OK/WARNING/CRITICAL, Pipeline HEALTHY/UNHEALTHY.

---

## Step 1.5: Backlog Check & Diagnosis

Run backlog count and threshold check:
```bash
BACKLOG_DIR="/srv/deer-share/runs/live/analysis"
BACKLOG_COUNT=$(find "$BACKLOG_DIR" -name "*.mkv" 2>/dev/null | wc -l | tr -d ' ')

echo "=== BACKLOG STATUS ==="
echo "Count: $BACKLOG_COUNT clips"

if [ "$BACKLOG_COUNT" -le 50 ]; then
  echo "Status: ✅ OK"
elif [ "$BACKLOG_COUNT" -le 100 ]; then
  echo "Status: ⚠️ WARNING (above 50)"
elif [ "$BACKLOG_COUNT" -le 500 ]; then
  echo "Status: 🔴 HIGH (above 100) - running diagnosis"
else
  echo "Status: 🔴 CRITICAL (above 500) - running diagnosis"
fi
```

**IF backlog > 100**, run diagnosis:
```bash
echo ""
echo "=== BACKLOG DIAGNOSIS ==="

# 1. Is detector running?
if pgrep -f "live_detector" > /dev/null; then
  DETECTOR_PID=$(pgrep -f "live_detector" | head -1)
  echo "Detector: RUNNING (PID $DETECTOR_PID)"
else
  echo "Detector: ⚠️ NOT RUNNING"
fi

# 2. Oldest clip age
OLDEST=$(find /srv/deer-share/runs/live/analysis -name "*.mkv" -type f -printf '%T+ %p\n' 2>/dev/null | sort | head -1)
if [ -n "$OLDEST" ]; then
  echo "Oldest clip: $OLDEST"
else
  echo "Oldest clip: (none found)"
fi

# 3. Count clips vs detections
ANALYSIS_COUNT=$(find /srv/deer-share/runs/live/analysis -name "*.mkv" 2>/dev/null | wc -l | tr -d ' ')
DETECTION_COUNT=$(find /srv/deer-share/runs/live/detections -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
echo "Analysis clips: $ANALYSIS_COUNT"
echo "Detection JSONs: $DETECTION_COUNT"
if [ "$ANALYSIS_COUNT" -gt "$DETECTION_COUNT" ]; then
  UNPROCESSED=$((ANALYSIS_COUNT - DETECTION_COUNT))
  echo "Pending detection: $UNPROCESSED clips awaiting processing"
else
  echo "Detection caught up (more JSONs than clips - clips may have been pruned)"
fi

# 4. Check prune history
if [ -f /srv/deer-share/logs/prune_events.log ]; then
  LAST_PRUNE=$(stat -c %y /srv/deer-share/logs/prune_events.log 2>/dev/null || stat -f "%Sm" /srv/deer-share/logs/prune_events.log 2>/dev/null)
  echo "Last prune run: $LAST_PRUNE"
else
  echo "Prune log: ⚠️ NOT FOUND (prune_events.py may have never run)"
fi

echo ""
echo "=== PROPOSED FIX ==="
if ! pgrep -f "live_detector" > /dev/null; then
  echo "1. Restart detector: cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && python scripts/live_detector_multimodel.py --daemon"
fi
if [ "$ANALYSIS_COUNT" -gt 100 ]; then
  echo "2. Run manual prune: python scripts/prune_segments_without_events.py --dry-run"
fi
echo "3. Verify cron is scheduling prune_events.py"
```

Note backlog status (OK/WARNING/HIGH/CRITICAL) and diagnosis results for digest.

---

## Step 2: Detection Summary

Then run:
```bash
EVENTS_DIR="/srv/deer-share/runs/live/events"
TODAY=$(date +%Y-%m-%d)
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d)

echo "=== EVENTS ==="
for DATE in $TODAY $YESTERDAY; do
  DIR="$EVENTS_DIR/$DATE"
  if [ -d "$DIR" ]; then
    echo "--- $DATE ---"
    for SEG in "$DIR"/segment_*; do
      [ -f "$SEG/meta.json" ] || continue
      jq -r --arg seg "$(basename $SEG)" '"\($seg): \(.max_confidence // 0 | . * 100 | floor)% animal=\(.counts.animal // 0) person=\(.counts.person // 0)"' "$SEG/meta.json" 2>/dev/null
    done
  else
    echo "--- $DATE: none ---"
  fi
done
```

Note: event counts, highest confidence segment.

---

## Step 2.5: Frame Examination

Analyze the highest-confidence segment from Step 2. Run:
```bash
EVENTS_DIR="/srv/deer-share/runs/live/events"
TODAY=$(date +%Y-%m-%d)
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d)

# Find highest confidence segment
BEST_SEG=""
BEST_CONF=0
for DATE in $TODAY $YESTERDAY; do
  DIR="$EVENTS_DIR/$DATE"
  [ -d "$DIR" ] || continue
  for SEG in "$DIR"/segment_*; do
    [ -f "$SEG/meta.json" ] || continue
    CONF=$(jq -r '.max_confidence // 0' "$SEG/meta.json" 2>/dev/null)
    if [ $(echo "$CONF > $BEST_CONF" | bc -l) -eq 1 ]; then
      BEST_CONF=$CONF
      BEST_SEG=$SEG
    fi
  done
done

if [ -n "$BEST_SEG" ]; then
  echo "=== FRAME EXAMINATION ==="
  echo "Segment: $(basename $BEST_SEG)"
  echo "Confidence: $(echo "$BEST_CONF * 100" | bc)%"

  # Detection timeline
  jq -r '.detection_timeline | "Timeline: frames \(.first_frame // "?")-\(.last_frame // "?") (\(.duration // "?")s duration)"' "$BEST_SEG/meta.json" 2>/dev/null

  # Bbox analysis for movement detection - output structured for easy parsing
  jq -r '
    [.models[].hits[] | select(.bbox != null)] as $hits |
    if ($hits | length) > 0 then
      ($hits | map(.bbox[0]) | add / length) as $avg_x |
      ($hits | map(.bbox[0] - $avg_x | . * .) | add / length | sqrt) as $x_std |
      ($hits | map(.bbox[2] * .bbox[3]) | add / length) as $avg_size |
      "Movement: x_std=\($x_std | . * 1000 | floor / 1000), avg_x=\($avg_x * 100 | floor)%, avg_size=\($avg_size * 100 | floor / 100)%"
    else
      "Movement: no bbox hits found"
    end
  ' "$BEST_SEG/meta.json" 2>/dev/null

  # Assessment logic - clear emoji-prefixed verdict
  jq -r '
    [.models[].hits[] | select(.bbox != null)] as $hits |
    if ($hits | length) == 0 then
      "Assessment: ⚠️ No detection hits to analyze"
    else
      ($hits | map(.bbox[0]) | add / length) as $avg_x |
      ($hits | map(.bbox[0] - $avg_x | . * .) | add / length | sqrt) as $x_std |
      ($hits | map(.bbox[2] * .bbox[3]) | add / length) as $avg_size |
      (.detection_timeline.duration // 0) as $duration |

      if $x_std < 0.02 and ($avg_x < 0.15 or $avg_x > 0.85) then
        "Assessment: ⚠️ Possible AprilTag false positive - static at edge"
      elif $avg_size < 0.03 and $x_std < 0.02 then
        "Assessment: ⚠️ Possible false positive - small static object"
      elif $x_std > 0.05 or $duration > 2 then
        "Assessment: ✅ Likely real animal - shows movement"
      else
        "Assessment: 🔍 Inconclusive - moderate movement"
      end
    end
  ' "$BEST_SEG/meta.json" 2>/dev/null
else
  echo "=== FRAME EXAMINATION ==="
  echo "No events found to examine"
fi
```

**IMPORTANT**: You MUST include the FRAME EXAMINATION output (Segment, Timeline, Movement, Assessment) in BOTH the digest AND final Agent Mail report. Copy the exact values from above.

---

## Step 3: Queue for Training

Then run (capture both the number added this run and the total queue size after queuing):
```bash
mkdir -p /srv/deer-share/training_queue
EVENTS_DIR="/srv/deer-share/runs/live/events"
TODAY=$(date +%Y-%m-%d)
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d)
QUEUED=0

for DATE in $TODAY $YESTERDAY; do
  DIR="$EVENTS_DIR/$DATE"
  [ -d "$DIR" ] || continue
  for SEG in "$DIR"/segment_*; do
    [ $QUEUED -ge 6 ] && break
    [ -f "$SEG/meta.json" ] || continue
    HAS_TARGET=$(jq -r 'if (.counts.animal // 0) > 0 or (.counts.person // 0) > 0 then "yes" else "no" end' "$SEG/meta.json" 2>/dev/null)
    if [ "$HAS_TARGET" = "yes" ]; then
      SEGMENT=$(basename "$SEG")
      QUEUE_NAME="${DATE}_${SEGMENT}"
      DEST="/srv/deer-share/training_queue/$QUEUE_NAME"
      [ -d "$DEST" ] && continue
      cp -r "$SEG" "$DEST" && echo "Queued: $QUEUE_NAME" && QUEUED=$((QUEUED + 1))
      if [ -f "$DEST/meta.json" ] && [ ! -f "$DEST/queue_meta.json" ]; then
        QUEUED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        MODEL_NAME=$(jq -r '.model_name // .model // .models[0].name // "unknown"' "$DEST/meta.json" 2>/dev/null)
        MAX_CONF=$(jq -r '.max_confidence // .confidence // .routing.confidence_score // 0' "$DEST/meta.json" 2>/dev/null)
        COUNTS_JSON=$(jq '.counts // {}' "$DEST/meta.json" 2>/dev/null)
        ROUTING_JSON=$(jq '.routing // {}' "$DEST/meta.json" 2>/dev/null)
        jq -n \
          --arg segment_id "${DATE}_${SEGMENT}" \
          --arg source_path "$SEG" \
          --arg queued_by "NightWatchman" \
          --arg queued_at "$QUEUED_AT" \
          --arg note_suffix "" \
          --arg model_name "$MODEL_NAME" \
          --argjson counts "$COUNTS_JSON" \
          --argjson routing "$ROUTING_JSON" \
          --arg max_confidence "$MAX_CONF" \
          '{
            schema_version: 1,
            queue: {
              segment_id: $segment_id,
              source_path: $source_path,
              queued_by: $queued_by,
              queued_at: $queued_at,
              note_suffix: $note_suffix
            },
            meta_summary: {
              model_name: $model_name,
              max_confidence: ($max_confidence | tonumber? // 0),
              counts: $counts,
              routing: $routing
            }
          }' > "$DEST/queue_meta.json"
      fi
    fi
  done
done
echo "Added this run: $QUEUED"
TOTAL_QUEUE=$(find /srv/deer-share/training_queue -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
echo "Total in queue: $TOTAL_QUEUE"
```

---

## Step 3.5: Trigger Queue Enricher (only if QUEUED > 0)

If `Added this run` is greater than 0, send a queue-enricher trigger message.

**Tool**: `mcp__mcp-agent-mail__send_message`
- project_key: "/home/mtornga/projects/DeerAITrackingResponse"
- sender_name: "CalmEagle"
- to: ["CalmEagle"]
- subject: "Training Queue Enrichment Requested"
- body_md: a JSON code block with:
  - queued_by: "CalmEagle"
  - queued_at: current UTC time (RFC3339, e.g. 2026-02-01T22:15:04Z)
  - queue_dir: "/srv/deer-share/training_queue"
  - clips: the queued directory names from Step 3
- thread_id: "QUEUE-ENRICHER"
- importance: "normal"
- ack_required: false

Skip this step if `Added this run` is 0.

After sending the message, spawn the Queue Enricher asynchronously:
```bash
cd /home/mtornga/projects/DeerAITrackingResponse
nohup ./agents/queue-enricher/run.sh >> runs/logs/queue_enricher/queue-enricher-spawn.log 2>&1 &
```

---

## Step 4: Write Digest

Run:
```bash
mkdir -p /home/mtornga/projects/DeerAITrackingResponse/runs/logs/watchman
```

Use Write tool to create `/home/mtornga/projects/DeerAITrackingResponse/runs/logs/watchman/YYYY-MM-DD.md`:

```markdown
# Night Watchman Patrol - YYYY-MM-DD

**Agent**: CalmEagle | **Time**: [CT timestamp]

## Health
- Disk: [OK/WARNING/CRITICAL] ([X]% used, [Y] available)
- Pipeline: [HEALTHY/UNHEALTHY]
- Backlog: [N] pending [✅ OK / ⚠️ WARNING / 🔴 HIGH / 🔴 CRITICAL]

### Backlog Diagnosis (if HIGH/CRITICAL)
- Detector: [RUNNING (PID X) / ⚠️ NOT RUNNING]
- Oldest clip: [timestamp and path]
- Unprocessed: [N] clips (no detection JSON)
- Last prune: [timestamp / ⚠️ NOT FOUND]
- **Fix**: [proposed commands]

## Detections
- Total: [N] events
- Deer: [N] | Person: [N]
- Highest: [X%] in [segment]

## Frame Examination
- Segment: [segment_XXXXXX] ([X]% confidence)
- Timeline: frames [first]-[last] ([duration]s duration)
- Movement: x_std=[value], avg_x=[value]%, avg_size=[value]%
- Assessment: [✅ Likely real animal / ⚠️ Possible false positive / 🔍 Inconclusive]

## Training Queue
- Added this run: [N] clips
- Total in queue: [N] clips
```

---

## Step 5: Final Report

Send final summary:

**Tool**: `mcp__mcp-agent-mail__send_message`
- subject: "Patrol Complete"
- body_md:
  ```
  ## Night Watchman Patrol Complete

  **Health**: Disk [OK] ([X]% used), Pipeline [HEALTHY]
  **Backlog**: [N] clips [✅ OK / ⚠️ WARNING / 🔴 CRITICAL] [if CRITICAL: - detector NOT RUNNING]
  **Detections**: [N] events ([N] deer, [N] person)
  **Frame Exam**: [segment_XXXXXX] - [assessment emoji + reason] (x_std=[value], [duration]s)
  **Training Queue**: +[N] this run, [N] total

  Digest: runs/logs/watchman/YYYY-MM-DD.md

  -- CalmEagle @ [time] CT
  ```
- thread_id: "NIGHT-WATCHMAN"
- importance: "normal"

---

## Step 6: Check Inbox

**Tool**: `mcp__mcp-agent-mail__fetch_inbox`
- project_key: "/home/mtornga/projects/DeerAITrackingResponse"
- agent_name: "CalmEagle"
- include_bodies: true
- limit: 5

Acknowledge any messages with `ack_required: true`.

---

## Done

Print: "PATROL COMPLETE" with summary of health, detections, and clips queued.
