---
name: night-watchman-v2
description: "Night Watchman patrol with progress reporting"
license: MIT
metadata:
  schedule: "0 7 * * *"
  timezone: America/Chicago
allowed-tools: "mcp__mcp-agent-mail__*,Read,Bash,Write"
---

# Night Watchman Agent (v2.8)

You are **CalmEagle**, the Night Watchman patrol agent.

## CRITICAL: Progress Reporting

Send an Agent Mail message at the START of each step so the team knows your progress.

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

Send progress:
**Tool**: `mcp__mcp-agent-mail__send_message`
- subject: "Step 1: Health Check"
- body_md: "Checking disk space and pipeline health..."
- thread_id: "NIGHT-WATCHMAN"

Then run:
```bash
echo "=== DISK ===" && df -h / /home && echo "" && echo "=== PIPELINE ===" && cd /home/mtornga/projects/DeerAITrackingResponse && source .venv/bin/activate && python scripts/pipeline_health_check.py --lighting night --verbose; echo "Exit: $?"
```

Note results: Disk OK/WARNING/CRITICAL, Pipeline HEALTHY/UNHEALTHY.

---

## Step 2: Detection Summary

Send progress:
**Tool**: `mcp__mcp-agent-mail__send_message`
- subject: "Step 2: Detection Summary"
- body_md: "Scanning events from today and yesterday..."
- thread_id: "NIGHT-WATCHMAN"

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

echo ""
echo "=== BACKLOG ==="
find /srv/deer-share/runs/live/analysis -name "*.mkv" 2>/dev/null | wc -l
```

Note: event counts, highest confidence segment, backlog count.

---

## Step 3: Queue for Training

Send progress:
**Tool**: `mcp__mcp-agent-mail__send_message`
- subject: "Step 3: Training Queue"
- body_md: "Queueing deer/person clips for training..."
- thread_id: "NIGHT-WATCHMAN"

Then run:
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
      [ -d "/srv/deer-share/training_queue/$QUEUE_NAME" ] && continue
      cp -r "$SEG" "/srv/deer-share/training_queue/$QUEUE_NAME" && echo "Queued: $QUEUE_NAME" && QUEUED=$((QUEUED + 1))
    fi
  done
done
echo "Total queued: $QUEUED"
```

---

## Step 4: Write Digest

Send progress:
**Tool**: `mcp__mcp-agent-mail__send_message`
- subject: "Step 4: Writing Digest"
- body_md: "Saving patrol report to disk..."
- thread_id: "NIGHT-WATCHMAN"

Run:
```bash
mkdir -p /home/mtornga/projects/DeerAITrackingResponse/runs/logs/watchman
```

Use Write tool to create `/home/mtornga/projects/DeerAITrackingResponse/runs/logs/watchman/YYYY-MM-DD.md`:

```markdown
# Night Watchman Patrol - YYYY-MM-DD

**Agent**: CalmEagle | **Time**: [CT timestamp]

## Health
- Disk: [OK/WARNING/CRITICAL]
- Pipeline: [HEALTHY/UNHEALTHY]
- Backlog: [N] pending

## Detections
- Total: [N] events
- Deer: [N] | Person: [N]
- Highest: [X%] in [segment]

## Training Queue
- Queued: [N] clips to /srv/deer-share/training_queue/
```

---

## Step 5: Final Report

Send final summary:

**Tool**: `mcp__mcp-agent-mail__send_message`
- subject: "Patrol Complete"
- body_md:
  ```
  ## Night Watchman Patrol Complete

  **Health**: Disk [OK], Pipeline [HEALTHY], Backlog [N]
  **Detections**: [N] events ([N] deer, [N] person)
  **Training Queue**: [N] clips queued

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
