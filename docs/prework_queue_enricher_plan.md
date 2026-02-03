# Pre-Work Implementation Plan: Queue Enricher + Metadata (v1)

## Context (current state)
- Training queue lives at `/srv/deer-share/training_queue/` and each entry is a directory
  containing `segment_*.mkv` and `meta.json`.
- Day/Night Watchman copy queue clips when detections meet criteria.
- Live detector currently uses a single YOLO model (RT-DETR removed from live path).
- False positives are dominated by snow/precipitation/IR artifacts, even when
  detection counts are high.
- Golden clips live at `/srv/deer-share/golden_clips/` with existing category layout.
- We do NOT want to archive training queue clips after use; the audit trail should
  come from TC run logs + golden manifest metadata.
- Taxonomy decision: keep directory layout stable; store richer taxonomy as metadata tags.
- Vision triage should be automated (no human in loop). Use full-frame, low-detail
  GPT-4o mini image input, three frames per clip. Use Batch (and Flex if available).
- The OpenAI FrameReview key exists in repo `.env` under
  `FrameReview_OPEN_AI_API_KEY` (two OpenAI keys total). We should read only that
  key (do not log or write it).

## Goals
1. Add `queue_meta.json` at enqueue time to carry canonical metadata.
2. Add a **Queue Enricher agent** that runs async when Watchman adds new clips.
3. Enrich queue clips exactly once (idempotent), then stop.
4. Perform automated vision triage + deterministic snow/precip scoring.
5. Preserve existing golden directory layout; add tags for finer taxonomy.

## Non-goals (for this phase)
- No changes to Training Coordinator (TC) logic yet.
- No model-agreement logic (single-model only).
- No changes to golden directory structure or training labels.

## Deliverables
- New `queue_meta.json` schema and population in Day/Night Watchman.
- New `agents/queue-enricher/` agent that:
  - processes only newly queued clips,
  - extracts previews,
  - calls GPT-4o mini vision via Batch API,
  - computes snow/motion metrics,
  - writes enrichment back into `queue_meta.json`.
- Watchman integration that triggers Queue Enricher only if new clips were queued.
- Enricher run logs in `runs/logs/queue_enricher/`.

## Taxonomy (Option A: metadata tags)
Directory layout remains unchanged (e.g., `golden_clips/<category>/<day|night>/<clip_id>`).
Use tags in metadata for finer detail:
- Deer splits: `deer`, `buck`, `doe`.
- Movement: `trot`, `sprint`, `grazing`, `standing`.
- False positives: `false_positive`, `night`, `snow`, `flies`, `rain`, `day`.
- Other wildlife: `other_wildlife`, plus specific tags like `coyote`, `fox`, etc.

## Queue metadata schema
Each queue directory gets a `queue_meta.json` file. Example:

```json
{
  "schema_version": 1,
  "queue": {
    "segment_id": "2026-01-04_segment_085055",
    "source_path": "/srv/deer-share/runs/live/events/2026-01-04/segment_085055",
    "queued_by": "DayWatchman",
    "queued_at": "2026-02-01T22:15:04Z",
    "note_suffix": "NightDeer"
  },
  "meta_summary": {
    "model_name": "yolov8n_wildlife_v7_night",
    "max_confidence": 0.842,
    "counts": {"animal": 14, "person": 0},
    "routing": {"decision": "review", "confidence_score": 0.842}
  },
  "enrichment": {
    "status": "complete",
    "batch_job_id": "batch_abc123",
    "enriched_at": "2026-02-01T22:18:33Z",
    "review_model": "gpt-4o-mini",
    "triage": "keep",
    "category": "deer",
    "lighting": "night",
    "tags": ["deer", "buck", "trot"],
    "review_confidence": 0.78,
    "snow_score": 0.12,
    "snow_reason": "low small-box ratio; persistent bbox track",
    "motion_stats": {
      "small_box_ratio": 0.05,
      "hit_persistence": 0.62,
      "spatial_dispersion": 0.08
    },
    "preview_frames": [
      "preview_001.jpg",
      "preview_002.jpg",
      "preview_003.jpg"
    ]
  }
}
```

Notes:
- `queue_meta.json` is the canonical record of queue + enrichment status.
- Enricher should add only the `enrichment` section (idempotent).

## Queue Enricher agent (new)
**Location:** `agents/queue-enricher/`

**Files:**
- `run.py`: main worker.
- `run.sh`: optional wrapper for cron/daemon execution.
- `README.md`: usage + environment variables.

**Identity:** register as a separate MCP agent using a valid adjective+noun name,
with task_description set to "Training Queue Enricher". It should send summary
mail only once per run to a dedicated thread (e.g., `QUEUE-ENRICHER`).

### Trigger / invocation
- Day/Night Watchman sends an Agent Mail message to Queue Enricher when it
  queued clips (only if `queued_count > 0`).
- The message includes clip names (queue dir names) + optional `queued_at` timestamp.
- Watchman spawns the Enricher asynchronously (detached). The Enricher reads inbox,
  processes pending clip lists, then acknowledges the message.

### Idempotency
Skip a clip if any of the following is true:
- `queue_meta.json` exists and contains `enrichment.enriched_at`.
- `preview_*.jpg` already exist.
- A lock file (e.g., `.enriching`) exists and is recent.

### Preview frames (three per clip)
- Use peak detection frame if available from `meta.json` (peak_frame).
- Use middle frame and early frame as fallbacks.
- Save as `preview_001.jpg`, `preview_002.jpg`, `preview_003.jpg` in the queue dir.
- Full-frame, low detail (no cropping in v1).

### GPT-4o mini vision triage
- Use OpenAI Batch API for cost control (Flex if available).
- One request per clip, containing three images.
- Request output as strict JSON with keys:
  `triage`, `category`, `lighting`, `tags`, `confidence`, `reason`.
- Validate/normalize output; fall back to `triage=unknown` on parse errors.
- Because Batch responses are async, mark `enrichment.status=pending` with a
  `batch_job_id`, then update to `complete` when results arrive.

### Negative mining from rejects (snow/precip FPs)
- Keep a capped subset of rejected clips as **negatives** for training when:
  `triage=reject` AND `category=false_positive` AND tags include snow/rain/flies.
- Route these to a `golden/negatives/snow` (or similar) bucket instead of dropping.
- Cap per TC run (e.g., 10–20) to avoid over-weighting negatives.

### Snow/precip heuristic
Compute `snow_score` from `meta.json` hits:
- `small_box_ratio`: fraction of bboxes below a tiny area threshold.
- `hit_persistence`: median consecutive-frame run length of hits.
- `spatial_dispersion`: stddev of bbox centers.
- `burstiness`: spikes in hits per frame.
Heuristic: high small_box_ratio + low persistence + high dispersion => high snow score.

### Lighting inference
- Primary: GPT-4o mini output.
- Secondary: frame heuristic (mean saturation/brightness) + time-of-day.
- No reliable RTSP IR flag discovered for Reolink cameras.

## Watchman integration
Update Watchman to send a trigger message only when new clips are queued.
Suggested message payload (JSON in body):

```json
{
  "queued_by": "DayWatchman",
  "queued_at": "2026-02-01T22:15:04Z",
  "queue_dir": "/srv/deer-share/training_queue",
  "clips": [
    "2026-01-04_segment_085055NightDeer",
    "2026-01-04_segment_085816NightDeer"
  ]
}
```

## Logging & audit trail
- Enricher writes per-run logs to `runs/logs/queue_enricher/`:
  - `queue-enricher-YYYY-MM-DD.json`: per-clip outcomes + summary stats.
- No queue archive; the audit trail is:
  - Enricher logs
  - `queue_meta.json`
  - (Later) TC run logs when clips move into golden

## Security / secrets
- Read FrameReview key from `.env` into `FrameReview_OPEN_AI_API_KEY`.
- Do not log keys or write them to disk.
- Do not commit `.env` changes.

## Validation checklist
- Enricher processes only new clips.
- Running Enricher twice yields no changes (idempotent).
- Three preview frames saved per clip.
- GPT output parsed into JSON consistently.
- Snow score correlates with known snow FP clips.

## Decisions locked for implementation
1. Env var name: `FrameReview_OPEN_AI_API_KEY`.
2. Trigger: Watchman spawns Enricher asynchronously (detached).
3. Batch async handling: mark `enrichment.status=pending` with `batch_job_id`,
   then update to `complete` when results arrive.
4. Preview frames: three full-frame, low-detail images per clip.

## Naming note
- MCP agent names must follow the adjective+noun format. Use a valid agent name
  for registration and set `task_description` to "Training Queue Enricher".
