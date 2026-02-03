# Training Coordinator Plan (v3)

## 1. Inventory & baselines
- Scan `/srv/deer-share/training_queue/` into a queue manifest:
  - clip id, note suffix, `queue_meta.json` (enrichment + snow/motion), `meta.json` stats (counts, max_confidence, detection_timeline), size, timestamps, model list.
- Record current live model set (files in `/srv/deer-share/models/` + any pipeline env/config used by the detector).
- Note: RT-DETR is no longer used in the live detector pipeline; do not assume model agreement signals.

## 2. Review & triage
- Deterministic pass: use metadata to flag likely keep/maybe/reject (counts, max confidence, snow/precip heuristics).
- Visual pass: **automated** via Queue Enricher (GPT-4o-mini vision on preview frames).
  - No human-in-the-loop by default; optional spot checks only if confidence is low or unknown.
- Assign category and lighting using enrichment output:
  - Category: deer | person | other_wildlife | false_positive | unknown.
  - No top-level “predator”; use specific animal names under `other_wildlife` via tags (e.g., fox, coyote).
  - False positives should be tagged with conditions: day/night + snow/rain/flies.
- Parse suffix notes (e.g., NightDeer, DeerTrot) as hints only.

## 3. Promote to golden with rollback metadata
- Move/copy accepted clips into `/srv/deer-share/golden_clips/<day|night>/<category>/<clip_id>/`.
- For each promoted clip, write a manifest entry that includes:
  - `tc_run_id`, `tc_agent`, `queued_from`, `queued_note`, `queued_at`, `source_path`, `clip_id`, `category`, `lighting`, and any review notes.
- Create a TC run log (e.g., `runs/logs/training_coordinator/TC-YYYYMMDD-HHMM.json`) listing all promoted clip ids + destinations + queue sources.
- Rollback rule: to undo a TC run, remove all golden dirs listed in that run log and strip those entries from `golden_clips/manifest.json`.

## 4. Queue cleanup
- After promotion decisions are finalized, delete items from training queue (no archive).
- Keep the TC run log as the audit trail; it’s the rollback handle.

## 5. Dataset build
- Refresh day/night golden datasets from `golden_clips` directories.
- Keep **directory layout stable**:
  `/srv/deer-share/golden_clips/<day|night>/<category>/<clip_id>/`.
- Store deeper taxonomy in metadata/tags (e.g., deer/buck/sprint; false_positive/night/snow).
- Use existing artifacts/scripts as guidance:
  - `trainingruncommand.txt` (canonical training command template).
  - `YOLO_TRAINING_SETUP.txt` (setup + dataset notes).
  - `docs/wildlife_v7_training.md` (prior run notes).

## 6. Training (RunPod)
- Kick off two model trainings (confirm exactly which two — likely day + night YOLO).
- Use prior scripts for reference (do not assume RT-DETR):
  - `train_yolov8n_golden.py`, `train_yolov8n_golden_v4.py`, `train_yolov8n_golden_v5.py`
  - `train_rtdetr_golden.py` (reference only; RT-DETR may be inactive in live pipeline)
- Capture commands, dataset hashes/paths, and run ids in the TC run log.

## 7. Eval (health check only for now)
- Run `scripts/pipeline_health_check.py` against new models.
- Compare with current live models.

## 8. Promote or discard
- If improved, update pipeline models/config and re-run health check.
- If not, keep models archived and leave pipeline unchanged.

## 9. Reporting
- Single final summary: clips accepted/rejected, queue size delta, training status, health check results, and whether promotion happened.

---

# Pre-work Notes & Implementation Hints

## Queue Enricher outputs (for TC to consume)
- Each clip in the queue has `queue_meta.json` with:
  - `queue.*` fields: `segment_id`, `queued_by`, `queued_at`, `source_path`, `note_suffix`.
  - `meta_summary.*` fields: model name, counts, max confidence, routing decision.
  - `enrichment.*` fields: status, preview_frames, snow/motion stats, triage/category/lighting/tags/confidence/reason.
- Preview images live alongside the clip:
  - `preview_full_*.jpg`, `preview_crop_*.jpg`.

## Negative mining rule (snow/precip false positives)
- Keep a capped subset of rejected clips as **negatives** for training when:
  `triage=reject` AND `category=false_positive` AND tags include snow/rain/flies.
- Route these to a `golden_clips/negatives/snow` (or similar) bucket instead of dropping.
- Cap per TC run (e.g., 10–20) to avoid over-weighting negatives.

## Taxonomy guidance
- Deer splits: buck/doe (keep tags even if current training class remains `deer`).
- Motion tags: trot/sprint (future pose work).
- Prefer `other_wildlife` for named animals (coyote/fox/etc), avoid top-level “predator.”
- False positives should be tagged by condition: `snow`, `rain`, `flies`, `night`, `day`.

## Lighting
- Use `enrichment.lighting` when present; fall back to simple heuristics if missing.

## Audit trail
- No queue archive. The rollback is via TC run logs and the golden manifest.

## Batch vs live
- Queue Enricher supports batch (preferred) and live review.
- Batch outputs are ingested into `queue_meta.json` and should be the source of truth for TC.
