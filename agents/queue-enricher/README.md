# Queue Enricher Agent

This worker enriches training queue clips by extracting preview images (2 full frames + crops), running
GPT-4o mini vision triage via Batch API, and writing results to `queue_meta.json`.

## Usage

```bash
./agents/queue-enricher/run.sh
./agents/queue-enricher/run.sh --dry-run
./agents/queue-enricher/run.sh --skip-mail
./agents/queue-enricher/run.sh --no-submit-batch  # build batch JSONL only
./agents/queue-enricher/run.sh --poll-batch --batch-id batch_XXX  # poll + download results
./agents/queue-enricher/poll_batch.sh batch_XXX --interval 300  # loop until ready
```

## Environment variables

- `QUEUE_ENRICHER_AGENT`: MCP agent name (adjective+noun). Default: `QuietHarbor`.
- `QUEUE_ENRICHER_THREAD`: Thread id for reports. Default: `QUEUE-ENRICHER`.
- `QUEUE_ENRICHER_MCP_URL`: MCP endpoint. Default: `http://127.0.0.1:8765/mcp/`.
- `QUEUE_ENRICHER_MCP_TOKEN`: Optional bearer token.
- `QUEUE_ENRICHER_TO`: Comma-separated recipients for the summary message.
- `FrameReview_OPEN_AI_API_KEY`: OpenAI key used for GPT-4o mini vision triage.
- `QUEUE_ENRICHER_MODEL`: Model id (default: `gpt-4o-mini`).
- `QUEUE_ENRICHER_IMAGE_DETAIL`: Image detail hint (default: `low`).
- `QUEUE_ENRICHER_PREVIEW_COUNT`: Crops per clip (default: `5`).

Preview selection notes:
- Crops come from motion-heavy frames; if `person` detections are expected, the enricher runs a lightweight local YOLO pass on sampled frames within the detection timeline to locate person boxes for crops.

Live review:
- Use `--live-review-all` to call the API per clip (non-batch).
- `--live-review-delay` throttles per-clip calls (default 2.0s) to avoid rate limits.
- `QUEUE_ENRICHER_SUBMIT_BATCH`: Submit batch to OpenAI (default: `true`).
- `QUEUE_ENRICHER_SCAN_QUEUE`: Scan full queue for missing enrichment (default: `true`).
- `QUEUE_ENRICHER_BATCH_WINDOW`: Batch completion window (default: `24h`).
- `REPO_DIR`, `VENV_DIR`, `LOG_DIR`, `SHELL_TIMEOUT`: Used by `run.sh` for execution.

## Logs

The worker writes `runs/logs/queue_enricher/queue-enricher-latest.json` when not
running with `--dry-run`. Batch request JSONL files are stored in the same folder.

## Notes

- The wrapper assumes `python` is on PATH after the virtualenv is activated.
- Batch responses are async. The first run marks `enrichment.status=pending` with
  `batch_job_id`; a later run with `--batch-results` finalizes `enrichment.status=complete`.
- Use `--poll-batch --batch-id <id>` to fetch status and download output when ready.
