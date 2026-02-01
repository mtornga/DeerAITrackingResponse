# Queue Enricher Agent (Skeleton)

This folder hosts the Queue Enricher worker. The current version is a scaffold
that registers with MCP Agent Mail and emits a single summary per run. Clip
processing logic will be added in follow-up tasks.

## Usage

```bash
./agents/queue-enricher/run.sh
./agents/queue-enricher/run.sh --dry-run
./agents/queue-enricher/run.sh --skip-mail
```

## Environment variables

- `QUEUE_ENRICHER_AGENT`: MCP agent name (adjective+noun). Default: `QuietHarbor`.
- `QUEUE_ENRICHER_THREAD`: Thread id for reports. Default: `QUEUE-ENRICHER`.
- `QUEUE_ENRICHER_MCP_URL`: MCP endpoint. Default: `http://127.0.0.1:8765/mcp/`.
- `QUEUE_ENRICHER_MCP_TOKEN`: Optional bearer token.
- `QUEUE_ENRICHER_TO`: Comma-separated recipients for the summary message.
- `REPO_DIR`, `VENV_DIR`, `LOG_DIR`, `SHELL_TIMEOUT`: Used by `run.sh` for execution.

## Logs

The skeleton writes `runs/logs/queue_enricher/queue-enricher-latest.json` when not
running with `--dry-run`.

## Notes

- The current wrapper assumes `python` is on PATH after the virtualenv is activated.
- Future tasks will wire enrichment into `agents/queue-enricher/run.py`.
