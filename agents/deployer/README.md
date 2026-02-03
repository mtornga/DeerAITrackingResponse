# Deployer Agent

Promotes candidate models to live, updates `DETECTOR_MODELS` in `.env`, and
runs pipeline health checks.

## Usage (ubuntubasement)

Dry run:
```bash
ssh mtornga@192.168.68.71 \
  "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && \
   python agents/deployer/run.py --dry-run"
```

Deploy latest evaluator pass:
```bash
ssh mtornga@192.168.68.71 \
  "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && \
   python agents/deployer/run.py"
```

