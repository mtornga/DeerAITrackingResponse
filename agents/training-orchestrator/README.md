# Training Orchestrator

Runs the full loop when the queue threshold is met:
Training Coordinator → Trainer → Evaluator → Deployer.

## Usage (ubuntubasement)

```bash
ssh mtornga@192.168.68.71 \
  "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && \
   python agents/training-orchestrator/run.py"
```

Dry run:
```bash
ssh mtornga@192.168.68.71 \
  "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && \
   python agents/training-orchestrator/run.py --dry-run"
```
