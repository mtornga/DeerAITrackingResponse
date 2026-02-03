# Evaluator Agent

Evaluates candidate models against baseline using `test_clips` and a fixed golden eval set.
Writes evaluation logs and gating decisions.

## Usage (ubuntubasement)

Dry run:
```bash
ssh mtornga@192.168.68.71 \
  "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && \
   python agents/evaluator/run.py --dry-run"
```

Evaluate latest trainer models:
```bash
ssh mtornga@192.168.68.71 \
  "cd ~/projects/DeerAITrackingResponse && source .venv/bin/activate && \
   python agents/evaluator/run.py"
```

## Config
- `configs/eval_sets.json` lists fixed day/night clip IDs.
- `configs/training_loop.json` controls gating thresholds.

