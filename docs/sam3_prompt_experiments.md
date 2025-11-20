# SAM3 Prompt Experiments – November 2025

## Goal
Reproduce Meta's SAM3 Playground behavior locally so we can prompt for "deer" on recorded clips and get per-frame masks/tracks that match the UI demo.

## Hardware & Constraints
- Mac: dev workstation (used for editing only)
- Ubuntu tower: RTX 3080 10 GB, Python 3.12 venv, `external/sam3` repo cloned
- No root access on Ubuntu (must avoid system-wide installs)

## Attempts

### 1. GPU tracker (scripts/sam3_prompt_video.py)
- Wrapped Meta's `build_sam3_video_predictor` in a CLI script.
- Added fp16/autocast patches to reduce VRAM, then reverted after dtype mismatches.
- Outcome: tracker consistently hits CUDA OOM on the 10 GB 3080 even with downsampled frames. Needs ≥12 GB.

### 2. CPU fallback (SAM3 image model)
- Added `--cpu-only` flag to run `build_sam3_image_model` per frame, no temporal tracking.
- Hid GPUs via `CUDA_VISIBLE_DEVICES=""`, forced model `.to(torch.device("cpu"))`.
- Blocker: upstream SAM3 modules (position encoding, decoder buffers) contain literal `device="cuda"` allocations, so the import still tries to use CUDA and crashes (`No CUDA GPUs are available`). Fixing this requires patching multiple files under `external/sam3` and maintaining a fork.

## Artifacts
- `scripts/sam3_prompt_video.py` (GPU tracker) with `--cpu-only` flag (non-functional without vendor patches).
- Added docs/record for future runs.

## Remaining Blockers
- GPU: 10 GB is insufficient for SAM3 video predictor; need ≥12 GB GPU.
- CPU path: upstream code assumes CUDA; would need extensive changes in `external/sam3` (position encoding, decoder, VL combiner). Not practical without forking Meta's repo.

## Recommendation
- Acquire a larger GPU or wait for Meta's Mobile/SAM3-S release.
- For CPU-only experiments, switch to SAM v1/SAM v2 ONNX models that officially support CPU inference.
