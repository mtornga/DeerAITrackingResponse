# Project Analysis and Weekly Plan

## Current State Analysis

**How does the current bevy of files meet the challenges?**

The current state is **partially aligned but disjointed**. 

*   **Local (Mac):** The `DEERAITRACKINGRESPONSE` directory contains the most recent documentation (`NewNotes.txt`, `README.md`) and a structured layout (`scripts`, `perception`, `calibration`, etc.). It appears to be the "brain" or "control center".
*   **Remote (Ubuntu):** The `~/projects/deer-vision/` directory seems cluttered, primarily containing a `tmp` directory with external tools (`MegaDetector`, `yolov5`, `ai4eutils`) and a `ui` folder. It lacks the structured logic found on the Mac.
*   **Gap:** There is a significant divergence. The Ubuntu machine (the "muscle" with the GPU) does not seem to have the latest code or structure to execute the vision tasks defined on the Mac. The `NewNotes.txt` vision of agents autonomously using the GPU is hindered by this lack of synchronization and structure.

## Restructuring Proposal

**How should we restructure the project across machines and directories?**

We should move towards a **Unified Controller-Worker Architecture**.

1.  **Mirror Structure:** The file structure on the Ubuntu machine should mirror the local `DEERAITRACKINGRESPONSE` (likely renaming `deer-vision` to match or syncing contents). This ensures scripts and paths are consistent.
2.  **Centralized Code, Distributed Data:** Keep the code in git/sync. Keep heavy data (video clips, weights) on the Samba share (`/srv/deer-share`) or S3, mounted/accessible by both.
3.  **External Tools Management:** Move the contents of `tmp` (like `yolov5`, `MegaDetector`) into a dedicated `external` or `tools` directory, properly ignored by git but accessible for import/usage.
4.  **Execution Flow:** 
    *   **Mac:** Runs the "Agent" logic, high-level orchestration, and user review (Streamlit/CVAT).
    *   **Ubuntu:** Runs the heavy compute (inference, training) via SSH commands triggered by the Mac or autonomous agents.

## 10-Step Medium-Level Plan

This plan focuses on bridging the gap between the Mac (Control) and Ubuntu (Compute) to enable the workflow described in `NewNotes.txt`.

1.  **Standardize Remote Directory:** Rename/Archive the current `~/projects/deer-vision` on Ubuntu. Create a clean `~/projects/DeerAITrackingResponse` that mirrors the local structure.
    - Status: Local Mac repo (`DeerAITrackingResponse`, branch `main`) is now clean and pushed to GitHub; safe to treat as canonical structure.
    - TODO (remote): SSH to Ubuntu and rename the old directory (e.g., `mv ~/projects/deer-vision ~/projects/deer-vision-legacy`) so new work starts in a fresh tree.
    - TODO (remote): Create `~/projects/DeerAITrackingResponse` as the new root for all CV/ML code (will be populated by `git clone` in Step 2).
    - Note: Keep `*-legacy` around temporarily in case we need to scavenge scripts or configs from older experiments.

2.  **Sync Codebase:** Push the local `DEERAITRACKINGRESPONSE` code to the new remote directory (via git or rsync for now) to ensure the Ubuntu machine has the latest scripts and `perception` modules.
    - Status: Latest `main` is already pushed to `git@github.com:mtornga/DeerAITrackingResponse.git` with large datasets excluded from git history.
    - TODO (remote): From Ubuntu, run `cd ~/projects && git clone git@github.com:mtornga/DeerAITrackingResponse.git` so structure mirrors the Mac.
    - TODO (remote): Verify key folders exist after clone: `scripts/`, `perception/`, `calibration/`, `outdoor/`, `demo/`.
    - Future refinement: Decide whether to rely purely on `git` for code or keep an `rsync` helper for quick one-off syncs during heavy iteration.

3.  **Environment Unification:** Create a setup script to install `requirements.txt` on the Ubuntu machine, ensuring the python environment matches the Mac's expectations.
    - Plan: Encapsulate the AGENTS instructions into a reusable script, e.g. `scripts/setup_env_remote.sh`.
    - Status (repo): `scripts/setup_env_remote.sh` now exists; it creates `.venv` under the repo root, upgrades `pip`, installs the pinned core stack from `constraints.txt`, then installs the rest of `requirements.txt`.
    - Status (remote): The setup script has been run on Ubuntu in `~/projects/DeerAITrackingResponse`, creating a `.venv` (~5.2G) with `torch==2.2.2+cu121`, `opencv-python==4.8.1`, and GPU support (`torch.cuda.is_available() == True`).
    - TODO (remote): Ensure any shell helpers (`deervision_dashboard_tmux.sh`, cron jobs, etc.) activate `.venv` before running Python scripts (or explicitly document which env to use).
    - Note: Capture any Ubuntu-specific packages (e.g., system `ffmpeg`, `libgl1`) in comments within the setup script for future agents.
    - Observation: Ubuntu currently has ~7.4G under `/home/mtornga/.local/share/mamba` (mamba envs) plus small `.conda`/`.mamba` dirs; these survived our project cleanup and are a major disk consumer.
    - Design decision: Do **not** run Python virtualenvs directly from the Samba share; network filesystems are fragile for Python envs and can cause weird import/locking issues.
    - Space strategy: Keep active envs on the Ubuntu SSD, but consider:
        * Archiving old/unused mamba envs from `.local/share/mamba/envs` to `/srv/deer-share/env-archives/` as compressed tarballs.
        * Documenting a “rebuild env” path (using `constraints.txt`) so agents feel safe deleting stale envs when space runs low.
    - Future TODO: Inventory which mamba envs are actually in use for Deer Vision vs past experiments, and record a recommended “default” env layout in `AGENTS_REMOTE.md`.

4.  **External Tools Organization:** Move the useful parts of the remote `tmp` (YOLOv5, MegaDetector) into a structured `external/` directory on the remote machine and document how to reference them.
    - Plan: Standardize on `external/` under `~/projects/DeerAITrackingResponse` for heavy third-party repos (YOLOv5, MegaDetector, ai4eutils, etc.).
    - TODO (remote): Move existing tools from `~/projects/deer-vision/tmp/` into `~/projects/DeerAITrackingResponse/external/` (preserving their internal git histories).
    - TODO (repo): Ensure `.gitignore` keeps `external/` out of version control while allowing lightweight wrappers or configs.
    - Note: Long-term, agents should call tools via thin wrappers in `scripts/` so paths like `external/yolov5` are never hard-coded in many places.

5.  **Samba Share Verification:** Verify the Samba share mount on both machines. Create a test script `scripts/verify_storage.sh` that writes/reads a file from both ends to confirm shared access.
    - Plan: Treat `/srv/deer-share` (or equivalent) as the single source of truth for heavy clips and evaluation artifacts.
    - TODO (repo): Add `scripts/verify_storage.sh` that writes a timestamped test file to the share and reads it back, exiting non-zero on failure.
    - TODO (Mac/remote): Run the script from both Mac and Ubuntu to validate read/write symmetry and record the mount paths in `README.md` or a `.env` key.
    - Note: This step is critical for the daily-review flow in `NewNotes.txt`, since agents need reliable shared storage for new clips and annotations.

6.  **Remote Execution Prototype:** Create a script `scripts/run_remote_inference.sh` on the Mac that successfully triggers a simple python script on the Ubuntu machine (e.g., "Hello GPU") via SSH.
    - Plan: Use passwordless SSH (key-based auth) from Mac to Ubuntu so agents can safely trigger remote jobs.
    - TODO (repo): Add `scripts/run_remote_inference.sh` that reads SSH target and project path from `.env` (e.g., `DEER_REMOTE_HOST`, `DEER_REMOTE_PROJECT_ROOT`) and runs a tiny GPU test script.
    - TODO (remote): Implement a simple `scripts/hello_gpu.py` that prints device info (e.g., `torch.cuda.is_available()`, current GPU name).
    - Note: This is the first concrete bridge from the "controller" Mac to the "muscle" Ubuntu, enabling the agent workflows described in `NewNotes.txt`.

7.  **Daily Review Workflow Skeleton:** Create a placeholder script `scripts/daily_review.py` that simulates the "morning review" process: finding new clips on the share, and listing them for the user.
    - Idea: Start with a CLI that lists unreviewed clips and prints suggested actions; later promote to a Streamlit or other UI.
    - TODO (repo): Design a simple convention for flagging "needs human review" clips (e.g., a JSON index on the Samba share).

8.  **Dashboard Update:** Update `scripts/deervision_dashboard_tmux.sh` to point to the new remote directory paths and verify it correctly reports Ubuntu status.
    - Idea: Dashboard should assume `~/projects/DeerAITrackingResponse` as the default root on Ubuntu going forward.
    - TODO (repo/remote): Update any stale `deer-vision` paths in the script and test the tmux layout on Ubuntu.

9.  **Agent Access Config:** Create a `AGENTS_REMOTE.md` file specifically for agents, detailing the SSH commands, paths, and constraints for using the Ubuntu server autonomously.
    - Idea: This document becomes the contract for future agents (what they may run, where data lives, and safety constraints).
    - TODO (repo): Draft `AGENTS_REMOTE.md` with concrete examples (e.g., how to launch training, run eval, or inspect logs).

10. **Documentation Sync:** Update `README.md` to reflect the new unified structure, the role of the `external` directory, and the standard commands for remote execution.
    - TODO (repo): Fold the Medium-level plan and NewNotes vision into a concise "Architecture" and "Daily Workflow" section.
    - Note: Keep `README.md` focused on onboarding humans; cross-link deeper agent instructions to `AGENTS.md` and `AGENTS_REMOTE.md`.
