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
2.  **Sync Codebase:** Push the local `DEERAITRACKINGRESPONSE` code to the new remote directory (via git or rsync for now) to ensure the Ubuntu machine has the latest scripts and `perception` modules.
3.  **Environment Unification:** Create a setup script to install `requirements.txt` on the Ubuntu machine, ensuring the python environment matches the Mac's expectations.
4.  **External Tools Organization:** Move the useful parts of the remote `tmp` (YOLOv5, MegaDetector) into a structured `external/` directory on the remote machine and document how to reference them.
5.  **Samba Share Verification:** Verify the Samba share mount on both machines. Create a test script `scripts/verify_storage.sh` that writes/reads a file from both ends to confirm shared access.
6.  **Remote Execution Prototype:** Create a script `scripts/run_remote_inference.sh` on the Mac that successfully triggers a simple python script on the Ubuntu machine (e.g., "Hello GPU") via SSH.
7.  **Daily Review Workflow Skeleton:** Create a placeholder script `scripts/daily_review.py` that simulates the "morning review" process: finding new clips on the share, and listing them for the user.
8.  **Dashboard Update:** Update `scripts/deervision_dashboard_tmux.sh` to point to the new remote directory paths and verify it correctly reports Ubuntu status.
9.  **Agent Access Config:** Create a `AGENTS_REMOTE.md` file specifically for agents, detailing the SSH commands, paths, and constraints for using the Ubuntu server autonomously.
10. **Documentation Sync:** Update `README.md` to reflect the new unified structure, the role of the `external` directory, and the standard commands for remote execution.
