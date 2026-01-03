#!/usr/bin/env bash
#
# Night Watchman Agent - Shell Wrapper
#
# This script is designed to be called by cron. It:
# 1. Changes to the repository directory
# 2. Activates the Python virtual environment
# 3. Runs the Night Watchman agent
#
# Usage:
#   ./agents/night-watchman/run.sh
#   ./agents/night-watchman/run.sh --dry-run
#
# Cron example (1am Central = 7am UTC):
#   0 7 * * * /home/mtornga/projects/DeerAITrackingResponse/agents/night-watchman/run.sh >> /var/log/night-watchman.log 2>&1
#

set -euo pipefail

# Configuration
REPO_DIR="${REPO_DIR:-/home/mtornga/projects/DeerAITrackingResponse}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/runs/logs/watchman}"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Log start time
echo "========================================"
echo "Night Watchman starting at $(date)"
echo "========================================"

# Change to repo directory
cd "${REPO_DIR}"

# Activate virtual environment
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: Virtual environment not found at ${VENV_DIR}"
    exit 1
fi

# Verify Claude Code is available
if ! command -v claude &> /dev/null; then
    echo "ERROR: 'claude' command not found in PATH"
    echo "PATH: ${PATH}"
    exit 1
fi

# Run the agent
python agents/night-watchman/run.py "$@"
exit_code=$?

# Log completion
echo "========================================"
echo "Night Watchman finished at $(date) with exit code ${exit_code}"
echo "========================================"

exit ${exit_code}
