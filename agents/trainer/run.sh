#!/usr/bin/env bash
#
# Trainer Agent - Shell Wrapper
#
# Usage:
#   ./agents/trainer/run.sh
#   ./agents/trainer/run.sh --dry-run
#   ./agents/trainer/run.sh --execute
#
# Environment variables:
#   REPO_DIR      - Repository directory (default: /home/mtornga/projects/DeerAITrackingResponse)
#   VENV_DIR      - Virtual environment directory (default: ${REPO_DIR}/.venv)
#   LOG_DIR       - Log output directory (default: ${REPO_DIR}/runs/logs/trainer)
#   SHELL_TIMEOUT - Shell-level timeout in seconds (default: 1800 = 30 min)
#

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/mtornga/projects/DeerAITrackingResponse}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/runs/logs/trainer}"
SHELL_TIMEOUT="${SHELL_TIMEOUT:-1800}"

mkdir -p "${LOG_DIR}"

cd "${REPO_DIR}"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: Virtual environment not found at ${VENV_DIR}"
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

if command -v timeout >/dev/null 2>&1; then
    set +e
    timeout "${SHELL_TIMEOUT}" "${PYTHON_BIN}" agents/trainer/run.py "$@"
    exit_code=$?
    set -e
    if [[ ${exit_code} -eq 124 ]]; then
        echo "ERROR: Shell-level timeout (${SHELL_TIMEOUT}s) exceeded"
    fi
else
    echo "WARNING: 'timeout' command not available, running without shell-level timeout"
    set +e
    "${PYTHON_BIN}" agents/trainer/run.py "$@"
    exit_code=$?
    set -e
fi

echo "Trainer finished with exit code ${exit_code}"
exit ${exit_code}
