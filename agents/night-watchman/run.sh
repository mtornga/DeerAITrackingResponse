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

resolve_claude_bin() {
    local candidate=""
    if [[ -n "${CLAUDE_BIN:-}" ]]; then
        candidate="${CLAUDE_BIN}"
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
        if command -v "${candidate}" >/dev/null 2>&1; then
            command -v "${candidate}"
            return 0
        fi
    fi

    if command -v claude >/dev/null 2>&1; then
        command -v claude
        return 0
    fi

    local nvm_root="${HOME}/.nvm/versions/node"
    if [[ -d "${nvm_root}" ]]; then
        candidate="$(find "${nvm_root}" -mindepth 2 -maxdepth 3 -path '*/bin/claude' -type f 2>/dev/null | head -n 1)"
        if [[ -n "${candidate}" && -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    fi

    return 1
}

CLAUDE_BIN_PATH="$(resolve_claude_bin || true)"
if [[ -z "${CLAUDE_BIN_PATH}" ]]; then
    echo "ERROR: Could not locate the 'claude' CLI. Install it or set CLAUDE_BIN to the binary path."
    exit 1
fi
export CLAUDE_BIN="${CLAUDE_BIN_PATH}"
CLAUDE_BIN_DIR="$(dirname "${CLAUDE_BIN_PATH}")"
case ":${PATH}:" in
    *":${CLAUDE_BIN_DIR}:"*) ;;
    *) export PATH="${CLAUDE_BIN_DIR}:${PATH}" ;;
esac

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
# claude should now be resolvable; provide a helpful log line
echo "Using claude CLI at ${CLAUDE_BIN_PATH}"

# Run the agent
python agents/night-watchman/run.py "$@"
exit_code=$?

# Log completion
echo "========================================"
echo "Night Watchman finished at $(date) with exit code ${exit_code}"
echo "========================================"

exit ${exit_code}
