#!/usr/bin/env bash
#
# Day Watchman Agent - Shell Wrapper
#
# Designed for cron. It:
# 1. Changes to the repository directory
# 2. Activates the Python virtual environment
# 3. Runs the Day Watchman collector
# 4. Runs the Day Watchman reporter (Codex + MCP Agent Mail)
#
# Usage:
#   ./agents/daywatchman/run.sh
#   ./agents/daywatchman/run.sh --dry-run
#   ./agents/daywatchman/run.sh --verbose
#
# Cron example (3pm Central time):
#   See agents/daywatchman/cron.example
#
# Environment variables:
#   REPO_DIR      - Repository directory (default: /home/mtornga/projects/DeerAITrackingResponse)
#   VENV_DIR      - Virtual environment directory (default: ${REPO_DIR}/.venv)
#   LOG_DIR       - Log output directory (default: ${REPO_DIR}/runs/logs/watchman)
#   SHELL_TIMEOUT - Shell-level timeout in seconds (default: 900 = 15 min)
#

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/mtornga/projects/DeerAITrackingResponse}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/runs/logs/watchman}"
SHELL_TIMEOUT="${SHELL_TIMEOUT:-900}"

mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Day Watchman starting at $(date)"
echo "========================================"

cd "${REPO_DIR}"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: Virtual environment not found at ${VENV_DIR}"
    exit 1
fi

# Ensure Codex CLI is discoverable for the reporter (user-level npm prefix).
CODEX_BIN_PATH="${CODEX_BIN:-${HOME}/.npm-global/bin/codex}"
if [[ -x "${CODEX_BIN_PATH}" ]]; then
    export CODEX_BIN="${CODEX_BIN_PATH}"
    CODEX_BIN_DIR="$(dirname "${CODEX_BIN_PATH}")"
    case ":${PATH}:" in
        *":${CODEX_BIN_DIR}:"*) ;;
        *) export PATH="${CODEX_BIN_DIR}:${PATH}" ;;
    esac
fi

echo "Shell timeout: ${SHELL_TIMEOUT}s"

collector_args=("$@")
report_args=()
skip_report="false"
idx=0
while [[ ${idx} -lt ${#collector_args[@]} ]]; do
    arg="${collector_args[$idx]}"
    case "${arg}" in
        --dry-run|--collect-only)
            skip_report="true"
            ;;
        --skip-mail|--verbose)
            report_args+=("${arg}")
            ;;
        --agent-name|--thread-id|--mcp-url|--recipients|--timeout|--analysis|--digest|--snapshot)
            report_args+=("${arg}")
            idx=$((idx + 1))
            if [[ ${idx} -lt ${#collector_args[@]} ]]; then
                report_args+=("${collector_args[$idx]}")
            fi
            ;;
    esac
    idx=$((idx + 1))
done

if command -v timeout >/dev/null 2>&1; then
    set +e
    timeout "${SHELL_TIMEOUT}" python agents/daywatchman/run.py --collect-only "${collector_args[@]}"
    collector_exit=$?
    set -e
    if [[ ${collector_exit} -eq 124 ]]; then
        echo "========================================"
        echo "ERROR: Shell-level timeout (${SHELL_TIMEOUT}s) exceeded!"
        echo "========================================"
    fi
else
    echo "WARNING: 'timeout' command not available, running without shell-level timeout"
    set +e
    python agents/daywatchman/run.py --collect-only "${collector_args[@]}"
    collector_exit=$?
    set -e
fi

if [[ "${skip_report}" == "true" ]]; then
    exit ${collector_exit}
fi

if [[ ! -f "${REPO_DIR}/runs/logs/watchman/day-watchman-latest.json" ]]; then
    echo "WARNING: Snapshot missing, skipping reporter"
    exit ${collector_exit}
fi

if command -v timeout >/dev/null 2>&1; then
    set +e
    timeout "${SHELL_TIMEOUT}" python agents/daywatchman/report.py "${report_args[@]}"
    report_exit=$?
    set -e
else
    set +e
    python agents/daywatchman/report.py "${report_args[@]}"
    report_exit=$?
    set -e
fi

if [[ ${report_exit} -ne 0 ]]; then
    exit_code=${report_exit}
else
    exit_code=${collector_exit}
fi

echo "========================================"
echo "Day Watchman finished at $(date) with exit code ${exit_code}"
echo "========================================"

exit ${exit_code}
