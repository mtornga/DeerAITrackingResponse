#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "${LOG_DIR}"

shopt -s nullglob

while true; do
    log_files=("${LOG_DIR}"/*.log)
    if (( ${#log_files[@]} == 0 )); then
        printf "No *.log files found in %s (as of %s)\n" "${LOG_DIR}" "$(date)"
        sleep 15
        continue
    fi

    printf "Tailing %s (Ctrl+C to stop)\n" "${log_files[*]}"
    tail -n 50 -F "${log_files[@]}"
    # If tail exits (e.g., files deleted), loop and check again.
done

