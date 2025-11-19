#!/usr/bin/env bash

set -euo pipefail

# Simple environment setup script for the Ubuntu server.
# Usage (from repo root on Ubuntu):
#   bash scripts/setup_env_remote.sh
#
# This follows the AGENTS instructions:
#   python3 -m venv .venv
#   source .venv/bin/activate
#   pip install --no-cache-dir --force-reinstall -r constraints.txt
# and then installs the additional packages from requirements.txt.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[setup_env_remote] Project root: ${PROJECT_ROOT}"
echo "[setup_env_remote] Python binary: ${PYTHON_BIN}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[setup_env_remote] ERROR: '${PYTHON_BIN}' not found on PATH" >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "[setup_env_remote] Creating virtualenv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "[setup_env_remote] Reusing existing virtualenv at ${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "[setup_env_remote] Upgrading pip"
pip install --no-cache-dir --upgrade pip

if [ -f "${PROJECT_ROOT}/constraints.txt" ]; then
  echo "[setup_env_remote] Installing pinned core stack from constraints.txt"
  pip install --no-cache-dir --force-reinstall -r "${PROJECT_ROOT}/constraints.txt"
fi

if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
  echo "[setup_env_remote] Installing additional requirements from requirements.txt"
  pip install --no-cache-dir -r "${PROJECT_ROOT}/requirements.txt"
fi

echo "[setup_env_remote] Environment setup complete."
