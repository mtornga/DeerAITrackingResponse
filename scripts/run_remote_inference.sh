#!/usr/bin/env bash
set -euo pipefail

# Simple "Hello GPU" style remote execution helper.
# Run this on the Mac (controller) from the repo root:
#   scripts/run_remote_inference.sh
#
# It will:
#   - Load .env (if present) to pick up DEERVISION_UBUNTU_* vars
#   - SSH into the Ubuntu server
#   - cd into the remote repo
#   - Activate the project virtualenv if available
#   - Run scripts/hello_gpu.py and stream the output

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/.env"
  set +a
fi

UBUNTU_HOST="${DEERVISION_UBUNTU_HOST:-192.168.68.71}"
UBUNTU_USER="${DEERVISION_UBUNTU_USER:-mtornga}"

# Prefer a sane default repo path on the remote; treat any older
# deer-vision path from .env as legacy and ignore it.
if [[ -n "${DEERVISION_UBUNTU_REPO_PATH:-}" && "${DEERVISION_UBUNTU_REPO_PATH}" != *"deer-vision"* ]]; then
  UBUNTU_REPO_PATH="${DEERVISION_UBUNTU_REPO_PATH}"
else
  UBUNTU_REPO_PATH="/home/${UBUNTU_USER}/projects/DeerAITrackingResponse"
fi

SSH_OPTS="-o ConnectTimeout=5 -o StrictHostKeyChecking=no"

echo "[run_remote_inference] Target: ${UBUNTU_USER}@${UBUNTU_HOST}"
echo "[run_remote_inference] Remote repo: ${UBUNTU_REPO_PATH}"

ssh ${SSH_OPTS} "${UBUNTU_USER}@${UBUNTU_HOST}" "cd ${UBUNTU_REPO_PATH} && \
  git pull --ff-only >/dev/null 2>&1 || true; \
  if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi; \
  python3 scripts/hello_gpu.py"
