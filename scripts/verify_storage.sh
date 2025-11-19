#!/usr/bin/env bash
set -euo pipefail

# Simple cross-machine storage smoke test.
#
# Usage (from repo root on Mac or Ubuntu):
#   scripts/verify_storage.sh
#
# The script looks for a candidate shared directory (Samba/USB) and then:
#   - Writes a small test file tagged with hostname + timestamp
#   - Reads it back and verifies the contents
#
# This is meant to be run separately on the Mac and on the Ubuntu server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Try to load .env so DEER_SHARE_* vars are available when present.
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/.env"
  set +a
fi

HOSTNAME_SHORT="$(hostname 2>/dev/null || echo unknown-host)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ 2>/dev/null || date +%s)"

declare -a CANDIDATES=()

if [[ -n "${DEER_SHARE_SERVER_PATH:-}" ]]; then
  CANDIDATES+=("${DEER_SHARE_SERVER_PATH}")
fi
if [[ -n "${DEER_SHARE_LOCAL_MOUNT:-}" ]]; then
  CANDIDATES+=("${DEER_SHARE_LOCAL_MOUNT}")
fi

# Reasonable defaults if env vars are missing.
CANDIDATES+=("/srv/deer-share" "${HOME}/DeerShare")

TEST_DIR=""
for d in "${CANDIDATES[@]}"; do
  if [[ -d "${d}" ]]; then
    TEST_DIR="${d}"
    break
  fi
done

if [[ -z "${TEST_DIR}" ]]; then
  echo "[verify_storage] ERROR: No candidate share directory found."
  echo "  Checked: ${CANDIDATES[*]}"
  exit 1
fi

echo "[verify_storage] Using test directory: ${TEST_DIR}"

TEST_FILE="${TEST_DIR}/deer_share_smoketest_${HOSTNAME_SHORT}_${TIMESTAMP}.txt"
PAYLOAD="deer-share-smoketest host=${HOSTNAME_SHORT} ts=${TIMESTAMP}"

echo "[verify_storage] Writing test file: ${TEST_FILE}"
if ! printf "%s\n" "${PAYLOAD}" > "${TEST_FILE}"; then
  echo "[verify_storage] ERROR: Failed to write test file (permissions or mount issue?)."
  exit 1
fi

echo "[verify_storage] Reading back test file..."
READBACK="$(cat "${TEST_FILE}" 2>/dev/null || true)"

if [[ "${READBACK}" != "${PAYLOAD}" ]]; then
  echo "[verify_storage] ERROR: Content mismatch."
  echo "  Expected: ${PAYLOAD}"
  echo "  Got:      ${READBACK}"
  exit 1
fi

echo "[verify_storage] SUCCESS: Read/write ok on ${TEST_DIR}"
