#!/usr/bin/env bash
#
# Queue Enricher Batch Poller
#
# Usage:
#   ./agents/queue-enricher/poll_batch.sh batch_XXX
#   ./agents/queue-enricher/poll_batch.sh batch_XXX --once
#   ./agents/queue-enricher/poll_batch.sh batch_XXX --interval 300
#
# Defaults to polling every 5 minutes until output is available.
#

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <batch_id> [--once] [--interval SECONDS]"
  exit 1
fi

BATCH_ID="$1"
shift

INTERVAL=300
ONCE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)
      ONCE="true"
      ;;
    --interval)
      shift
      INTERVAL="${1:-300}"
      ;;
  esac
  shift || true
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

set -a
source "${REPO_DIR}/.env"
set +a

while true; do
  "${REPO_DIR}/agents/queue-enricher/run.sh" --poll-batch --batch-id "${BATCH_ID}"

  if [[ "${ONCE}" == "true" ]]; then
    exit 0
  fi

  sleep "${INTERVAL}"
done
