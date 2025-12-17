#!/usr/bin/env bash
#
# Sync feedback frames to Ubuntu server and trigger training cycle
#
# This script:
# 1. Syncs feedback frames from Mac to Ubuntu server
# 2. Optionally starts the feedback cycle on the server
#
# Usage:
#   ./scripts/sync_and_train.sh           # Sync only
#   ./scripts/sync_and_train.sh --train   # Sync and start training
#   ./scripts/sync_and_train.sh --tmux    # Sync and start training in tmux
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

# Server config from .env
UBUNTU_HOST="${DEERVISION_UBUNTU_HOST:-192.168.68.71}"
UBUNTU_USER="${DEERVISION_UBUNTU_USER:-mtornga}"
UBUNTU_REPO="${DEERVISION_UBUNTU_REPO_PATH:-~/projects/DeerAITrackingResponse}"

# Local paths
LOCAL_FEEDBACK_DIR="$REPO_ROOT/outdoor/deer-vision/data/raw/feedback"

# Parse args
DO_TRAIN=""
USE_TMUX=""
CONTINUOUS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --train)
            DO_TRAIN="yes"
            shift
            ;;
        --tmux)
            DO_TRAIN="yes"
            USE_TMUX="yes"
            shift
            ;;
        --continuous)
            CONTINUOUS="--continuous"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Sync Feedback Frames & Train"
echo "=============================================="
echo "Ubuntu server: $UBUNTU_USER@$UBUNTU_HOST"
echo "Remote repo:   $UBUNTU_REPO"
echo ""

# Check SSH connectivity
echo "Checking SSH connectivity..."
if ! ssh -o ConnectTimeout=5 "$UBUNTU_USER@$UBUNTU_HOST" "echo 'Connected'" 2>/dev/null; then
    echo "ERROR: Cannot connect to $UBUNTU_USER@$UBUNTU_HOST"
    echo "Make sure SSH keys are set up and the server is reachable."
    exit 1
fi
echo "SSH connection OK"
echo ""

# Check local feedback frames
if [[ ! -d "$LOCAL_FEEDBACK_DIR" ]]; then
    echo "No feedback frames found at $LOCAL_FEEDBACK_DIR"
    echo "Run extract_feedback_frames.py first."
    exit 1
fi

FRAME_COUNT=$(find "$LOCAL_FEEDBACK_DIR" -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
echo "Found $FRAME_COUNT feedback frames to sync"

if [[ "$FRAME_COUNT" -eq 0 ]]; then
    echo "No frames to sync."
    exit 0
fi

# Sync feedback frames
echo ""
echo "Syncing feedback frames to server..."
rsync -avz --progress \
    "$LOCAL_FEEDBACK_DIR/" \
    "$UBUNTU_USER@$UBUNTU_HOST:$UBUNTU_REPO/outdoor/deer-vision/data/raw/feedback/"

echo ""
echo "Sync complete!"

# Trigger training if requested
if [[ -n "$DO_TRAIN" ]]; then
    echo ""
    echo "=============================================="
    echo "Starting Training on Server"
    echo "=============================================="

    if [[ -n "$USE_TMUX" ]]; then
        echo "Starting feedback cycle in tmux session..."
        ssh "$UBUNTU_USER@$UBUNTU_HOST" "cd $UBUNTU_REPO && ./scripts/run_feedback_cycle.sh --tmux $CONTINUOUS"
    else
        echo "Running feedback cycle (will block until complete)..."
        ssh -t "$UBUNTU_USER@$UBUNTU_HOST" "cd $UBUNTU_REPO && ./scripts/run_feedback_cycle.sh $CONTINUOUS"
    fi
fi

echo ""
echo "Done!"
