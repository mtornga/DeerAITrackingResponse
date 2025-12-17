#!/usr/bin/env bash
#
# Run the feedback cycle on the Ubuntu server with GTX 3080
#
# Usage:
#   ./scripts/run_feedback_cycle.sh              # Single run
#   ./scripts/run_feedback_cycle.sh --continuous # Continuous mode
#   ./scripts/run_feedback_cycle.sh --tmux       # Start in tmux session
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_ROOT/logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Load environment
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

# Activate virtual environment
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    source "$REPO_ROOT/.venv/bin/activate"
elif [[ -f "$HOME/.venv/bin/activate" ]]; then
    source "$HOME/.venv/bin/activate"
fi

# Parse arguments
CONTINUOUS=""
USE_TMUX=""
EPOCHS=80
BATCH_SIZE=16
MODEL_VARIANTS="yolov8n,yolov8s"

while [[ $# -gt 0 ]]; do
    case $1 in
        --continuous)
            CONTINUOUS="--continuous"
            shift
            ;;
        --tmux)
            USE_TMUX="yes"
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model-variants)
            MODEL_VARIANTS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        return 0
    else
        echo "No NVIDIA GPU detected, will use CPU"
        return 1
    fi
}

# Main function
run_cycle() {
    echo "=============================================="
    echo "Deer Vision Feedback Cycle"
    echo "=============================================="
    echo "Repo root: $REPO_ROOT"
    echo "Epochs: $EPOCHS"
    echo "Batch size: $BATCH_SIZE"
    echo "Model variants: $MODEL_VARIANTS"
    echo ""

    check_gpu || true
    echo ""

    cd "$REPO_ROOT"

    python scripts/feedback_cycle.py \
        $CONTINUOUS \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --model-variants "$MODEL_VARIANTS" \
        --log-file "$LOG_DIR/feedback_cycle.log" \
        --poll-interval 3600
}

# Run in tmux if requested
if [[ -n "$USE_TMUX" ]]; then
    SESSION_NAME="feedback-cycle"

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Session $SESSION_NAME already exists. Attaching..."
        exec tmux attach -t "$SESSION_NAME"
    fi

    echo "Starting feedback cycle in tmux session: $SESSION_NAME"
    tmux new-session -d -s "$SESSION_NAME" -n "cycle"

    # Send the run command
    tmux send-keys -t "$SESSION_NAME:cycle" "cd $REPO_ROOT && source .venv/bin/activate && ./scripts/run_feedback_cycle.sh --continuous" Enter

    echo "Session started. Attach with: tmux attach -t $SESSION_NAME"
    echo "Detach with: Ctrl-b d"

    # Optionally attach
    read -p "Attach now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        exec tmux attach -t "$SESSION_NAME"
    fi
else
    run_cycle
fi
