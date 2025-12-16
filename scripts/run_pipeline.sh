#!/usr/bin/env bash
set -euo pipefail

#
# run_pipeline.sh - Start the Deer Vision continuous pipeline
#
# This script launches the ingest and detector processes in a tmux session
# on the Ubuntu server. It's designed for 24/7 operation.
#
# Usage:
#   ./scripts/run_pipeline.sh [start|stop|status|attach]
#
# The pipeline consists of:
#   1. Stream Ingest - Captures RTSP stream into 60-second segments
#   2. Detector - Runs YOLOv8 on new segments, promotes events
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load environment
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

# Configuration
SESSION_NAME="deervision-pipeline"
VENV_PATH="${REPO_ROOT}/.venv"
SHARE_ROOT="${DEER_SHARE_SERVER_PATH:-/srv/deer-share}"
LOG_DIR="${SHARE_ROOT}/runs/live/logs"

# Pipeline parameters
SEGMENT_LENGTH="${SEGMENT_LENGTH:-60}"           # seconds per segment
RETENTION_HOURS="${RETENTION_HOURS:-72}"         # keep 3 days of segments
DETECTOR_MODELS="${DETECTOR_MODELS:-yolov8n.pt,yolov8n_deer.pt,yolov8s_deer.pt,rtdetr_deer.pt}"  # base YOLO (person) + deer models
DETECTOR_CONFIDENCE="${DETECTOR_CONFIDENCE:-0.25}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"             # seconds between detector checks
ENABLE_ROUTING="${ENABLE_ROUTING:-true}"         # enable autonomous routing
AUTO_ACCEPT_THRESHOLD="${AUTO_ACCEPT_THRESHOLD:-0.85}"  # min confidence for auto-accept
DETECTOR_DEVICE="${DETECTOR_DEVICE:-cuda:0}"     # pytorch device (cuda:0 or cpu)

# Camera - default to REOLINK_3
CAMERA_URL="${REOLINK_3_RTSP:-}"

usage() {
    cat <<EOF
Usage: $0 [command]

Commands:
  start   - Start the pipeline (default)
  stop    - Stop the pipeline
  status  - Check if pipeline is running
  attach  - Attach to the pipeline tmux session
  logs    - Tail the pipeline logs

Environment variables:
  SEGMENT_LENGTH        - Seconds per segment (default: 60)
  RETENTION_HOURS       - Hours to retain segments (default: 72)
  DETECTOR_MODELS       - Comma-separated model paths (default: yolov8n_deer.pt,yolov8s_deer.pt,rtdetr_deer.pt)
  DETECTOR_CONFIDENCE   - Min confidence for events (default: 0.25)
  DETECTOR_DEVICE       - PyTorch device (default: cuda:0)
  ENABLE_ROUTING        - Enable autonomous routing (default: true)
  AUTO_ACCEPT_THRESHOLD - Min confidence for auto-accept (default: 0.85)
EOF
    exit 1
}

check_deps() {
    if ! command -v tmux &>/dev/null; then
        echo "ERROR: tmux is required but not installed"
        exit 1
    fi

    if [[ ! -d "${VENV_PATH}" ]]; then
        echo "ERROR: Python venv not found at ${VENV_PATH}"
        exit 1
    fi

    if [[ -z "${CAMERA_URL}" ]]; then
        echo "ERROR: No camera URL configured. Set REOLINK_3_RTSP in .env"
        exit 1
    fi
}

start_pipeline() {
    check_deps

    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "Pipeline already running. Use '$0 attach' to view or '$0 stop' first."
        exit 1
    fi

    # Ensure log directory exists
    mkdir -p "${LOG_DIR}"

    echo "Starting Deer Vision pipeline..."
    echo "  Camera: ${CAMERA_URL:0:50}..."
    echo "  Segment length: ${SEGMENT_LENGTH}s"
    echo "  Retention: ${RETENTION_HOURS}h"
    echo "  Detector models: ${DETECTOR_MODELS}"
    echo "  Device: ${DETECTOR_DEVICE}"
    echo "  Routing: ${ENABLE_ROUTING} (auto-accept threshold: ${AUTO_ACCEPT_THRESHOLD})"
    echo "  Logs: ${LOG_DIR}/"

    # Create tmux session with ingest pane
    tmux new-session -d -s "${SESSION_NAME}" -n pipeline

    # Pane 0: Stream Ingest (with auto-restart loop)
    local ingest_cmd="cd '${REPO_ROOT}' && source '${VENV_PATH}/bin/activate' && while true; do
        echo '[\$(date)] Starting ingest...' | tee -a '${LOG_DIR}/ingest.log'
        python scripts/reolink_stream_ingest.py \\
            --stream-url '${CAMERA_URL}' \\
            --segment-length ${SEGMENT_LENGTH} \\
            --retention-hours ${RETENTION_HOURS} \\
            --output-dir '${SHARE_ROOT}/runs/live/segments' \\
            --analysis-dir '${SHARE_ROOT}/runs/live/analysis' \\
            --log-file '${LOG_DIR}/ingest.log' \\
            2>&1 | tee -a '${LOG_DIR}/ingest.log'
        echo '[\$(date)] Ingest exited, restarting in 10s...' | tee -a '${LOG_DIR}/ingest.log'
        sleep 10
    done"

    tmux send-keys -t "${SESSION_NAME}:0.0" "${ingest_cmd}" C-m

    # Split and create Pane 1: Detector (with auto-restart loop)
    tmux split-window -v -t "${SESSION_NAME}:0"

    # Build routing flags
    local routing_flags=""
    if [[ "${ENABLE_ROUTING}" == "true" ]]; then
        routing_flags="--enable-routing --auto-accept-threshold ${AUTO_ACCEPT_THRESHOLD}"
    fi

    local detector_cmd="cd '${REPO_ROOT}' && source '${VENV_PATH}/bin/activate' && sleep 15 && while true; do
        echo '[\$(date)] Starting multi-model detector...' | tee -a '${LOG_DIR}/detector.log'
        python scripts/live_detector_multimodel.py \\
            --models '${DETECTOR_MODELS}' \\
            --segments-dir '${SHARE_ROOT}/runs/live/analysis' \\
            --detections-dir '${SHARE_ROOT}/runs/live/detections' \\
            --events-dir '${SHARE_ROOT}/runs/live/events' \\
            --event-threshold ${DETECTOR_CONFIDENCE} \\
            --poll-interval ${POLL_INTERVAL} \\
            --device ${DETECTOR_DEVICE} \\
            ${routing_flags} \\
            --log-file '${LOG_DIR}/detector.log' \\
            2>&1 | tee -a '${LOG_DIR}/detector.log'
        echo '[\$(date)] Detector exited, restarting in 10s...' | tee -a '${LOG_DIR}/detector.log'
        sleep 10
    done"

    tmux send-keys -t "${SESSION_NAME}:0.1" "${detector_cmd}" C-m

    # Split and create Pane 2: Status monitor
    tmux split-window -h -t "${SESSION_NAME}:0.0"

    local status_cmd="watch -n 30 'echo \"=== Deer Vision Pipeline Status ===\"
echo \"\"
echo \"Segments (last 5):\"
ls -lt ${SHARE_ROOT}/runs/live/segments/\$(date +%Y-%m-%d)/ 2>/dev/null | head -6
echo \"\"
echo \"Events today:\"
ls ${SHARE_ROOT}/runs/live/events/\$(date +%Y-%m-%d)/ 2>/dev/null | wc -l
echo \"\"
echo \"Disk usage:\"
df -h ${SHARE_ROOT} | tail -1
echo \"\"
echo \"Last log lines:\"
tail -3 ${LOG_DIR}/ingest.log 2>/dev/null
tail -3 ${LOG_DIR}/detector.log 2>/dev/null'"

    tmux send-keys -t "${SESSION_NAME}:0.2" "${status_cmd}" C-m

    # Arrange panes nicely
    tmux select-layout -t "${SESSION_NAME}:0" main-horizontal

    echo ""
    echo "Pipeline started in tmux session '${SESSION_NAME}'"
    echo "  - Use '$0 attach' to view"
    echo "  - Use '$0 stop' to stop"
    echo "  - Use '$0 logs' to tail logs"
}

stop_pipeline() {
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "Stopping pipeline..."
        tmux kill-session -t "${SESSION_NAME}"
        echo "Pipeline stopped."
    else
        echo "Pipeline is not running."
    fi
}

check_status() {
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "Pipeline is RUNNING"
        echo ""
        echo "Session: ${SESSION_NAME}"
        tmux list-panes -t "${SESSION_NAME}" -F "  Pane #{pane_index}: #{pane_current_command}"
        echo ""
        echo "Recent segments:"
        ls -lt "${SHARE_ROOT}/runs/live/segments/$(date +%Y-%m-%d)/" 2>/dev/null | head -4 || echo "  (none today)"
        echo ""
        echo "Events today:"
        ls "${SHARE_ROOT}/runs/live/events/$(date +%Y-%m-%d)/" 2>/dev/null | head -5 || echo "  (none)"
    else
        echo "Pipeline is STOPPED"
    fi
}

attach_session() {
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        exec tmux attach -t "${SESSION_NAME}"
    else
        echo "Pipeline is not running. Use '$0 start' first."
        exit 1
    fi
}

tail_logs() {
    if [[ -d "${LOG_DIR}" ]]; then
        tail -f "${LOG_DIR}"/*.log
    else
        echo "No logs found at ${LOG_DIR}"
        exit 1
    fi
}

# Main
case "${1:-start}" in
    start)
        start_pipeline
        ;;
    stop)
        stop_pipeline
        ;;
    status)
        check_status
        ;;
    attach)
        attach_session
        ;;
    logs)
        tail_logs
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: $1"
        usage
        ;;
esac
