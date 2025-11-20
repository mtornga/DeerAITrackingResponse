#!/usr/bin/env bash
set -euo pipefail

# Launch a tmux-based "situational awareness" dashboard for deervision.
# Panes:
#   - Mac status (this machine)
#   - Ubuntu status (remote server)
#   - Ubuntu CPU/GPU live view (htop / top / nvidia-smi)
#   - Ubuntu logs tail (logs/*.log)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/.env"
    set +a
fi

SESSION_NAME="deervision-dashboard"
UBUNTU_HOST="${DEERVISION_UBUNTU_HOST:-192.168.68.71}"
UBUNTU_USER="${DEERVISION_UBUNTU_USER:-mtornga}"

# Normalize the remote repo path. If DEERVISION_UBUNTU_REPO_PATH was set with a
# local-style path (e.g., /Users/marktornga/...), ignore it and fall back to the
# canonical remote location under /home/<user>/projects.
if [[ -n "${DEERVISION_UBUNTU_REPO_PATH:-}" && "${DEERVISION_UBUNTU_REPO_PATH}" != /Users/* ]]; then
    UBUNTU_REPO_PATH="${DEERVISION_UBUNTU_REPO_PATH}"
else
    UBUNTU_REPO_PATH="/home/${UBUNTU_USER}/projects/DeerAITrackingResponse"
fi

# SSH options tuned for unattended use in tmux panes.
SSH_OPTS="-o ConnectTimeout=5 -o StrictHostKeyChecking=no"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed. Please install tmux to use this dashboard."
    exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Attaching to existing tmux session: ${SESSION_NAME}"
    exec tmux attach -t "${SESSION_NAME}"
fi

echo "Creating tmux session: ${SESSION_NAME}"

# Window 0, pane 0: Mac status (refresh loop).
tmux new-session -d -s "${SESSION_NAME}" -n dashboard
tmux send-keys -t "${SESSION_NAME}:0.0" "cd '${REPO_ROOT}'" C-m
tmux send-keys -t "${SESSION_NAME}:0.0" "while true; do clear; scripts/deervision_status_mac.sh; sleep 15; done" C-m

# Pane 1 (right): Ubuntu status (refresh loop).
tmux split-window -h -t "${SESSION_NAME}:0"
ubuntu_status_cmd="while true; do clear; ssh ${SSH_OPTS} ${UBUNTU_USER}@${UBUNTU_HOST} \"cd ${UBUNTU_REPO_PATH} && scripts/deervision_status_ubuntu.sh\" || { echo 'Ubuntu status: SSH failed'; sleep 5; }; sleep 15; done"
tmux send-keys -t "${SESSION_NAME}:0.1" "${ubuntu_status_cmd}" C-m

# Pane 2 (bottom-left): Ubuntu CPU/GPU live view.
tmux split-window -v -t "${SESSION_NAME}:0.0"
ubuntu_monitor_cmd="ssh ${SSH_OPTS} ${UBUNTU_USER}@${UBUNTU_HOST} 'if command -v htop >/dev/null 2>&1; then htop; else top; fi'"
tmux send-keys -t "${SESSION_NAME}:0.2" "${ubuntu_monitor_cmd}" C-m

# Pane 3 (bottom-right): Ubuntu logs tail.
tmux split-window -v -t "${SESSION_NAME}:0.1"
ubuntu_logs_cmd="ssh ${SSH_OPTS} ${UBUNTU_USER}@${UBUNTU_HOST} \"cd ${UBUNTU_REPO_PATH} && scripts/deervision_tail_logs.sh\""
tmux send-keys -t "${SESSION_NAME}:0.3" "${ubuntu_logs_cmd}" C-m

tmux select-layout -t "${SESSION_NAME}:0" tiled >/dev/null 2>&1 || true

echo "Dashboard session created. Attaching..."
exec tmux attach -t "${SESSION_NAME}"
