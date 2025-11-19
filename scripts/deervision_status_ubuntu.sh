#!/usr/bin/env bash
set -euo pipefail

# Basic, dependency-light health summary for the Ubuntu server that
# runs the deervision pipeline.
#
# It is safe to run repeatedly (e.g., in a tmux pane).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load shared configuration if present (camera RTSP URLs, CVAT URL, etc.).
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/.env"
    set +a
fi

print_header() {
    printf "\n=== %s ===\n" "$1"
}

check_disk() {
    local path="$1"
    local label="$2"
    if ! df_out=$(df -h "$path" 2>/dev/null | awk 'NR==2 {print $5, $3 "/" $2, $6}'); then
        printf "%-12s %-4s %s\n" "$label" "ERR" "unavailable"
        return
    fi

    local used_pct used pct mount status
    used_pct=$(printf "%s\n" "$df_out" | awk '{print $1}' | tr -d '%')
    used=$(printf "%s\n" "$df_out" | awk '{print $2}')
    mount=$(printf "%s\n" "$df_out" | awk '{print $3}')

    status="OK"
    if [[ "$used_pct" =~ ^[0-9]+$ ]]; then
        if (( used_pct >= 90 )); then
            status="CRIT"
        elif (( used_pct >= 80 )); then
            status="WARN"
        fi
    else
        status="UNK"
    fi

    printf "%-12s %-4s %3s%% used (%s) on %s\n" "$label" "$status" "$used_pct" "$used" "$mount"
}

check_service() {
    local url="$1"
    local label="$2"

    if [[ -z "$url" ]]; then
        printf "%-16s %-4s %s\n" "$label" "SKIP" "no URL configured"
        return
    fi

    if command -v curl >/dev/null 2>&1 && \
       curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
        printf "%-16s %-4s %s\n" "$label" "UP" "$url"
    else
        printf "%-16s %-4s %s\n" "$label" "DOWN" "$url"
    fi
}

check_process() {
    local pattern="$1"
    local label="$2"

    if pgrep -f "$pattern" >/dev/null 2>&1; then
        printf "%-24s %-7s (%s)\n" "$label" "RUNNING" "$pattern"
    else
        printf "%-24s %-7s (%s)\n" "$label" "DOWN" "$pattern"
    fi
}

print_camera_statuses() {
    # Treat any *_RTSP env vars as camera endpoints and probe RTSP port 554.
    local names var url host status

    if ! command -v nc >/dev/null 2>&1; then
        echo "nc (netcat) not found; skipping camera socket checks."
        return
    fi

    names=$(compgen -v | grep '_RTSP$' || true)
    if [[ -z "$names" ]]; then
        echo "No *_RTSP env vars loaded."
        return
    fi

    for var in $names; do
        url="${!var-}"
        [[ -z "$url" ]] && continue

        host=$(printf "%s\n" "$url" | sed -E 's#rtsp://([^/@]*@)?([^/:]+).*#\2#')
        if [[ -z "$host" ]]; then
            printf "%-24s %-15s %s\n" "$var" "-" "invalid RTSP URL"
            continue
        fi

        if nc -z -w 2 "$host" 554 >/dev/null 2>&1; then
            status="UP"
        else
            status="DOWN"
        fi

        printf "%-24s %-15s %s\n" "$var" "$host" "$status"
    done
}

main() {
    printf "Deervision Ubuntu Status — %s\n" "$(date)"

    print_header "Host"
    printf "Hostname: %s\n" "$(hostname)"
    if command -v uptime >/dev/null 2>&1; then
        printf "Uptime:   %s\n" "$(uptime -p 2>/dev/null || uptime)"
        printf "Load:     %s\n" "$(uptime | sed 's/.*load average[s]*: //')"
    fi

    print_header "CPU / Memory"
    if command -v free >/dev/null 2>&1; then
        free -h | awk 'NR==2 {printf "Mem:      %s used / %s total\n", $3, $2}'
    fi

    print_header "Disk"
    check_disk "/" "root"
    check_disk "/srv/deer-share" "deer-share"

    print_header "GPU"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits 2>/dev/null | head -n 1 | \
            awk -F',' '{printf "GPU: %s | util %s%% | mem %s / %s MiB\n", $1, $2, $3, $4}'
    else
        echo "nvidia-smi not found."
    fi

    print_header "Services"
    check_service "${DEERVISION_CVAT_URL:-http://localhost:8080}" "CVAT"
    check_service "${DEERVISION_STREAMLIT_URL:-http://localhost:8501}" "Streamlit"

    print_header "Pipeline Processes"
    # Edit this list to match the long-running jobs you care about.
    local processes=(
        "reolink_ingest:reolink_stream_ingest.py"
        "megadetector_live:live_megadetector.py"
        "mdv5_batch:mdv5_process_video.py"
        "prune_segments:prune_segments_without_events.py"
    )
    local entry label pattern
    for entry in "${processes[@]}"; do
        label="${entry%%:*}"
        pattern="${entry#*:}"
        check_process "$pattern" "$label"
    done

    print_header "Cameras (RTSP endpoints)"
    print_camera_statuses
}

main "$@"

