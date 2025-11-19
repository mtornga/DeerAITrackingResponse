#!/usr/bin/env bash
set -euo pipefail

# Compact status view for the Mac that orchestrates the deervision workflow.
# Intended to run in a tmux pane and refresh in a loop.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/.env"
    set +a
fi

# Prefer the project-local virtualenv if it exists.
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/.venv/bin/activate"
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

    local used_pct used mount status
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

print_camera_statuses() {
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
    printf "Deervision Mac Status — %s\n" "$(date)"

    print_header "Host"
    printf "Hostname: %s\n" "$(hostname)"
    if command -v uptime >/dev/null 2>&1; then
        printf "Uptime:   %s\n" "$(uptime)"
    fi

    print_header "CPU / Memory"
    if command -v uptime >/dev/null 2>&1; then
        printf "Load:     %s\n" "$(uptime | sed 's/.*load average[s]*: //')"
    fi
    if command -v top >/dev/null 2>&1; then
        top -l 1 | awk '/PhysMem/ {print "PhysMem: " $2 " used, " $6 " wired, " $8 " free"; exit}' || true
    fi

    print_header "Disk"
    check_disk "/" "root"

    print_header "Ubuntu Server"
    local ubuntu_host="${DEERVISION_UBUNTU_HOST:-192.168.68.71}"
    if command -v nc >/dev/null 2>&1 && \
       nc -z -w 2 "$ubuntu_host" 22 >/dev/null 2>&1; then
        printf "SSH (%s:22): UP\n" "$ubuntu_host"
    else
        printf "SSH (%s:22): DOWN\n" "$ubuntu_host"
    fi

    print_header "Local Services"
    check_service "${DEERVISION_STREAMLIT_URL:-http://localhost:8501}" "Streamlit"

    print_header "Cameras (RTSP endpoints)"
    print_camera_statuses
}

main "$@"
