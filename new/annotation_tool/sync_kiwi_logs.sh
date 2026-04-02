#!/bin/bash

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-kiwi-rmt}"
REMOTE_BASE="${REMOTE_BASE:-~/workspace/auto-annotation-research/new/annotation_tool}"
LOCAL_BASE="${LOCAL_BASE:-$(cd "$(dirname "$0")" && pwd)/kiwi_logs}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-5}"
MODE="${1:---watch}"
REMOTE_LOG_PATH="${REMOTE_BASE}/annotation.log"

mkdir -p "$LOCAL_BASE"

sync_once() {
    rsync -av \
        --include='annotation.log' \
        --include='inference_process.jsonl' \
        --include='annotated_samples_auto.csv' \
        --exclude='*' \
        "${REMOTE_HOST}:${REMOTE_BASE}/" \
        "${LOCAL_BASE}/"
}

get_remote_line_no() {
    local pattern="$1"
    ssh "$REMOTE_HOST" "grep -n \"$pattern\" $REMOTE_LOG_PATH | tail -n 1 | cut -d: -f1"
}

get_last_start_line() {
    get_remote_line_no "Starting auto-annotation for" || true
}

get_last_complete_line() {
    get_remote_line_no "Auto-annotation completed:" || true
}

normalize_line_no() {
    local value="${1:-}"
    if [[ -z "$value" ]]; then
        echo 0
    else
        echo "$value"
    fi
}

print_status() {
    echo "Remote host : ${REMOTE_HOST}"
    echo "Remote base : ${REMOTE_BASE}"
    echo "Local base  : ${LOCAL_BASE}"
}

case "$MODE" in
    --once)
        print_status
        sync_once
        ;;
    --wait-complete)
        print_status
        baseline_start="$(normalize_line_no "$(get_last_start_line)")"
        baseline_complete="$(normalize_line_no "$(get_last_complete_line)")"
        last_start="$baseline_start"

        if (( baseline_start > baseline_complete )); then
            run_in_progress=1
            echo "Detected an in-progress auto-annotation run on kiwi."
        else
            run_in_progress=0
            echo "Waiting for a new auto-annotation run to start on kiwi."
        fi

        while true; do
            current_start="$(normalize_line_no "$(get_last_start_line)")"
            current_complete="$(normalize_line_no "$(get_last_complete_line)")"

            if (( current_start > last_start )); then
                last_start="$current_start"
                run_in_progress=1
                echo "Detected a new auto-annotation run. Waiting for completion..."
            fi

            if (( run_in_progress == 1 && current_complete > baseline_complete && current_complete > last_start )); then
                echo "Detected auto-annotation completion on kiwi. Syncing logs..."
                sync_once
                echo "Sync complete."
                exit 0
            fi

            sleep "$INTERVAL_SECONDS"
        done
        ;;
    --watch)
        print_status
        echo "Watching kiwi logs. Press Ctrl+C to stop."
        while true; do
            sync_once
            sleep "$INTERVAL_SECONDS"
        done
        ;;
    *)
        echo "Usage: bash sync_kiwi_logs.sh [--once|--watch|--wait-complete]"
        exit 1
        ;;
esac
