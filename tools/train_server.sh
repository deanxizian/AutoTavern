#!/usr/bin/env bash
# Linux server helper; no credentials or host-specific settings belong here.
set -euo pipefail
cd "$(dirname "$0")/.."
action=${1:-status}
run_dir=${2:-runs/ppo-baseline}
if [ "$#" -ge 2 ]; then shift 2; else set --; fi
mkdir -p "$run_dir"
run_dir=$(realpath "$run_dir")
pid_file="$run_dir/pid"
is_running() {
  [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null &&
    tr '\0' ' ' < "/proc/$(cat "$pid_file")/cmdline" | grep -Fq 'training.train'
}
case "$action" in
  start|resume)
    if is_running; then echo "Already running: $(cat "$pid_file")"; exit 1; fi
    if [ -e "$run_dir/STOP" ]; then
      echo "Remove $run_dir/STOP to explicitly resume this stopped run."; exit 1
    fi
    extra=()
    if [ "$action" = resume ]; then extra=(--resume "$run_dir/latest.pt"); fi
    nohup .venv/bin/python -u -m training.train --run-dir "$run_dir" "${extra[@]}" "$@" \
      >> "$run_dir/train.log" 2>&1 < /dev/null &
    echo "$!" > "$pid_file"
    echo "Started PID $!; log: $run_dir/train.log"
    ;;
  status)
    if is_running; then echo "Running PID $(cat "$pid_file")"; else echo "Not running"; fi
    if [ -f "$run_dir/status.json" ]; then cat "$run_dir/status.json"; fi
    ;;
  stop)
    touch "$run_dir/STOP"
    echo "Stop requested; checkpoint is saved at the next rollout/update boundary."
    ;;
  *) echo "Usage: $0 {start|resume|status|stop} [run-dir] [trainer options]"; exit 2 ;;
esac
