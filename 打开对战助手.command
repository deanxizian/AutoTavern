#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$TASK_ROOT/script/build_and_run.sh"
