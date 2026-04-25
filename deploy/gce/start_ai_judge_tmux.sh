#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SESSION_NAME="${TMUX_SESSION_NAME:-ai_judge}"
exec bash "${SCRIPT_DIR}/start_tmux_job.sh" "${SESSION_NAME}" bash "${SCRIPT_DIR}/run_ai_judge.sh" "$@"
