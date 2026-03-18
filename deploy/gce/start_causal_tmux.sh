#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SESSION_NAME="${TMUX_SESSION_NAME:-causal_eval}"
exec bash "${SCRIPT_DIR}/start_tmux_job.sh" "${SESSION_NAME}" bash "${SCRIPT_DIR}/run_causal.sh" "$@"
