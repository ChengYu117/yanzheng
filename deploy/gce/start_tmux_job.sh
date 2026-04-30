#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_gce_env
require_tmux
ensure_repo_root

if [[ $# -lt 2 ]]; then
  echo "Usage: bash deploy/gce/start_tmux_job.sh <session-name> <command> [args...]" >&2
  exit 1
fi

SESSION_NAME="$1"
shift

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
  exit 1
fi

LOG_ROOT="${OUTPUT_ROOT}/tmux_launcher_logs"
mkdir -p "${LOG_ROOT}"
LAUNCH_LOG="${LOG_ROOT}/${SESSION_NAME}_launcher.log"

COMMAND_STRING=""
for arg in "$@"; do
  if [[ -n "${COMMAND_STRING}" ]]; then
    COMMAND_STRING+=" "
  fi
  COMMAND_STRING+="$(printf '%q' "${arg}")"
done

RUNNER="export ENV_FILE=$(printf '%q' "${ENV_FILE}"); export RUN_SAE=$(printf '%q' "${RUN_SAE}"); export RUN_INTERPRETABILITY=$(printf '%q' "${RUN_INTERPRETABILITY}"); export RUN_CAUSAL=$(printf '%q' "${RUN_CAUSAL}"); export RUN_AI_JUDGE=$(printf '%q' "${RUN_AI_JUDGE}"); export CAUSAL_LABEL=$(printf '%q' "${CAUSAL_LABEL}"); export CAUSAL_DATA_DIR=$(printf '%q' "${CAUSAL_DATA_DIR}"); export ALLOW_LEGACY_CAUSAL_DATA=$(printf '%q' "${ALLOW_LEGACY_CAUSAL_DATA}"); cd $(printf '%q' "${PROJECT_ROOT}"); ${COMMAND_STRING}"

printf -v RUNNER_QUOTED '%q' "${RUNNER}"
tmux new-session -d -s "${SESSION_NAME}" "bash -lc ${RUNNER_QUOTED}" >>"${LAUNCH_LOG}" 2>&1

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "List sessions: tmux ls"
echo "Launcher log: ${LAUNCH_LOG}"
