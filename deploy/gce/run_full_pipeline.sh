#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_gce_env
activate_venv
ensure_repo_root

mkdir -p "${PIPELINE_OUTPUT_DIR}"
PIPELINE_LOG="${PIPELINE_OUTPUT_DIR}/pipeline.log"
PIPELINE_STATUS="${PIPELINE_OUTPUT_DIR}/pipeline_status.json"
PIPELINE_EVENTS="${PIPELINE_OUTPUT_DIR}/pipeline_events.jsonl"

touch "${PIPELINE_LOG}" "${PIPELINE_EVENTS}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

PIPELINE_STARTED_AT="$(date +%s)"

json_write() {
  local path="$1"
  local tmp_path="${path}.tmp"
  shift
  python - "$tmp_path" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
items = sys.argv[2:]
payload = {}
for item in items:
    key, value = item.split("=", 1)
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        payload[key] = int(value)
    else:
        payload[key] = value
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
  mv "${tmp_path}" "${path}"
}

json_append_event() {
  local event="$1"
  local stage="$2"
  local status="$3"
  local message="$4"
  local exit_code="${5:-0}"
  local now
  now="$(date +%s)"
  python - "${PIPELINE_EVENTS}" "${event}" "${stage}" "${status}" "${message}" "${exit_code}" "$((now - PIPELINE_STARTED_AT))" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "event": sys.argv[2],
    "stage": sys.argv[3],
    "status": sys.argv[4],
    "message": sys.argv[5],
    "exit_code": int(sys.argv[6]),
    "elapsed_seconds": int(sys.argv[7]),
}
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
PY
}

write_status() {
  local status="$1"
  local stage="$2"
  local message="$3"
  local exit_code="${4:-0}"
  local now
  now="$(date +%s)"
  json_write "${PIPELINE_STATUS}" \
    "status=${status}" \
    "stage=${stage}" \
    "message=${message}" \
    "exit_code=${exit_code}" \
    "elapsed_seconds=$((now - PIPELINE_STARTED_AT))" \
    "pipeline_log=${PIPELINE_LOG}" \
    "pipeline_events=${PIPELINE_EVENTS}" \
    "sae_output_dir=${SAE_OUTPUT_DIR}" \
    "mapping_output_dir=${MAPPING_OUTPUT_DIR}" \
    "followup_output_dir=${FOLLOWUP_OUTPUT_DIR}" \
    "causal_candidate_output_dir=${CAUSAL_CANDIDATE_OUTPUT_DIR}" \
    "causal_output_dir=${CAUSAL_OUTPUT_DIR}"
}

run_stage() {
  local stage="$1"
  local enabled="$2"
  shift 2

  if [[ "${enabled}" != "1" ]]; then
    echo
    echo "=== Skipping ${stage} (${enabled}) ==="
    json_append_event "stage_skipped" "${stage}" "skipped" "stage switch is ${enabled}"
    return 0
  fi

  echo
  echo "=== Starting ${stage} ==="
  printf 'Running command:'
  printf ' %q' "$@"
  printf '\n'
  json_append_event "stage_start" "${stage}" "running" "stage started"
  write_status "running" "${stage}" "stage started"

  local stage_started_at
  stage_started_at="$(date +%s)"
  set +e
  "$@"
  local exit_code=$?
  set -e

  local stage_elapsed
  stage_elapsed=$(($(date +%s) - stage_started_at))
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "=== Failed ${stage} after ${stage_elapsed}s with exit code ${exit_code} ==="
    json_append_event "stage_failed" "${stage}" "failed" "stage failed" "${exit_code}"
    write_status "failed" "${stage}" "stage failed" "${exit_code}"
    exit "${exit_code}"
  fi

  echo "=== Completed ${stage} in ${stage_elapsed}s ==="
  json_append_event "stage_done" "${stage}" "completed" "stage completed"
  write_status "running" "${stage}" "stage completed"
}

echo "MISC full cloud pipeline started at $(date -Is)"
echo "Pipeline output dir: ${PIPELINE_OUTPUT_DIR}"
echo "Pipeline log: ${PIPELINE_LOG}"
echo "SAE output dir: ${SAE_OUTPUT_DIR}"
echo "Mapping output dir: ${MAPPING_OUTPUT_DIR}"
echo "Follow-up output dir: ${FOLLOWUP_OUTPUT_DIR}"
echo "Causal candidate output dir: ${CAUSAL_CANDIDATE_OUTPUT_DIR}"
echo "Causal output dir: ${CAUSAL_OUTPUT_DIR}"
echo "Causal data dir: ${CAUSAL_DATA_DIR}"
echo "Stage switches: RUN_SAE=${RUN_SAE}, RUN_INTERPRETABILITY=${RUN_INTERPRETABILITY}, RUN_CAUSAL=${RUN_CAUSAL}, RUN_AI_JUDGE=${RUN_AI_JUDGE}"

json_append_event "pipeline_start" "initializing" "running" "pipeline started"
write_status "running" "initializing" "pipeline started"

run_stage "sae_full_eval" "${RUN_SAE}" bash "${SCRIPT_DIR}/run_full_eval.sh"
run_stage "interpretability" "${RUN_INTERPRETABILITY}" bash "${SCRIPT_DIR}/run_interpretability.sh"
run_stage "causal_validation" "${RUN_CAUSAL}" bash "${SCRIPT_DIR}/run_causal.sh"
run_stage "ai_judge" "${RUN_AI_JUDGE}" bash "${SCRIPT_DIR}/run_ai_judge.sh"

json_append_event "pipeline_completed" "completed" "completed" "pipeline completed"
write_status "completed" "completed" "pipeline completed"

echo
echo "MISC full cloud pipeline completed."
echo "Check outputs:"
echo "  ${SAE_OUTPUT_DIR}/metrics_structural.json"
echo "  ${SAE_OUTPUT_DIR}/functional/misc_label_mapping/latent_label_matrix.csv"
echo "  ${MAPPING_OUTPUT_DIR}/mapping_structure_report.md"
echo "  ${FOLLOWUP_OUTPUT_DIR}/followup_interpretability_report.md"
echo "  ${CAUSAL_CANDIDATE_OUTPUT_DIR}/causal_candidate_groups.json"
echo "  ${CAUSAL_OUTPUT_DIR}/run_status.json"
echo "  ${CAUSAL_OUTPUT_DIR}/summary_tables.md"
