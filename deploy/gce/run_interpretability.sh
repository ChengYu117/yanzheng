#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_gce_env
activate_venv
ensure_repo_root

MAPPING_INPUT_DIR="${MAPPING_INPUT_DIR:-${SAE_OUTPUT_DIR}/functional/misc_label_mapping}"
MAPPING_MATRIX="${MAPPING_INPUT_DIR}/latent_label_matrix.csv"

if [[ ! -f "${MAPPING_MATRIX}" ]]; then
  echo "MISC latent-label matrix not found: ${MAPPING_MATRIX}" >&2
  echo "Run deploy/gce/run_full_eval.sh first, or set MAPPING_INPUT_DIR explicitly." >&2
  exit 1
fi

mkdir -p "${SAE_OUTPUT_DIR}/interpretability" "${MAPPING_OUTPUT_DIR}" "${FOLLOWUP_OUTPUT_DIR}" "${CAUSAL_CANDIDATE_OUTPUT_DIR}"
INTERPRETABILITY_LOG="${SAE_OUTPUT_DIR}/interpretability/run.log"
exec > >(tee "${INTERPRETABILITY_LOG}") 2>&1

print_runtime_summary
echo "Interpretability log: ${INTERPRETABILITY_LOG}"

mapping_cmd=(
  python -u run_misc_mapping_structure_analysis.py
  --mapping-dir "${MAPPING_INPUT_DIR}"
  --output-dir "${MAPPING_OUTPUT_DIR}"
  --doc-report ""
)

followup_cmd=(
  python -u run_misc_interpretability_analysis.py
  --eval-dir "${SAE_OUTPUT_DIR}"
  --output-dir "${FOLLOWUP_OUTPUT_DIR}"
  --doc-report ""
)

candidate_cmd=(
  python -u run_misc_causal_candidate_export.py
  --eval-dir "${SAE_OUTPUT_DIR}"
  --mapping-dir "${MAPPING_INPUT_DIR}"
  --mapping-structure-dir "${MAPPING_OUTPUT_DIR}"
  --followup-dir "${FOLLOWUP_OUTPUT_DIR}"
  --output-dir "${CAUSAL_CANDIDATE_OUTPUT_DIR}"
  --doc-report ""
)

run_cmd() {
  local label="$1"
  shift
  echo
  echo "=== ${label} ==="
  printf 'Running command:'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

run_cmd "Mapping Structure analysis" "${mapping_cmd[@]}"
run_cmd "Follow-up interpretability analysis" "${followup_cmd[@]}"
run_cmd "Causal candidate export" "${candidate_cmd[@]}"

echo
echo "Interpretability pipeline completed."
echo "Mapping report: ${MAPPING_OUTPUT_DIR}/mapping_structure_report.md"
echo "Follow-up report: ${FOLLOWUP_OUTPUT_DIR}/followup_interpretability_report.md"
echo "Causal candidates: ${CAUSAL_CANDIDATE_OUTPUT_DIR}/causal_candidate_groups.json"
