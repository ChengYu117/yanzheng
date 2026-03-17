#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_gce_env
activate_venv
ensure_repo_root

mkdir -p "${CAUSAL_OUTPUT_DIR}"

CANDIDATE_CSV="${CANDIDATE_CSV:-${SAE_OUTPUT_DIR}/candidate_latents.csv}"
if [[ ! -f "${CANDIDATE_CSV}" ]]; then
  echo "Candidate CSV not found: ${CANDIDATE_CSV}" >&2
  echo "Run deploy/gce/run_full_eval.sh first or set CANDIDATE_CSV explicitly." >&2
  exit 1
fi

cmd=(
  python causal/run_experiment.py
  --model-dir "${MODEL_DIR}"
  --candidate-csv "${CANDIDATE_CSV}"
  --data-dir "data/mi_re"
  --output-dir "${CAUSAL_OUTPUT_DIR}"
  --batch-size "${CAUSAL_BATCH_SIZE}"
  --max-seq-len "${CAUSAL_MAX_SEQ_LEN}"
)

if [[ -n "${CAUSAL_LAMBDAS:-}" ]]; then
  cmd+=(--lambdas)
  # shellcheck disable=SC2206
  lambda_args=(${CAUSAL_LAMBDAS})
  cmd+=("${lambda_args[@]}")
fi

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

printf 'Running command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" | tee "${CAUSAL_OUTPUT_DIR}/run.log"
