#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_gce_env
activate_venv
ensure_repo_root

mkdir -p "${CAUSAL_OUTPUT_DIR}"
print_runtime_summary

if [[ "${ALLOW_LEGACY_CAUSAL_DATA:-0}" != "1" ]]; then
  case "${CAUSAL_DATA_DIR}" in
    *derived/re_nonre*|*data/mi_re*|*data/cactus*)
      echo "Refusing to run causal validation on legacy/balanced data: ${CAUSAL_DATA_DIR}" >&2
      echo "The cloud causal stage is now intended to evaluate the full MISC dataset." >&2
      echo "Set CAUSAL_DATA_DIR=data/mi_quality_counseling_misc, or set ALLOW_LEGACY_CAUSAL_DATA=1 for an explicit legacy rerun." >&2
      exit 1
      ;;
  esac
fi

for arg in "$@"; do
  if [[ "${arg}" == "--data-dir" || "${arg}" == --data-dir=* ]]; then
    echo "Do not pass --data-dir to deploy/gce/run_causal.sh." >&2
    echo "Set CAUSAL_DATA_DIR in deploy/gce/.env or as an environment override instead." >&2
    exit 1
  fi
done

if [[ -n "${CANDIDATE_CSV:-}" ]]; then
  CANDIDATE_SOURCE="explicit CANDIDATE_CSV"
elif [[ -f "${CAUSAL_CANDIDATE_OUTPUT_DIR}/label_candidates/${CAUSAL_LABEL}_candidate_latents.csv" ]]; then
  CANDIDATE_CSV="${CAUSAL_CANDIDATE_OUTPUT_DIR}/label_candidates/${CAUSAL_LABEL}_candidate_latents.csv"
  CANDIDATE_SOURCE="MISC causal candidate export for ${CAUSAL_LABEL}"
else
  CANDIDATE_CSV="${SAE_OUTPUT_DIR}/candidate_latents.csv"
  CANDIDATE_SOURCE="SAE root candidate_latents.csv fallback"
fi

if [[ ! -f "${CANDIDATE_CSV}" ]]; then
  echo "Candidate CSV not found: ${CANDIDATE_CSV}" >&2
  echo "Run deploy/gce/run_full_eval.sh and deploy/gce/run_interpretability.sh first, or set CANDIDATE_CSV explicitly." >&2
  exit 1
fi

echo "Causal label: ${CAUSAL_LABEL}"
echo "Causal data dir: ${CAUSAL_DATA_DIR}"
echo "Candidate source: ${CANDIDATE_SOURCE}"
echo "Candidate CSV: ${CANDIDATE_CSV}"

cmd=(
  python -u causal/run_experiment.py
  --model-dir "${MODEL_DIR}"
  --device "${DEVICE}"
  --candidate-csv "${CANDIDATE_CSV}"
  --data-dir "${CAUSAL_DATA_DIR}"
  --output-dir "${CAUSAL_OUTPUT_DIR}"
  --batch-size "${CAUSAL_BATCH_SIZE}"
  --max-seq-len "${CAUSAL_MAX_SEQ_LEN}"
  --checkpoint-topk-semantics "${CHECKPOINT_TOPK_SEMANTICS}"
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

"${cmd[@]}" 2>&1 | tee "${CAUSAL_OUTPUT_DIR}/run.log"
