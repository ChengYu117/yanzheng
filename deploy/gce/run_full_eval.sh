#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_gce_env
activate_venv
ensure_repo_root

mkdir -p "${SAE_OUTPUT_DIR}"
print_runtime_summary

cmd=(
  python -u run_sae_evaluation.py
  --model-dir "${MODEL_DIR}"
  --device "${DEVICE}"
  --data-dir "${DATA_DIR}"
  --output-dir "${SAE_OUTPUT_DIR}"
  --batch-size "${SAE_BATCH_SIZE}"
  --max-seq-len "${SAE_MAX_SEQ_LEN}"
  --checkpoint-topk-semantics "${CHECKPOINT_TOPK_SEMANTICS}"
  --full-structural
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

printf 'Running command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" 2>&1 | tee "${SAE_OUTPUT_DIR}/run.log"
