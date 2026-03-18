#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_gce_env
activate_venv
ensure_repo_root

mkdir -p "${SAE_OUTPUT_DIR}"

cmd=(
  python run_sae_evaluation.py
  --model-dir "${MODEL_DIR}"
  --data-dir "data/mi_re"
  --output-dir "${SAE_OUTPUT_DIR}"
  --batch-size "${SAE_BATCH_SIZE}"
  --max-seq-len "${SAE_MAX_SEQ_LEN}"
  --full-structural
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

printf 'Running command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" | tee "${SAE_OUTPUT_DIR}/run.log"
