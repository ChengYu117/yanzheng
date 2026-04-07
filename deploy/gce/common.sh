#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"

load_gce_env() {
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing environment file: ${ENV_FILE}" >&2
    echo "Run: bash deploy/gce/configure_env.sh --hf-token <YOUR_TOKEN>" >&2
    exit 1
  fi

  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a

  : "${DATA_ROOT:=/mnt/disks/data}"
  : "${MODEL_DIR:=${DATA_ROOT}/models/Llama-3.1-8B}"
  : "${HF_HOME:=${DATA_ROOT}/hf-cache}"
  : "${OUTPUT_ROOT:=${DATA_ROOT}/outputs}"
  : "${VENV_DIR:=${PROJECT_ROOT}/.venv}"
  : "${PYTHON_BIN:=python3.10}"
  : "${PYTORCH_INDEX_URL:=https://download.pytorch.org/whl/cu124}"
  : "${MODEL_HF_REPO_ID:=meta-llama/Llama-3.1-8B}"
  : "${MODEL_REVISION:=main}"
  : "${SAE_OUTPUT_DIR:=${OUTPUT_ROOT}/sae_eval_full}"
  : "${CAUSAL_OUTPUT_DIR:=${OUTPUT_ROOT}/causal_validation_full}"
  : "${SAE_BATCH_SIZE:=4}"
  : "${SAE_MAX_SEQ_LEN:=128}"
  : "${SAE_INFERENCE_MODE:=legacy}"
  : "${SAE_COMPARE_MEAN:=0}"
  : "${CAUSAL_BATCH_SIZE:=4}"
  : "${CAUSAL_MAX_SEQ_LEN:=128}"
  : "${CAUSAL_N_BOOTSTRAP:=10}"
  : "${CAUSAL_SIDE_EFFECT_MAX_SAMPLES:=16}"

  export SCRIPT_DIR PROJECT_ROOT DATA_ROOT MODEL_DIR HF_HOME OUTPUT_ROOT
  export VENV_DIR PYTHON_BIN PYTORCH_INDEX_URL MODEL_HF_REPO_ID MODEL_REVISION
  export SAE_OUTPUT_DIR CAUSAL_OUTPUT_DIR SAE_BATCH_SIZE SAE_MAX_SEQ_LEN
  export SAE_INFERENCE_MODE SAE_COMPARE_MEAN
  export CAUSAL_BATCH_SIZE CAUSAL_MAX_SEQ_LEN CAUSAL_N_BOOTSTRAP
  export CAUSAL_SIDE_EFFECT_MAX_SAMPLES
}

activate_venv() {
  if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Virtual environment not found: ${VENV_DIR}" >&2
    echo "Run deploy/gce/bootstrap.sh first." >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

ensure_repo_root() {
  cd "${PROJECT_ROOT}"
}
