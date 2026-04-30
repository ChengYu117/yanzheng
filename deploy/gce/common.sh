#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"

load_gce_env() {
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing environment file: ${ENV_FILE}" >&2
    echo "Copy deploy/gce/env.example to deploy/gce/.env and fill in your values." >&2
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
  : "${PYTHON_BIN:=python3.11}"
  : "${PYTORCH_INDEX_URL:=https://download.pytorch.org/whl/cu124}"
  : "${CONDA_ENV_NAME:=}"
  : "${CONDA_BASE:=}"
  : "${DATA_DIR:=data/mi_quality_counseling_misc}"
  : "${DEVICE:=cuda}"
  : "${CHECKPOINT_TOPK_SEMANTICS:=hard}"
  : "${MODEL_HF_REPO_ID:=meta-llama/Llama-3.1-8B}"
  : "${MODEL_REVISION:=main}"
  : "${HF_ENDPOINT:=}"
  : "${HF_HUB_DOWNLOAD_TIMEOUT:=60}"
  : "${HF_HUB_ETAG_TIMEOUT:=60}"
  : "${HF_HUB_DOWNLOAD_MAX_WORKERS:=4}"
  : "${SAE_OUTPUT_DIR:=${OUTPUT_ROOT}/sae_eval_full}"
  : "${CAUSAL_OUTPUT_DIR:=${OUTPUT_ROOT}/causal_validation_full}"
  : "${SAE_BATCH_SIZE:=4}"
  : "${SAE_MAX_SEQ_LEN:=128}"
  : "${CAUSAL_BATCH_SIZE:=4}"
  : "${CAUSAL_MAX_SEQ_LEN:=128}"

  export SCRIPT_DIR PROJECT_ROOT DATA_ROOT MODEL_DIR HF_HOME OUTPUT_ROOT
  export VENV_DIR PYTHON_BIN PYTORCH_INDEX_URL CONDA_ENV_NAME CONDA_BASE
  export DATA_DIR DEVICE CHECKPOINT_TOPK_SEMANTICS
  export MODEL_HF_REPO_ID MODEL_REVISION HF_ENDPOINT
  export HF_HUB_DOWNLOAD_TIMEOUT HF_HUB_ETAG_TIMEOUT HF_HUB_DOWNLOAD_MAX_WORKERS
  export SAE_OUTPUT_DIR CAUSAL_OUTPUT_DIR SAE_BATCH_SIZE SAE_MAX_SEQ_LEN
  export CAUSAL_BATCH_SIZE CAUSAL_MAX_SEQ_LEN
}

activate_venv() {
  if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
    local conda_base="${CONDA_BASE:-}"
    if [[ -z "${conda_base}" ]]; then
      if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base)"
      elif [[ -d "${HOME}/miniconda3" ]]; then
        conda_base="${HOME}/miniconda3"
      elif [[ -d "${HOME}/anaconda3" ]]; then
        conda_base="${HOME}/anaconda3"
      elif [[ -d "/opt/conda" ]]; then
        conda_base="/opt/conda"
      elif [[ -d "/usr/local/miniconda3" ]]; then
        conda_base="/usr/local/miniconda3"
      elif [[ -d "/usr/local/anaconda3" ]]; then
        conda_base="/usr/local/anaconda3"
      else
        echo "CONDA_ENV_NAME is set to '${CONDA_ENV_NAME}', but conda was not found." >&2
        echo "Set CONDA_BASE in ${ENV_FILE}, or leave CONDA_ENV_NAME empty to use VENV_DIR." >&2
        exit 1
      fi
    fi

    if [[ ! -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      echo "Conda activation script not found: ${conda_base}/etc/profile.d/conda.sh" >&2
      echo "Check CONDA_BASE in ${ENV_FILE}." >&2
      exit 1
    fi

    # shellcheck disable=SC1090
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    return
  fi

  if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Virtual environment not found: ${VENV_DIR}" >&2
    echo "Run deploy/gce/bootstrap.sh first." >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

print_runtime_summary() {
  if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
    echo "Runtime env: conda:${CONDA_ENV_NAME}"
  else
    echo "Runtime env: venv:${VENV_DIR}"
  fi
  echo "Python executable: $(command -v python)"
  python - <<'PY'
import importlib.util
import sys

print("Python version:", sys.version.split()[0])
required = (("lm_saes", "lm_saes"), ("transformer_lens", "transformer_lens"))
optional = (
    ("sentencepiece", "sentencepiece"),
    ("tiktoken", "tiktoken"),
    ("protobuf", "google.protobuf"),
)
missing = [name for name, import_name in required if importlib.util.find_spec(import_name) is None]
for name, import_name in required + optional:
    print(f"{name}_present:", importlib.util.find_spec(import_name) is not None)
if missing:
    raise SystemExit(
        "Missing required official SAE runtime packages: "
        + ", ".join(missing)
        + ". Run bash deploy/gce/bootstrap.sh in the same environment."
    )
PY
}

ensure_repo_root() {
  cd "${PROJECT_ROOT}"
}

require_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required but not installed. Run deploy/gce/bootstrap.sh first." >&2
    exit 1
  fi
}
