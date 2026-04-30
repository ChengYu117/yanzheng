#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_gce_env

sudo apt-get update
sudo apt-get install -y tmux git

mkdir -p "$(dirname "${MODEL_DIR}")" "${HF_HOME}" "${OUTPUT_ROOT}"

if [[ -z "${CONDA_ENV_NAME:-}" ]]; then
  if ! apt-cache show "${PYTHON_BIN}" >/dev/null 2>&1 || ! apt-cache show "${PYTHON_BIN}-venv" >/dev/null 2>&1; then
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
  fi
  sudo apt-get install -y "${PYTHON_BIN}" "${PYTHON_BIN}-venv" python3-pip

  if ! "${PYTHON_BIN}" -Im ensurepip --version >/dev/null 2>&1; then
    echo "Python ensurepip is unavailable for ${PYTHON_BIN}." >&2
    echo "Install the matching venv package and recreate the environment:" >&2
    echo "  sudo apt-get update" >&2
    echo "  sudo apt-get install -y ${PYTHON_BIN}-venv" >&2
    echo "  rm -rf ${VENV_DIR}" >&2
    echo "  bash deploy/gce/bootstrap.sh" >&2
    exit 1
  fi

  if [[ -d "${VENV_DIR}" && ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Removing incomplete virtual environment: ${VENV_DIR}" >&2
    rm -rf "${VENV_DIR}"
  fi

  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Using existing conda environment: ${CONDA_ENV_NAME}"
fi

activate_venv
ensure_repo_root

python - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit(
        "The official lm-saes runtime requires Python 3.11+ in this project. "
        f"Current Python is {sys.version.split()[0]}."
    )
PY

python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==2.6.0" --index-url "${PYTORCH_INDEX_URL}"
python -m pip install -r requirements.txt
python -m pip install -e .

python - <<'PY'
import os
import importlib.util
import torch

print("=== Bootstrap Validation ===")
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count:", torch.cuda.device_count())
    print("cuda_device_name:", torch.cuda.get_device_name(0))
print("MODEL_DIR:", os.environ.get("MODEL_DIR"))
print("HF_HOME:", os.environ.get("HF_HOME"))
print("OUTPUT_ROOT:", os.environ.get("OUTPUT_ROOT"))
print("DATA_DIR:", os.environ.get("DATA_DIR"))
print("CHECKPOINT_TOPK_SEMANTICS:", os.environ.get("CHECKPOINT_TOPK_SEMANTICS"))
print("HF_TOKEN_present:", bool(os.environ.get("HF_TOKEN")))
for package_name, import_name in (
    ("lm-saes", "lm_saes"),
    ("transformer-lens", "transformer_lens"),
):
    if importlib.util.find_spec(import_name) is None:
        raise RuntimeError(
            f"{package_name} is not installed. Use Python 3.11+ so requirements.txt installs it."
        )
    print(f"{package_name}_present: True")
PY

python run_sae_evaluation.py --help > /dev/null
python run_misc_mapping_structure_analysis.py --help > /dev/null
python run_misc_interpretability_analysis.py --help > /dev/null
python run_misc_causal_candidate_export.py --help > /dev/null
python causal/run_experiment.py --help > /dev/null
python run_ai_re_judge.py --help > /dev/null

echo
echo "Bootstrap completed successfully."
echo "Next steps:"
echo "  1. Copy deploy/gce/env.example to deploy/gce/.env and set HF_TOKEN if you have not done so."
echo "  2. Run: bash deploy/gce/download_model.sh"
echo "  3. Run full MISC pipeline: bash deploy/gce/start_full_pipeline_tmux.sh"
