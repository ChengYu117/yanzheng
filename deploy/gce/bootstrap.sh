#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_gce_env

sudo apt-get update
sudo apt-get install -y "${PYTHON_BIN}" python3-pip python3.10-venv tmux git

mkdir -p "$(dirname "${MODEL_DIR}")" "${HF_HOME}" "${OUTPUT_ROOT}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
activate_venv
ensure_repo_root

python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==2.6.0" --index-url "${PYTORCH_INDEX_URL}"
python -m pip install -r requirements.txt
python -m pip install -e .

python - <<'PY'
import os
import torch

print("=== Bootstrap Validation ===")
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count:", torch.cuda.device_count())
    print("cuda_device_name:", torch.cuda.get_device_name(0))
print("MODEL_DIR:", os.environ.get("MODEL_DIR"))
print("HF_HOME:", os.environ.get("HF_HOME"))
print("OUTPUT_ROOT:", os.environ.get("OUTPUT_ROOT"))
print("HF_TOKEN_present:", bool(os.environ.get("HF_TOKEN")))
PY

python run_sae_evaluation.py --help > /dev/null
python causal/run_experiment.py --help > /dev/null
python run_ai_re_judge.py --help > /dev/null

echo
echo "Bootstrap completed successfully."
echo "Next steps:"
echo "  1. Copy deploy/gce/env.example to deploy/gce/.env and set HF_TOKEN if you have not done so."
echo "  2. Run: bash deploy/gce/download_model.sh"
