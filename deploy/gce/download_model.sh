#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_gce_env
activate_venv
ensure_repo_root

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ENV_FILE=${ENV_FILE}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "HF_HOME=${HF_HOME}"
echo "HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is empty. Set it in ${ENV_FILE} before downloading the model." >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}" "${HF_HOME}"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

token = os.environ["HF_TOKEN"]
repo_id = os.environ["MODEL_HF_REPO_ID"]
revision = os.environ.get("MODEL_REVISION") or None
model_dir = os.environ["MODEL_DIR"]
cache_dir = os.environ["HF_HOME"]
endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
max_workers = int(os.environ.get("HF_HUB_DOWNLOAD_MAX_WORKERS") or "4")

print(f"Downloading {repo_id} to {model_dir}")
print(f"HF_ENDPOINT={endpoint}")
print(f"HF_HOME={cache_dir}")
print(f"max_workers={max_workers}")
snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=model_dir,
    cache_dir=cache_dir,
    token=token,
    max_workers=max_workers,
)
print("Model download complete.")
PY
