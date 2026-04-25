#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

load_gce_env
activate_venv
ensure_repo_root

INPUT_DIR="${JUDGE_INPUT_DIR:-${SAE_OUTPUT_DIR}}"
OUTPUT_DIR="${JUDGE_OUTPUT_DIR:-${INPUT_DIR}/ai_judge}"

: "${JUDGE_TOP_LATENTS:=20}"
: "${JUDGE_TOP_N:=10}"
: "${JUDGE_CONTROL_N:=5}"
: "${JUDGE_GROUPS:=G1,G5,G20}"
: "${JUDGE_TEMPERATURE:=0}"
: "${JUDGE_MAX_RETRIES:=3}"
: "${JUDGE_REQUEST_TIMEOUT:=600}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is empty. Set it in ${ENV_FILE} before running AI judge." >&2
  exit 1
fi

if [[ -z "${OPENAI_MODEL:-}" ]]; then
  echo "OPENAI_MODEL is empty. Set it in ${ENV_FILE} before running AI judge." >&2
  exit 1
fi

if [[ ! -f "${INPUT_DIR}/judge_bundle/manifest.json" && ! -f "${INPUT_DIR}/manifest.json" ]]; then
  echo "judge_bundle is missing under '${INPUT_DIR}'." >&2
  echo "Run deploy/gce/run_full_eval.sh first with the current codebase." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
print_runtime_summary

cmd=(
  python -u run_ai_re_judge.py
  --input-dir "${INPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --model "${OPENAI_MODEL}"
  --top-latents "${JUDGE_TOP_LATENTS}"
  --top-n "${JUDGE_TOP_N}"
  --control-n "${JUDGE_CONTROL_N}"
  --groups "${JUDGE_GROUPS}"
  --temperature "${JUDGE_TEMPERATURE}"
  --max-retries "${JUDGE_MAX_RETRIES}"
  --request-timeout "${JUDGE_REQUEST_TIMEOUT}"
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

printf 'Running command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
