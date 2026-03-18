#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/bailian_qwen35plus.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

INPUT_DIR="${1:-${JUDGE_INPUT_DIR:-outputs/sae_eval_full_max}}"
OUTPUT_DIR="${2:-${JUDGE_OUTPUT_DIR:-${INPUT_DIR}/ai_judge_qwen35plus}}"

: "${OPENAI_BASE_URL:=https://dashscope.aliyuncs.com/compatible-mode/v1}"
: "${OPENAI_MODEL:=qwen3.5-plus}"
: "${OPENAI_EXTRA_BODY_JSON:={"enable_thinking":false}}"
: "${JUDGE_TOP_LATENTS:=20}"
: "${JUDGE_TOP_N:=10}"
: "${JUDGE_CONTROL_N:=5}"
: "${JUDGE_GROUPS:=G1,G5,G20}"
: "${JUDGE_TEMPERATURE:=0}"
: "${JUDGE_MAX_RETRIES:=3}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is missing. Put it in ${ENV_FILE} or export it before running." >&2
  exit 1
fi

if [[ ! -f "${INPUT_DIR}/judge_bundle/manifest.json" && ! -f "${INPUT_DIR}/manifest.json" ]]; then
  echo "judge_bundle is missing under '${INPUT_DIR}'." >&2
  echo "Re-run run_sae_evaluation.py with the current codebase so it exports judge_bundle/ first." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_EXE:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" run_ai_re_judge.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model "${OPENAI_MODEL}" \
  --top-latents "${JUDGE_TOP_LATENTS}" \
  --top-n "${JUDGE_TOP_N}" \
  --control-n "${JUDGE_CONTROL_N}" \
  --groups "${JUDGE_GROUPS}" \
  --temperature "${JUDGE_TEMPERATURE}" \
  --max-retries "${JUDGE_MAX_RETRIES}"
