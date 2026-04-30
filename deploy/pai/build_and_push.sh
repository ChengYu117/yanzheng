#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IGNORE_FILE="${SCRIPT_DIR}/.dockerignore"

: "${ACR_REGISTRY:?Set ACR_REGISTRY, e.g. registry-vpc.cn-hangzhou.aliyuncs.com}"
: "${ACR_NAMESPACE:?Set ACR_NAMESPACE}"
: "${ACR_REPOSITORY:?Set ACR_REPOSITORY}"

IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"
IMAGE_URI="${ACR_REGISTRY}/${ACR_NAMESPACE}/${ACR_REPOSITORY}:${IMAGE_TAG}"

BUILD_CONTEXT="$(mktemp -d)"
trap 'rm -rf "${BUILD_CONTEXT}"' EXIT

mkdir -p "${BUILD_CONTEXT}/repo"
tar -C "${REPO_ROOT}" --exclude-from="${IGNORE_FILE}" -cf - . | tar -C "${BUILD_CONTEXT}/repo" -xf -

echo "Building image: ${IMAGE_URI}"
docker build -f "${BUILD_CONTEXT}/repo/deploy/pai/Dockerfile" -t "${IMAGE_URI}" "${BUILD_CONTEXT}/repo"

if [[ -n "${ACR_USERNAME:-}" && -n "${ACR_PASSWORD:-}" ]]; then
  echo "${ACR_PASSWORD}" | docker login "${ACR_REGISTRY}" --username "${ACR_USERNAME}" --password-stdin
else
  echo "ACR_USERNAME / ACR_PASSWORD not set. Please run docker login ${ACR_REGISTRY} manually if needed."
fi

docker push "${IMAGE_URI}"

echo "Pushed image: ${IMAGE_URI}"
