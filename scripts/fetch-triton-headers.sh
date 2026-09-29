#!/usr/bin/env bash
#
# Extract the Triton C++ backend headers from a Triton container image so the
# backend can be built without a local Triton installation.
#
# Usage:
#   scripts/fetch-triton-headers.sh [output-include-dir]
#
# Environment:
#   TRITON_IMAGE  Triton image to extract from
#                 (default: nvcr.io/nvidia/tritonserver:24.10-py3)
#
set -euo pipefail

IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.10-py3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${REPO_ROOT}/third_party/triton/include}"

if [ -f "${DEST}/triton/core/tritonbackend.h" ]; then
    echo "Triton backend headers already present in ${DEST}"
    exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required to extract headers from ${IMAGE}" >&2
    exit 1
fi

tmp_dir="$(mktemp -d)"
container="triton-header-export-$$"
cleanup() {
    docker rm -f "${container}" >/dev/null 2>&1 || true
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

docker create --name "${container}" --entrypoint /bin/true "${IMAGE}" >/dev/null
docker cp "${container}:/opt/tritonserver/include/triton" "${tmp_dir}/triton"

mkdir -p "${DEST}"
cp -a "${tmp_dir}/triton" "${DEST}/"

echo "Extracted Triton backend headers from ${IMAGE}"
echo "  to ${DEST}/triton/core"
echo
echo "Configure the build with:"
echo "  -DTRITON_CORE_INCLUDE_DIR=${DEST}"
