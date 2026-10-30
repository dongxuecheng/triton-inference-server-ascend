#!/bin/bash
# Builds the Triton Ascend YOLO backends and deploys each library into the model
# that uses it.
#
# Usage:
#   ASCEND_HOME_PATH=/path/to/Ascend/ascend-toolkit/latest ./build.sh
#
# Optional environment:
#   ASCEND_HOME_PATH   CANN toolkit prefix (default: /usr/local/Ascend/ascend-toolkit/latest)
#   TRITON_DIR         Triton installation holding include/triton/core/tritonbackend.h
#                      (default: third_party/triton, populated by
#                      scripts/fetch-triton-headers.sh)
#   SOC_VERSION        Ascend target (default: Ascend310P3)
#   BUILD_DIR          Build tree (default: <repo>/build)

set -e

SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
MODELS_DIR="${SCRIPT_DIR}/workspace/models"

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
SOC_VERSION="${SOC_VERSION:-Ascend310P3}"

HOST_ARCH=$(uname -m)
ACL_INCLUDE_DIR=""
for candidate in "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/${HOST_ARCH}-linux/include"; do
    if [ -f "${candidate}/acl/acl.h" ]; then
        ACL_INCLUDE_DIR="${candidate}"
        break
    fi
done
if [ -z "${ACL_INCLUDE_DIR}" ]; then
    echo "[build.sh] ERROR: acl/acl.h not found under ${ASCEND_HOME_PATH}" >&2
    echo "[build.sh] Checked:" >&2
    echo "  ${ASCEND_HOME_PATH}/include/acl/acl.h" >&2
    echo "  ${ASCEND_HOME_PATH}/${HOST_ARCH}-linux/include/acl/acl.h" >&2
    exit 1
fi
echo "[build.sh] ACL include: ${ACL_INCLUDE_DIR}"

# Prefer the bundled headers, but fall back to a Triton installation if present.
if [ -z "${TRITON_DIR}" ]; then
    if [ -f "${SCRIPT_DIR}/third_party/triton/include/triton/core/tritonbackend.h" ]; then
        TRITON_DIR="${SCRIPT_DIR}/third_party/triton"
    else
        for candidate in /opt/tritonserver /usr /usr/local; do
            if [ -f "${candidate}/include/triton/core/tritonbackend.h" ]; then
                TRITON_DIR="${candidate}"
                break
            fi
        done
    fi
fi

if [ ! -f "${TRITON_DIR}/include/triton/core/tritonbackend.h" ]; then
    echo "[build.sh] ERROR: tritonbackend.h not found under ${TRITON_DIR}" >&2
    echo "[build.sh] Run scripts/fetch-triton-headers.sh, or set TRITON_DIR" >&2
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "[build.sh] ERROR: cmake not found" >&2
    exit 1
fi

# The Ascend C compiler stages temporary files in TMPDIR; keep it on a
# filesystem with room to spare when /tmp is small.
export TMPDIR="${TMPDIR:-/dev/shm}"

echo "[build.sh] CANN:       ${ASCEND_HOME_PATH}"
echo "[build.sh] Triton:     ${TRITON_DIR}"
echo "[build.sh] TMPDIR:     ${TMPDIR}"
echo "[build.sh] SOC:        ${SOC_VERSION}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
    -DTRITON_CORE_INCLUDE_DIR="${TRITON_DIR}/include" \
    -DSOC_VERSION="${SOC_VERSION}"
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

# Install rather than copying from the build tree: installation is what applies
# the `$ORIGIN` rpath that lets each backend find the shared canvas pool and the
# Ascend C kernel sitting next to it in the model's version directory.
INSTALL_DIR="${BUILD_DIR}/install"
rm -rf "${INSTALL_DIR}"
cmake --install "${BUILD_DIR}" --prefix "${INSTALL_DIR}"

# The Ascend C kernel is emitted under <install>/lib alongside the backends.
LIB_DIR="${INSTALL_DIR}/lib"

# Each custom backend is deployed inside the version directory of the model that
# uses it, which is how the reference project organises its model repository.
# The shared canvas pool accompanies both device backends; its SONAME keeps the
# loader from instantiating it twice.
deploy() {
    local model="$1"
    shift
    local target="${MODELS_DIR}/${model}/1"
    [ -d "${MODELS_DIR}/${model}" ] || return 0
    mkdir -p "${target}"
    for library in "$@"; do
        if [ ! -f "${LIB_DIR}/${library}" ]; then
            echo "[build.sh] ERROR: ${library} missing from ${LIB_DIR}" >&2
            exit 1
        fi
        cp -f "${LIB_DIR}/${library}" "${target}/"
    done
    echo "[build.sh] deployed ${model} <- $*"
}

deploy YOLO_640_LETTERBOX_PREPROCESS \
    libtriton_preprocess.so libyolo_preprocess_kernel.so libtriton_ascend_canvas_pool.so
# Same backend and the same kernel library, behind a YUV input instead of BGR.
# The kernel library carries a SONAME, so every model that loads it needs the
# same build of it: one stale copy in any of these directories is the one the
# whole process gets.
deploy YUV_640_LETTERBOX_PREPROCESS \
    libtriton_preprocess.so libyolo_preprocess_kernel.so libtriton_ascend_canvas_pool.so
deploy YOLO_640_LETTERBOX_PREPROCESS_RGB \
    libtriton_rgb_preprocess.so libyolo_rgb_preprocess_kernel.so
deploy YOLO11_DET_PRE \
    libtriton_yoloinfer.so libtriton_ascend_canvas_pool.so
deploy YOLO11_DET_PRE_HOST \
    libtriton_yoloinfer.so libtriton_ascend_canvas_pool.so
deploy YOLO11_DET_PRE_POSTPROCESS \
    libtriton_yolo11_postprocess.so
deploy YOLO26_DET_PRE \
    libtriton_yolo26_infer.so libtriton_ascend_canvas_pool.so
deploy YOLO26_DET_PRE_HOST \
    libtriton_yolo26_infer.so libtriton_ascend_canvas_pool.so
deploy YOLO26_DET_PRE_RGB \
    libtriton_yolo26_infer.so libtriton_ascend_canvas_pool.so
deploy YOLO26_DET_PRE_POSTPROCESS \
    libtriton_yolo26_postprocess.so

echo
echo "[build.sh] Done. Model repository: ${MODELS_DIR}"
