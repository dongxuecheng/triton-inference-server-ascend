#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/cann-env.sh"

# 显式命令用于无 NPU 的镜像检查、导出或设备预检。
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

model_file="${YOLO26_MODEL_FILE:-/models/YOLO26_DET_PRE/1/yolo26s.om}"
if [ ! -s "${model_file}" ] || [ ! -f "${model_file}" ]; then
    echo "[entrypoint] YOLO26 .om is missing or empty: ${model_file}" >&2
    echo "[entrypoint] Check the host file before creating the bind mount." >&2
    exit 1
fi
echo "[entrypoint] CANN=${ASCEND_HOME_PATH}; ACL logical device=${ASCEND_DEVICE_ID:-0}"
exec /bin/bash "${SCRIPT_DIR}/serve.sh"
