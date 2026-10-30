#!/bin/bash
# 由构建/启动脚本 source；不能只写到 .bashrc，非交互启动不会加载它。
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
if [ ! -d "${ASCEND_HOME_PATH}" ]; then
    echo "[cann-env] CANN directory not found: ${ASCEND_HOME_PATH}" >&2
    return 1
fi

if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" || return 1
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh || return 1
else
    echo "[cann-env] CANN set_env.sh not found" >&2
    return 1
fi

# set_env.sh 可能重设 ASCEND_HOME_PATH；统一解析链接后使用真实版本目录。
export ASCEND_HOME_PATH="$(readlink -f "${ASCEND_HOME_PATH}")"
# 按查找优先级从左到右组装：宿主机真实驱动优先，不能被 develop 中的 HAL 覆盖。
library_prefix=""
for lib_dir in \
    /usr/local/Ascend/driver/lib64/driver \
    /usr/local/Ascend/driver/lib64/common \
    /usr/local/Ascend/driver/lib64 \
    "${ASCEND_HOME_PATH}/lib64" \
    "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64" \
    "${ASCEND_HOME_PATH}/runtime/lib64" \
    "${ASCEND_HOME_PATH}/fwkacllib/lib64"; do
    if [ -d "${lib_dir}" ]; then
        library_prefix="${library_prefix}${library_prefix:+:}${lib_dir}"
    fi
done
if [ -n "${library_prefix}" ]; then
    export LD_LIBRARY_PATH="${library_prefix}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# 不主动加入 develop、devlib 或 stub；已有环境路径保留在上述明确顺序之后。
unset lib_dir library_prefix
