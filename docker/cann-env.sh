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
for lib_dir in \
    "${ASCEND_HOME_PATH}/lib64" \
    "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64" \
    "${ASCEND_HOME_PATH}/runtime/lib64" \
    "${ASCEND_HOME_PATH}/fwkacllib/lib64" \
    /usr/local/Ascend/driver/lib64 \
    /usr/local/Ascend/driver/lib64/common \
    /usr/local/Ascend/driver/lib64/driver \
    /usr/local/Ascend/develop/lib64; do
    if [ -d "${lib_dir}" ]; then
        export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
done
# 不把 devlib/stub 加入运行时路径，它们不能替代宿主机驱动。
unset lib_dir
