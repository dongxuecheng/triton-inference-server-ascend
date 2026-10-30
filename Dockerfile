# 不使用 BuildKit 专用语法，兼容 Docker 18.09。
ARG CANN_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-310p-ubuntu22.04-py3.11
ARG TRITON_BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.10-py3

FROM ${CANN_IMAGE} AS builder
USER root
ARG SOC_VERSION=Ascend310P3
ENV DEBIAN_FRONTEND=noninteractive SOC_VERSION=${SOC_VERSION}
RUN test "$(uname -m)" = aarch64 && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake binutils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/triton-ascend-src
COPY CMakeLists.txt build.sh ./
COPY cmake ./cmake
COPY src ./src
COPY third_party ./third_party
COPY workspace/models ./workspace/models
COPY docker/cann-env.sh /opt/triton-ascend/cann-env.sh
# 不需要 NPU 或宿主机 Toolkit；避免构建时 /dev/shm 默认只有 64 MB。
RUN bash -c 'source /opt/triton-ascend/cann-env.sh && \
    export TMPDIR=/tmp && bash ./build.sh'

FROM ${TRITON_BASE_IMAGE} AS runtime
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN test "$(uname -m)" = aarch64 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libnuma1 libelf1 libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
# 保留 CANN 9 的版本目录和兼容链接，不从宿主机挂载 Toolkit。
COPY --from=builder /usr/local/Ascend/cann-9.0.0 /usr/local/Ascend/cann-9.0.0
COPY --from=builder /usr/local/Ascend/ascend-toolkit /usr/local/Ascend/ascend-toolkit
RUN ln -sfn /usr/local/Ascend/cann-9.0.0 /usr/local/Ascend/cann
ENV ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0 \
    ASCEND_DEVICE_ID=2
COPY --from=builder /opt/triton-ascend-src/workspace/models /models
COPY scripts/serve.sh /opt/triton-ascend/serve.sh
COPY docker /opt/triton-ascend
COPY workspace/models_to_load.txt /models_to_load.txt
EXPOSE 8000 8001 8002
ENTRYPOINT ["/bin/bash", "/opt/triton-ascend/entrypoint.sh"]
