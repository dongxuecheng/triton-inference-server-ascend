#!/bin/bash
# Starts Triton for the model repository in this project.
#
# Runs in explicit model-control mode and loads the models listed in
# MODELS_FILE, so a model whose `.om` is not present yet (for example the YOLO26
# chain) does not stop the server from starting. Models can still be loaded and
# unloaded at runtime through the model repository API.
#
# Environment:
#   MODEL_REPOSITORY  repository path inside the container (default /models)
#   MODELS_FILE       load list inside the container (default /models_to_load.txt)
#   TRITON_HTTP_PORT / TRITON_GRPC_PORT / TRITON_METRICS_PORT
#   TRITON_LOG_VERBOSE
#
# CUSTOM_LABELS is always appended to the load list: it is the service clients
# use to turn class ids into names.

set -e

MODEL_REPOSITORY="${MODEL_REPOSITORY:-/models}"
MODELS_FILE="${MODELS_FILE:-/models_to_load.txt}"

models=()
if [ -f "${MODELS_FILE}" ]; then
    while IFS= read -r line || [ -n "${line}" ]; do
        # Strip comments and surrounding whitespace.
        name="${line%%#*}"
        name="$(printf '%s' "${name}" | tr -d '[:space:]')"
        if [ -n "${name}" ]; then
            models+=("${name}")
        fi
    done < "${MODELS_FILE}"
    echo "[serve.sh] read $((${#models[@]})) model(s) from ${MODELS_FILE}"
else
    echo "[serve.sh] ${MODELS_FILE} not found, starting with no models loaded"
fi

# Appended rather than required in the list, so forgetting it there cannot
# silently break label lookup.
case " ${models[*]} " in
    *" CUSTOM_LABELS "*) ;;
    *) models+=(CUSTOM_LABELS) ;;
esac

load_args=()
for name in "${models[@]}"; do
    load_args+=("--load-model=${name}")
done
echo "[serve.sh] loading ${#models[@]} model(s)"

exec /opt/tritonserver/bin/tritonserver \
    --model-repository="${MODEL_REPOSITORY}" \
    --model-control-mode=explicit \
    "${load_args[@]}" \
    --http-port="${TRITON_HTTP_PORT:-8000}" \
    --grpc-port="${TRITON_GRPC_PORT:-8001}" \
    --metrics-port="${TRITON_METRICS_PORT:-8002}" \
    --log-verbose="${TRITON_LOG_VERBOSE:-0}"
