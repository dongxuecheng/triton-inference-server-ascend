#!/bin/bash
# Converts a YOLO26 ONNX export into the Ascend `.om` that `yolo26_infer` loads.
#
# Usage:
#   scripts/convert_yolo26.sh <model.onnx> [output_name]
#
# Environment:
#   AIPP_CFG=<path>   static AIPP to insert (default workspace/atc/aipp_nv12_640.cfg),
#                     which makes the model consume the NV12 canvas that
#                     YOLO_640_LETTERBOX_PREPROCESS writes.
#   AIPP_CFG=none     convert with no AIPP at all. The model then reads the ONNX
#                     image input directly, so serve it with
#                     YOLO_640_LETTERBOX_PREPROCESS_RGB, which produces that
#                     tensor. Run `probe_om` on the result to see the signature:
#                     it is FP32 [1, 3, 640, 640] rather than the UINT8 canvas.
#
# The export is normalised before ATC sees it, because CANN 8.2's ATC has two
# limits that YOLO26 exports hit by default:
#
#   * it cannot parse opset 20, so the declared opset is lowered;
#   * it cannot infer shapes through the dynamic-shape subgraph that YOLO26's
#     end-to-end head builds from Shape/Tile/Range/Mod/TopK/GatherElements, so
#     the input dims are fixed and the graph is constant-folded with onnxsim.
#
# After that the model converts as-is. Neither step changes the network, so
# re-exporting with `opset=17` and `dynamic=False` produces the same result and
# lets you skip the normalisation.

set -e

MODEL="${1:?usage: convert_yolo26.sh <model.onnx> [output_name]}"
NAME="${2:-$(basename "${MODEL}" .onnx)}"

SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)
ROOT_DIR=$(dirname "${SCRIPT_DIR}")
DEFAULT_AIPP_CFG="${ROOT_DIR}/workspace/atc/aipp_nv12_640.cfg"
OUTPUT_DIR=$(cd "$(dirname "$(realpath "${MODEL}")")" && pwd)

# `AIPP_CFG=none` converts without AIPP; anything else is a path to insert.
AIPP_CFG="${AIPP_CFG:-${DEFAULT_AIPP_CFG}}"
if [ "${AIPP_CFG}" = "none" ] || [ "${AIPP_CFG}" = "" ]; then
    AIPP_CFG=""
    echo "[convert] converting WITHOUT AIPP (use YOLO_640_LETTERBOX_PREPROCESS_RGB)"
else
    [ -f "${AIPP_CFG}" ] || {
        echo "[convert] ERROR: ${AIPP_CFG} is missing" >&2
        exit 1
    }
    echo "[convert] inserting static AIPP from ${AIPP_CFG}"
fi

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
SOC_VERSION="${SOC_VERSION:-Ascend310P3}"
TRITON_INPUT_SHAPE="${TRITON_INPUT_SHAPE:-images:1,3,640,640}"
INPUT_NAME="${TRITON_INPUT_SHAPE%%:*}"
: "${PRECISION_MODE:=allow_fp32_to_fp16}"

[ -f "${ASCEND_HOME_PATH}/compiler/bin/atc" ] || {
    echo "[convert] ERROR: atc not found under ${ASCEND_HOME_PATH}" >&2
    exit 1
}

# The Ascend C compiler and ATC stage temporary files in TMPDIR.
export TMPDIR="${TMPDIR:-/dev/shm}"
export PATH="${ASCEND_HOME_PATH}/compiler/bin:${ASCEND_HOME_PATH}/bin:${PATH}"

NORMALISED="${OUTPUT_DIR}/${NAME}_atc.onnx"
python3 - "${MODEL}" "${NORMALISED}" "${INPUT_NAME}" "${TRITON_INPUT_SHAPE#*:}" <<'PY'
import sys

import onnx
import onnxsim

source, target, input_name, shape = sys.argv[1:5]
dims = [int(value) for value in shape.split(",")]

model = onnx.load(source)
declared = [(p.domain or "ai.onnx", p.version) for p in model.opset_import]
print("[convert] source opset %s, %d nodes" % (declared, len(model.graph.node)))

needs_shape_fix = any(
    any(d.dim_param for d in i.type.tensor_type.shape.dim)
    for i in model.graph.input
)
if needs_shape_fix:
    print("[convert] fixing dynamic input dims with onnxsim")
    simplified, ok = onnxsim.simplify(
        source, overwrite_input_shapes={input_name: dims})
    assert ok, "onnxsim could not simplify the model"
    model = simplified

# ATC's parser is versioned per opset and knows nothing above 17.
for entry in model.opset_import:
    if entry.version > 17:
        print("[convert] lowering declared opset %d -> 17" % entry.version)
        entry.version = 17

onnx.save(model, target)
print("[convert] wrote %s" % target)
PY

echo "[convert] running atc -> ${OUTPUT_DIR}/${NAME}.om"
atc_args=(
    --model="${NORMALISED}"
    --framework=5
    --output="${OUTPUT_DIR}/${NAME}"
    --input_shape="${TRITON_INPUT_SHAPE}"
    --soc_version="${SOC_VERSION}"
    --precision_mode="${PRECISION_MODE}"
)
if [ -n "${AIPP_CFG}" ]; then
    atc_args+=(--insert_op_conf="${AIPP_CFG}")
fi
atc "${atc_args[@]}"
rm -f "${NORMALISED}"
echo "[convert] done: ${OUTPUT_DIR}/${NAME}.om"
if [ -n "${AIPP_CFG}" ]; then
    echo "[convert] next: point YOLO26_DET_PRE's model_path at it and list the"
    echo "[convert]       YOLO26 models in workspace/models_to_load.txt"
else
    echo "[convert] next: point the model's model_path at it, declare an FP32"
    echo "[convert]       RGB_CHW input of dims [1, 3, 640, 640], and feed it from"
    echo "[convert]       YOLO_640_LETTERBOX_PREPROCESS_RGB"
fi
