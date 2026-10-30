# Root CMakeLists for the Triton Ascend YOLO backends.
#
# Builds the custom backend shared libraries that make up the YOLO11 pipeline:
#   libtriton_preprocess.so         Ascend C letterbox preprocessing (AIPP)
#   libtriton_rgb_preprocess.so     Ascend C letterbox preprocessing (no AIPP)
#   libtriton_yoloinfer.so          ACL .om inference
#   libtriton_yolo11_postprocess.so host CPU decode + NMS
#
# plus libtriton_ascend_canvas_pool.so, the device canvas pool the preprocess
# and inference backends share.
#
# Recommended build environment: the CANN toolkit plus a C++17 host compiler.
# Use ./build.sh, which also copies each library into the model's `1/` folder.

cmake_minimum_required(VERSION 3.16)
project(triton_ascend_yolo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit path")
set(TRITON_CORE_INCLUDE_DIR "" CACHE PATH "Triton core/include directory")
set(SOC_VERSION "Ascend310P3" CACHE STRING "Ascend device target")

if(NOT ASCEND_CANN_PACKAGE_PATH)
    message(FATAL_ERROR "Set ASCEND_HOME_PATH or ASCEND_CANN_PACKAGE_PATH to the CANN toolkit")
endif()
if(NOT EXISTS "${TRITON_CORE_INCLUDE_DIR}/triton/core/tritonbackend.h")
    message(FATAL_ERROR "Set TRITON_CORE_INCLUDE_DIR to Triton core/include")
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/AscendToolkit.cmake")
triton_find_acl("${ASCEND_CANN_PACKAGE_PATH}")

# Headers are included as "common/..." and "<backend>/...", so `src` is the
# include root.
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TRITON_CORE_INCLUDE_DIR}
    ${ACL_INCLUDE_DIR}
)

# Shared device canvas pool: lets the preprocess backend hand the NV12 canvas to
# the inference backend as a device address instead of copying it to the host
# and back. It must stay one shared object so both backends share a single pool;
# the SONAME guarantees that even when a copy sits in each model directory.
add_library(triton_ascend_canvas_pool SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/ascend_canvas_pool.cc)
set_target_properties(triton_ascend_canvas_pool PROPERTIES
    OUTPUT_NAME "triton_ascend_canvas_pool"
    BUILD_RPATH "${ASCEND_RUNTIME_LIBRARY_DIR}")
target_link_libraries(triton_ascend_canvas_pool PRIVATE "${ASCENDCL_LIBRARY}")

# Ascend C kernel library used by the preprocess backend only.
# It also declares the AIPP-free kernel library; see the note in that file.
add_subdirectory(src/preprocess)

# -------------------- Preprocess backend --------------------
set(PREPROCESS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/triton_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/triton_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/model_config.cpp
)

add_library(triton_preprocess SHARED ${PREPROCESS_SOURCES})
set_target_properties(triton_preprocess PROPERTIES
    OUTPUT_NAME "triton_preprocess"
    BUILD_RPATH "${ASCEND_RUNTIME_LIBRARY_DIR}"
    INSTALL_RPATH "${ASCEND_RUNTIME_LIBRARY_DIR};$ORIGIN")
target_link_libraries(triton_preprocess PRIVATE
    yolo_preprocess_kernel
    triton_ascend_canvas_pool
    "${ASCENDCL_LIBRARY}"
    dl
    pthread)

# -------------------- AIPP-free preprocess backend --------------------
# Sibling of `triton_preprocess` for models converted without AIPP. Those have no
# AIPP to convert and normalise the input, so they read the ONNX image tensor
# itself and this backend produces it: FP32, RGB, planar, scaled to [0, 1].
# It links no canvas pool, because that tensor goes over in host memory.
set(RGB_PREPROCESS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rgb_preprocess/triton_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rgb_preprocess/triton_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/model_config.cpp
)

add_library(triton_rgb_preprocess SHARED ${RGB_PREPROCESS_SOURCES})
set_target_properties(triton_rgb_preprocess PROPERTIES
    OUTPUT_NAME "triton_rgb_preprocess"
    BUILD_RPATH "${ASCEND_RUNTIME_LIBRARY_DIR}"
    INSTALL_RPATH "${ASCEND_RUNTIME_LIBRARY_DIR};$ORIGIN")
target_link_libraries(triton_rgb_preprocess PRIVATE
    yolo_rgb_preprocess_kernel
    "${ASCENDCL_LIBRARY}"
    dl
    pthread)

# -------------------- Inference backend --------------------
set(YOLOINFER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yoloinfer/triton_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yoloinfer/triton_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/model_config.cpp
)

add_library(triton_yoloinfer SHARED ${YOLOINFER_SOURCES})
set_target_properties(triton_yoloinfer PROPERTIES
    OUTPUT_NAME "triton_yoloinfer"
    BUILD_RPATH "${ASCEND_RUNTIME_LIBRARY_DIR}"
    INSTALL_RPATH "${ASCEND_RUNTIME_LIBRARY_DIR};$ORIGIN")
target_link_libraries(triton_yoloinfer PRIVATE
    triton_ascend_canvas_pool
    "${ASCENDCL_LIBRARY}"
    dl
    pthread)

# -------------------- YOLO26 inference backend --------------------
# Same ACL machinery as `yoloinfer`, but the model is an end-to-end detector
# whose output rows are finished detections rather than an attribute-major
# tensor.
set(YOLO26_INFER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo26_infer/triton_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo26_infer/triton_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/ascend_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/model_config.cpp
)

add_library(triton_yolo26_infer SHARED ${YOLO26_INFER_SOURCES})
set_target_properties(triton_yolo26_infer PROPERTIES
    OUTPUT_NAME "triton_yolo26_infer"
    BUILD_RPATH "${ASCEND_RUNTIME_LIBRARY_DIR}"
    INSTALL_RPATH "${ASCEND_RUNTIME_LIBRARY_DIR};$ORIGIN")
target_link_libraries(triton_yolo26_infer PRIVATE
    triton_ascend_canvas_pool
    "${ASCENDCL_LIBRARY}"
    dl
    pthread)

# -------------------- YOLO11 detection postprocess backend --------------------
# Decoding and NMS are host CPU work, so this backend links neither ACL nor the
# canvas pool.
set(YOLO11_POSTPROCESS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo11_postprocess/triton_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo11_postprocess/triton_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo11_postprocess/yolo11_postprocess_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/model_config.cpp
)

add_library(triton_yolo11_postprocess SHARED ${YOLO11_POSTPROCESS_SOURCES})
set_target_properties(triton_yolo11_postprocess PROPERTIES
    OUTPUT_NAME "triton_yolo11_postprocess"
    INSTALL_RPATH "$ORIGIN")

# -------------------- YOLO26 detection postprocess backend --------------------
# YOLO26 is end-to-end, so this one only thresholds and re-maps coordinates:
# no anchor decoding and no NMS. Still pure host CPU work.
set(YOLO26_POSTPROCESS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo26_postprocess/triton_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo26_postprocess/triton_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo26_postprocess/yolo26_postprocess_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/model_config.cpp
)

add_library(triton_yolo26_postprocess SHARED ${YOLO26_POSTPROCESS_SOURCES})
set_target_properties(triton_yolo26_postprocess PROPERTIES
    OUTPUT_NAME "triton_yolo26_postprocess"
    INSTALL_RPATH "$ORIGIN")

install(TARGETS
    triton_ascend_canvas_pool
    triton_preprocess
    triton_rgb_preprocess
    triton_yoloinfer
    triton_yolo26_infer
    triton_yolo11_postprocess
    triton_yolo26_postprocess
    yolo_preprocess_kernel
    yolo_rgb_preprocess_kernel
    LIBRARY DESTINATION lib)
