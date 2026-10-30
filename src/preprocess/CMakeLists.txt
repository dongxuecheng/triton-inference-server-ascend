cmake_minimum_required(VERSION 3.16)
project(triton_yolo_preprocess LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(SOC_VERSION "Ascend310P3" CACHE STRING "Ascend device target")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit path")

if(NOT ASCEND_CANN_PACKAGE_PATH)
    message(FATAL_ERROR "Set ASCEND_HOME_PATH or ASCEND_CANN_PACKAGE_PATH to the CANN toolkit")
endif()

set(ASCENDC_CMAKE_DIR "" CACHE PATH "Directory containing ascendc.cmake (optional override)")
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/AscendToolkit.cmake")
triton_find_ascendc("${ASCEND_CANN_PACKAGE_PATH}")

include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
ascendc_library(yolo_preprocess_kernel SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/yolo_preprocess_kernel.cpp")

# The AIPP-free kernel is declared here rather than from src/rgb_preprocess/
# because ascendc.cmake can only be included once per project: it creates
# fixed-name targets (host_intf_pub, ascendc_runtime_static) and sets
# directory-scoped variables that ascendc_library() reads. Declaring both here
# keeps them in the one scope where the include happened; only the source file
# lives with its backend.
ascendc_library(yolo_rgb_preprocess_kernel SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/../rgb_preprocess/yolo_rgb_preprocess_kernel.cpp")
