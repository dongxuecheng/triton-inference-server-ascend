# 运行库不从系统目录或 devlib/stub 查找，避免混用另一套 CANN。
function(triton_find_acl root)
    find_path(ACL_INCLUDE_DIR acl/acl.h
        HINTS "${root}/include" "${root}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
        NO_DEFAULT_PATH REQUIRED)
    find_library(ASCENDCL_LIBRARY ascendcl
        HINTS "${root}/lib64" "${root}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
              "${root}/runtime/lib64" "${root}/fwkacllib/lib64"
        NO_DEFAULT_PATH REQUIRED)
    get_filename_component(runtime_dir "${ASCENDCL_LIBRARY}" DIRECTORY)
    set(ACL_INCLUDE_DIR "${ACL_INCLUDE_DIR}" PARENT_SCOPE)
    set(ASCENDCL_LIBRARY "${ASCENDCL_LIBRARY}" PARENT_SCOPE)
    set(ASCEND_RUNTIME_LIBRARY_DIR "${runtime_dir}" PARENT_SCOPE)
    message(STATUS "ACL include: ${ACL_INCLUDE_DIR}")
    message(STATUS "ACL library: ${ASCENDCL_LIBRARY}")
endfunction()

function(triton_find_ascendc root)
    if(ASCENDC_CMAKE_DIR)
        if(NOT EXISTS "${ASCENDC_CMAKE_DIR}/ascendc.cmake")
            message(FATAL_ERROR "Invalid ASCENDC_CMAKE_DIR: ${ASCENDC_CMAKE_DIR}")
        endif()
        message(STATUS "Found AscendC CMake: ${ASCENDC_CMAKE_DIR}/ascendc.cmake")
        return()
    endif()
    foreach(relative_dir tools/tikcpp/ascendc_kernel_cmake compiler/tikcpp/ascendc_kernel_cmake
            "${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/ascendc_kernel_cmake")
        if(EXISTS "${root}/${relative_dir}/ascendc.cmake")
            set(ASCENDC_CMAKE_DIR "${root}/${relative_dir}" PARENT_SCOPE)
            message(STATUS "Found AscendC CMake: ${root}/${relative_dir}/ascendc.cmake")
            return()
        endif()
    endforeach()
    # CANN 9 子包布局兜底；有多个版本时要求显式选择，不能任取一个。
    file(GLOB_RECURSE candidates "${root}/*ascendc.cmake")
    list(LENGTH candidates count)
    if(NOT count EQUAL 1)
        message(FATAL_ERROR "Cannot uniquely locate ascendc.cmake under ${root}; set ASCENDC_CMAKE_DIR")
    endif()
    list(GET candidates 0 cmake_file)
    get_filename_component(cmake_dir "${cmake_file}" DIRECTORY)
    set(ASCENDC_CMAKE_DIR "${cmake_dir}" PARENT_SCOPE)
    message(STATUS "Found AscendC CMake: ${cmake_file}")
endfunction()
