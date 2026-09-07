cmake_minimum_required(VERSION 3.22.1)

project(fallingball_native LANGUAGES CXX)

# QNN SDK headers are supplied locally under their upstream SDK terms.
# QNN_SDK_ROOT points to the SDK directory containing include/QNN.
if(DEFINED ENV{QNN_SDK_ROOT} AND NOT "$ENV{QNN_SDK_ROOT}" STREQUAL "")
    set(QNN_INCLUDE_DIR "$ENV{QNN_SDK_ROOT}/include")
else()
    set(QNN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/qnn/include")
endif()
if(NOT EXISTS "${QNN_INCLUDE_DIR}/QNN/QnnInterface.h")
    message(FATAL_ERROR
        "QNN SDK headers are missing. Set QNN_SDK_ROOT to your QAIRT/QNN SDK "
        "directory (containing include/QNN), or place its headers in "
        "app/src/main/cpp/third_party/qnn/include. See README.md.")
endif()

add_library(
    fallingball_native
    SHARED
    ncnn_tinyheatmap.cpp
    yuv_tensor_preprocessor.cpp
    qnn_parent_executor.cpp
    cpu_affinity.cpp
)

add_library(ncnn SHARED IMPORTED)
set_target_properties(
    ncnn
    PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/generated/nativeRuntime/jniLibs/${ANDROID_ABI}/libncnn.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/include"
)

find_library(android-lib android)
find_library(dl-lib dl)
find_library(log-lib log)

target_include_directories(
    fallingball_native
    PRIVATE
    "${QNN_INCLUDE_DIR}"
    # QNN/System headers include their siblings by bare name ("QnnDevice.h"),
    # so the QNN directory itself must also be on the search path.
    "${QNN_INCLUDE_DIR}/QNN"
)

target_compile_features(fallingball_native PRIVATE cxx_std_17)
target_compile_options(
    fallingball_native
    PRIVATE
    -O3
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(
    fallingball_native
    PRIVATE
    ncnn
    ${android-lib}
    ${dl-lib}
    ${log-lib}
)
