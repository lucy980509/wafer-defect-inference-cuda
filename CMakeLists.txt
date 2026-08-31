cmake_minimum_required(VERSION 3.20)

project(WaferCppInference LANGUAGES CXX)

# =========================================================
# Build Configuration
# =========================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8)
endif()


# =========================================================
# Dependencies
# =========================================================

# OpenCV
find_package(OpenCV REQUIRED)

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# ONNX Runtime
#
# Example:
# cmake -S . -B build ^
#   -DONNXRUNTIME_DIR="C:/onnxruntime/onnxruntime-win-x64-gpu-1.26.0"
#
set(
    ONNXRUNTIME_DIR
    ""
    CACHE PATH
    "Path to ONNX Runtime installation"
)

if(NOT ONNXRUNTIME_DIR)
    message(
        FATAL_ERROR
        "ONNXRUNTIME_DIR is not set. "
        "Configure CMake with -DONNXRUNTIME_DIR=<path-to-onnxruntime>"
    )
endif()

set(
    ONNXRUNTIME_INCLUDE_DIR
    "${ONNXRUNTIME_DIR}/include"
)

set(
    ONNXRUNTIME_LIBRARY
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
)

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}")
    message(
        FATAL_ERROR
        "ONNX Runtime include directory not found: "
        "${ONNXRUNTIME_INCLUDE_DIR}"
    )
endif()

if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
    message(
        FATAL_ERROR
        "ONNX Runtime library not found: "
        "${ONNXRUNTIME_LIBRARY}"
    )
endif()


# =========================================================
# Common Include Directories
# =========================================================

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)


# =========================================================
# Main Executables
# =========================================================

# 1. FP32 Single-Image Inference
add_executable(
    wafer_inference
    src/main.cpp
)

target_link_libraries(
    wafer_inference
    PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)


# 2. FP16 Single-Image Inference
add_executable(
    wafer_inference_fp16
    src/main_fp16.cpp
)

target_link_libraries(
    wafer_inference_fp16
    PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)


# 3. ONNX Shape Inspector
add_executable(
    check_shape
    src/check_shape.cpp
)

target_link_libraries(
    check_shape
    PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)


# =========================================================
# Benchmark Executables
# =========================================================

# 4. Baseline Batch-Scaling Benchmark
add_executable(
    batch_benchmark
    benchmarks/batch_benchmark.cpp
)

target_link_libraries(
    batch_benchmark
    PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)


# 5. Latency Breakdown / I/O Binding Benchmark
add_executable(
    batch_benchmark_optimized
    benchmarks/batch_benchmark_optimized.cpp
)

target_link_libraries(
    batch_benchmark_optimized
    PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
    CUDA::cudart
)


# 6. RAM-Cached Benchmark
add_executable(
    batch_benchmark_cached
    benchmarks/batch_benchmark_cached.cpp
)

target_link_libraries(
    batch_benchmark_cached
    PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
    CUDA::cudart
)


# 7. FP32 vs. FP16 Precision Comparison
add_executable(
    batch_benchmark_precision_compare
    benchmarks/batch_benchmark_precision_compare.cpp
)

target_link_libraries(
    batch_benchmark_precision_compare
    PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
    CUDA::cudart
)


# =========================================================
# Copy Model Files to Build Directory
# =========================================================

set(MODELS_DIR "${CMAKE_SOURCE_DIR}/models")

foreach(
    TARGET_NAME

    wafer_inference
    wafer_inference_fp16
    check_shape
    batch_benchmark
    batch_benchmark_optimized
    batch_benchmark_cached
    batch_benchmark_precision_compare
)

    add_custom_command(
        TARGET ${TARGET_NAME}
        POST_BUILD

        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${MODELS_DIR}"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>/models"
    )

endforeach()
