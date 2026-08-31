cmake_minimum_required(VERSION 3.20)

project(WaferCppInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# OpenCV
find_package(OpenCV REQUIRED)

# CUDA
find_package(CUDAToolkit REQUIRED)

# ONNX Runtime path must be provided by user
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime installation")

if(NOT ONNXRUNTIME_DIR)
    message(FATAL_ERROR "Please set ONNXRUNTIME_DIR to your ONNX Runtime installation path.")
endif()

include_directories(
    ${ONNXRUNTIME_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# FP32 inference
add_executable(
    wafer_inference
    src/main.cpp
)

target_link_libraries(
    wafer_inference
    PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
)

# FP16 inference
add_executable(
    wafer_inference_fp16
    src/main_fp16.cpp
)

target_link_libraries(
    wafer_inference_fp16
    PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
)

# Shape inspector
add_executable(
    check_shape
    src/check_shape.cpp
)

target_link_libraries(
    check_shape
    PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
)

# Baseline benchmark
add_executable(
    batch_benchmark
    benchmarks/batch_benchmark.cpp
)

target_link_libraries(
    batch_benchmark
    PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
)

# Breakdown benchmark
add_executable(
    batch_benchmark_optimized
    benchmarks/batch_benchmark_optimized.cpp
)

target_link_libraries(
    batch_benchmark_optimized
    PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
    CUDA::cudart
)

# RAM-cached benchmark
add_executable(
    batch_benchmark_cached
    benchmarks/batch_benchmark_cached.cpp
)

target_link_libraries(
    batch_benchmark_cached
    PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
    CUDA::cudart
)

# FP32 vs FP16 benchmark
add_executable(
    batch_benchmark_precision_compare
    benchmarks/batch_benchmark_precision_compare.cpp
)

target_link_libraries(
    batch_benchmark_precision_compare
    PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
    CUDA::cudart
)
