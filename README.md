# High-Throughput Semiconductor Wafer Defect Classifier

### C++17 · CUDA · ONNX Runtime · OpenCV

A profiling-driven C++/CUDA inference pipeline for semiconductor wafer defect classification using the WM-811K wafer map dataset.

This project covers the complete engineering path from PyTorch model development to standalone C++ inference and GPU performance optimization:

**P0 → Model Development → P1 → C++ Inference → P2 → GPU Profiling & Optimization**

The primary objective of P2 is not simply to maximize GPU throughput, but to identify the actual system bottleneck through controlled experiments and make workload-specific optimization decisions based on measured evidence.

---

## Project Overview

The model classifies wafer maps into eight defect categories:

- Center
- Donut
- Edge-Loc
- Edge-Ring
- Loc
- Near-full
- Random
- Scratch

### Model Specifications

- **Dataset:** WM-811K Wafer Maps
- **Input:** `24 × 24 × 1`
- **Output:** 8 classes
- **Architecture:** Lightweight CNN
- **Training:** PyTorch
- **Deployment:** ONNX
- **Runtime:** ONNX Runtime
- **GPU Backend:** CUDA Execution Provider

---

# Project Evolution

## P0 — PyTorch Model Development

The project began with the development and evaluation of a lightweight CNN for wafer defect classification.

### Key Components

- WM-811K wafer map dataset processing
- Lightweight CNN design
- Class-weighted Cross-Entropy Loss to handle severe class imbalance
- Data augmentation
- 8-class classification
- FP32 ONNX export
- FP16 ONNX export

### Exported Models

```text
models/
├── wafer_fault_cnn.onnx
└── wafer_fault_cnn_fp16.onnx
```

# P1 — Standalone C++ Inference Engine

The trained model was deployed outside the Python training environment using C++ to eliminate runtime dependencies.

## Inference Pipeline

```text
Input Image
    ↓
OpenCV Preprocessing
    ↓
Tensor Construction
    ↓
ONNX Runtime C++ API
    ↓
Inference Execution
    ↓
Class Output
```

## Key Goals

- Remove Python runtime dependency
- Build a standalone C++ inference executable
- Validate ONNX Runtime C++ API integration
- Establish a production-ready deployable inference pipeline

---

# P2 — GPU Acceleration & Performance Engineering

P2 focuses on profiling and optimizing the inference system.

The optimization process follows a measurement-driven workflow:

```text
Baseline
   ↓
Profile
   ↓
Identify Bottleneck
   ↓
Form Hypothesis
   ↓
Controlled Experiment
   ↓
Optimize
   ↓
Re-measure
   ↓
Engineering Decision
```

## P2.1 — CUDA Execution Provider Integration

ONNX Runtime CUDA Execution Provider was integrated into the C++ inference engine.

CUDA execution was verified through runtime execution logs and NVIDIA Nsight Systems profiling.

---

## P2.2 — Batch Benchmark Baseline

The baseline benchmark measured inference performance across multiple batch sizes:

**1 / 8 / 16 / 32 / 64**

Both FP32 and FP16 models were evaluated.

The initial benchmark showed that simply switching to FP16 did not guarantee higher throughput, motivating deeper profiling.

---

## P2.3 — GPU I/O Binding

`Ort::IoBinding` was introduced to eliminate host/device memory reallocation overhead.

The optimized benchmark measured:

- CPU preprocessing latency
- Host-to-Device (H2D) transfer latency
- Pure GPU inference latency
- Total End-to-End (E2E) latency
- Batch throughput

This experiment proved that GPU memory transfer was not the primary system bottleneck.

---

## P2.4 — Bottleneck Identification

Detailed latency breakdown revealed that the original end-to-end pipeline was heavily dominated by CPU-side image loading and decoding.

### Representative Batch 64 Measurement

| Stage | Latency | Share |
|---|---:|---:|
| CPU Preprocessing (Disk Read + OpenCV Decode) | 10.717 ms | ~95.4% |
| H2D Transfer | ~0.037 ms | ~0.3% |
| GPU Inference | 0.482 ms | ~4.3% |
| **Total E2E** | **11.236 ms** | **100.0%** |

The GPU completed inference substantially faster than the CPU input pipeline could feed tensors, indicating that further GPU-side optimization alone would yield negligible end-to-end impact.

---

## P2.5 — RAM-Cached Bottleneck Isolation

To determine whether the preprocessing bottleneck originated from tensor array operations or disk I/O, a controlled RAM-cached experiment was performed.

All **3,828 test images** were loaded into system memory before benchmarking.

### Pipeline Comparison

#### Original Pipeline

```text
Disk
 ↓
Image Decode
 ↓
CPU Preprocessing
 ↓
H2D
 ↓
GPU Inference
```

#### [RAM-Cached Pipeline] 
```text
RAM Cache 
  ↓
CPU Preprocessing
  ↓
H2D
  ↓
GPU Inference
```

### Batch 64 Comparison

| Metric | Original Pipeline | RAM-Cached Pipeline | Improvement |
|---|---:|---:|---:|
| Preprocessing Latency | 10.717 ms | 0.028 ms | ~382× faster |
| GPU Inference Latency | 0.482 ms | 0.321 ms | ~1.5× faster |
| **Total E2E Latency** | **11.236 ms** | **0.359 ms** | **~31× faster** |

RAM caching reduced the measured preprocessing stage by over two orders of magnitude, experimentally demonstrating that disk image loading and decoding dominated the original pipeline.

> **Note:** The RAM-cached benchmark intentionally isolates inference performance from disk I/O. Its throughput represents isolated hardware inference capacity rather than end-to-end file-reading throughput.

---

## P2.6 — GPU-Focused FP32 vs. FP16 Evaluation

With disk I/O isolated, FP32 and FP16 precision modes were evaluated under identical RAM-cached conditions.

### Final Benchmark Matrix

| Batch Size | FP32 Pure Infer | FP16 Pure Infer | FP32 Throughput | FP16 Throughput | FP16 / FP32 Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.139 ms | 0.180 ms | 6,944 FPS | 5,390 FPS | 0.77× |
| 8 | 0.165 ms | 0.225 ms | 44,413 FPS | 33,539 FPS | 0.76× |
| 16 | 0.215 ms | 0.273 ms | 67,744 FPS | 54,699 FPS | 0.81× |
| 32 | 0.259 ms | 0.384 ms | 112,033 FPS | 77,823 FPS | 0.69× |
| 64 | **0.349 ms** | **0.543 ms** | **161,305 FPS** | **107,882 FPS** | **0.67×** |

Under controlled execution, FP16 was consistently **0.67×–0.81×** slower than FP32.

---

# Deep Dive: FP16 Performance Inversion Analysis

The FP16 throughput drop was initially counterintuitive since the target GPU supports Tensor Core execution.

NVIDIA Nsight Systems profiling confirmed active Tensor Core FP16 kernels during execution:

```text
sm75_xmma_fprop_implicit_gemm
```

This verified that the performance drop was not caused by failure to execute Tensor Core-capable kernels.

Instead, the behavior aligns with specific workload characteristics:

### Small Spatial Dimension

**Input:** `24 × 24 × 1`

The compute volume per layer is relatively small, limiting the opportunity to fully saturate Tensor Core matrix-math units.

### Type Conversion Overhead

The ONNX Runtime CUDA EP execution path introduces internal FP32 ↔ FP16 precision conversion and runtime dispatch overhead.

For a lightweight CNN, these additional operations can become significant relative to the actual mathematical workload.

### Launch Latency Dominance

For lightweight networks, CUDA kernel launch and runtime dispatch overhead can represent a significant portion of total execution time, potentially outweighing the mathematical compute advantage of FP16.

---

## Engineering Takeaway

> **Tensor Core capability does not guarantee end-to-end speedups for all workloads.**

For this lightweight CNN with a small computational workload, FP32 remains the optimal configuration based on the measured benchmark results.

# Numerical Precision & Accuracy

FP16 output logits were evaluated against the FP32 reference to verify numerical stability.

### Maximum Absolute Logit Error

**Batch 64: `1.172e-02`**

### Mean Absolute Logit Error

**~`1.89e-03`**

### Model Performance Metrics

| Metric | FP32 Model | FP16 Model | Absolute Delta |
|---|---:|---:|---:|
| Accuracy | 85.08% | 85.06% | -0.02% |
| Weighted F1-Score | 0.8544 | 0.8542 | -0.0002 |

The FP16 model maintained virtually identical classification quality while providing no throughput benefit for this specific network size.

# Key Findings

- **Initial Bottleneck Was Input I/O:** Disk loading and decoding accounted for ~95.4% of total E2E latency at Batch 64.
- **Experimental Isolation Confirmed I/O Impact:** RAM caching reduced E2E latency from 11.236 ms to 0.359 ms.
- **H2D Transfers Were Negligible:** Host-to-Device transfer accounted for < 0.5% of E2E latency, making further memory transfer tuning low priority.
- **FP16 Inversion Verified via Profiling:** Profiling confirmed active Tensor Core kernels, demonstrating that runtime overhead and workload characteristics contributed to the observed FP16 performance degradation.
- **Data-Driven Configuration Selection:** FP32 was selected as the optimal production engine configuration based on empirical measurements.

# Performance Engineering Summary
```text
Initial E2E Pipeline (~5K FPS)
       │
       ▼ (Identify I/O Bottleneck)
RAM-Cached Isolation
       │
       ▼ (Measure GPU Execution)
Peak Isolated Throughput (161K FPS @ Batch 64 FP32)
       │
       ▼ (Controlled Precision Comparison)
FP32 Selected Over FP16 (1.49× higher throughput at Batch 64)
```

# Project Structure
```text
wafer-defect-inference-cuda/
│
├── models/
│   ├── wafer_fault_cnn.onnx        # FP32 ONNX Model
│   └── wafer_fault_cnn_fp16.onnx   # FP16 ONNX Model
│
├── src/
│   ├── main.cpp                    # Single-image FP32 CUDA executable
│   ├── main_fp16.cpp               # Single-image FP16 CUDA executable
│   └── check_shape.cpp             # Input shape, data type & memory layout validation
│
├── benchmarks/
│   ├── batch_benchmark.cpp         # Baseline batch benchmark
│   ├── batch_benchmark_optimized.cpp # Ort::IoBinding latency breakdown
│   ├── batch_benchmark_cached.cpp  # RAM-cached isolated benchmark
│   └── batch_benchmark_precision_compare.cpp # Final FP32 vs. FP16 comparison
│
├── results/                        # Benchmark execution logs
├── CMakeLists.txt                  # Build configuration
└── README.md                       # Documentation
```

## 📁 Source File Roles

| File | Primary Role |
|---|---|
| `main.cpp` | Standalone FP32 CUDA inference runner |
| `main_fp16.cpp` | Standalone FP16 CUDA inference runner |
| `check_shape.cpp` | Input tensor shape, data type, and memory layout validation utility |
| `batch_benchmark.cpp` | Baseline batch benchmark |
| `batch_benchmark_optimized.cpp` | Latency breakdown profiler with `Ort::IoBinding` |
| `batch_benchmark_cached.cpp` | RAM-cached benchmark for pure GPU throughput isolation |
| `batch_benchmark_precision_compare.cpp` | Final FP32 vs. FP16 precision and performance comparison |
---

## Profiling & System Metrics

NVIDIA Nsight Systems was used to profile CUDA API usage and GPU timeline events, including:

- CUDA kernel launch latency and runtime dispatch overhead
- Host-to-device memory transfer behavior
- CPU-GPU synchronization
- FP16 Tensor Core kernel execution
- `sm75_xmma_fprop_implicit_gemm`

Profiling traces were used to drive optimization hypotheses rather than relying solely on theoretical hardware performance.

---

## Build & Execution

### Requirements

- C++17-compatible compiler
- CMake >= 3.20
- OpenCV
- ONNX Runtime with CUDA Execution Provider
- NVIDIA CUDA Toolkit
- NVIDIA cuDNN
- CUDA-capable NVIDIA GPU

### Tested Environment

| Component | Version / Configuration |
|---|---|
| OS | Windows |
| GPU | NVIDIA GeForce GTX 1660 (6 GB VRAM) |
| NVIDIA Driver | 561.17 |
| CUDA Toolkit | 12.6 |
| cuDNN | 9.8 |
| ONNX Runtime GPU | 1.26.0 |
| OpenCV | 4.14.0 |
| C++ Standard | C++17 |
| CMake | 4.4.2 |

> The versions above correspond to the environment used to build and benchmark this project. Other compatible versions may also work.

### Build Instructions

Clone the repository:

```bash
git clone https://github.com/lucy980509/wafer-defect-inference-cuda.git
cd wafer-defect-inference-cuda
```

Configure the project with CMake and provide the path to your ONNX Runtime installation:

```bash
cmake -S . -B build -DONNXRUNTIME_DIR="C:/path/to/onnxruntime"
```

Build the project in Release mode:

```bash
cmake --build build --config Release
```

The generated executables will be located in the build output directory.

### Running

After building the project, the executables can be run from the Release output directory.

#### Single-Image Inference

```bash
build/Release/wafer_inference.exe
build/Release/wafer_inference_fp16.exe
```

#### Benchmarking

```bash
build/Release/batch_benchmark.exe
build/Release/batch_benchmark_optimized.exe
build/Release/batch_benchmark_cached.exe
build/Release/batch_benchmark_precision_compare.exe
```

The benchmark executables correspond to different stages of the performance analysis, from baseline batch measurements to RAM-cached bottleneck isolation and the final FP32 vs. FP16 comparison.

---

# Conclusion

High-performance machine learning deployment is fundamentally a systems engineering problem.

This project demonstrated that:

- **System bottlenecks must be identified through empirical profiling rather than assumptions.**
- **Experimental isolation**, such as RAM caching, is necessary to accurately measure individual pipeline components.
- **Theoretical hardware capabilities**, such as Tensor Cores, do not automatically guarantee end-to-end application speedups without considering workload size and runtime overheads.

The final P2 results demonstrate a complete measurement-driven optimization cycle:

```text
Baseline
   ↓
Profile
   ↓
Identify Bottleneck
   ↓
Controlled Experiment
   ↓
Isolate System Components
   ↓
Measure GPU Performance
   ↓
Evaluate FP32 vs. FP16
   ↓
Select Optimal Configuration
```

Rather than optimizing solely for theoretical GPU capability, the project used profiling and controlled experiments to determine which optimization strategies actually benefited the target workload.

For this lightweight wafer defect classifier, the final engineering decision was to retain FP32 inference, which achieved higher throughput than FP16 while maintaining the same classification quality.
