# wafer-defect-inference-cuda

# 🔬 High-Throughput Semiconductor Wafer Defect Classifier

### C++17 · CUDA · ONNX Runtime · OpenCV

A profiling-driven C++/CUDA inference pipeline for semiconductor wafer defect classification using the WM-811K wafer map dataset.

This project covers the complete engineering path from PyTorch model development to standalone C++ inference and GPU performance optimization:

**P0 → Model Development → P1 → C++ Inference → P2 → GPU Profiling & Optimization**

The primary objective of P2 is not simply to maximize GPU throughput, but to identify the actual system bottleneck through controlled experiments and make workload-specific optimization decisions based on measured evidence.

---

## 📌 Project Overview

The model classifies wafer maps into eight defect categories:

- Center
- Donut
- Edge-Loc
- Edge-Ring
- Loc
- Near-full
- Random
- Scratch

### Model

- **Dataset:** WM-811K Wafer Maps
- **Input:** `24 × 24 × 1`
- **Output:** 8 classes
- **Architecture:** Lightweight CNN
- **Training:** PyTorch
- **Deployment:** ONNX
- **Runtime:** ONNX Runtime
- **GPU Backend:** CUDA Execution Provider

---

# 🚀 Project Evolution

## P0 — PyTorch Model Development

The project began with development and evaluation of a lightweight CNN for wafer defect classification.

### Components

- WM-811K wafer map dataset
- Lightweight CNN
- Class-weighted Cross-Entropy Loss
- Data augmentation
- 8-class classification
- FP32 ONNX export
- FP16 ONNX export

### Exported Models

```text
models/
├── wafer_fault_cnn.onnx
└── wafer_fault_cnn_fp16.onnx
P1 — Standalone C++ Inference Engine

The trained model was deployed outside the Python training environment using C++.

Inference Pipeline
Input Image
    ↓
OpenCV Preprocessing
    ↓
Tensor Construction
    ↓
ONNX Runtime C++ API
    ↓
Inference
    ↓
Classification
Goals
Remove Python runtime dependency
Build a standalone C++ inference executable
Validate ONNX Runtime C++ API integration
Establish a deployable inference pipeline
P2 — GPU Acceleration & Performance Engineering

P2 focuses on profiling and optimizing the inference system.

The optimization process follows a measurement-driven workflow:

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
Make Engineering Decision
P2.1 — CUDA Execution Provider

ONNX Runtime CUDA Execution Provider was integrated into the C++ inference engine.

CUDA execution was verified through runtime execution and NVIDIA Nsight Systems profiling.

P2.2 — Batch Benchmark Baseline

The baseline benchmark measured inference performance across multiple batch sizes:

1 / 8 / 16 / 32 / 64

Both FP32 and FP16 models were evaluated.

The initial benchmark showed that simply switching to FP16 did not guarantee higher throughput, motivating deeper profiling.

P2.3 — GPU I/O Binding

Ort::IoBinding was introduced to reduce unnecessary host/device memory handling overhead.

The optimized benchmark measured:

CPU preprocessing
H2D transfer
GPU inference
End-to-end latency
Batch throughput

This experiment showed that GPU memory transfer was not the primary bottleneck.

P2.4 — Bottleneck Identification

Detailed latency measurements revealed that the original end-to-end pipeline was dominated by CPU-side image loading and preprocessing.

Representative Batch 64 Measurement
Stage	Latency
CPU Preprocessing	10.717 ms
H2D Transfer	~0.037 ms
GPU Inference	0.482 ms
Total E2E	11.236 ms

The GPU completed inference substantially faster than the CPU input pipeline.

This indicated that further GPU optimization alone would have limited impact on the original end-to-end pipeline.

P2.5 — RAM-Cached Bottleneck Isolation

To determine whether the preprocessing bottleneck came from actual tensor preprocessing or from disk image loading and decoding, a controlled RAM-cached experiment was performed.

All 3,828 test images were loaded into RAM before benchmarking.

Original Pipeline
Disk
 ↓
Image Decode
 ↓
CPU Preprocessing
 ↓
H2D
 ↓
GPU Inference
RAM-Cached Pipeline
RAM Cache
 ↓
CPU Preprocessing
 ↓
H2D
 ↓
GPU Inference
Batch 64 Comparison
Metric	Original	RAM Cached
Preprocessing	10.717 ms	0.028 ms
GPU Inference	0.482 ms	0.321 ms
Total E2E	11.236 ms	0.359 ms

RAM caching reduced the measured preprocessing stage by more than two orders of magnitude.

This experimentally demonstrated that disk image loading and decoding dominated the original end-to-end pipeline.

The RAM-cached benchmark intentionally removes disk I/O from the timed path. Its throughput therefore represents an isolated inference benchmark rather than production end-to-end throughput.

P2.6 — GPU-Focused FP32 vs FP16 Evaluation

After isolating disk I/O, FP32 and FP16 were compared under the same RAM-cached conditions.

Final Benchmark
Batch Size	FP32 Pure Infer	FP16 Pure Infer	FP32 FPS	FP16 FPS	FP16 / FP32
1	0.139 ms	0.180 ms	6,944	5,390	0.77×
8	0.165 ms	0.225 ms	44,413	33,539	0.76×
16	0.215 ms	0.273 ms	67,744	54,699	0.81×
32	0.259 ms	0.384 ms	112,033	77,823	0.69×
64	0.349 ms	0.543 ms	161,305	107,882	0.67×

Under the controlled RAM-cached workload, FP16 was consistently slower than FP32.

🔍 FP16 Performance Analysis

The FP16 result was initially counterintuitive because the tested NVIDIA GPU is capable of Tensor Core execution.

NVIDIA Nsight Systems profiling confirmed the presence of Tensor Core-capable FP16 kernel activity, including:

sm75_xmma_fprop_implicit_gemm

Therefore, the result was not simply caused by FP16 failing to use Tensor Core-capable kernels.

Instead, the measured behavior is consistent with the characteristics of the workload:

Very small input size: 24 × 24 × 1
Lightweight CNN architecture
Small individual layer workloads
CUDA kernel launch / runtime dispatch overhead
Type conversion overhead in the FP16 execution path
Limited opportunity to amortize overhead across the small network

The important engineering conclusion is:

Tensor Core capability does not guarantee FP16 end-to-end acceleration for every workload.

For this particular lightweight CNN, FP32 produced higher throughput across all tested batch sizes.

Therefore, FP32 was selected as the preferred inference configuration for the current workload.

📊 Numerical Accuracy

FP16 outputs were compared against the FP32 reference.

Maximum Absolute Error

At Batch 64:

1.172e-02
Model-Level Comparison
Metric	FP32	FP16
Accuracy	85.08%	85.06%
Weighted F1	0.8544	0.8542

The FP16 model maintained similar classification quality while providing no throughput advantage in the tested workload.

📈 Key Findings
1. The Initial Bottleneck Was Not GPU Compute

The original pipeline spent most of its time loading and decoding images.

At Batch 64:

CPU preprocessing: 10.717 ms
GPU inference:       0.482 ms

The GPU was therefore frequently waiting for the input pipeline.

2. RAM Caching Experimentally Isolated the Bottleneck

After removing disk I/O from the timed path:

Preprocessing:
10.717 ms → 0.028 ms

E2E latency:
11.236 ms → 0.359 ms

This provided strong experimental evidence that disk I/O and image decoding dominated the original pipeline.

3. H2D Transfer Was Not the Main Limitation

H2D transfer represented only a small portion of the measured latency.

Therefore, additional host-to-device transfer optimization was not expected to produce a large end-to-end improvement for this workload.

4. FP16 Was Not Faster Despite Tensor Core-Capable Execution

Nsight profiling showed Tensor Core-capable FP16 kernels, but the overall FP16 pipeline remained slower than FP32.

This demonstrates that theoretical hardware capability does not automatically translate into application-level speedup.

5. Optimization Must Be Workload-Specific

The final configuration was selected based on measured behavior:

Current workload:
Lightweight CNN
24 × 24 × 1 input

Preferred configuration:
FP32
🧪 Performance Engineering Summary

The major performance transition was:

Initial End-to-End Pipeline
        ↓
Disk / Image Decode Bottleneck
        ↓
~5K FPS
        ↓
RAM-Cached Bottleneck Isolation
        ↓
GPU-Focused Benchmark
        ↓
161K FPS Isolated FP32 Throughput
        ↓
FP32 / FP16 Controlled Comparison
        ↓
FP32 Selected

The key result is therefore not simply the peak FPS.

The project demonstrates a complete performance-engineering cycle:

Measure
   ↓
Profile
   ↓
Identify
   ↓
Hypothesize
   ↓
Isolate
   ↓
Optimize
   ↓
Validate
   ↓
Decide
📁 Project Structure
wafer-defect-inference-cuda/
│
├── models/
│   ├── wafer_fault_cnn.onnx
│   └── wafer_fault_cnn_fp16.onnx
│
├── src/
│   ├── main.cpp
│   ├── main_fp16.cpp
│   ├── batch_benchmark.cpp
│   ├── batch_benchmark_optimized.cpp
│   └── batch_benchmark_cached.cpp
│
├── test_images/
│   └── 3,828 wafer test images
│
├── results/
│   └── benchmark outputs
│
├── CMakeLists.txt
└── README.md
Source File Roles
File	Purpose
main.cpp	FP32 CUDA inference executable
main_fp16.cpp	FP16 CUDA inference executable
batch_benchmark.cpp	Baseline batch benchmark
batch_benchmark_optimized.cpp	I/O Binding and latency breakdown
batch_benchmark_cached.cpp	RAM-cached controlled benchmark
🖥️ Profiling

NVIDIA Nsight Systems was used to analyze:

CUDA kernel execution
CUDA API activity
Kernel launch behavior
GPU execution timeline
Host/device activity
FP16 Tensor Core-capable kernels

Representative observed kernel:

sm75_xmma_fprop_implicit_gemm

Profiling results were used to guide subsequent experiments rather than relying solely on theoretical GPU specifications.

🛠️ Build
Requirements
C++17
CMake 3.18+
MSVC compatible compiler
CUDA Toolkit 12.x
NVIDIA GPU
OpenCV 4.x
ONNX Runtime GPU package
Configure
cmake -B build
Build
cmake --build build --config Release
▶️ Run
FP32 Inference
build\Release\wafer_inference.exe
FP16 Inference
build\Release\wafer_inference_fp16.exe
Baseline Benchmark
build\Release\batch_benchmark.exe
I/O Binding Benchmark
build\Release\batch_benchmark_optimized.exe
RAM-Cached Benchmark
build\Release\batch_benchmark_cached.exe
🏁 Final Conclusion

This project demonstrates that high-performance ML inference is fundamentally a systems problem.

The investigation showed that:

The original pipeline was dominated by disk image loading and decoding.
Profiling identified the CPU/input pipeline as the primary bottleneck.
RAM caching experimentally isolated and removed that bottleneck.
GPU inference then became the dominant stage.
FP32 and FP16 were compared under controlled GPU-focused conditions.
FP16 remained slower despite Tensor Core-capable execution.
FP32 was selected as the better configuration for the tested lightweight CNN.

The final engineering principle demonstrated by P2 is:

Do not optimize the component that looks theoretically fastest to optimize. Profile the system, identify the actual bottleneck, isolate it experimentally, and optimize based on measured evidence.
