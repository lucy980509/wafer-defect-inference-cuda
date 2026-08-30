# wafer-defect-inference-cuda

# 🔬 High-Throughput Semiconductor Wafer Defect Classifier

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
Key GoalsRemove Python runtime dependencyBuild a standalone C++ inference executableValidate ONNX Runtime C++ API integrationEstablish a production-ready deployable inference pipelineP2 — GPU Acceleration & Performance EngineeringP2 focuses on profiling and optimizing the inference system. The optimization process follows a measurement-driven workflow:PlaintextBaseline → Profile → Identify Bottleneck → Form Hypothesis → Controlled Experiment → Optimize → Re-measure → Engineering Decision
P2.1 — CUDA Execution Provider IntegrationONNX Runtime CUDA Execution Provider was integrated into the C++ inference engine. CUDA execution was verified through runtime execution logs and NVIDIA Nsight Systems profiling.P2.2 — Batch Benchmark BaselineThe baseline benchmark measured inference performance across multiple batch sizes (1 / 8 / 16 / 32 / 64). Both FP32 and FP16 models were evaluated.The initial benchmark showed that simply switching to FP16 did not guarantee higher throughput, motivating deeper profiling.P2.3 — GPU I/O BindingOrt::IoBinding was introduced to eliminate host/device memory reallocation overhead. The optimized benchmark measured:CPU preprocessing latencyHost-to-Device (H2D) transfer latencyPure GPU inference latencyTotal End-to-End (E2E) latencyBatch throughputThis experiment proved that GPU memory transfer was not the primary system bottleneck.P2.4 — Bottleneck IdentificationDetailed latency breakdown revealed that the original end-to-end pipeline was heavily dominated by CPU-side image loading and decoding.Representative Batch 64 MeasurementStageLatencyShareCPU Preprocessing (Disk Read + OpenCV Decode)10.717 ms~95.4%H2D Transfer~0.037 ms~0.3%GPU Inference0.482 ms~4.3%Total E2E11.236 ms100.0%The GPU completed inference substantially faster than the CPU input pipeline could feed tensors, indicating that further GPU-side optimization alone would yield negligible end-to-end impact.P2.5 — RAM-Cached Bottleneck IsolationTo determine whether the preprocessing bottleneck originated from tensor array operations or disk I/O, a controlled RAM-cached experiment was performed. All 3,828 test images were loaded into system memory before benchmarking.Plaintext[Original Pipeline]   Disk → Image Decode → CPU Preprocessing → H2D → GPU Inference
[RAM-Cached Pipeline] RAM Cache -----------> CPU Preprocessing → H2D → GPU Inference
Batch 64 ComparisonMetricOriginal PipelineRAM-Cached PipelineImprovementPreprocessing Latency10.717 ms0.028 ms~382× fasterGPU Inference Latency0.482 ms0.321 ms~1.5× fasterTotal E2E Latency11.236 ms0.359 ms~31× fasterRAM caching reduced the measured preprocessing stage by over two orders of magnitude, experimentally demonstrating that disk image loading and decoding dominated the original pipeline.Note: The RAM-cached benchmark intentionally isolates inference performance from disk I/O. Its throughput represents isolated hardware inference capacity rather than end-to-end file-reading throughput.P2.6 — GPU-Focused FP32 vs. FP16 EvaluationWith disk I/O isolated, FP32 and FP16 precision modes were evaluated under identical RAM-cached conditions.Final Benchmark MatrixBatch SizeFP32 Pure InferFP16 Pure InferFP32 ThroughputFP16 ThroughputFP16 / FP32 Speedup10.139 ms0.180 ms6,944 FPS5,390 FPS0.77×80.165 ms0.225 ms44,413 FPS33,539 FPS0.76×160.215 ms0.273 ms67,744 FPS54,699 FPS0.81×320.259 ms0.384 ms112,033 FPS77,823 FPS0.69×640.349 ms0.543 ms161,305 FPS107,882 FPS0.67×Under controlled execution, FP16 was consistently 0.67× ~ 0.81× slower than FP32.🔍 Deep Dive: FP16 Performance Inversion AnalysisThe FP16 throughput drop was initially counterintuitive since the target GPU supports Tensor Core execution.NVIDIA Nsight Systems profiling confirmed active Tensor Core FP16 kernels during execution:Plaintextsm75_xmma_fprop_implicit_gemm
This verified that the performance drop was not caused by failure to execute Tensor Core-capable kernels. Instead, the behavior aligns with specific workload characteristics:Small Spatial Dimension (24 × 24 × 1): The compute volume per layer is too small to saturate Tensor Core matrix math units.Type Conversion Overhead: ONNX Runtime CUDA EP introduces internal FP32 ↔ FP16 precision casting and kernel dispatch overheads.Launch Latency Dominance: For lightweight networks, kernel launch overhead constitutes a significant portion of total execution time, outweighing mathematical compute speedups.Engineering TakeawayTensor Core capability does not guarantee end-to-end speedups for all workloads. For lightweight CNNs with low memory bandwidth requirements, FP32 remains the optimal configuration.📊 Numerical Precision & AccuracyFP16 output logits were evaluated against the FP32 reference to verify numerical stability.Max Absolute Logit Error (Batch 64): 1.172e-02Mean Absolute Logit Error: ~1.89e-03Model Performance MetricsMetricFP32 ModelFP16 ModelAbsolute DeltaAccuracy85.08%85.06%-0.02%Weighted F1-Score0.85440.8542-0.0002The FP16 quantized model maintained virtually identical classification quality while providing no throughput benefit for this specific network size.📈 Key FindingsInitial Bottleneck Was Input I/O, Not GPU Compute: Disk loading and decoding accounted for ~95.4% of total E2E latency at Batch 64.Experimental Isolation Confirmed I/O Impact: RAM caching reduced E2E latency from 11.236 ms to 0.359 ms.H2D Transfers Were Negligible: Host-to-Device transfer accounted for < 0.5% of E2E latency, making further memory transfer tuning low priority.FP16 Inversion Verified via Profiling: Profiling confirmed active Tensor Core kernels, proving that overhead (dispatch/casting/small matrix dimensions) caused the FP16 performance degradation.Data-Driven Configuration Selection: FP32 was selected as the optimal production engine configuration based on empirical measurements.

# Performance Engineering Summary
PlaintextInitial E2E Pipeline (~5K FPS)
       │
       ▼ (Identify I/O Bottleneck)
RAM-Cached Isolation
       │
       ▼ (Measure GPU Execution)
Peak Isolated Throughput (161K FPS @ Batch 64 FP32)
       │
       ▼ (Controlled Precision Comparison)
FP32 Selected Over FP16 (1.49× faster at Batch 64)

📁 Project StructurePlaintextwafer-defect-inference-cuda/
│
├── models/
│   ├── wafer_fault_cnn.onnx        # FP32 ONNX Model
│   └── wafer_fault_cnn_fp16.onnx   # Quantized FP16 ONNX Model
│
├── src/
│   ├── main.cpp                    # Single-image FP32 CUDA executable
│   ├── main_fp16.cpp               # Single-image FP16 CUDA executable
│   ├── batch_benchmark.cpp         # Baseline batch benchmark
│   ├── batch_benchmark_optimized.cpp # Ort::IoBinding latency breakdown
│   └── batch_benchmark_cached.cpp  # RAM-cached isolated benchmark
│
├── test_images/                    # Evaluation image dataset (3,828 samples)
├── results/                        # Benchmark execution logs
├── CMakeLists.txt                  # Build configuration
└── README.md                       # Documentation

Source File RolesFilePrimary Rolemain.cppStandalone FP32 CUDA inference runnermain_fp16.cppStandalone FP16 CUDA inference runnerbatch_benchmark.cppBaseline pipeline performance evaluatorbatch_benchmark_optimized.cppLatency breakdown profiler with Ort::IoBindingbatch_benchmark_cached.cppRAM-cached benchmark for pure GPU throughput isolation🖥️ Profiling & System MetricsNVIDIA Nsight Systems was used to profile CUDA API usage and GPU timeline events:Kernel launch latency and dispatch overheadHost-to-device memory copy synchronizationFP16 Tensor Core kernel verification (sm75_xmma_fprop_implicit_gemm)Profiling traces were used to drive optimization hypotheses rather than relying on theoretical hardware performance specifications.🛠️ Build & ExecutionPrerequisitesLanguage: C++17Build System: CMake >= 3.18Compiler: MSVC (Windows) or GCC/Clang (Linux)GPU Driver & Toolkit: CUDA Toolkit 12.xLibraries: OpenCV 4.x, ONNX Runtime GPU PackageBuild InstructionsDOScmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
Running ExecutablesDOS# Single-Image Run
build\Release\wafer_inference.exe
build\Release\wafer_inference_fp16.exe

# Benchmarks
build\Release\batch_benchmark.exe
build\Release\batch_benchmark_optimized.exe
build\Release\batch_benchmark_cached.exe

---

# 🏁 Conclusion

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
