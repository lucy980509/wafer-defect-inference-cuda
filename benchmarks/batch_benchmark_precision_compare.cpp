#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

struct DetailedMetrics {
    double prepMs;
    double h2dMs;
    double inferMs;
    double e2eMs;
    double fps;
    double vramUsedMB;
    std::vector<float> lastBatchOutputsFP32; // Accuracy Comparison Buffer
};

// Helper: Measure VRAM Usage in MB
double GetVRAMUsageMB() {
    size_t freeBytes = 0, totalBytes = 0;
    cudaMemGetInfo(&freeBytes, &totalBytes);
    return static_cast<double>(totalBytes - freeBytes) / (1024.0 * 1024.0);
}

// Helper: Measure Isolated Performance & VRAM for a given model (FP32 or FP16)
DetailedMetrics RunSingleModelCachedBenchmark(
    const std::wstring& modelPath,
    const std::vector<cv::Mat>& cachedImages,
    int batchSize,
    std::vector<float>& outRawLogits)
{
    void* d_input = nullptr;
    void* d_output = nullptr;

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "CachedComparisonBenchmark");
        Ort::SessionOptions sessionOptions;
        
        OrtCUDAProviderOptions cudaOptions;
        cudaOptions.device_id = 0;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);

        Ort::Session session(env, modelPath.c_str(), sessionOptions);
        Ort::AllocatorWithDefaultOptions allocator;

        auto inputNameAlloc = session.GetInputNameAllocated(0, allocator);
        auto outputNameAlloc = session.GetOutputNameAllocated(0, allocator);
        const char* inputName = inputNameAlloc.get();
        const char* outputName = outputNameAlloc.get();

        // Check actual input tensor element type required by ONNX model
        auto typeInfo = session.GetInputTypeInfo(0);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType modelInputType = tensorInfo.GetElementType();

        bool isFP16Input = (modelInputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

        size_t totalImages = cachedImages.size();
        size_t totalBatches = (totalImages + batchSize - 1) / batchSize;

        size_t maxInputElements = batchSize * 1 * 24 * 24;
        size_t maxOutputElements = batchSize * 8;
        size_t inputElementSize = isFP16Input ? sizeof(uint16_t) : sizeof(float);
        size_t outputElementSize = isFP16Input ? sizeof(uint16_t) : sizeof(float);

        cudaMalloc(&d_input, maxInputElements * inputElementSize);
        cudaMalloc(&d_output, maxOutputElements * outputElementSize);

        Ort::MemoryInfo cudaMemInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
        
        // Host buffers
        std::vector<float> hostInputFP32(maxInputElements);
        std::vector<uint16_t> hostInputFP16(maxInputElements);

        std::vector<double> listPrepMs, listH2dMs, listInferMs, listE2eMs;
        listPrepMs.reserve(totalBatches);
        listH2dMs.reserve(totalBatches);
        listInferMs.reserve(totalBatches);
        listE2eMs.reserve(totalBatches);

        // Warm-up Execution
        {
            std::vector<int64_t> warmupInputShape = { batchSize, 1, 24, 24 };
            std::vector<int64_t> warmupOutputShape = { batchSize, 8 };

            auto gpuInputTensor = Ort::Value::CreateTensor(
                cudaMemInfo, d_input, maxInputElements * inputElementSize, 
                warmupInputShape.data(), warmupInputShape.size(), modelInputType);
            
            auto gpuOutputTensor = Ort::Value::CreateTensor(
                cudaMemInfo, d_output, maxOutputElements * outputElementSize, 
                warmupOutputShape.data(), warmupOutputShape.size(), modelInputType);

            Ort::IoBinding ioBinding(session);
            ioBinding.BindInput(inputName, gpuInputTensor);
            ioBinding.BindOutput(outputName, gpuOutputTensor);
            session.Run(Ort::RunOptions{ nullptr }, ioBinding);
            cudaDeviceSynchronize();
        }

        double vramUsedMB = GetVRAMUsageMB();

        // Host Output Buffer for Accuracy Check
        std::vector<float> hostOutputFP32(maxOutputElements);
        std::vector<uint16_t> hostOutputFP16(maxOutputElements);

        for (size_t b = 0; b < totalImages; b += batchSize) {
            size_t currentBatchSize = std::min((size_t)batchSize, totalImages - b);

            // 1. Preprocessing (Format Conversion & Normalization)
            auto t0 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < currentBatchSize; ++i) {
                const cv::Mat& img = cachedImages[b + i];
                cv::Mat imgFloat;
                img.convertTo(imgFloat, CV_32F, 1.0 / 255.0);

                size_t offset = i * 24 * 24;
                if (!isFP16Input) {
                    std::memcpy(hostInputFP32.data() + offset, imgFloat.data, 24 * 24 * sizeof(float));
                } else {
                    const float* srcPtr = reinterpret_cast<const float*>(imgFloat.data);
                    for (int p = 0; p < 24 * 24; ++p) {
                        cv::Mat f32Mat(1, 1, CV_32F, const_cast<float*>(&srcPtr[p]));
                        cv::Mat f16Mat;
                        f32Mat.convertTo(f16Mat, CV_16F);
                        hostInputFP16[offset + p] = f16Mat.at<uint16_t>(0);
                    }
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();

            // 2. Host-to-Device Memory Transfer
            size_t currentInputElements = currentBatchSize * 1 * 24 * 24;
            if (!isFP16Input) {
                cudaMemcpy(d_input, hostInputFP32.data(), currentInputElements * sizeof(float), cudaMemcpyHostToDevice);
            } else {
                cudaMemcpy(d_input, hostInputFP16.data(), currentInputElements * sizeof(uint16_t), cudaMemcpyHostToDevice);
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            // 3. Isolated GPU Inference
            std::vector<int64_t> inputShape = { static_cast<int64_t>(currentBatchSize), 1, 24, 24 };
            std::vector<int64_t> outputShape = { static_cast<int64_t>(currentBatchSize), 8 };

            auto gpuInputTensor = Ort::Value::CreateTensor(
                cudaMemInfo, d_input, currentInputElements * inputElementSize, 
                inputShape.data(), inputShape.size(), modelInputType);
            
            auto gpuOutputTensor = Ort::Value::CreateTensor(
                cudaMemInfo, d_output, currentBatchSize * 8 * outputElementSize, 
                outputShape.data(), outputShape.size(), modelInputType);

            Ort::IoBinding ioBinding(session);
            ioBinding.BindInput(inputName, gpuInputTensor);
            ioBinding.BindOutput(outputName, gpuOutputTensor);

            session.Run(Ort::RunOptions{ nullptr }, ioBinding);
            cudaDeviceSynchronize();
            auto t3 = std::chrono::high_resolution_clock::now();

            // Fetch Last Batch Output for Accuracy Divergence Check
            if (b + batchSize >= totalImages) {
                size_t activeOutputs = currentBatchSize * 8;
                outRawLogits.resize(activeOutputs);
                if (!isFP16Input) {
                    cudaMemcpy(hostOutputFP32.data(), d_output, activeOutputs * sizeof(float), cudaMemcpyDeviceToHost);
                    std::copy(hostOutputFP32.begin(), hostOutputFP32.begin() + activeOutputs, outRawLogits.begin());
                } else {
                    cudaMemcpy(hostOutputFP16.data(), d_output, activeOutputs * sizeof(uint16_t), cudaMemcpyDeviceToHost);
                    for (size_t o = 0; o < activeOutputs; ++o) {
                        cv::Mat f16Mat(1, 1, CV_16F, &hostOutputFP16[o]);
                        cv::Mat f32Mat;
                        f16Mat.convertTo(f32Mat, CV_32F);
                        outRawLogits[o] = f32Mat.at<float>(0);
                    }
                }
            }

            double prep = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double h2d  = std::chrono::duration<double, std::milli>(t2 - t1).count();
            double infer= std::chrono::duration<double, std::milli>(t3 - t2).count();
            double e2e  = std::chrono::duration<double, std::milli>(t3 - t0).count();

            listPrepMs.push_back(prep);
            listH2dMs.push_back(h2d);
            listInferMs.push_back(infer);
            listE2eMs.push_back(e2e);
        }

        auto avg = [](const std::vector<double>& v) {
            return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        };

        double totalTimeMs = std::accumulate(listE2eMs.begin(), listE2eMs.end(), 0.0);
        double totalSeconds = totalTimeMs / 1000.0;
        double fps = static_cast<double>(totalImages) / totalSeconds;

        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);

        return { avg(listPrepMs), avg(listH2dMs), avg(listInferMs), avg(listE2eMs), fps, vramUsedMB, outRawLogits };
    }
    catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        return { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {} };
    }
}

int main() {
    std::string dataDir = "test_images";
    std::vector<std::string> imagePaths;

    for (const auto& entry : fs::recursive_directory_iterator(dataDir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".bmp") {
                imagePaths.push_back(entry.path().string());
            }
        }
    }

    std::cout << "Caching " << imagePaths.size() << " images into RAM..." << std::endl;
    std::vector<cv::Mat> cachedImages;
    cachedImages.reserve(imagePaths.size());

    for (const auto& path : imagePaths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (!img.empty()) {
            cachedImages.push_back(img);
        }
    }
    std::cout << "Successfully cached " << cachedImages.size() << " images in RAM.\n" << std::endl;

    std::vector<int> batchSizes = { 1, 8, 16, 32, 64 };
    std::wstring modelFP32 = L"models/wafer_fault_cnn.onnx";
    std::wstring modelFP16 = L"models/wafer_fault_cnn_fp16.onnx";

    std::cout << "=====================================================================================================================" << std::endl;
    std::cout << "                          RAM-Cached Fair FP32 vs. FP16 Benchmark (Latency, VRAM & Accuracy)                       " << std::endl;
    std::cout << "=====================================================================================================================" << std::endl;
    std::cout << std::fixed << std::setprecision(3);

    for (int b : batchSizes) {
        std::vector<float> logitsFP32, logitsFP16;

        DetailedMetrics fp32Res = RunSingleModelCachedBenchmark(modelFP32, cachedImages, b, logitsFP32);
        DetailedMetrics fp16Res = RunSingleModelCachedBenchmark(modelFP16, cachedImages, b, logitsFP16);

        double inferSpeedup = (fp16Res.inferMs > 0) ? (fp32Res.inferMs / fp16Res.inferMs) : 0.0;
        double e2eSpeedup   = (fp16Res.e2eMs > 0) ? (fp32Res.e2eMs / fp16Res.e2eMs) : 0.0;

        // Calculate Accuracy Divergence (Max & Mean Absolute Error)
        float maxAbsError = 0.0f;
        float totalAbsError = 0.0f;
        size_t errorElements = std::min(logitsFP32.size(), logitsFP16.size());

        for (size_t i = 0; i < errorElements; ++i) {
            float diff = std::abs(logitsFP32[i] - logitsFP16[i]);
            if (diff > maxAbsError) maxAbsError = diff;
            totalAbsError += diff;
        }
        float meanAbsError = (errorElements > 0) ? (totalAbsError / errorElements) : 0.0f;

        std::cout << "\n[Batch Size: " << b << "]" << std::endl;
        std::cout << " Precision | Prep (ms) | H2D (ms) | Pure GPU Infer (ms) | Total E2E (ms) | Isolated FPS | VRAM Allocated (MB)" << std::endl;
        std::cout << "---------------------------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << " FP32      | " << std::setw(9) << fp32Res.prepMs << " | "
                                    << std::setw(8) << fp32Res.h2dMs << " | "
                                    << std::setw(19) << fp32Res.inferMs << " | "
                                    << std::setw(14) << fp32Res.e2eMs << " | "
                                    << std::setw(12) << fp32Res.fps << " | "
                                    << std::setw(19) << fp32Res.vramUsedMB << std::endl;
        std::cout << " FP16      | " << std::setw(9) << fp16Res.prepMs << " | "
                                    << std::setw(8) << fp16Res.h2dMs << " | "
                                    << std::setw(19) << fp16Res.inferMs << " | "
                                    << std::setw(14) << fp16Res.e2eMs << " | "
                                    << std::setw(12) << fp16Res.fps << " | "
                                    << std::setw(19) << fp16Res.vramUsedMB << std::endl;
        std::cout << "---------------------------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << " -> Pure GPU Infer Speedup (FP16/FP32) : " << std::setprecision(2) << inferSpeedup << "x" << std::endl;
        std::cout << " -> Total E2E Speedup      (FP16/FP32) : " << std::setprecision(2) << e2eSpeedup << "x" << std::endl;
        std::cout << std::scientific << std::setprecision(6);
        std::cout << " -> Accuracy Output Divergence        : Max Absolute Error = " << maxAbsError 
                  << " | Mean Absolute Error = " << meanAbsError << std::endl;
        std::cout << std::fixed << std::setprecision(3);
    }

    return 0;
}
