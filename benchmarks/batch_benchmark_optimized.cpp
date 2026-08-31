#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

struct BreakdownResult {
    int batchSize;
    double prepMs;
    double h2dMs;
    double inferMs;
    double e2eMs;
    bool success;
};

BreakdownResult RunBreakdownBenchmark(
    const std::wstring& modelPath, 
    const std::vector<std::string>& imagePaths, 
    int batchSize) 
{
    float* d_input = nullptr;
    float* d_output = nullptr;

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "BreakdownBenchmark");
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

        size_t totalImages = imagePaths.size();
        size_t totalBatches = (totalImages + batchSize - 1) / batchSize;

        size_t maxInputElements = batchSize * 1 * 24 * 24;
        size_t maxOutputElements = batchSize * 8;

        cudaMalloc(&d_input, maxInputElements * sizeof(float));
        cudaMalloc(&d_output, maxOutputElements * sizeof(float));

        Ort::MemoryInfo cudaMemInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);

        std::vector<float> hostInputBuffer(maxInputElements);
        
        std::vector<double> listPrepMs, listH2dMs, listInferMs, listE2eMs;
        listPrepMs.reserve(totalBatches);
        listH2dMs.reserve(totalBatches);
        listInferMs.reserve(totalBatches);
        listE2eMs.reserve(totalBatches);

        // Warm-up Execution
        {
            std::vector<int64_t> warmupInputShape = { batchSize, 1, 24, 24 };
            std::vector<int64_t> warmupOutputShape = { batchSize, 8 };

            auto gpuInputTensor = Ort::Value::CreateTensor<float>(
                cudaMemInfo, d_input, maxInputElements, warmupInputShape.data(), warmupInputShape.size());
            auto gpuOutputTensor = Ort::Value::CreateTensor<float>(
                cudaMemInfo, d_output, maxOutputElements, warmupOutputShape.data(), warmupOutputShape.size());

            Ort::IoBinding ioBinding(session);
            ioBinding.BindInput(inputName, gpuInputTensor);
            ioBinding.BindOutput(outputName, gpuOutputTensor);
            session.Run(Ort::RunOptions{ nullptr }, ioBinding);
            cudaDeviceSynchronize();
        }

        for (size_t b = 0; b < totalImages; b += batchSize) {
            size_t currentBatchSize = std::min((size_t)batchSize, totalImages - b);

            // 1. Preprocessing (cv::imread + float conversion + format layout)
            auto t0 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < currentBatchSize; ++i) {
                cv::Mat img = cv::imread(imagePaths[b + i], cv::IMREAD_GRAYSCALE);
                if (img.empty()) continue;

                cv::Mat imgFloat;
                img.convertTo(imgFloat, CV_32F, 1.0 / 255.0);

                size_t offset = i * 24 * 24;
                std::memcpy(hostInputBuffer.data() + offset, imgFloat.data, 24 * 24 * sizeof(float));
            }
            auto t1 = std::chrono::high_resolution_clock::now();

            // 2. Host to Device Data Copy
            size_t currentInputSize = currentBatchSize * 1 * 24 * 24;
            cudaMemcpy(d_input, hostInputBuffer.data(), currentInputSize * sizeof(float), cudaMemcpyHostToDevice);
            auto t2 = std::chrono::high_resolution_clock::now();

            // 3. Pure GPU Inference Execution
            std::vector<int64_t> inputShape = { static_cast<int64_t>(currentBatchSize), 1, 24, 24 };
            std::vector<int64_t> outputShape = { static_cast<int64_t>(currentBatchSize), 8 };

            auto gpuInputTensor = Ort::Value::CreateTensor<float>(
                cudaMemInfo, d_input, currentInputSize, inputShape.data(), inputShape.size());
            
            auto gpuOutputTensor = Ort::Value::CreateTensor<float>(
                cudaMemInfo, d_output, currentBatchSize * 8, outputShape.data(), outputShape.size());

            Ort::IoBinding ioBinding(session);
            ioBinding.BindInput(inputName, gpuInputTensor);
            ioBinding.BindOutput(outputName, gpuOutputTensor);

            session.Run(Ort::RunOptions{ nullptr }, ioBinding);
            cudaDeviceSynchronize();
            auto t3 = std::chrono::high_resolution_clock::now();

            // Latency Breakdown Calculations
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

        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);

        return { batchSize, avg(listPrepMs), avg(listH2dMs), avg(listInferMs), avg(listE2eMs), true };
    }
    catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        return { batchSize, 0.0, 0.0, 0.0, 0.0, false };
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

    std::cout << "=== Latency Breakdown Analysis (FP32) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Batch | Preprocess (A) | H2D Transfer | GPU Infer (B) | Total E2E (C) | Prep Share" << std::endl;
    std::cout << "----------------------------------------------------------------------------------" << std::endl;

    std::vector<int> batchSizes = { 1, 8, 16, 32, 64 };

    for (int b : batchSizes) {
        BreakdownResult res = RunBreakdownBenchmark(L"models/wafer_fault_cnn.onnx", imagePaths, b);
        double prepShare = (res.prepMs / res.e2eMs) * 100.0;
        
        std::cout << "Batch " << std::setw(2) << b << " | "
                  << std::setw(8) << res.prepMs << " ms | "
                  << std::setw(7) << res.h2dMs << " ms | "
                  << std::setw(8) << res.inferMs << " ms | "
                  << std::setw(8) << res.e2eMs << " ms | "
                  << std::setw(8) << prepShare << " %" << std::endl;
    }

    return 0;
}
