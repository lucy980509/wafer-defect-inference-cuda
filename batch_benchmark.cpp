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

namespace fs = std::filesystem;

struct BenchmarkResult {
    int batchSize;
    double avgLatencyMs;
    double fps;
    bool success;
};

BenchmarkResult RunBaselineBenchmark(
    const std::wstring& modelPath, 
    const std::vector<std::string>& imagePaths, 
    int batchSize) 
{
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "BaselineBenchmark");
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

        Ort::MemoryInfo cpuMemInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<double> batchLatenciesMs;
        batchLatenciesMs.reserve(totalBatches);

        // Warm-up
        {
            std::vector<float> warmupInput(batchSize * 1 * 24 * 24, 0.0f);
            std::vector<int64_t> warmupShape = { batchSize, 1, 24, 24 };
            auto warmupTensor = Ort::Value::CreateTensor<float>(
                cpuMemInfo, warmupInput.data(), warmupInput.size(), warmupShape.data(), warmupShape.size());
            
            const char* inputNames[] = { inputName };
            const char* outputNames[] = { outputName };
            session.Run(Ort::RunOptions{ nullptr }, inputNames, &warmupTensor, 1, outputNames, 1);
        }

        for (size_t b = 0; b < totalImages; b += batchSize) {
            size_t currentBatchSize = std::min((size_t)batchSize, totalImages - b);

            auto start = std::chrono::high_resolution_clock::now();

            std::vector<float> inputTensorValues(currentBatchSize * 1 * 24 * 24);

            for (size_t i = 0; i < currentBatchSize; ++i) {
                cv::Mat img = cv::imread(imagePaths[b + i], cv::IMREAD_GRAYSCALE);
                if (img.empty()) continue;

                cv::Mat imgFloat;
                img.convertTo(imgFloat, CV_32F, 1.0 / 255.0);

                size_t offset = i * 24 * 24;
                std::memcpy(inputTensorValues.data() + offset, imgFloat.data, 24 * 24 * sizeof(float));
            }

            std::vector<int64_t> inputShape = { static_cast<int64_t>(currentBatchSize), 1, 24, 24 };
            auto inputTensor = Ort::Value::CreateTensor<float>(
                cpuMemInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size());

            const char* inputNames[] = { inputName };
            const char* outputNames[] = { outputName };

            auto outputTensors = session.Run(
                Ort::RunOptions{ nullptr }, 
                inputNames, &inputTensor, 1, 
                outputNames, 1
            );

            auto end = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration<double, std::milli>(end - start).count();
            batchLatenciesMs.push_back(latency);
        }

        double totalTimeMs = std::accumulate(batchLatenciesMs.begin(), batchLatenciesMs.end(), 0.0);
        double avgLatencyMs = totalTimeMs / totalBatches;
        double totalSeconds = totalTimeMs / 1000.0;
        double fps = static_cast<double>(totalImages) / totalSeconds;

        return { batchSize, avgLatencyMs, fps, true };
    }
    catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return { batchSize, 0.0, 0.0, false };
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

    std::cout << "=== Baseline Batch Scaling Benchmark (FP32) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Batch Size | Avg Latency (ms) | Throughput (FPS)" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    std::vector<int> batchSizes = { 1, 8, 16, 32, 64 };

    for (int b : batchSizes) {
        BenchmarkResult res = RunBaselineBenchmark(L"models/wafer_fault_cnn.onnx", imagePaths, b);
        std::cout << "Batch " << std::setw(2) << b << "   | "
                  << std::setw(16) << res.avgLatencyMs << " | "
                  << std::setw(16) << res.fps << " FPS" << std::endl;
    }

    return 0;
}
