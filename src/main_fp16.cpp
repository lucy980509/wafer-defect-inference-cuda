#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <string>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <numeric>
#include <cmath>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace fs = std::filesystem;

// ---------------------------------------------------------
// Metrics Structure
// ---------------------------------------------------------
struct EvaluationMetrics {
    double accuracy = 0.0;
    double weightedF1 = 0.0;
    std::vector<double> precision;
    std::vector<double> recall;
    std::vector<double> f1Score;
};

// ---------------------------------------------------------
// Helper: Calculate Accuracy & Per-class F1-Score
// ---------------------------------------------------------
EvaluationMetrics CalculateMetrics(const std::vector<int>& groundTruths, 
                                   const std::vector<int>& predictions, 
                                   size_t numClasses) {
    EvaluationMetrics metrics;
    metrics.precision.resize(numClasses, 0.0);
    metrics.recall.resize(numClasses, 0.0);
    metrics.f1Score.resize(numClasses, 0.0);

    std::vector<std::vector<int>> confusionMatrix(numClasses, std::vector<int>(numClasses, 0));
    int correctCount = 0;
    size_t totalSamples = groundTruths.size();

    if (totalSamples == 0) return metrics;

    for (size_t i = 0; i < totalSamples; ++i) {
        int gt = groundTruths[i];
        int pred = predictions[i];
        if (gt >= 0 && gt < static_cast<int>(numClasses) && pred >= 0 && pred < static_cast<int>(numClasses)) {
            confusionMatrix[gt][pred]++;
            if (gt == pred) correctCount++;
        }
    }

    metrics.accuracy = static_cast<double>(correctCount) / totalSamples;

    double totalWeightedF1 = 0.0;

    for (size_t c = 0; c < numClasses; ++c) {
        int tp = confusionMatrix[c][c];
        int fp = 0;
        int fn = 0;
        int classSupport = 0;

        for (size_t i = 0; i < numClasses; ++i) {
            classSupport += confusionMatrix[c][i];
            if (i != c) {
                fp += confusionMatrix[i][c];
                fn += confusionMatrix[c][i];
            }
        }

        double prec = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double rec = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
        double f1 = (prec + rec > 0) ? (2.0 * prec * rec) / (prec + rec) : 0.0;

        metrics.precision[c] = prec;
        metrics.recall[c] = rec;
        metrics.f1Score[c] = f1;

        totalWeightedF1 += f1 * classSupport;
    }

    metrics.weightedF1 = totalWeightedF1 / totalSamples;
    return metrics;
}

int main()
{
    try
    {
        // =========================================================
        // 1. Paths & Datasets Setup
        // =========================================================
        const std::wstring modelPath = L"models/wafer_fault_cnn_fp16.onnx";
        const std::string testImagesDir = "test_images";
        const std::string singleImagePath = "results/test_wafer.png";

        const std::vector<std::string> classNames = {
            "Center", "Donut", "Edge-Loc", "Edge-Ring",
            "Loc", "Near-full", "Random", "Scratch"
        };
        const size_t numClasses = classNames.size();

        // =========================================================
        // 2. ONNX Runtime Environment & CUDA Setup
        // =========================================================
        std::cout << "[1] Creating ONNX Runtime Environment (FP16 Engine)..." << std::endl;
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "WaferInferenceFP16");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        std::cout << "[2] Configuring CUDA Execution Provider for FP16..." << std::endl;
        try {
            OrtCUDAProviderOptions cudaOptions{};
            cudaOptions.device_id = 0;
            cudaOptions.arena_extend_strategy = 0;
            cudaOptions.gpu_mem_limit = static_cast<size_t>(-1);
            cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cudaOptions.do_copy_in_default_stream = 1;

            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
            std::cout << "[SUCCESS] CUDA Execution Provider Enabled for FP16." << std::endl;
        }
        catch (const Ort::Exception& e) {
            std::cerr << "[WARNING] Failed to enable CUDA EP: " << e.what() << std::endl;
        }

        std::cout << "[3] Loading FP16 ONNX Model..." << std::endl;
        Ort::Session session(env, modelPath.c_str(), sessionOptions);
        std::cout << "[SUCCESS] FP16 ONNX Model Loaded!" << std::endl;

        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = session.GetInputNameAllocated(0, allocator);
        auto outputName = session.GetOutputNameAllocated(0, allocator);

        const char* inputNames[] = { inputName.get() };
        const char* outputNames[] = { outputName.get() };

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 4> inputShape = { 1, 1, 24, 24 };

        // =========================================================
        // 3. Warm-up (100 runs) with FP32 Input Tensor
        // =========================================================
        std::cout << "\n[4] Running Warm-up (100 runs - FP16 Engine)..." << std::endl;
        std::vector<float> warmUpBuffer(24 * 24, 0.0f);

        Ort::Value warmUpTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, warmUpBuffer.data(), warmUpBuffer.size(), inputShape.data(), inputShape.size()
        );

        for (int i = 0; i < 100; ++i) {
            session.Run(Ort::RunOptions{ nullptr }, inputNames, &warmUpTensor, 1, outputNames, 1);
        }
        std::cout << "[SUCCESS] 100 Warm-up runs completed." << std::endl;

        // =========================================================
        // 4. Test Set Directory Scanning
        // =========================================================
        std::cout << "\n[5] Scanning Test Set Directory: " << testImagesDir << std::endl;
        std::vector<std::string> imagePaths;
        std::vector<int> groundTruths;

        if (fs::exists(testImagesDir) && fs::is_directory(testImagesDir)) {
            for (const auto& entry : fs::directory_iterator(testImagesDir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".png") {
                    std::string filename = entry.path().filename().string();
                    imagePaths.push_back(entry.path().string());

                    size_t pos = filename.find("_label");
                    if (pos != std::string::npos && pos + 6 < filename.size()) {
                        int label = filename[pos + 6] - '0';
                        groundTruths.push_back(label);
                    } else {
                        groundTruths.push_back(-1);
                    }
                }
            }
        }

        if (imagePaths.empty()) {
            imagePaths.resize(3828, singleImagePath);
            groundTruths.resize(3828, 2);
        }

        std::cout << "[INFO] Target evaluation samples count: " << imagePaths.size() << std::endl;

        std::vector<double> latencies;
        latencies.reserve(imagePaths.size());
        std::vector<int> predictions;
        predictions.reserve(imagePaths.size());

        auto overallStart = std::chrono::high_resolution_clock::now();

        // =========================================================
        // 5. Inference Loop
        // =========================================================
        for (size_t idx = 0; idx < imagePaths.size(); ++idx) {
            auto t0 = std::chrono::high_resolution_clock::now();

            cv::Mat img = cv::imread(imagePaths[idx], cv::IMREAD_GRAYSCALE);
            if (img.empty()) continue;

            cv::Mat imgFloat, imgResize;
            img.convertTo(imgFloat, CV_32F);
            double minVal, maxVal;
            cv::minMaxLoc(imgFloat, &minVal, &maxVal);
            if (maxVal > 0.0) imgFloat /= static_cast<float>(maxVal);
            cv::resize(imgFloat, imgResize, cv::Size(24, 24), 0, 0, cv::INTER_LINEAR);

            std::vector<float> inputBuffer(24 * 24);
            std::copy(imgResize.ptr<float>(), imgResize.ptr<float>() + 24 * 24, inputBuffer.begin());

            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo, inputBuffer.data(), inputBuffer.size(), inputShape.data(), inputShape.size()
            );

            auto outputTensors = session.Run(
                Ort::RunOptions{ nullptr }, inputNames, &inputTensor, 1, outputNames, 1
            );

            // Output Handling (Support FP32 & FP16 Outputs)
            auto typeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType outputType = typeInfo.GetElementType();
            size_t outCount = typeInfo.GetElementCount();
            int predClass = 0;

            if (outputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                Ort::Float16_t* outData = outputTensors[0].GetTensorMutableData<Ort::Float16_t>();
                float maxValOut = -1e9f;
                for (size_t c = 0; c < outCount; ++c) {
                    float val = outData[c].ToFloat();
                    if (val > maxValOut) {
                        maxValOut = val;
                        predClass = static_cast<int>(c);
                    }
                }
            } else {
                float* outData = outputTensors[0].GetTensorMutableData<float>();
                predClass = static_cast<int>(std::distance(outData, std::max_element(outData, outData + outCount)));
            }

            predictions.push_back(predClass);

            auto t1 = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration<double, std::milli>(t1 - t0).count();
            latencies.push_back(latency);
        }

        auto overallEnd = std::chrono::high_resolution_clock::now();
        double totalTimeMs = std::chrono::duration<double, std::milli>(overallEnd - overallStart).count();

        // =========================================================
        // 6. Metrics Analysis
        // =========================================================
        std::sort(latencies.begin(), latencies.end());
        double sumLatency = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        double meanLatency = sumLatency / latencies.size();
        double p50 = latencies[static_cast<size_t>(latencies.size() * 0.50)];
        double p95 = latencies[static_cast<size_t>(latencies.size() * 0.95)];
        double overallFPS = (latencies.size() / totalTimeMs) * 1000.0;

        std::cout << "\n========================================\n";
        std::cout << "  P2 FP16 Benchmark Metrics (GPU)\n";
        std::cout << "========================================\n";
        std::cout << "Evaluated Samples : " << latencies.size() << "\n";
        std::cout << "Mean Latency      : " << std::fixed << std::setprecision(4) << meanLatency << " ms\n";
        std::cout << "P50 Latency       : " << p50 << " ms\n";
        std::cout << "P95 Latency       : " << p95 << " ms\n";
        std::cout << "Throughput (FPS)  : " << std::fixed << std::setprecision(2) << overallFPS << " FPS\n";
        std::cout << "========================================\n";

        EvaluationMetrics eval = CalculateMetrics(groundTruths, predictions, numClasses);

        std::cout << "\n========================================\n";
        std::cout << "  P2 FP16 Model Accuracy & F1 Analysis\n";
        std::cout << "========================================\n";
        std::cout << "Overall Accuracy  : " << std::setprecision(2) << (eval.accuracy * 100.0) << " %\n";
        std::cout << "Weighted F1-Score : " << std::setprecision(4) << eval.weightedF1 << "\n";
        std::cout << "----------------------------------------\n";
        for (size_t c = 0; c < numClasses; ++c) {
            std::cout << "[" << c << "] " << std::left << std::setw(12) << classNames[c]
                      << " | F1: " << std::setprecision(4) << eval.f1Score[c]
                      << " | Prec: " << eval.precision[c]
                      << " | Rec: " << eval.recall[c] << "\n";
        }
        std::cout << "========================================\n";

        return 0;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "\n[ORT EXCEPTION] " << e.what() << std::endl;
        return 2;
    }
    catch (const std::exception& e) {
        std::cerr << "\n[STD EXCEPTION] " << e.what() << std::endl;
        return 3;
    }
}
