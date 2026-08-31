#include <iostream>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>

void InspectModel(const std::wstring& modelPath, const std::string& modelLabel) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ShapeChecker");
    Ort::SessionOptions sessionOptions;
    
    try {
        Ort::Session session(env, modelPath.c_str(), sessionOptions);
        Ort::AllocatorWithDefaultOptions allocator;
        
        auto inputName = session.GetInputNameAllocated(0, allocator);
        auto typeInfo = session.GetInputTypeInfo(0);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> inputShape = tensorInfo.GetShape();
        
        std::cout << "[" << modelLabel << "] Input Name: " << inputName.get() << std::endl;
        std::cout << "[" << modelLabel << "] Input Shape: [";
        for (size_t i = 0; i < inputShape.size(); ++i) {
            std::cout << inputShape[i] << (i + 1 < inputShape.size() ? ", " : "");
        }
        std::cout << "]" << std::endl;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[" << modelLabel << "] Error: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  ONNX Model Input Shape Inspector" << std::endl;
    std::cout << "========================================" << std::endl;
    
    InspectModel(L"models/wafer_fault_cnn.onnx", "FP32 Model");
    InspectModel(L"models/wafer_fault_cnn_fp16.onnx", "FP16 Model");
    
    std::cout << "========================================" << std::endl;
    return 0;
}
