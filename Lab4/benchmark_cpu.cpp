// benchmark_cpu.cpp
// CPU-only benchmark for multi-class SVM models.
//
// Usage:
//   ./benchmark_cpu <ModelFileList> <testset.db> <testset.label> [--verbose]
//
// ModelFileList format:
//   Each model takes two non-empty lines: state file path and label file path.
//   Exactly 10 models are expected.
//
// Dataset format:
//   testset.db   : magic 00 00 0D 02 (little-endian float), sample count (LE),
//                  dimension (LE), then sample_count * dim little-endian floats.
//   testset.label: magic 00 00 08 01 (big-endian unsigned char), sample count (BE),
//                  then sample_count signed bytes.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

namespace svm {
#include "SVM/SVM.h"
}

namespace {

constexpr uint32_t kMaxSamples = 100000000u; // 1e8
constexpr uint32_t kMaxDim = 10000000u;     // 1e7

std::string trim(const std::string& s) {
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return {};
    }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

bool readNonEmptyLine(std::ifstream& in, std::string& line) {
    while (std::getline(in, line)) {
        line = trim(line);
        if (!line.empty()) {
            return true;
        }
    }
    return false;
}

bool readMagic(std::ifstream& in, const unsigned char expected[4]) {
    unsigned char magic[4];
    if (!in.read(reinterpret_cast<char*>(magic), 4)) {
        return false;
    }
    return magic[0] == expected[0] && magic[1] == expected[1] &&
           magic[2] == expected[2] && magic[3] == expected[3];
}

bool readU32LE(std::ifstream& in, uint32_t& value) {
    unsigned char b[4];
    if (!in.read(reinterpret_cast<char*>(b), 4)) {
        return false;
    }
    value = static_cast<uint32_t>(b[0]) |
            (static_cast<uint32_t>(b[1]) << 8) |
            (static_cast<uint32_t>(b[2]) << 16) |
            (static_cast<uint32_t>(b[3]) << 24);
    return true;
}

bool readU32BE(std::ifstream& in, uint32_t& value) {
    unsigned char b[4];
    if (!in.read(reinterpret_cast<char*>(b), 4)) {
        return false;
    }
    value = (static_cast<uint32_t>(b[0]) << 24) |
            (static_cast<uint32_t>(b[1]) << 16) |
            (static_cast<uint32_t>(b[2]) << 8) |
            static_cast<uint32_t>(b[3]);
    return true;
}

bool hostIsLittleEndian() {
    static const bool little = []() {
        const uint16_t x = 1;
        return *reinterpret_cast<const unsigned char*>(&x) == 1;
    }();
    return little;
}

void swapFloatArrayIfNeeded(float* arr, size_t count) {
    if (hostIsLittleEndian()) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        uint32_t u = 0;
        std::memcpy(&u, arr + i, sizeof(u));
        u = ((u & 0x000000FFu) << 24) |
            ((u & 0x0000FF00u) << 8) |
            ((u & 0x00FF0000u) >> 8) |
            ((u & 0xFF000000u) >> 24);
        std::memcpy(arr + i, &u, sizeof(u));
    }
}

void manualFreeDataSet(svm::DataSet& ds) {
    if (ds.Data != nullptr) {
        for (int i = 0; i < ds.Num; ++i) {
            if (ds.Data[i] != nullptr) {
                if (ds.Data[i]->X != nullptr) {
                    std::free(ds.Data[i]->X);
                    ds.Data[i]->X = nullptr;
                }
                std::free(ds.Data[i]);
                ds.Data[i] = nullptr;
            }
        }
        std::free(ds.Data);
        ds.Data = nullptr;
    }
    if (ds.Y != nullptr) {
        std::free(ds.Y);
        ds.Y = nullptr;
    }
    ds.Num = 0;
}

void releaseModels(svm::Model* models[], int count) {
    for (int i = 0; i < count; ++i) {
        if (models[i] != nullptr) {
            svm::SVMFreeModel(models[i]);
            models[i] = nullptr;
        }
    }
}

bool loadDataSet(const char* dataPath, const char* labelPath,
                 svm::DataSet& testset, std::string& error) {
    std::ifstream data(dataPath, std::ios::binary);
    std::ifstream label(labelPath, std::ios::binary);
    if (!data.is_open()) {
        error = std::string("Cannot open data file: ") + dataPath;
        return false;
    }
    if (!label.is_open()) {
        error = std::string("Cannot open label file: ") + labelPath;
        return false;
    }

    const unsigned char expectedDataMagic[4] = {0x00, 0x00, 0x0D, 0x02};
    const unsigned char expectedLabelMagic[4] = {0x00, 0x00, 0x08, 0x01};

    if (!readMagic(data, expectedDataMagic)) {
        error = "Invalid data file magic (expected 00 00 0D 02).";
        return false;
    }
    if (!readMagic(label, expectedLabelMagic)) {
        error = "Invalid label file magic (expected 00 00 08 01).";
        return false;
    }

    uint32_t dataSamples = 0;
    uint32_t labelSamples = 0;
    uint32_t dim = 0;

    if (!readU32LE(data, dataSamples)) {
        error = "Failed to read sample count from data file.";
        return false;
    }
    if (!readU32BE(label, labelSamples)) {
        error = "Failed to read sample count from label file.";
        return false;
    }
    if (dataSamples != labelSamples) {
        error = "Sample count mismatch between data and label files: " +
                std::to_string(dataSamples) + " vs " + std::to_string(labelSamples) + ".";
        return false;
    }
    if (dataSamples == 0) {
        error = "Dataset is empty (sample count is zero).";
        return false;
    }
    if (dataSamples > kMaxSamples) {
        error = "Sample count exceeds supported limit: " + std::to_string(dataSamples) + ".";
        return false;
    }
    if (!readU32LE(data, dim)) {
        error = "Failed to read feature dimension from data file.";
        return false;
    }
    if (dim == 0) {
        error = "Feature dimension is zero.";
        return false;
    }
    if (dim > kMaxDim) {
        error = "Feature dimension exceeds supported limit: " + std::to_string(dim) + ".";
        return false;
    }
    if (dataSamples > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        dim > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        error = "Sample count or dimension exceeds int range.";
        return false;
    }

    std::memset(&testset, 0, sizeof(testset));
    testset.Num = static_cast<int>(dataSamples);

    testset.Data = static_cast<svm::Vec**>(
        std::calloc(static_cast<size_t>(dataSamples), sizeof(svm::Vec*)));
    if (testset.Data == nullptr) {
        testset.Num = 0;
        error = "Out of memory while allocating DataSet data array.";
        return false;
    }

    testset.Y = static_cast<signed char*>(
        std::malloc(static_cast<size_t>(dataSamples) * sizeof(signed char)));
    if (testset.Y == nullptr) {
        manualFreeDataSet(testset);
        error = "Out of memory while allocating DataSet label array.";
        return false;
    }

    const std::streamsize featureBytes =
        static_cast<std::streamsize>(dim) * sizeof(float);

    for (uint32_t i = 0; i < dataSamples; ++i) {
        svm::Vec* vec = static_cast<svm::Vec*>(std::calloc(1, sizeof(svm::Vec)));
        if (vec == nullptr) {
            manualFreeDataSet(testset);
            error = "Out of memory while allocating Vec for sample " + std::to_string(i) + ".";
            return false;
        }

        testset.Data[i] = vec;
        vec->Dim = static_cast<int>(dim);
        vec->X = static_cast<float*>(
            std::malloc(static_cast<size_t>(dim) * sizeof(float)));
        if (vec->X == nullptr) {
            manualFreeDataSet(testset);
            error = "Out of memory while allocating feature vector for sample " +
                    std::to_string(i) + ".";
            return false;
        }

        if (!data.read(reinterpret_cast<char*>(vec->X), featureBytes)) {
            manualFreeDataSet(testset);
            error = "Failed to read feature vector for sample " + std::to_string(i) + ".";
            return false;
        }

        swapFloatArrayIfNeeded(vec->X, static_cast<size_t>(dim));
    }

    const std::streamsize labelBytes =
        static_cast<std::streamsize>(dataSamples) * sizeof(signed char);
    if (!label.read(reinterpret_cast<char*>(testset.Y), labelBytes)) {
        manualFreeDataSet(testset);
        error = "Failed to read label values.";
        return false;
    }

    return true;
}

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <ModelFileList> <testset.db> <testset.label> [--verbose]"
              << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        printUsage(argv[0]);
        return 1;
    }

    bool verbose = false;
    if (argc == 5) {
        if (std::strcmp(argv[4], "--verbose") == 0) {
            verbose = true;
        } else {
            std::cerr << "Error: unknown option: " << argv[4] << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    const char* modelListPath = argv[1];
    const char* dataPath = argv[2];
    const char* labelPath = argv[3];

    svm::DataSet testset{};
    std::string error;
    if (!loadDataSet(dataPath, labelPath, testset, error)) {
        std::cerr << "Error: " << error << std::endl;
        return 1;
    }

    std::cout << "Read test set successfully: " << testset.Num
              << " samples, dimension " << testset.Data[0]->Dim << "." << std::endl;

    svm::Model* SV[10] = {nullptr};
    svm::Model models;
    std::memset(&models, 0, sizeof(models));

    std::ifstream modellist(modelListPath);
    if (!modellist.is_open()) {
        std::cerr << "Error: cannot open model file list: " << modelListPath << std::endl;
        svm::SVMCleanUpDataSet(&testset);
        return 1;
    }

    int loadedCount = 0;
    bool modelLoadFailed = false;
    std::string modelError;

    for (int i = 0; i < 10; ++i) {
        std::string statePath;
        std::string labelFilePath;

        if (!readNonEmptyLine(modellist, statePath)) {
            modelError = "Model file list is too short; missing state file for model " +
                         std::to_string(i) + ".";
            modelLoadFailed = true;
            break;
        }
        if (!readNonEmptyLine(modellist, labelFilePath)) {
            modelError = "Model file list is too short; missing label file for model " +
                         std::to_string(i) + ".";
            modelLoadFailed = true;
            break;
        }

        std::memset(&models, 0, sizeof(models));

        if (svm::LoadState(&models, statePath.c_str(), labelFilePath.c_str()) == -1) {
            modelError = "Failed to load model " + std::to_string(i) + ": " + statePath;
            svm::SVMCleanUpModel(&models);
            std::memset(&models, 0, sizeof(models));
            modelLoadFailed = true;
            break;
        }

        SV[i] = static_cast<svm::Model*>(std::calloc(1, sizeof(svm::Model)));
        if (SV[i] == nullptr) {
            modelError = "Out of memory while allocating extracted SV model " +
                         std::to_string(i) + ".";
            svm::SVMCleanUpModel(&models);
            std::memset(&models, 0, sizeof(models));
            modelLoadFailed = true;
            break;
        }

        svm::extractSV(&models, SV[i]);
        svm::SVMCleanUpModel(&models);
        std::memset(&models, 0, sizeof(models));

        ++loadedCount;
        std::cout << "Loaded model " << i << " successfully." << std::endl;
    }
    modellist.close();

    if (modelLoadFailed) {
        std::cerr << "Error: " << modelError << std::endl;
        releaseModels(SV, loadedCount);
        svm::SVMCleanUpDataSet(&testset);
        return 1;
    }

    if (loadedCount != 10) {
        std::cerr << "Error: expected 10 models, loaded " << loadedCount << "." << std::endl;
        releaseModels(SV, loadedCount);
        svm::SVMCleanUpDataSet(&testset);
        return 1;
    }

    const int err = svm::benchmark(SV, &testset, 10);
    if (err < 0) {
        std::cerr << "Error: benchmark failed." << std::endl;
        releaseModels(SV, 10);
        svm::SVMCleanUpDataSet(&testset);
        return 1;
    }

    const int total = testset.Num;
    const double errorRate = total > 0 ? static_cast<double>(err) / total : 0.0;
    const double accuracy = 1.0 - errorRate;

    std::cout << "Benchmark errors: " << err << "/" << total
              << " (error rate " << (errorRate * 100.0) << "%, accuracy "
              << (accuracy * 100.0) << "%)" << std::endl;

    if (verbose) {
        std::cout << "Verbose mode: printing misclassified samples..." << std::endl;
        int verboseErr = 0;

        for (int i = 0; i < total; ++i) {
            int top1 = 0;
            float maxWeight = svm::Predict(SV[0], testset.Data[i]);
            float secondWeight = -std::numeric_limits<float>::infinity();
            int second = -1;

            for (int j = 1; j < 10; ++j) {
                const float w = svm::Predict(SV[j], testset.Data[i]);
                if (w > maxWeight) {
                    secondWeight = maxWeight;
                    second = top1;
                    maxWeight = w;
                    top1 = j;
                } else if (w > secondWeight) {
                    secondWeight = w;
                    second = j;
                }
            }

            const int expected = static_cast<int>(testset.Y[i]);
            if (top1 != expected) {
                ++verboseErr;
                std::cout << "Error #" << verboseErr << " at data index " << i
                          << ": expected " << expected
                          << ", predicted " << top1
                          << " (weight=" << maxWeight;
                if (second >= 0) {
                    std::cout << ", second=" << second
                              << " (weight=" << secondWeight << ")";
                } else {
                    std::cout << ", second=N/A";
                }
                std::cout << ")" << std::endl;
            }
        }

        if (verboseErr != err) {
            std::cerr << "Warning: verbose error count (" << verboseErr
                      << ") differs from benchmark result (" << err << ")." << std::endl;
        }
    }

    releaseModels(SV, 10);
    svm::SVMCleanUpDataSet(&testset);

    return 0;
}
