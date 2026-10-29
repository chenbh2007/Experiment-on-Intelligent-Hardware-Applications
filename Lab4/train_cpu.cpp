#include <iostream>
#include <csignal>
#include <cstdlib>
#include <cerrno>
#include <cmath>

namespace svm {
#include "SVM/SVM.h"
}

static volatile std::sig_atomic_t is_running = 1;

static void interruptHandler(int signalNumber) {
    (void)signalNumber;
    is_running = 0;
}

static void printUsage(const char *program) {
    std::cerr << "Usage: " << (program ? program : "train_cpu")
              << " <data.db> <label.db> <model.pth> <model.label> <C>"
              << std::endl;
    std::cerr << "  C must be a positive finite float." << std::endl;
}

static bool parsePositiveFloat(const char *text, float &value) {
    if (!text) {
        return false;
    }

    errno = 0;
    char *end = nullptr;
    const float parsed = std::strtof(text, &end);

    if (end == nullptr || end == text || *end != '\0') {
        return false;
    }
    if (errno == ERANGE || !std::isfinite(parsed) || parsed <= 0.0f) {
        return false;
    }

    value = parsed;
    return true;
}

static bool validateModelForTraining(const svm::Model &m) {
    if (!m.Data || !m.Data->Data || !m.Data->Y || m.Data->Num <= 0) {
        return false;
    }
    if (!m.Data->Data[0] || !m.Data->Data[0]->X) {
        return false;
    }

    const int n = m.Data->Num;
    const int dim = m.Data->Data[0]->Dim;
    if (dim <= 0) {
        return false;
    }

    for (int i = 0; i < n; ++i) {
        const svm::Vec *v = m.Data->Data[i];
        if (!v || !v->X || v->Dim != dim) {
            return false;
        }
    }

    if (!m.Alpha || !m.E || !m.IS_VALID_E) {
        return false;
    }

    return true;
}

static int countSupportVectors(const svm::Model &m) {
    if (!m.Data || !m.Alpha || m.Data->Num <= 0) {
        return 0;
    }

    const float alphaEps = static_cast<float>(SVM_Eps);
    int count = 0;

    for (int i = 0; i < m.Data->Num; ++i) {
        if (m.Alpha[i] > alphaEps) {
            ++count;
        }
    }

    return count;
}

static void printModelInfo(const svm::Model &m, const char *prefix) {
    if (!m.Data || !m.Data->Data || !m.Data->Data[0] || m.Data->Num <= 0) {
        std::cout << prefix << ": no model data." << std::endl;
        return;
    }

    std::cout << prefix
              << ": samples=" << m.Data->Num
              << ", dim=" << m.Data->Data[0]->Dim
              << ", support_vectors=" << countSupportVectors(m)
              << ", C=" << m.C
              << ", B=" << m.B
              << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        printUsage(argc > 0 ? argv[0] : nullptr);
        return -1;
    }

    float C = 0.0f;
    if (!parsePositiveFloat(argv[5], C)) {
        std::cerr << "Invalid C: '" << argv[5]
                  << "'. C must be a positive finite float." << std::endl;
        return -1;
    }

    if (std::signal(SIGINT, interruptHandler) == SIG_ERR) {
        std::cerr << "Warning: failed to install SIGINT handler." << std::endl;
    }

    svm::Model m = {};
    bool resumed = false;

    if (svm::LoadState(&m, argv[3], argv[4]) == 0) {
        resumed = true;
        std::cout << "Checkpoint loaded from '" << argv[3]
                  << "' and '" << argv[4] << "'." << std::endl;
    } else {
        std::cout << "Checkpoint unavailable or invalid; initializing from raw data."
                  << std::endl;

        svm::SVMCleanUpModel(&m);

        m.Data = static_cast<svm::DataSet*>(std::calloc(1, sizeof(svm::DataSet)));
        if (!m.Data) {
            std::cerr << "Failed to allocate DataSet." << std::endl;
            return -1;
        }

        if (svm::SetUpData(m.Data, argv[1], argv[2]) != 0) {
            std::cerr << "Failed to load training data: '"
                      << argv[1] << "', '" << argv[2] << "'." << std::endl;
            svm::SVMCleanUpModel(&m);
            return -1;
        }

        svm::ModelInit(&m, C);
        std::cout << "Model initialized from raw data." << std::endl;
    }

    m.C = C;

    if (!validateModelForTraining(m)) {
        std::cerr << "Invalid or incomplete model state." << std::endl;
        svm::SVMCleanUpModel(&m);
        return -1;
    }

    std::cout << (resumed ? "Resumed model" : "New model")
              << " uses C=" << C << "." << std::endl;
    printModelInfo(m, resumed ? "Resumed model" : "New model");

    while (is_running && svm::Train(&m, 2000)) {
        svm::SaveState(&m, argv[3], argv[4]);
        std::cout << "Save Successfully." << std::endl;
    }

    if (!is_running) {
        std::cout << "Training interrupted by SIGINT." << std::endl;
    } else {
        std::cout << "Training converged." << std::endl;
    }

    svm::SaveState(&m, argv[3], argv[4]);
    std::cout << "Save Successfully." << std::endl;
    printModelInfo(m, "Final model");

    svm::SVMCleanUpModel(&m);
    return 0;
}
