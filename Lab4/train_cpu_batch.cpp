/* =============================================================================
 *  train_cpu.cpp -- 纯 CPU 环境的 RBF-SVM 增量训练驱动程序
 *
 *  对应头文件: SVM/SVM.h  (纯 C 实现, 与 SVM.c 链接)
 *
 *  用法:
 *      ./train_cpu <data.db> <label.db> <model.pth> <model.label> [C]
 *          argv[1] data.db        训练样本 (float 数据库)
 *          argv[2] label.db       训练标签 (signed char)
 *          argv[3] model.pth      模型权重输出/续训输入
 *          argv[4] model.label    模型标签输出/续训输入
 *          argv[5] C              惩罚系数, 缺省 1.0f
 *
 *  行为 (沿用 Lab4/train.cpp 的思路):
 *      1. 把整个训练集按 db.Num / 5 切块, 每轮 (pass) 顺序训练 5 个 batch;
 *      2. 每个 batch 训练前用 extractSV() 从上一轮模型里提取支持向量,
 *         再与当前 batch 通过 mergeDataSet() 合并成新的训练集, ModelInit() 重置
 *         对偶变量后重新训练 (incremental / 自适应训练集缩减);
 *      3. 单个 batch 内反复调用 Train(M, 1000) 直到 KKT 收敛, 每次未收敛都
 *         SaveState() 落盘检查点;
 *      4. 一轮结束后若 |ΔB| 小于 SVM_Tol/5 则认为整轮收敛, 结束训练;
 *      5. SIGINT/SIGTERM 只置位 volatile sig_atomic_t 标志, 训练在安全的
 *         批边界/轮边界退出, 并且退出前一定会把当前模型落盘.
 *
 *  内存所有权约定 (与 SVM.c 的释放函数严格配对):
 *      - 栈上 DataSet  (db / batch / SV.Data ...) -> SVMCleanUpDataSet();
 *      - 堆上 DataSet  (calloc 出来, 挂到 Model::Data) -> SVMFreeDataSet();
 *        因为 SVMCleanUpModel() 内部就是用 SVMFreeDataSet() 释放 M->Data,
 *        所以本文件中所有"进入 Model::Data 的 DataSet"一律用 calloc 分配.
 *      - Model 本体 (Alpha / E / IS_VALID_E / K_ij / Data) -> SVMCleanUpModel();
 *        该函数调用后所有指针均被置 NULL, 因此可安全重复调用.
 * ============================================================================= */

#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace svm {
#include "SVM/SVM.h"
}

namespace {

/* ------------------------------------------------------------------------- *
 * 可调参数
 * ------------------------------------------------------------------------- */
constexpr int   kDefaultBatchCount = 5;      /* 每轮切分的 batch 数            */
constexpr int   kTrainCycles       = 1000;   /* 单次 Train() 的 SMO 迭代次数   */
constexpr int   kMaxTrainRounds    = 1000;   /* Train() 重试次数上限 (安全阀)  */
constexpr float kDefaultC          = 1.0f;   /* 缺省惩罚系数                   */
constexpr float kConvergenceSlack  = 5.0f;   /* |ΔB| <= SVM_Tol/5 视为收敛     */

struct Options {
    const char *dataPath       = nullptr;
    const char *labelPath      = nullptr;
    const char *modelPath      = nullptr;
    const char *modelLabelPath = nullptr;
    float       C              = kDefaultC;
    int         batchCount     = kDefaultBatchCount;
};

/* ------------------------------------------------------------------------- *
 * 信号处理: 只写一个 sig_atomic_t, 满足异步信号安全要求
 * ------------------------------------------------------------------------- */
volatile std::sig_atomic_t is_running = 1;

void InterruptHandler(int sig)
{
    (void)sig;
    is_running = 0;
}

bool InstallSignalHandlers()
{
    if (std::signal(SIGINT, InterruptHandler) == SIG_ERR) {
        return false;
    }
#ifdef SIGTERM
    if (std::signal(SIGTERM, InterruptHandler) == SIG_ERR) {
        return false;
    }
#endif
    return true;
}

/* ------------------------------------------------------------------------- *
 * RAII 守卫: 保证任何返回/异常路径上资源都被正确释放
 * ------------------------------------------------------------------------- */
struct ModelGuard {
    svm::Model M;
    ModelGuard() { std::memset(&M, 0, sizeof(M)); }
    ~ModelGuard() { svm::SVMCleanUpModel(&M); }
    ModelGuard(const ModelGuard &)            = delete;
    ModelGuard &operator=(const ModelGuard &) = delete;
};

struct DataSetGuard {
    svm::DataSet D;
    DataSetGuard() { std::memset(&D, 0, sizeof(D)); }
    ~DataSetGuard() { svm::SVMCleanUpDataSet(&D); }
    DataSetGuard(const DataSetGuard &)            = delete;
    DataSetGuard &operator=(const DataSetGuard &) = delete;
};

/* ------------------------------------------------------------------------- *
 * 小型工具函数
 * ------------------------------------------------------------------------- */

/* 分配一个"归 Model 所有"的 DataSet (配对释放: SVMFreeDataSet) */
svm::DataSet *AllocDataSet()
{
    return static_cast<svm::DataSet *>(std::calloc(1, sizeof(svm::DataSet)));
}

/* 数据集结构完整性检查 (含每条向量与维度一致性) */
bool DataSetIsUsable(const svm::DataSet *d)
{
    if (d == nullptr || d->Num <= 0 || d->Data == nullptr || d->Y == nullptr) {
        return false;
    }
    const int dim = (d->Data[0] != nullptr) ? d->Data[0]->Dim : 0;
    if (dim <= 0) {
        return false;
    }
    for (int i = 0; i < d->Num; ++i) {
        const svm::Vec *v = d->Data[i];
        if (v == nullptr || v->X == nullptr || v->Dim != dim) {
            return false;
        }
    }
    return true;
}

/* 模型可用性检查: 对偶变量与数据集必须同时就绪 */
bool ModelIsUsable(const svm::Model *M)
{
    if (M == nullptr) {
        return false;
    }
    if (M->Data == nullptr || M->Alpha == nullptr || M->E == nullptr ||
        M->IS_VALID_E == nullptr) {
        return false;
    }
    return DataSetIsUsable(M->Data);
}

/* 从 src 中切出 [beg, end) 生成一个新的堆上 DataSet (所有权交给 Model) */
svm::DataSet *SliceDataSet(svm::DataSet *src, int beg, int end)
{
    if (src == nullptr || beg < 0 || beg >= end || end > src->Num) {
        return nullptr;
    }
    svm::DataSet *dst = AllocDataSet();
    if (dst == nullptr) {
        return nullptr;
    }
    svm::dataSetSlice(src, dst, beg, end);
    if (!DataSetIsUsable(dst)) {
        svm::SVMFreeDataSet(dst);   /* 配对释放 calloc 出来的 DataSet */
        return nullptr;
    }
    return dst;
}

/* ------------------------------------------------------------------------- *
 * 单个 batch 内的训练: 反复 Train 直到 KKT 收敛, 每轮落盘检查点
 * ------------------------------------------------------------------------- */
void TrainUntilConverged(svm::Model *M, const Options &opt, bool *fatal)
{
    if (fatal != nullptr) {
        *fatal = false;
    }

    for (int round = 0; round < kMaxTrainRounds; ++round) {
        if (is_running == 0) {
            return;     /* 被中断: 由调用者负责落盘并退出 */
        }

        if (svm::Train(M, kTrainCycles) == 0) {
            return;     /* 返回 0 表示在本轮迭代内已满足 KKT 条件 */
        }

        /* Train() 返回 -1 且 Gram 矩阵仍为空 => K_init 分配失败, 无法恢复 */
        if (M->K_ij == nullptr) {
            std::cerr << "[error] Train() failed to allocate the Gram matrix (n = "
                      << ((M->Data != nullptr) ? M->Data->Num : 0) << ")." << std::endl;
            if (fatal != nullptr) {
                *fatal = true;
            }
            return;
        }

        /* 未收敛: 保存检查点后继续 (与 train.cpp 语义一致) */
        svm::SaveState(M, opt.modelPath, opt.modelLabelPath);
    }

    std::cout << "[warn] Train() still reported a non-converged state after "
              << kMaxTrainRounds << " rounds; keeping the current model." << std::endl;
}

/* ------------------------------------------------------------------------- *
 * 用"上一轮支持向量 + 当前 batch"重建模型的训练集
 *
 *   前置: M 持有可用的模型 (M->Data / M->Alpha ...)
 *   后置: 成功时 M 的训练集 = SV ∪ batch, 对偶变量由 ModelInit() 重置,
 *         K_ij 为 NULL (下次 Train() 会按新规模重建);
 *         失败时 M 上的部分资源也已释放, 由 ModelGuard 统一收尾.
 * ------------------------------------------------------------------------- */
bool RebuildModelWithBatch(svm::Model *M, svm::DataSet *batch, float C)
{
    ModelGuard sv;                      /* 持有本轮提取出的支持向量模型 */

    svm::extractSV(M, &sv.M);
    if (sv.M.Data == nullptr) {
        std::cerr << "[error] extractSV() failed." << std::endl;
        return false;
    }

    const int nsv = sv.M.Data->Num;
    const bool svConsistent = (nsv == 0) ||
                              (DataSetIsUsable(sv.M.Data) &&
                               sv.M.Alpha != nullptr && sv.M.E != nullptr);
    if (nsv < 0 || !svConsistent) {
        std::cerr << "[error] extractSV() returned an inconsistent support vector set." << std::endl;
        return false;
    }
    std::cout << "Number of SVs: " << nsv << std::endl;

    /* 释放旧训练集与对偶变量; 调用后 M 内所有指针均为 NULL */
    svm::SVMCleanUpModel(M);

    svm::DataSet *merged = AllocDataSet();
    if (merged == nullptr) {
        std::cerr << "[error] out of memory while merging data sets." << std::endl;
        return false;
    }
    svm::mergeDataSet(sv.M.Data, batch, merged);

    if (!DataSetIsUsable(merged) || merged->Num != nsv + batch->Num) {
        std::cerr << "[error] mergeDataSet() failed." << std::endl;
        svm::SVMFreeDataSet(merged);
        return false;
    }

    M->Data = merged;                   /* 所有权移交给 Model */
    svm::ModelInit(M, C);               /* 重置 Alpha/E/IS_VALID_E, B=0, C=C */
    if (!ModelIsUsable(M)) {
        std::cerr << "[error] ModelInit() failed." << std::endl;
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------------- *
 * 一轮 (pass): 顺序处理全部 batch
 *
 *   每个 batch 都只被消费一次:
 *     - 首次训练 (haveModel == false): 直接以第一个 batch 作为训练集;
 *     - 之后每次: SV(上一轮) ∪ batch 后重新训练.
 *   注: train.cpp 中 init/cold 两个分支会让每轮有一个 batch 被跳过
 *       (cold 分支只对已收敛的模型重复调用一次 Train), 此处已修正.
 * ------------------------------------------------------------------------- */
bool RunPass(svm::DataSet *db, svm::Model *M, bool &haveModel,
             const Options &opt, float bPassStart)
{
    const int batchSize = db->Num / opt.batchCount;

    for (int i = 0; i < opt.batchCount; ++i) {
        if (is_running == 0) {
            return true;
        }

        const int beg = i * batchSize;
        int end = (i + 1) * batchSize;
        if (i + 1 == opt.batchCount || end > db->Num) {
            end = db->Num;              /* 最后一块吸收 db.Num % batchCount 的余数 */
        }
        if (beg >= end) {
            continue;                   /* 空块 (数据量极少时才会出现) */
        }

        if (!haveModel) {
            svm::DataSet *first = SliceDataSet(db, beg, end);
            if (first == nullptr) {
                std::cerr << "[error] failed to slice batch " << i << " of the data set." << std::endl;
                return false;
            }
            M->Data = first;            /* 所有权移交给 Model */
            svm::ModelInit(M, opt.C);
            if (!ModelIsUsable(M)) {
                std::cerr << "[error] ModelInit() failed on batch " << i << "." << std::endl;
                return false;
            }
            haveModel = true;
            std::cout << "Init Finished." << std::endl;
        } else {
            DataSetGuard batch;         /* 作用域结束自动 SVMCleanUpDataSet */
            svm::dataSetSlice(db, &batch.D, beg, end);
            if (!DataSetIsUsable(&batch.D)) {
                std::cerr << "[error] failed to slice batch " << i << " of the data set." << std::endl;
                return false;
            }
            if (!RebuildModelWithBatch(M, &batch.D, opt.C)) {
                std::cerr << "[error] failed to prepare the model for batch " << i << "." << std::endl;
                return false;
            }
        }

        bool fatal = false;
        TrainUntilConverged(M, opt, &fatal);

        std::cout << "Batch " << i << " complete. " << std::endl;
        std::cout << "delta b: " << (M->B - bPassStart) << std::endl;

        /* 批边界落盘: 即使随后被 SIGINT 打断, 磁盘上的状态也是一致的 */
        svm::SaveState(M, opt.modelPath, opt.modelLabelPath);

        if (fatal) {
            std::cerr << "[error] training aborted on batch " << i << "." << std::endl;
            return false;
        }
        if (is_running == 0) {
            return true;
        }
    }
    return true;
}

} /* namespace */

/* ------------------------------------------------------------------------- *
 * main
 * ------------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    if (argc < 5) {
        std::cout << "Too few argument." << std::endl;
        std::cout << "Usage: " << ((argc > 0 && argv[0] != nullptr) ? argv[0] : "train_cpu")
                  << " <data.db> <label.db> <model.pth> <model.label> [C]" << std::endl;
        return -1;
    }

    Options opt;
    opt.dataPath       = argv[1];
    opt.labelPath      = argv[2];
    opt.modelPath      = argv[3];
    opt.modelLabelPath = argv[4];

    /* ---------------- 解析可选参数 C ---------------- */
    bool cGiven = false;
    if (argc >= 6) {
        errno = 0;
        char *parseEnd = nullptr;
        const float value = std::strtof(argv[5], &parseEnd);
        if (parseEnd == argv[5] || *parseEnd != '\0' || errno == ERANGE ||
            !std::isfinite(value) || value <= 0.0f) {
            std::cerr << "Invalid penalty parameter C = \"" << argv[5]
                      << "\" (a finite number greater than 0 is required)." << std::endl;
            return -1;
        }
        opt.C  = value;
        cGiven = true;
    }

    /* ---------------- 信号处理 ---------------- */
    is_running = 1;
    if (!InstallSignalHandlers()) {
        std::cerr << "[warn] failed to install signal handlers." << std::endl;
    }

    /* ---------------- 载入训练数据 ---------------- */
    DataSetGuard db;
    if (svm::SetUpData(&db.D, opt.dataPath, opt.labelPath) != 0) {
        std::cerr << "Failed to load data set (" << opt.dataPath
                  << ", " << opt.labelPath << ")." << std::endl;
        return -1;
    }
    if (!DataSetIsUsable(&db.D)) {
        std::cerr << "Data set is inconsistent." << std::endl;
        return -1;
    }
    std::cout << "Loaded " << db.D.Num << " samples of dimension "
              << db.D.Data[0]->Dim << "." << std::endl;

    /* ---------------- 载入/恢复模型 ---------------- */
    ModelGuard model;
    bool haveModel = false;
    if (svm::LoadState(&model.M, opt.modelPath, opt.modelLabelPath) == 0 &&
        ModelIsUsable(&model.M)) {
        haveModel = true;
        std::cout << "Resumed from " << opt.modelPath << " with "
                  << model.M.Data->Num << " samples (B = " << model.M.B
                  << ", C = " << model.M.C << ")." << std::endl;
    } else {
        svm::SVMCleanUpModel(&model.M);   /* 丢弃可能残留的半成品状态 */
        std::cout << "No usable state in " << opt.modelPath
                  << "; starting from scratch." << std::endl;
    }

    /* ---------------- 决定本次训练使用的 C ---------------- */
    if (haveModel) {
        if (cGiven) {
            model.M.C = opt.C;            /* 命令行显式覆盖状态文件中的 C */
        } else if (std::isfinite(model.M.C) && model.M.C > 0.0f) {
            opt.C = model.M.C;            /* 续训沿用状态文件中的 C */
        } else {
            std::cout << "[warn] invalid C in state file (" << model.M.C
                      << "), falling back to " << kDefaultC << "." << std::endl;
            opt.C     = kDefaultC;
            model.M.C = opt.C;
        }
    }
    std::cout << "Penalty parameter C = " << opt.C << std::endl;

    /* ---------------- 计算分块规模 ---------------- */
    if (db.D.Num / opt.batchCount <= 0) {
        opt.batchCount = 1;               /* 样本数不足 5 时退化为单块 */
    }
    std::cout << "Samples per batch: " << (db.D.Num / opt.batchCount)
              << " (" << opt.batchCount << " batch(es) per pass)." << std::endl;

    /* ---------------- 多轮增量训练 ---------------- */
    bool firstPass = true;
    bool ok        = true;
    while (is_running != 0) {
        const float bPassStart = model.M.B;

        if (!RunPass(&db.D, &model.M, haveModel, opt, bPassStart)) {
            ok = false;
            break;
        }
        if (is_running == 0) {
            break;
        }

        if (!firstPass) {
            const float delta = std::fabs(bPassStart - model.M.B);
            if (delta <= static_cast<float>(SVM_Tol) / kConvergenceSlack) {
                std::cout << "Converged: |delta B| = " << delta << "." << std::endl;
                break;
            }
        }
        firstPass = false;
    }

    /* ---------------- 收尾 ---------------- */
    if (!ok) {
        std::cerr << "[error] training terminated abnormally." << std::endl;
        return -1;
    }

    if (is_running == 0) {
        /* 被信号打断: 再做一次防御性落盘, 保证磁盘状态 = 内存状态 */
        if (ModelIsUsable(&model.M)) {
            svm::SaveState(&model.M, opt.modelPath, opt.modelLabelPath);
        }
        std::cout << "Interrupted by signal; the model has been saved to "
                  << opt.modelPath << "." << std::endl;
    }

    std::cout << "Training finished. Final B = " << model.M.B
              << ", support vectors = "
              << ((model.M.Data != nullptr) ? model.M.Data->Num : 0) << "." << std::endl;

    /* ModelGuard / DataSetGuard 在此处完成 SVMCleanUpModel / SVMCleanUpDataSet */
    return 0;
}
