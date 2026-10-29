#include"SVM.h"
#include<stdio.h>
#include<stdint.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define ABS(a)    ((a) > 0 ? (a) : (-(a)))

/* =============================================================
 * Kernel: RBF (Gaussian) Kernel on CPU
 * ============================================================= */
static inline float Kernel(Vec *v1, Vec *v2){
    int d = v1->Dim;
    int i;
    float dist = 0.0f;
    for (i = 0; i < d; i++){
        float dx = v1->X[i] - v2->X[i];
        dist += dx * dx;
    }
    return expf(-SVM_Gamma * dist);
}

/* =============================================================
 * SetUpData: Read binary database and label files
 *   data format:  4B header (0x00 0x00 0x0D 0x02)
 *                 uint32 size
 *                 uint32 dim
 *                 size * dim floats
 *   label format: 4B header (0x00 0x00 0x09 0x01)
 *                 uint32 size
 *                 size signed char
 * ============================================================= */
int SetUpData(DataSet *db, const char *dataname, const char *labelname){
    FILE *data = NULL, *lab = NULL;
    uint8_t cache[4];
    uint32_t size = 0, tmp = 0, dim = 0;
    int i, ok = 1;

    if (!db) return -1;
    db->Data = NULL;
    db->Y = NULL;
    db->Num = 0;

    data = fopen(dataname, "rb");
    lab  = fopen(labelname, "rb");
    if (!data || !lab){
        printf("Data file not found.\n");
        if (data) fclose(data);
        if (lab)  fclose(lab);
        return -1;
    }

    if (fread(cache, 1, 4, data) != 4){
        printf("Failed to read data header.\n");
        fclose(data); fclose(lab);
        return -1;
    }
    if (cache[2] != 0x0D || cache[3] != 0x02){
        printf("Data type is not float.\nAbort.\n");
        fclose(data); fclose(lab);
        return -1;
    }

    if (fread(cache, 1, 4, lab) != 4){
        printf("Failed to read label header.\n");
        fclose(data); fclose(lab);
        return -1;
    }
    if (cache[2] != 0x09 || cache[3] != 0x01){
        printf("Label type is not signed char.\nAbort.\n");
        fclose(data); fclose(lab);
        return -1;
    }

    if (fread(&size, sizeof(size), 1, data) != 1 ||
        fread(&tmp,  sizeof(tmp),  1, lab)  != 1){
        printf("Failed to read sample count.\n");
        fclose(data); fclose(lab);
        return -1;
    }
    if (size != tmp){
        printf("Data doesn't fit label.\nAbort.\n");
        fclose(data); fclose(lab);
        return -1;
    }
    if (fread(&dim, sizeof(dim), 1, data) != 1){
        printf("Failed to read dimension.\n");
        fclose(data); fclose(lab);
        return -1;
    }
    if (size == 0 || dim == 0){
        printf("Empty data set.\n");
        fclose(data); fclose(lab);
        return -1;
    }

    db->Num  = (int)size;
    db->Y    = (signed char*)malloc((size_t)size * sizeof(signed char));
    db->Data = (Vec**)malloc((size_t)size * sizeof(Vec*));
    if (!db->Y || !db->Data){
        printf("Out of memory.\n");
        if (db->Y)    { free(db->Y);    db->Y = NULL; }
        if (db->Data) { free(db->Data); db->Data = NULL; }
        db->Num = 0;
        fclose(data); fclose(lab);
        return -1;
    }

    if (fread(db->Y, 1, (size_t)size, lab) != (size_t)size){
        printf("Failed to read labels.\n");
        ok = 0;
    }

    for (i = 0; i < (int)size; i++) db->Data[i] = NULL;

    for (i = 0; ok && i < (int)size; i++){
        Vec *v = (Vec*)malloc(sizeof(Vec));
        if (!v){ ok = 0; break; }
        v->Dim = (int)dim;
        v->X   = (float*)malloc((size_t)dim * sizeof(float));
        if (!v->X){ free(v); ok = 0; break; }
        if (fread(v->X, sizeof(float), (size_t)dim, data) != (size_t)dim){
            free(v->X); free(v); ok = 0; break;
        }
        db->Data[i] = v;
    }

    fclose(data);
    fclose(lab);

    if (!ok){
        SVMCleanUpDataSet(db);
        db->Num = 0;
        return -1;
    }
    return 0;
}

/* =============================================================
 * ModelInit
 * ============================================================= */
void ModelInit(Model *M, float _C){
    int n, i;
    if (!M || !M->Data){
        printf("Model Init Failed: Data Set hasn't Set up.\n");
        return;
    }
    n = M->Data->Num;

    if (M->Alpha)      { free(M->Alpha);      M->Alpha = NULL; }
    if (M->E)          { free(M->E);          M->E = NULL; }
    if (M->IS_VALID_E) { free(M->IS_VALID_E); M->IS_VALID_E = NULL; }
    if (M->K_ij)       { free(M->K_ij);       M->K_ij = NULL; }

    M->B = 0.0f;
    M->C = _C;

    if (n <= 0) return;

    M->Alpha      = (float*)malloc((size_t)n * sizeof(float));
    M->E          = (float*)malloc((size_t)n * sizeof(float));
    M->IS_VALID_E = (unsigned char*)malloc((size_t)n * sizeof(unsigned char));

    for (i = 0; i < n; i++){
        M->Alpha[i] = 0.0f;
        M->E[i]     = -(float)M->Data->Y[i];
    }
    memset(M->IS_VALID_E, 0, (size_t)n * sizeof(unsigned char));
}

/* =============================================================
 * SaveState: Save model in cuSVM-compatible binary format
 * ============================================================= */
void SaveState(Model *M, const char *FileName, const char *labname){
    FILE *f, *fl;
    uint8_t buffer[4];
    uint32_t dim, n;
    int i;

    if (!M || !M->Data || M->Data->Num <= 0){
        printf("Nothing to save.\n");
        return;
    }

    f  = fopen(FileName, "wb");
    fl = fopen(labname,  "wb");
    if (!f || !fl){
        if (f)  fclose(f);
        if (fl) fclose(fl);
        printf("Failed to save.\n");
        return;
    }

    buffer[0] = 0x00; buffer[1] = 0x00; buffer[2] = 0x0D; buffer[3] = 0x02;
    fwrite(buffer, 1, 4, f);

    dim = (uint32_t)(M->Data->Data[0]->Dim + 2);
    n   = (uint32_t)M->Data->Num;
    fwrite(&n,   sizeof(n),   1, f);
    fwrite(&dim, sizeof(dim), 1, f);

    buffer[2] = 0x09; buffer[3] = 0x01;
    fwrite(buffer, 1, 4, fl);
    fwrite(&n, sizeof(n), 1, fl);

    for (i = 0; i < (int)n; i++){
        fwrite(M->Data->Data[i]->X, sizeof(float), (size_t)(dim - 2), f);
        fwrite(&M->Alpha[i], sizeof(float), 1, f);
        fwrite(&M->E[i],     sizeof(float), 1, f);
    }

    fwrite(M->Data->Y, 1, (size_t)n, fl);
    fwrite(&M->B, sizeof(float), 1, fl);
    fwrite(&M->C, sizeof(float), 1, fl);

    fclose(f);
    fclose(fl);
}

/* =============================================================
 * LoadState: Load model from cuSVM-compatible binary format
 * ============================================================= */
int LoadState(Model *M, const char *StateFile, const char *labfile){
    FILE *f;
    int i, n;

    if (!M) return -1;

    if (!M->Data){
        M->Data = (DataSet*)calloc(1, sizeof(DataSet));
        if (!M->Data) return -1;
    } else {
        M->Data->Data = NULL;
        M->Data->Y    = NULL;
        M->Data->Num  = 0;
    }

    if (SetUpData(M->Data, StateFile, labfile) != 0){
        free(M->Data);
        M->Data = NULL;
        return -1;
    }

    f = fopen(labfile, "rb");
    if (!f){
        SVMFreeDataSet(M->Data);
        M->Data = NULL;
        return -1;
    }
    if (fseek(f, -(long)(2 * sizeof(float)), SEEK_END) != 0 ||
        fread(&M->B, sizeof(float), 1, f) != 1 ||
        fread(&M->C, sizeof(float), 1, f) != 1){
        printf("Failed to read B/C.\n");
        fclose(f);
        SVMFreeDataSet(M->Data);
        M->Data = NULL;
        return -1;
    }
    fclose(f);

    n = M->Data->Num;
    if (M->Alpha)      free(M->Alpha);
    if (M->E)          free(M->E);
    if (M->IS_VALID_E) free(M->IS_VALID_E);
    if (M->K_ij)       free(M->K_ij);

    M->Alpha      = (float*)malloc((size_t)n * sizeof(float));
    M->E          = (float*)malloc((size_t)n * sizeof(float));
    M->IS_VALID_E = (unsigned char*)malloc((size_t)n * sizeof(unsigned char));
    M->K_ij       = NULL;

    for (i = 0; i < n; i++){
        int d = M->Data->Data[i]->Dim;
        M->Alpha[i] = M->Data->Data[i]->X[d - 2];
        M->E[i]     = M->Data->Data[i]->X[d - 1];
        M->Data->Data[i]->Dim -= 2;
        M->IS_VALID_E[i] = (M->Alpha[i] > SVM_Eps) ? 1 : 0;
    }
    return 0;
}

/* =============================================================
 * K_init: Compute Gram matrix on CPU (flat 1D array of size n*n)
 * ============================================================= */
void K_init(Model *M, float *d_K){
    int n, i, j;
    if (!M || !M->Data || M->Data->Num <= 0) return;
    n = M->Data->Num;

    if (!M->K_ij){
        M->K_ij = (float*)malloc((size_t)n * (size_t)n * sizeof(float));
        if (!M->K_ij) return;
    }

    for (i = 0; i < n; i++){
        M->K_ij[(size_t)i * n + i] = 1.00f;
        for (j = 0; j < i; j++){
            float k = Kernel(M->Data->Data[i], M->Data->Data[j]);
            M->K_ij[(size_t)i * n + j] = k;
            M->K_ij[(size_t)j * n + i] = k;
        }
    }

    if (d_K) memcpy(d_K, M->K_ij, (size_t)n * (size_t)n * sizeof(float));
}

/* =============================================================
 * CPU SMO Optimization Routines
 * ============================================================= */
static int select_alpha1_cpu(float *alpha,
                             signed char *Y,
                             float *E,
                             int n,
                             float *cache,
                             float C,
                             int skip_a1){
    int i;
    float best;
    int bestIdx;

    if (skip_a1 == -1){
        for (i = 0; i < n; i++){
            float tmp = -1.0f;
            if (alpha[i] > SVM_Eps)        tmp = (float)Y[i] * E[i];
            if (alpha[i] < C - SVM_Eps)    tmp = MAX(tmp, -(float)Y[i] * E[i]);
            cache[i] = tmp;
        }
    } else {
        if (skip_a1 >= 0 && skip_a1 < n) cache[skip_a1] = 0.0f;
    }

    best    = SVM_Tol;
    bestIdx = -1;
    for (i = 0; i < n; i++){
        if (cache[i] > best){
            best    = cache[i];
            bestIdx = i;
        }
    }
    return bestIdx;
}

static int select_alpha2_cpu(float *alpha,
                             float *E,
                             float *K,
                             signed char *Y,
                             float C,
                             int n,
                             int a1,
                             float *delta1,
                             float *delta2,
                             float *deltab,
                             float *B){
    int i;
    float best;
    int a2;

    for (i = 0; i < n; i++){
        float Kval = K[(size_t)a1 * n + i];
        float d1, d2, b1, b2, L, H, tmpalpha1, tmpalpha2;
        signed char Y1;
        float E1, alpha1;

        if ((Kval - 1.0f > -SVM_Eps && Kval - 1.0f < SVM_Eps) || i == a1){
            delta1[i] = 0.0f;
            delta2[i] = 0.0f;
            deltab[i] = 0.0f;
            continue;
        }

        E1     = E[a1];
        Y1     = Y[a1];
        alpha1 = alpha[a1];

        d2 = (float)Y[i] * (E1 - E[i]) / (2.0f - 2.0f * Kval);

        if (Y1 == Y[i]){
            L = MAX(0.0f, alpha1 + alpha[i] - C);
            H = MIN(C,    alpha1 + alpha[i]);
        } else {
            L = MAX(0.0f, alpha[i] - alpha1);
            H = MIN(C,    C + alpha[i] - alpha1);
        }

        if (L > H - SVM_Eps){
            delta1[i] = 0.0f;
            delta2[i] = 0.0f;
            deltab[i] = 0.0f;
            continue;
        }

        tmpalpha2 = MAX(L, MIN(H, alpha[i] + d2));
        d2        = tmpalpha2 - alpha[i];
        d1        = -(float)Y1 * (float)Y[i] * d2;
        tmpalpha1 = d1 + alpha1;

        b1 = -E1   - (float)Y1 * d1        - (float)Y[i] * d2 * Kval;
        b2 = -E[i] - (float)Y1 * d1 * Kval - (float)Y[i] * d2;

        if (tmpalpha1 > SVM_Eps && tmpalpha1 < C - SVM_Eps){
            deltab[i] = b1;
        } else if (tmpalpha2 > SVM_Eps && tmpalpha2 < C - SVM_Eps){
            deltab[i] = b2;
        } else {
            deltab[i] = (b1 + b2) * 0.5f;
        }
        delta1[i] = d1;
        delta2[i] = d2;
    }

    best = SVM_Eps;
    a2   = -1;
    for (i = 0; i < n; i++){
        float v = ABS(delta2[i]);
        if (v > best){
            best = v;
            a2   = i;
        }
    }

    if (a2 != -1){
        float h_d1 = delta1[a2];
        float h_d2 = delta2[a2];
        float h_db = deltab[a2];

        alpha[a1] += h_d1;
        alpha[a2] += h_d2;
        *B        += h_db;

        for (i = 0; i < n; i++){
            E[i] += (float)Y[a1] * h_d1 * K[(size_t)a1 * n + i]
                  + (float)Y[a2] * h_d2 * K[(size_t)a2 * n + i]
                  + h_db;
        }
    }
    return a2;
}

/* =============================================================
 * Train
 * ============================================================= */
int Train(Model *M, int Cyc){
    int n, i;
    float *d_alpha, *d_E, *d_delta1, *d_delta2, *d_deltab, *d_cache;
    int a1 = -1, skip_a1 = -1;

    if (!M || !M->Data) return -1;
    n = M->Data->Num;
    if (n <= 0) return 0;

    if (!M->K_ij) K_init(M, NULL);
    if (!M->K_ij) return -1;

    d_alpha  = (float*)malloc((size_t)n * sizeof(float));
    d_E      = (float*)malloc((size_t)n * sizeof(float));
    d_delta1 = (float*)malloc((size_t)n * sizeof(float));
    d_delta2 = (float*)malloc((size_t)n * sizeof(float));
    d_deltab = (float*)malloc((size_t)n * sizeof(float));
    d_cache  = (float*)malloc((size_t)n * sizeof(float));

    if (!d_alpha || !d_E || !d_delta1 || !d_delta2 || !d_deltab || !d_cache){
        free(d_alpha);  free(d_E);
        free(d_delta1); free(d_delta2); free(d_deltab); free(d_cache);
        return -1;
    }

    memcpy(d_alpha, M->Alpha, (size_t)n * sizeof(float));
    memcpy(d_E,     M->E,     (size_t)n * sizeof(float));

    for (i = 0; i < Cyc; i++){
        int a2;
        a1 = select_alpha1_cpu(d_alpha, M->Data->Y, d_E, n, d_cache, M->C, skip_a1);
        if (a1 == -1) break;

        a2 = select_alpha2_cpu(d_alpha, d_E, M->K_ij, M->Data->Y, M->C, n, a1,
                               d_delta1, d_delta2, d_deltab, &M->B);
        if (a2 == -1) skip_a1 = a1;
        else          skip_a1 = -1;
    }

    memcpy(M->Alpha, d_alpha, (size_t)n * sizeof(float));
    memcpy(M->E,     d_E,     (size_t)n * sizeof(float));

    for (i = 0; i < n; i++){
        M->IS_VALID_E[i] = (M->Alpha[i] > SVM_Eps) ? 1 : 0;
    }

    free(d_alpha);  free(d_E);
    free(d_delta1); free(d_delta2); free(d_deltab); free(d_cache);

    return (a1 == -1) ? 0 : -1;
}

/* =============================================================
 * Predict
 * ============================================================= */
float Predict(Model *M, Vec *X){
    float Res;
    int i;
    if (!M || !X) return 0.0f;
    if (!M->Data || !M->Data->Data || M->Data->Num <= 0) return M->B;
    if (M->Data->Data[0]->Dim != X->Dim) return 0.0f;

    Res = M->B;
    for (i = 0; i < M->Data->Num; i++){
        if (M->Alpha[i] > SVM_Eps){
            Res += M->Alpha[i] * Kernel(X, M->Data->Data[i]) * (float)M->Data->Y[i];
        }
    }
    return Res;
}

/* =============================================================
 * benchmark: Pure CPU evaluation (n==1 binary, n>1 One-vs-Rest)
 * ============================================================= */
int benchmark(Model **M, DataSet *testSet, int n){
    int v, i, k;
    int errCount = 0;

    if (!M || !testSet || n < 1) return 0;
    if (!testSet->Data || testSet->Num <= 0) return 0;
    if (!M[0] || !M[0]->Data || !M[0]->Data->Data || M[0]->Data->Num <= 0) return 0;

    v = testSet->Data[0]->Dim;
    if (v != M[0]->Data->Data[0]->Dim) return -1;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:errCount) schedule(dynamic)
#endif
    for (i = 0; i < testSet->Num; i++){
        if (n == 1){
            float p = Predict(M[0], testSet->Data[i]);
            signed char yhat = (p >= 0.0f) ? (signed char)1 : (signed char)-1;
            if (yhat != testSet->Y[i]) errCount++;
        } else {
            int y = (int)testSet->Y[i];
            if (y < 0 || y >= n){
                errCount++;
                continue;
            }
            float max_score = Predict(M[y], testSet->Data[i]);
            for (k = 0; k < n; k++){
                if (k != y && Predict(M[k], testSet->Data[i]) > max_score){
                    errCount++;
                    break;
                }
            }
        }
    }
    return errCount;
}

/* =============================================================
 * Dataset & Vector Utilities
 * ============================================================= */
void copyVec(Vec *src, Vec *dst){
    if (!src || !dst) return;
    dst->Dim = src->Dim;
    dst->X   = (float*)malloc((size_t)src->Dim * sizeof(float));
    if (!dst->X){ dst->Dim = 0; return; }
    memcpy(dst->X, src->X, (size_t)src->Dim * sizeof(float));
}

void extractSV(Model *src, Model *dst){
    int n, i, cnt, idx;
    if (!src || !dst || !src->Data || !src->Alpha) return;

    n = src->Data->Num;
    cnt = 0;
    for (i = 0; i < n; i++){
        if (src->Alpha[i] > SVM_Eps) cnt++;
    }

    dst->Data = (DataSet*)malloc(sizeof(DataSet));
    dst->Data->Data = (cnt > 0) ? (Vec**)malloc((size_t)cnt * sizeof(Vec*)) : NULL;
    dst->Data->Y    = (cnt > 0) ? (signed char*)malloc((size_t)cnt * sizeof(signed char)) : NULL;
    dst->Data->Num  = cnt;

    dst->IS_VALID_E = (cnt > 0) ? (unsigned char*)malloc((size_t)cnt * sizeof(unsigned char)) : NULL;
    dst->Alpha      = (cnt > 0) ? (float*)malloc((size_t)cnt * sizeof(float)) : NULL;
    dst->E          = (cnt > 0) ? (float*)malloc((size_t)cnt * sizeof(float)) : NULL;
    dst->B          = src->B;
    dst->C          = src->C;
    dst->K_ij       = NULL;

    idx = 0;
    for (i = 0; i < n; i++){
        if (src->Alpha[i] > SVM_Eps){
            dst->Data->Data[idx] = (Vec*)malloc(sizeof(Vec));
            if (!dst->Data->Data[idx]){
                dst->Data->Data[idx] = NULL;
                dst->Data->Num = idx;
                return;
            }
            copyVec(src->Data->Data[i], dst->Data->Data[idx]);
            dst->Data->Y[idx]    = src->Data->Y[i];
            dst->E[idx]          = src->E[i];
            dst->IS_VALID_E[idx] = 1;
            dst->Alpha[idx]      = src->Alpha[i];
            idx++;
        }
    }
}

void mergeDataSet(DataSet *src1, DataSet *src2, DataSet *dst){
    int n1, n2, n, i;
    if (!src1 || !src2 || !dst) return;
    n1 = src1->Num;
    n2 = src2->Num;
    n  = n1 + n2;

    dst->Num  = n;
    dst->Data = (Vec**)malloc((size_t)n * sizeof(Vec*));
    dst->Y    = (signed char*)malloc((size_t)n * sizeof(signed char));

    for (i = 0; i < n1; i++){
        dst->Data[i] = (Vec*)malloc(sizeof(Vec));
        copyVec(src1->Data[i], dst->Data[i]);
        dst->Y[i] = src1->Y[i];
    }
    for (i = 0; i < n2; i++){
        dst->Data[n1 + i] = (Vec*)malloc(sizeof(Vec));
        copyVec(src2->Data[i], dst->Data[n1 + i]);
        dst->Y[n1 + i] = src2->Y[i];
    }
}

void dataSetSlice(DataSet *src, DataSet *dst, int beg, int end){
    int n, i;
    if (!src || !dst) return;
    if (beg < 0) beg = 0;
    if (end > src->Num) end = src->Num;
    n = end - beg;
    if (n < 0) n = 0;

    dst->Num  = n;
    dst->Data = (Vec**)malloc((size_t)n * sizeof(Vec*));
    dst->Y    = (signed char*)malloc((size_t)n * sizeof(signed char));

    for (i = 0; i < n; i++){
        dst->Data[i] = (Vec*)malloc(sizeof(Vec));
        copyVec(src->Data[i + beg], dst->Data[i]);
        dst->Y[i] = src->Y[i + beg];
    }
}

/* =============================================================
 * Memory Management
 * ============================================================= */
void SVMFreeVec(Vec *V){
    SVMCleanUpVec(V);
    free(V);
}

void SVMFreeDataSet(DataSet *D){
    SVMCleanUpDataSet(D);
    free(D);
}

void SVMFreeModel(Model *M){
    SVMCleanUpModel(M);
    free(M);
}

void SVMCleanUpModel(Model *M){
    if (!M) return;
    if (M->Data){
        SVMFreeDataSet(M->Data);
        M->Data = NULL;
    }
    if (M->Alpha){      free(M->Alpha);      M->Alpha = NULL; }
    if (M->E){          free(M->E);          M->E = NULL; }
    if (M->IS_VALID_E){ free(M->IS_VALID_E); M->IS_VALID_E = NULL; }
    if (M->K_ij){       free(M->K_ij);       M->K_ij = NULL; }
}

void SVMCleanUpDataSet(DataSet *D){
    int i;
    if (!D) return;
    if (D->Data){
        for (i = 0; i < D->Num; i++){
            if (D->Data[i]) SVMFreeVec(D->Data[i]);
        }
        free(D->Data);
        D->Data = NULL;
    }
    if (D->Y){
        free(D->Y);
        D->Y = NULL;
    }
    D->Num = 0;
}

void SVMCleanUpVec(Vec *V){
    if (!V) return;
    if (V->X){
        free(V->X);
        V->X = NULL;
    }
    V->Dim = 0;
}
