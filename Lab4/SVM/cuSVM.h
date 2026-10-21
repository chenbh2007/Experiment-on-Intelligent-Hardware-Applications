#ifndef cuSVM_H
#define cuSVM_H

#ifdef __cplusplus
extern "C"{
#endif
#ifndef SVM_Tol
#define SVM_Tol 1E-3
#endif
#ifndef SVM_Eps
#define SVM_Eps 1E-5
#endif
#ifndef SVM_Gamma
#define SVM_Gamma 1E-1
#endif

typedef struct{
    float *X;
    int Dim;
}Vec;

typedef struct{
    Vec **Data;
    int Num;
    signed char *Y;
}DataSet;

typedef struct{
    DataSet *Data;
    float *Alpha;
    float B;
    float C;
    float *K_ij;
    unsigned char *IS_VALID_E;
    float *E;
}Model;

int SetUpData(DataSet *db,const char *database,const char *labelbase);
void ModelInit(Model *M,float _C);

void SaveState(Model *M,const char *FileName,const char *labname);
int LoadState(Model *M,const char *StateFile,const char *labfile);

int Train(Model *M,int Step);
void K_init(Model *M,float *d_K);
float Predict(Model *M,Vec *X);
int benchmark(Model **M,DataSet *test,int n);

void copyVec(Vec *src,Vec *dst);
void extractSV(Model *src,Model *dst);
void mergeDataSet(DataSet *src1,DataSet *src2,DataSet *dst);
void dataSetSlice(DataSet *src,DataSet *dst,int beg,int end);

void SVMFreeVec(Vec *V);
void SVMFreeDataSet(DataSet *D);
void SVMFreeModel(Model *M);
void SVMCleanUpModel(Model *M);
void SVMCleanUpDataSet(DataSet *D);
void SVMCleanUpVec(Vec *V);
#ifdef __cplusplus
}
#endif
#endif
