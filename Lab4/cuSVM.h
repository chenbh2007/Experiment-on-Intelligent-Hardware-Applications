#ifndef cuSVM_H
#define cuSVM_H

#ifdef __cplusplus
extern "C"{
#endif

#define SVM_Tol 1E-3
#define SVM_Eps 1E-5
#define SVM_Gamma 1E-1

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

void SetUpData(Model *M,const char *FileName);
void ModelInit(Model *M,float _C);
void SaveState(Model *M,const char *FileName);
int LoadState(Model *M,const char *StateFile,const char *DataFile);
int Train(Model *M,int Step);
void K_init(Model *M,float *d_K);
float Predict(Model *M,Vec *X);
void SVMFreeVec(Vec *V);
void SVMFreeDataSet(DataSet *D);
void SVMFreeModel(Model *M,int n);
void SVMCleanUpModel(Model *M,int n);
void SVMCleanUpDataSet(DataSet *D);
void SVMCleanUpVec(Vec *V);
#ifdef __cplusplus
}
#endif
#endif
