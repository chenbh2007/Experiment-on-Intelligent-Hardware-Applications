#ifndef SVM_H
#define SVM_H

#define Tol 1E-3
#define Eps 1E-5
#define Gamma 1E-1

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
    float **K_ij;
    unsigned char *IS_VALID_E;
    float *E;
}Model;

void SetUpData(Model *M,const char *FileName);
void ModelInit(Model *M,float _C);
void SaveState(Model *M,const char *FileName);
int LoadState(Model *M,const char *StateFile,const char *DataFile);
int Train(Model *M,int Step);
int TrainOnline(Model *M,int Step);
void K_init(Model *M);
float Predict(Model *M,Vec *X);

#endif
