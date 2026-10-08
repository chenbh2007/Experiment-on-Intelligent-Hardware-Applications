#include<stdio.h>
#include<stdlib.h>
#include"SVM.h"
Model *BaseModel;
int main(){
    BaseModel=(Model *)malloc(sizeof(Model));
    LoadState(BaseModel,"checkpoint.pth","SvmDataSet");
    int v=BaseModel->Data->Data[0]->Dim;
    while (1){
       int i;
       Vec *X0=(Vec *)malloc(sizeof(Vec));
       X0->X=(double *)malloc(v*sizeof(double));
       X0->Dim=v;
       for (i=0;i<v;i++)
           scanf("%lf",&X0->X[i]);
       printf("%.2lf\n",Predict(BaseModel,X0));
    }
}
