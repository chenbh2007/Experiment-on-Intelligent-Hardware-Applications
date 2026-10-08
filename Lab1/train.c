#include"SVM.h"
#include<stdio.h>
Model SVM_model;
int main(){
    int tmp;
    scanf("%d",&tmp);
    char D_name[20],S_name[20];
    snprintf(D_name,sizeof(D_name),"DataBase%d.db",tmp);
    snprintf(S_name,sizeof(S_name),"checkpointRPS%d.pth",tmp);
    if(LoadState(&SVM_model,S_name,D_name)<0){
        SetUpData(&SVM_model,D_name);
        printf("Data Set Up\n");
        ModelInit(&SVM_model,2);
        printf("Init finished.\n");
    }
    int b=0;
    while (b==0){
        b=Train(&SVM_model,10000);
        SaveState(&SVM_model,S_name);
    }
    return 0;
}
