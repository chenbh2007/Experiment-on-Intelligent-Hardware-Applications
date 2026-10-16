#include<iostream>
#include<cstdio>
#include<cstdlib>
namespace svm{
    #include"SVM/SVM.h"
}
#include<fstream>
#include<sstream>
#include<vector>
int main(int argc,char **argv){
    if (argc<2){
        std::cout<<"Too few arguments."<<std::endl;
    }
    std::ifstream raw(argv[1]);
    svm::Model models[10];
    for (int i=0;i<10;i++){
        char modelname[256],dbname[256];
        std::snprintf(modelname,sizeof(modelname),"models/SVM%d.pth",i);
        std::snprintf(dbname,sizeof(dbname),"database/database%d.db",i);
        LoadState(&models[i],modelname,dbname);
    }
    svm::Vec testVec;
    int err=0,tot=0;
    testVec.Dim=models[0].Data->Data[0]->Dim;
    testVec.X=(float *)malloc(testVec.Dim*sizeof(float));
    while (raw>>testVec.X[0]){
        for (int i=1;i<testVec.Dim;i++){
            raw>>testVec.X[i];
        }
        int y;
        raw>>y;
        float max=svm::Predict(&models[0],&testVec);
        float idx=0;
        for (int i=1;i<10;i++){
            float tmp=svm::Predict(&models[i],&testVec);
            if (tmp>max){
                max=tmp;
                idx=i;
            }
        }
        tot++;
        if (idx!=y) err++;
    }
    std::cout<<"Error: "<<err<<" Total: "<<tot<<std::endl;
    raw.close();
    return 0;
}
