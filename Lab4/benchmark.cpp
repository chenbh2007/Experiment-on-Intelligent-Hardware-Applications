#include<iostream>
#include<cstdint>
#include<cstring>
namespace svm{
    #include"SVM/cuSVM.h"
}
#include<fstream>
#include<byteswap.h>
int main(int argc,char **argv){
    if (argc<4){
        std::cout<<"Too few arguments. "<<std::endl;
        return -1;
    }
    std::ifstream modellist(argv[1]);
    std::ifstream data(argv[2],std::ios::binary);
    std::ifstream label(argv[3],std::ios::binary);
    if (!modellist.is_open()){
        std::cout<<"Models don't exist."<<std::endl;
        return -1;
    }
    if (!data.is_open()||!label.is_open()){
        std::cout<<"Fail to load database."<<std::endl;
        return -1;
    }
    char cache[4];
    std::uint32_t s1,s2; 
    data.read(cache,sizeof(cache));
    if (cache[2]!=0x0D||cache[3]!=0x02){
        std::cout<<"Invalid data file."<<std::endl;
        return -1;
    }
    label.read(cache,sizeof(cache));
    if (cache[2]!=0x08||cache[3]!=0x01){
        std::cout<<"Invalid data file."<<std::endl;
        return -1;
    }
    data.read((char*)&s1,sizeof(s1));
    label.read((char*)&s2,sizeof(s2));
    s2=bswap_32(s2);
    if (s1!=s2){
        std::cout<<"Data doesn't fit the label. "<<std::endl;
        return -1;
    }
    svm::DataSet testset={};
    testset.Num=s1;
    testset.Data=(svm::Vec**)malloc(s1*sizeof(svm::Vec*));
    testset.Y=(signed char*)malloc(s1*sizeof(signed char));
    data.read((char*)&s2,sizeof(s2));
    for (int i=0;i<(int)s1;i++){
        testset.Data[i]=(svm::Vec*)malloc(sizeof(svm::Vec));
        testset.Data[i]->X=(float*)malloc((long)s2*sizeof(float));
        data.read((char*)testset.Data[i]->X,s2*sizeof(float));
        testset.Data[i]->Dim=s2;
    }
    label.read((char*)testset.Y,s1*sizeof(signed char));
    data.close();
    label.close();
    std::cout<<"Read test set successfully. "<<std::endl;
    svm::Model models,SV[10];
    memset(&models,0,sizeof(models));
    memset(SV,0,sizeof(SV));
    for (int i=0;i<10;i++){
        std::string state,lab;
        std::getline(modellist,state);
        std::getline(modellist,lab);
        if (svm::LoadState(&models,state.data(),lab.data())==-1){
            std::cout<<"Model "<<i<<" doesn't exist. "<<std::endl;
            return -1;
        }
        svm::extractSV(&models,&SV[i]);
        svm::SVMCleanUpModel(&models);
    }
    int err=0;
    for (int i=0;i<testset.Num;i++){
        float max=svm::Predict(&SV[0],testset.Data[i]),second=-1.0f;
        int idx=0,sidx=-1;
        for (int j=1;j<10;j++){
            float weight=svm::Predict(&SV[j],testset.Data[i]);
            if (weight>max){
                second=max;
                max=weight;
                sidx=idx;
                idx=j;
            }
        }
        if (idx!=(int)testset.Y[i]){
            err++;
            std::cout<<"Error "<<err<<" at data "<<i<<": Expect "<<(int)testset.Y[i]<<" but "<<idx<<" received. The second one is "<<sidx<<" with a weight of "<<second<<"."<<std::endl;
        }
    }
    std::cout<<"Error rate: "<<(float)err/(float)testset.Num<<std::endl;
    for (int i=0;i<10;i++){
        svm::SVMCleanUpModel(&SV[i]);
    }
    svm::SVMCleanUpDataSet(&testset);
    return 0;
}
