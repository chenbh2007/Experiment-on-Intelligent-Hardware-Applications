#include<iostream>
#include<sstream>
#include<csignal>
#include<cstring>
namespace svm{
    #include"SVM/cuSVM.h"
}
static volatile std::sig_atomic_t is_running=1;
void interruptHandler(int signal){
    (void)signal;
    is_running=0;
}
int main(int argc,char **argv){
    if (argc<5){
        std::cout<<"Too few argument."<<std::endl;
        return -1;
    }
    std::signal(SIGINT,interruptHandler);
    svm::DataSet db,batch;
    svm::Model M;
    std::memset(&M,0,sizeof(M));
    int init=0;
    if(svm::SetUpData(&db,argv[1],argv[2])==-1){
        return -1;
    }
    if(svm::LoadState(&M,argv[3],argv[4])==-1){
        init=1;
    }
    svm::Model SV;
    std::memset(&SV,0,sizeof(SV));
    int batchsize=db.Num/5;
    float tmp_b=0.0f;
    int first=1;
    int cold=1;
    while (is_running&&(first||(tmp_b-M.B)>SVM_Tol/5||(tmp_b-M.B)<-SVM_Tol/5)){
        tmp_b=M.B;
        for (int i=0;i<5&&is_running;i++){
            if (init){
                M.Data=(svm::DataSet*)malloc(sizeof(svm::DataSet));
                svm::dataSetSlice(&db,M.Data,i*batchsize,(i+1)*batchsize);
                svm::ModelInit(&M,1.0f);
                init=0;
                std::cout<<"Init Finished."<<std::endl;
            }
            else
            if (cold!=1)
            {
                svm::extractSV(&M,&SV);
                std::cout<<"Number of SVs: "<<SV.Data->Num<<std::endl;
                svm::SVMCleanUpModel(&M);
                M.Data=(svm::DataSet*)malloc(sizeof(svm::DataSet));
                svm::dataSetSlice(&db,&batch,i*batchsize,(i+1)*batchsize);
                svm::mergeDataSet(SV.Data,&batch,M.Data);
                svm::ModelInit(&M,1.0f);
            }
            else cold=0;
            while (is_running&&svm::Train(&M,1000)==-1){
                svm::SaveState(&M,argv[3],argv[4]);
            }
            std::cout<<"Batch "<<i<<" complete. "<<std::endl;
            std::cout<<"delta b"<<M.B-tmp_b<<std::endl;
            svm::SaveState(&M,argv[3],argv[4]);
            svm::SVMCleanUpDataSet(&batch);
            if (SV.Data!=nullptr) svm::SVMCleanUpModel(&SV);
        }
        if (first) first=0;
    }
    svm::SVMCleanUpModel(&M);
    svm::SVMCleanUpDataSet(&db);
    return 0;
}
