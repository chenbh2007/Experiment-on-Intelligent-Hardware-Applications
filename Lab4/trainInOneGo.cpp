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
    if (argc<6){
        std::cout<<"Too few argument."<<std::endl;
        return -1;
    }
    std::signal(SIGINT,interruptHandler);
    svm::Model m={};
    std::stringstream ss;
    ss<<argv[5];
    float C;
    ss>>C;
    if (svm::LoadState(&m,argv[3],argv[4])==-1){
    	m.Data=(svm::DataSet*)malloc(sizeof(svm::DataSet));
	svm::SetUpData(m.Data,argv[1],argv[2]);
	svm::ModelInit(&m,C);
    }
    m.C=C;
    while (is_running&&svm::Train(&m,2000)){
	svm::SaveState(&m,argv[3],argv[4]);
	std::cout<<"Save Successfully."<<std::endl;
    }
    svm::SaveState(&m,argv[3],argv[4]);
    std::cout<<"Save Successfully."<<std::endl;
    svm::SVMCleanUpModel(&m,m.Data->Num);
    return 0;
}
