#include<iostream>
#include<sstream>
#include<csignal>
namespace svm{
    #include"SVM/cuSVM.h"
}
static volatile std::sig_atomic_t is_running=1;
void interruptHandler(int signal){
    (void)signal;
    is_running=0;
}
int main(int argc,char **argv){
    if (argc<4){
        std::cout<<"Too few argument."<<std::endl;
        return -1;
    }
    std::signal(SIGINT,interruptHandler);
    std::stringstream ss;
    ss<<argv[1]<<std::endl<<argv[2]<<std::endl<<argv[3]<<std::endl;
    float c;
    int y,step;
    ss>>y>>c>>step;
    ss.clear();
    ss<<"database/database"<<y<<".db"<<std::endl;
    char modelname[256],dbname[256];
    ss>>dbname;
    ss.clear();
    ss<<"models/SVM"<<y<<".pth"<<std::endl;
    ss>>modelname;
    ss.clear();
    svm::Model M;
    M.K_ij=NULL;
    M.Data=NULL;
    M.IS_VALID_E=NULL;
    M.E=NULL;
    M.Alpha=NULL;
    if (svm::LoadState(&M,modelname,dbname)!=0){
        svm::SetUpData(&M,dbname);
        svm::ModelInit(&M,c);
    }
    int label=-1;
    while (is_running&&label!=0){
        label=svm::Train(&M,step);
        svm::SaveState(&M,modelname);
    }
    svm::SaveState(&M,modelname);
    svm::SVMCleanUpModel(&M,M.Data->Num);
    return 0;
}
