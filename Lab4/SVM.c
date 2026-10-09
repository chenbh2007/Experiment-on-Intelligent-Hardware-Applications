#include"SVM.h"
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<signal.h>
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
static inline float Kernel(Vec *v1,Vec *v2){
    int i,d=v1->Dim;
    float tmp=0,t=0;
    for (i=0;i<d;i++){
        tmp=v1->X[i]-v2->X[i];
        t+=tmp*tmp;
    }
    return exp(-SVM_Gamma*t);
}
void SetUpData(Model *M,const char *file_name){
    FILE *f=fopen(file_name,"r");
    if (f){
        int n,v;
        M->Data=(DataSet *)malloc(sizeof(DataSet));
        fscanf(f,"%d%d",&n,&v);
        M->Data->Num=n;
        M->Data->Data=(Vec **)malloc(n*sizeof(Vec *));
        M->Data->Y=(signed char *)malloc(n*sizeof(signed char));
        int i,tmp_y;
        for (i=0;i<n;i++){
            M->Data->Data[i]=(Vec *)malloc(sizeof(Vec));
            M->Data->Data[i]->Dim=v;
            M->Data->Data[i]->X=(float *)malloc(v*sizeof(float));
            int j;
            for (j=0;j<v;j++)
                fscanf(f,"%f",&M->Data->Data[i]->X[j]);
            fscanf(f,"%d",&tmp_y);
            M->Data->Y[i]=(signed char)tmp_y;
        }
        fclose(f);
    }
    return;
}
void ModelInit(Model *M,float _C){
    if (!M->Data){
        printf("Model Init Failed: Data Set hasn't Set up.\n");
        return;
    }
    int n=M->Data->Num;
    M->IS_VALID_E=(unsigned char*)malloc(n*sizeof(unsigned char));
    M->E=(float *)malloc(n*sizeof(float));
    M->B=0;
    M->C=_C;
    M->K_ij=NULL;
    int i;
    M->Alpha=(float *)malloc(n*sizeof(float));
    for (i=0;i<n;i++){
        M->Alpha[i]=0;
        M->E[i]=-M->Data->Y[i];
    }
    memset(M->IS_VALID_E,0,n*sizeof(unsigned char));
    return;
}
void K_init(Model *M){
    int n=M->Data->Num;
    M->K_ij=malloc(n*sizeof(float *));
    int i,j;
    for (i=0;i<n;i++) M->K_ij[i]=(float *)malloc(n*sizeof(float));
    for (i=0;i<n;i++){
        M->K_ij[i][i]=1.00;
        for (j=0;j<i;j++){
            M->K_ij[i][j]=Kernel(M->Data->Data[i],M->Data->Data[j]);
            M->K_ij[j][i]=M->K_ij[i][j];
        }
    }
    return;
}
static inline float loss_fun(Model *M,int k){
    int n=M->Data->Num;
    int i;
    float tmp=M->B;
    for (i=0;i<n;i++)
        tmp+=M->Data->Y[i]*M->Alpha[i]*M->K_ij[i][k];
    return tmp-M->Data->Y[k];
}
static inline float loss_fun_online(Model *M,int k){
    int n=M->Data->Num;
    int i;
    float tmp=M->B;
    for (i=0;i<n;i++)
        if (M->Alpha[i]>SVM_Eps)
            tmp+=M->Data->Y[i]*M->Alpha[i]*Kernel(M->Data->Data[i],M->Data->Data[k]);
    return tmp-M->Data->Y[k];
}

static inline int select_alpha1(Model *M,unsigned char if_full){
    float tmp_e;
    int i;
    int n=M->Data->Num;
    int index=rand()%n;
    for (i=0;i<n;i++){
        if ((M->Alpha[index]>SVM_Eps&&M->Alpha[index]<M->C-SVM_Eps)||if_full){
            if (!M->IS_VALID_E[index])
                tmp_e=loss_fun(M,index);
            else tmp_e=M->E[index];
            float tmp_alpha=M->Alpha[index];
            if ((tmp_alpha<M->C-SVM_Eps&&M->Data->Y[index]*tmp_e<-SVM_Tol)||(tmp_alpha>SVM_Eps&&M->Data->Y[index]*tmp_e>SVM_Tol)){
                if (!M->IS_VALID_E[index]){
                    M->IS_VALID_E[index]=1;
                    M->E[index]=tmp_e;
                }
                return index;
            }
        }
        index++;
        index=index%n;
    }
    return -1;
}
static inline int select_alpha1_online(Model *M,unsigned char if_full){
    float tmp_e;
    int i;
    int n=M->Data->Num;
    int index=n-1;
    for (i=0;i<n;i++){
        if ((M->Alpha[index]>SVM_Eps&&M->Alpha[index]<M->C-SVM_Eps)||if_full){
            if (!M->IS_VALID_E[index])
                tmp_e=loss_fun_online(M,index);
            else tmp_e=M->E[index];
            float tmp_alpha=M->Alpha[index];
            if ((tmp_alpha<M->C-SVM_Eps&&M->Data->Y[index]*tmp_e<-SVM_Tol)||(tmp_alpha>SVM_Eps&&M->Data->Y[index]*tmp_e>SVM_Tol)){
                if (!M->IS_VALID_E[index]){
                    M->IS_VALID_E[index]=1;
                    M->E[index]=tmp_e;
                }
                return index;
            }
        }
        index--;
    }
    return -1;
}


static inline void update_E(Model *M,float *d1,float *d2,float *db,int *a1,int *a2){
    int i,n=M->Data->Num;
    for (i=0;i<n;i++)
        if (M->IS_VALID_E[i])
            M->E[i]+=M->Data->Y[*a1]*(*d1)*M->K_ij[i][*a1]+M->Data->Y[*a2]*(*d2)*M->K_ij[i][*a2]+*db;
    return;
}
static inline void update_E_online(Model *M,float *d1,float *d2,float *db,int *a1,int *a2){
    int i,n=M->Data->Num;
    for (i=0;i<n;i++)
        if (M->IS_VALID_E[i])
            M->E[i]+=M->Data->Y[*a1]*(*d1)*Kernel(M->Data->Data[i],M->Data->Data[*a1])+M->Data->Y[*a2]*(*d2)*Kernel(M->Data->Data[i],M->Data->Data[*a2])+*db;
    return;
}

void SaveState(Model *M,const char *file_name){
    FILE *f=fopen(file_name,"w");
    if (f){
        fprintf(f,"%d %.8f %.8f\n",M->Data->Num,M->B,M->C);
        int i;
        for (i=0;i<M->Data->Num;i++)
            fprintf(f,"%.8f %d %d %.8f\n",M->Alpha[i],M->Data->Y[i],M->IS_VALID_E[i],M->E[i]);
        fclose(f);
        printf("Save Successfully.\n");
    }
    else
        printf("Save Failed!\n");
    return;
}
int LoadState(Model *M,const char *state_file,const char *Data_set){
    FILE *f1=fopen(state_file,"r"),*f2=fopen(Data_set,"r");
    if (!f1){
        printf("State File Not Found.\n");
        if (f2) fclose(f2);
        return -1;
    }
    if (!f2){
        printf("Data Set Not Found.\n");
        if (f1) fclose(f1);
        return -2;
    }
    fclose(f2);
    SetUpData(M,Data_set);
    int n;
    float b,c;
    fscanf(f1,"%d%f%f",&n,&b,&c);
    if (n!=M->Data->Num){
        printf("Data Set doesn't Match the State File.\n");
        fclose(f1);
        SVMFreeDataSet(M->Data);
        M->Data=NULL;
        return -3;
    }
    ModelInit(M,c);
    M->B=b;
    int tmp_y,i,tmp_valid_e;
    for (i=0;i<M->Data->Num;i++){
        fscanf(f1,"%f%d%d%f",&M->Alpha[i],&tmp_y,&tmp_valid_e,&M->E[i]);
        M->IS_VALID_E[i]=(unsigned char)tmp_valid_e;
    }
    printf("Load Successfully.\n");
    fclose(f1);
    return 0;
}
int Train(Model *M,int Cyc){
    if (!M->K_ij) K_init(M);
    int i,n=M->Data->Num;
    int alpha1=select_alpha1(M,0);
    if (alpha1==-1) 
        alpha1=select_alpha1(M,1);
    if (alpha1==-1){
        return -1;
    }
    unsigned char updated=0;
    for (i=2;i<Cyc+2;i++){
        if (Cyc==-1) i--;
        int j=rand()%n;
        int k;
        unsigned char entire=0;
        updated=0;
loop:
        for (k=0;k<n&&!updated;k++){
            if (((M->Alpha[j]>SVM_Eps&&M->Alpha[j]<M->C-SVM_Eps)||entire)&&j!=alpha1&&(M->K_ij[alpha1][j]-1>SVM_Eps||M->K_ij[alpha1][j]-1<-SVM_Eps)){
                unsigned char tmp_valid_e=M->IS_VALID_E[j];
                if (!M->IS_VALID_E[j]){
                    M->IS_VALID_E[j]=1;
                    M->E[j]=loss_fun(M,j);
                }
                float delta2=M->Data->Y[j]*(M->E[alpha1]-M->E[j])/(2-2*M->K_ij[alpha1][j]);
                float L,H;
                if (M->Data->Y[alpha1]*M->Data->Y[j]<0){
                    L=MAX(0,M->Alpha[j]-M->Alpha[alpha1]);
                    H=MIN(M->C,M->C+M->Alpha[j]-M->Alpha[alpha1]);
                }
                else{
                    L=MAX(0,M->Alpha[j]+M->Alpha[alpha1]-M->C);
                    H=MIN(M->C,M->Alpha[j]+M->Alpha[alpha1]);
                }
                if (L+SVM_Eps>=H){
                    M->IS_VALID_E[j]=tmp_valid_e;
                    j=(j+1)%n;
                    continue;
                }
                float alpha2_new=MIN(MAX(M->Alpha[j]+delta2,L),H);
                delta2=alpha2_new-M->Alpha[j];
                if (delta2>SVM_Eps||delta2<-SVM_Eps){
                    updated=1;
                    float delta1=-M->Data->Y[j]*M->Data->Y[alpha1]*delta2;
                    M->Alpha[j]=alpha2_new;
                    M->Alpha[alpha1]+=delta1;
                    float deltab;
                    if (M->Alpha[alpha1]>SVM_Eps&&M->Alpha[alpha1]<M->C-SVM_Eps)
                        deltab=-M->E[alpha1]-M->Data->Y[alpha1]*delta1-M->Data->Y[j]*delta2*M->K_ij[alpha1][j];
                    else
                        if (alpha2_new>SVM_Eps&&alpha2_new<M->C-SVM_Eps)
                            deltab=-M->E[j]-M->Data->Y[alpha1]*delta1*M->K_ij[alpha1][j]-M->Data->Y[j]*delta2;
                            else
                                deltab=-(M->E[alpha1]+M->E[j]+(M->Data->Y[alpha1]*delta1+M->Data->Y[j]*delta2)*(1+M->K_ij[alpha1][j]))*0.5;
                    M->B+=deltab;
                    update_E(M,&delta1,&delta2,&deltab,&alpha1,&j);
                    if (alpha2_new<SVM_Eps||alpha2_new>M->C-SVM_Eps) M->IS_VALID_E[j]=0;
                    if (M->Alpha[alpha1]<SVM_Eps||M->Alpha[alpha1]>M->C-SVM_Eps) M->IS_VALID_E[alpha1]=0;
                }
                else
                    M->IS_VALID_E[j]=tmp_valid_e;
            }
            j=(j+1)%n;
        }
        if (!updated&&!entire){
            entire=1;
            j=rand()%n;
            goto loop;
        }
        if (!updated){
            if (M->Alpha[alpha1]<SVM_Eps||M->Alpha[alpha1]>M->C-SVM_Eps) M->IS_VALID_E[alpha1]=0;
        }
        alpha1=select_alpha1(M,0);
        if (alpha1==-1)  
            alpha1=select_alpha1(M,1);
        if (alpha1==-1){
            return -1;
        }
    }
    return 0;
}
int TrainOnline(Model *M,int Cyc){
    int i,n=M->Data->Num;
    int alpha1=select_alpha1_online(M,0);
    if (alpha1==-1) 
        alpha1=select_alpha1_online(M,1);
    if (alpha1==-1){
        return -1;
    }
    unsigned char updated=0;
    for (i=2;i<Cyc+2;i++){
        if (Cyc==-1) i--;
        int j=rand()%n;
        int k;
        unsigned char entire=0;
        updated=0;
loop:
        for (k=0;k<n&&!updated;k++){
            if (((M->Alpha[j]>SVM_Eps&&M->Alpha[j]<M->C-SVM_Eps)||entire)&&j!=alpha1){
                float tmpk=Kernel(M->Data->Data[alpha1],M->Data->Data[j]);
                if (tmpk-1<SVM_Eps&&tmpk-1>-SVM_Eps){
                    j++;
                    j=j%n;
                    continue;
                }
                unsigned char tmp_valid_e=M->IS_VALID_E[j];
                if (!M->IS_VALID_E[j]){
                    M->IS_VALID_E[j]=1;
                    M->E[j]=loss_fun_online(M,j);
                }
                float delta2=M->Data->Y[j]*(M->E[alpha1]-M->E[j])/(2-2*tmpk);
                float L,H;
                if (M->Data->Y[alpha1]*M->Data->Y[j]<0){
                    L=MAX(0,M->Alpha[j]-M->Alpha[alpha1]);
                    H=MIN(M->C,M->C+M->Alpha[j]-M->Alpha[alpha1]);
                }
                else{
                    L=MAX(0,M->Alpha[j]+M->Alpha[alpha1]-M->C);
                    H=MIN(M->C,M->Alpha[j]+M->Alpha[alpha1]);
                }
                if (L+SVM_Eps>=H){
                    M->IS_VALID_E[j]=tmp_valid_e;
                    j=(j+1)%n;
                    continue;
                }
                float alpha2_new=MIN(MAX(M->Alpha[j]+delta2,L),H);
                delta2=alpha2_new-M->Alpha[j];
                if (delta2>SVM_Eps||delta2<-SVM_Eps){
                    updated=1;
                    float delta1=-M->Data->Y[j]*M->Data->Y[alpha1]*delta2;
                    M->Alpha[j]=alpha2_new;
                    M->Alpha[alpha1]+=delta1;
                    float deltab;
                    if (M->Alpha[alpha1]>SVM_Eps&&M->Alpha[alpha1]<M->C-SVM_Eps)
                        deltab=-M->E[alpha1]-M->Data->Y[alpha1]*delta1-M->Data->Y[j]*delta2*tmpk;
                    else
                        if (alpha2_new>SVM_Eps&&alpha2_new<M->C-SVM_Eps)
                            deltab=-M->E[j]-M->Data->Y[alpha1]*delta1*tmpk-M->Data->Y[j]*delta2;
                            else
                                deltab=-(M->E[alpha1]+M->E[j]+(M->Data->Y[alpha1]*delta1+M->Data->Y[j]*delta2)*(1+tmpk))*0.5;
                    M->B+=deltab;
                    update_E_online(M,&delta1,&delta2,&deltab,&alpha1,&j);
                    if (alpha2_new<SVM_Eps||alpha2_new>M->C-SVM_Eps) M->IS_VALID_E[j]=0;
                    if (M->Alpha[alpha1]<SVM_Eps||M->Alpha[alpha1]>M->C-SVM_Eps) M->IS_VALID_E[alpha1]=0;
                }
                else
                    M->IS_VALID_E[j]=tmp_valid_e;
            }
            j=(j+1)%n;
        }
        if (!updated&&!entire){
            entire=1;
            j=rand()%n;
            goto loop;
        }
        if (!updated){
            if (M->Alpha[alpha1]<SVM_Eps||M->Alpha[alpha1]>M->C-SVM_Eps) M->IS_VALID_E[alpha1]=0;
        }
        alpha1=select_alpha1_online(M,0);
        if (alpha1==-1)  
            alpha1=select_alpha1_online(M,1);
        if (alpha1==-1){
            return -1;
        }
    }
    return 0;
}

float Predict(Model *M,Vec *X){
    float Res=M->B;
    int i;
    if (M->Data->Data[0]->Dim!=X->Dim) return 0;
    for (i=0;i<M->Data->Num;i++)
        if (M->Alpha[i]>SVM_Eps)
            Res+=M->Alpha[i]*Kernel(X,M->Data->Data[i])*M->Data->Y[i];
    return Res;
}
void SVMFreeVec(Vec *V){
    SVMCleanUpVec(V);
    free(V);
    return;
}
void SVMFreeDataSet(DataSet *D){
    SVMCleanUpDataSet(D);
    free(D);
    return;
}
void SVMFreeModel(Model *M,int n){
    SVMCleanUpModel(M,n);
    free(M);
    return;
}
void SVMCleanUpModel(Model *M,int n){
    if (M->Data){
        SVMFreeDataSet(M->Data);
        M->Data=NULL;
    }
    if (M->Alpha){
        free(M->Alpha);
        M->Alpha=NULL;
    }
    if (M->IS_VALID_E){
        free(M->IS_VALID_E);
        M->IS_VALID_E=NULL;
    }
    if (M->E){
        free(M->E);
        M->E=NULL;
    }
    int i;
    if (M->K_ij){
        for (i=0;i<n;i++)
            if (M->K_ij[i]) free(M->K_ij[i]);
        free(M->K_ij);
        M->K_ij=NULL;
    }
    return;
}
void SVMCleanUpDataSet(DataSet *D){
    int i;
    if (D->Data){
        for (i=0;i<D->Num;i++){
            if (D->Data[i])
                SVMFreeVec(D->Data[i]);
        }
        free(D->Data);
        D->Data=NULL;
    }
    if (D->Y){
        free(D->Y);
        D->Y=NULL;
    }
    return;
}
void SVMCleanUpVec(Vec *V){
    if (V->X){
        free(V->X);
        V->X=NULL;
    }
    V->Dim=0;
    return;
}


