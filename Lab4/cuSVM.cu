#include"cuSVM.h"
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<signal.h>
#include<cuda_runtime.h>
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define ABS(a) ((a)>0 ? (a):(-a))

static inline float Kernel(Vec *v1,Vec *v2){
    int d=v1->Dim;
    int i;
    float dist=0.0f;
    for (i=0;i<d;i++){
        float dx=v1->X[i]-v2->X[i];
        dist+=dx*dx;
    }
    return expf(-SVM_Gamma*dist);
}

__device__ unsigned int g_block_done_count=0;

__device__ inline void warpReduceMax(float *val,int *idx){
    int offset;
    for (offset=16;offset>0;offset/=2){
        float other_val=__shfl_down_sync(0xffffffff,*val,offset);
        int other_idx=__shfl_down_sync(0xffffffff,*idx,offset);
        if (other_val>*val){
            *val=other_val;
            *idx=other_idx;
        }
    }
}
__global__ void gridReduceMax(const float tol,const unsigned char is_abs,const float *d_in,int n,float *d_block_max,int *d_block_idx,float *d_max,int *d_idx){
    __shared__ float s_max[32];
    __shared__ int s_idx[32];
    __shared__ bool s_is_last_block;
    int tid=threadIdx.x;
    int lane=tid%32;
    int wid=tid/32;
    float my_max=tol;
    int my_idx=-1;
    int i;
    for (i=blockIdx.x*blockDim.x+tid;i<n;i+=blockDim.x*gridDim.x){
        if (is_abs){
            if (ABS(d_in[i])>my_max){
                my_max=ABS(d_in[i]);
                my_idx=i;
            }
        }
        else{
            if (d_in[i]>my_max){
                my_max=d_in[i];
                my_idx=i;
            }
        }
    }
    warpReduceMax(&my_max,&my_idx);
    if (lane==0){
        s_max[wid]=my_max;
        s_idx[wid]=my_idx;
    }
    __syncthreads();
    if (wid==0){
        my_max=(tid<blockDim.x/32)?s_max[lane]:tol;
        my_idx=(tid<blockDim.x/32)?s_idx[lane]:-1;
        warpReduceMax(&my_max,&my_idx);
        if (tid==0){
            d_block_max[blockIdx.x]=my_max;
            d_block_idx[blockIdx.x]=my_idx;
        }
    }
    __threadfence();
    if (tid==0){
        unsigned int ticket=atomicInc(&g_block_done_count,gridDim.x);
        s_is_last_block=(ticket==gridDim.x-1);
    }
    __syncthreads();
    if (s_is_last_block){
        my_max=tol;
        my_idx=-1;
        for (i=tid;i<gridDim.x;i+=blockDim.x){
            if (is_abs){
                if (ABS(d_block_max[i])>my_max){
                    my_max=ABS(d_block_max[i]);
                    my_idx=d_block_idx[i];
                }
            }
            else{
                if (d_block_max[i]>my_max){
                    my_max=d_block_max[i];
                    my_idx=d_block_idx[i];
                }
            }
        }
        warpReduceMax(&my_max,&my_idx);
        if (lane==0){
            s_max[wid]=my_max;
            s_idx[wid]=my_idx;
        }
        __syncthreads();

        if (wid==0){
            my_max=(tid<blockDim.x/32)?s_max[lane]:tol;
            my_idx=(tid<blockDim.x/32)?s_idx[lane]:-1;
            warpReduceMax(&my_max,&my_idx);
            if (tid==0){
                *d_max=my_max;
                *d_idx=my_idx;
                g_block_done_count=0;
            }
        }
    }
}
__global__ void gaussKernel(float *d_Vec,const int n,const int l,float *d_K){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    if (i<n&&j<n){
        if (i==j) d_K[i*n+j]=1.0f;
        else{
            float dist=0;
            int k;
            for (k=0;k<l;k++){
                float dx=d_Vec[i*l+k]-d_Vec[j*l+k];
                dist+=dx*dx;
            }
            d_K[i*n+j]=expf(-SVM_Gamma*dist);
        }
    }
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
void K_init(Model *M,float *d_K){
    int n=M->Data->Num,Dim=M->Data->Data[0]->Dim;
    int i;
    if (!M->K_ij){
        M->K_ij=(float *)malloc(n*n*sizeof(float));
    }
    float *d_Vec;
    cudaMalloc((void **)&d_Vec,n*Dim*sizeof(float));
    for (i=0;i<n;i++){
        cudaMemcpy(d_Vec+i*Dim,M->Data->Data[i]->X,Dim*sizeof(float),cudaMemcpyHostToDevice);
    }
    dim3 block2d(16,16);
    dim3 grid2d((n+15)/16,(n+15)/16);
    gaussKernel<<<grid2d,block2d>>>(d_Vec,n,Dim,d_K);
    cudaMemcpy(M->K_ij,d_K,n*n*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_Vec);
}

__global__ void alpha1UpdateCache(float *d_Alpha,signed char *d_Y,float *d_E,int n,float C,float *d_res){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n){
        float tmp=-1.0f;
        if (d_Alpha[i]>SVM_Eps) tmp=d_Y[i]*d_E[i];
        if (d_Alpha[i]<C-SVM_Eps) tmp=MAX(tmp,-d_Y[i]*d_E[i]);
        d_res[i]=tmp;
    }
}

static inline int select_alpha1(float *d_Alpha,signed char *d_Y,float *d_E,int n,float *d_cache,float *d_block_max,int *d_block_idx,float *d_max,int *d_idx,float C){
    int block1d=256;
    int grid1d=(n+255)/256;
    alpha1UpdateCache<<<grid1d,block1d>>>(d_Alpha,d_Y,d_E,n,C,d_cache);
    gridReduceMax<<<grid1d,block1d>>>(SVM_Tol,0,d_cache,n,d_block_max,d_block_idx,d_max,d_idx);
    int index;
    cudaMemcpy(&index,d_idx,sizeof(int),cudaMemcpyDeviceToHost);
    return index;
}

__global__ void updateEKernel(float *d_E,signed char *d_Y,float *d_K,int n,float d1,float d2,float db,int a1,int a2){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n){
        d_E[i]+=d_Y[a1]*d1*d_K[a1*n+i]+d_Y[a2]*d2*d_K[a2*n+i]+db;
    }
}

static inline void update_E(float *d_E,signed char *d_Y,float *d_K,int n,float d1,float d2,float db,int a1,int a2){
    int block1d=256;
    int grid1d=(n+255)/256;
    updateEKernel<<<grid1d,block1d>>>(d_E,d_Y,d_K,n,d1,d2,db,a1,a2);
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

__global__ void updateDelta(float *d_alpha,float *d_E,float *d_K,signed char *d_Y,float C,int n,int a1,float *d_delta1,float *d_delta2,float *d_deltab){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n){
    if (d_K[a1*n+i]-1>-SVM_Eps&&d_K[a1*n+i]-1<SVM_Eps||i==a1){
        d_delta2[i]=0.0f;
        d_delta1[i]=0.0f;
        d_deltab[i]=0.0f;
    }
    else{
        float E1=d_E[a1];
        signed char Y1=d_Y[a1];
        float alpha1=d_alpha[a1];
        float d2=d_Y[i]*(E1-d_E[i])/(2-2*d_K[n*a1+i]);
        float L,H;
        if (Y1==d_Y[i]){
            L=MAX(0,alpha1+d_alpha[i]-C);
            H=MIN(C,alpha1+d_alpha[i]);
        }
        else{
            L=MAX(0,d_alpha[i]-alpha1);
            H=MIN(C,C+d_alpha[i]-alpha1);
        }
        if (L>H-SVM_Eps){
            d_delta1[i]=0.0f;
            d_deltab[i]=0.0f;
            d_delta2[i]=0.0f;
            return;
        }
        float tmpalpha2=MAX(L,MIN(H,d_alpha[i]+d2));
        d2=tmpalpha2-d_alpha[i];
        float d1=-Y1*d_Y[i]*d2;
        float tmpalpha1=d1+alpha1;
        float b1=-E1-Y1*d1-d_Y[i]*d2*d_K[n*a1+i];
        float b2=-d_E[i]-Y1*d1*d_K[n*a1+i]-d_Y[i]*d2;
        if (tmpalpha1>SVM_Eps&&tmpalpha1<C-SVM_Eps){
            d_deltab[i]=b1;
        }
        else if(tmpalpha2>SVM_Eps&&tmpalpha2<C-SVM_Eps){
            d_deltab[i]=b2;
        }
        else d_deltab[i]=(b1+b2)/2;
        d_delta1[i]=d1;
        d_delta2[i]=d2;
    }
    }
} 


static inline int selectalpha2(float *d_Alpha,float *d_E,float *d_K,signed char *d_Y,float C,int n,int alpha1,float *d_delta1,float *d_delta2,float *d_deltab,float *d_block_max,int *d_block_idx,float *d_max,int *d_idx,float *B){
    int block1d=256;
    int grid1d=(n+255)/256;
    updateDelta<<<grid1d,block1d>>>(d_Alpha,d_E,d_K,d_Y,C,n,alpha1,d_delta1,d_delta2,d_deltab);
    int a2;
    float h_delta2;
    gridReduceMax<<<grid1d,block1d>>>(SVM_Eps,1,d_delta2,n,d_block_max,d_block_idx,d_max,d_idx);
    cudaMemcpy(&a2,d_idx,sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_delta2,d_delta2+a2,sizeof(float),cudaMemcpyDeviceToHost);
    if (a2!=-1){
        float h_alpha1,h_alpha2;
        cudaMemcpy(&h_alpha1,d_Alpha+alpha1,sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_alpha2,d_Alpha+a2,sizeof(float),cudaMemcpyDeviceToHost);
        float h_delta1,h_deltab;
        cudaMemcpy(&h_delta1,d_delta1+a2,sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_deltab,d_deltab+a2,sizeof(float),cudaMemcpyDeviceToHost);
        h_alpha1+=h_delta1;
        h_alpha2+=h_delta2;
        cudaMemcpy(d_Alpha+alpha1,&h_alpha1,sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_Alpha+a2,&h_alpha2,sizeof(float),cudaMemcpyHostToDevice);
        *B=(*B)+h_deltab;
        update_E(d_E,d_Y,d_K,n,h_delta1,h_delta2,h_deltab,alpha1,a2);
    }
    return a2;
}

__global__ void updateISVALIDE(unsigned char *d_IS_VALID_E,float *d_alpha,int n,float C){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n){
        if (d_alpha[i]>SVM_Eps&&d_alpha[i]<C-SVM_Eps){
            d_IS_VALID_E[i]=1;
        }
        else d_IS_VALID_E[i]=0;
    }
}

int Train(Model *M,int Cyc){
    int n=M->Data->Num,i;
    float *d_K;
    cudaMalloc((void **)&d_K,n*n*sizeof(float));
    if (!M->K_ij) K_init(M,d_K);
    else{
        cudaMemcpy(d_K,M->K_ij,n*n*sizeof(float),cudaMemcpyHostToDevice);
    }

    float *d_alpha,*d_E,*d_delta1,*d_delta2,*d_deltab;
    cudaMalloc((void **)&d_delta1,n*sizeof(float));
    cudaMalloc((void **)&d_delta2,n*sizeof(float));
    cudaMalloc((void **)&d_deltab,n*sizeof(float));

    float *d_cache;
    cudaMalloc((void**)&d_cache,n*sizeof(float));

    float *d_block_max,*d_max;
    int *d_block_idx,*d_idx;
    cudaMalloc((void**)&d_block_max,(n+255)/256*sizeof(float));
    cudaMalloc((void**)&d_block_idx,(n+255)/256*sizeof(int));
    cudaMalloc((void**)&d_max,sizeof(float));
    cudaMalloc((void**)&d_idx,sizeof(int));

    signed char *d_Y;
    cudaMalloc((void **)&d_alpha,n*sizeof(float));
    cudaMalloc((void **)&d_E,n*sizeof(float));
    cudaMalloc((void **)&d_Y,n*sizeof(unsigned char));
    cudaMemcpy(d_alpha,M->Alpha,n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_E,M->E,n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y,M->Data->Y,n*sizeof(unsigned char),cudaMemcpyHostToDevice);
    for (i=0;i<Cyc;i++){
        int a1=select_alpha1(d_alpha,d_Y,d_E,n,d_cache,d_block_max,d_block_idx,d_max,d_idx,M->C);
        if (a1==-1) goto gracereturn;
        int a2=selectalpha2(d_alpha,d_E,d_K,d_Y,M->C,n,a1,d_delta1,d_delta2,d_deltab,d_block_max,d_block_idx,d_max,d_idx,&M->B);
        if (a2==-1) goto gracereturn;
    }
gracereturn:
    cudaMemcpy(M->Alpha,d_alpha,n*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(M->E,d_E,n*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_E);
    cudaFree(d_K);
    cudaFree(d_Y);
    int block1d=256;
    int grid1d=(n+255)/256;
    unsigned char *d_IS;
    cudaMalloc((void **)&d_IS,n*sizeof(unsigned char));
    updateISVALIDE<<<grid1d,block1d>>>(d_IS,d_alpha,n,M->C);
    cudaMemcpy(M->IS_VALID_E,d_IS,n*sizeof(unsigned char),cudaMemcpyDeviceToHost);
    cudaFree(d_IS);
    cudaFree(d_alpha);

    cudaFree(d_delta1);
    cudaFree(d_delta2);
    cudaFree(d_deltab);

    cudaFree(d_cache);

    cudaFree(d_block_max);
    cudaFree(d_block_idx);
    cudaFree(d_max);
    cudaFree(d_idx);
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
    if (M->K_ij){
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


