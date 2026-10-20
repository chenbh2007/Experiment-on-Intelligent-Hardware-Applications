#include"cuSVM.h"
#include<stdio.h>
#include<byteswap.h>
#include<stdint.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
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

__device__ static unsigned int g_block_done_count=0;

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

int SetUpData(DataSet *db,const char *dataname,const char *labelname){
    FILE *data=fopen(dataname,"rb+");
    FILE *lab=fopen(labelname,"rb+");
    if (data&&lab){
        uint8_t cache[4];
        fread(cache,1,4,data);
        if (cache[2]!=0x0D||cache[3]!=2){
            printf("Data type is not float.\nAbort.\n");
            fclose(data);
            fclose(lab);
            return -1;
        }
        fread(cache,1,4,lab);
        if (cache[2]!=0x09||cache[3]!=1){
            printf("Label type is not signed char.\nAbort.\n");
            fclose(data);
            fclose(lab);
            return -1;
        }
        uint32_t size,tmp;
        fread(&size,sizeof(size),1,data);
        fread(&tmp,sizeof(tmp),1,lab);
        if (size!=tmp){
            printf("Data doesn't fit label.\nAbort.\n");
            fclose(data);
            fclose(lab);
            return -1;
        }
        uint32_t dim;
        fread(&dim,sizeof(dim),1,data);
        int i;
        db->Num=(int32_t)size;
        db->Y=(signed char*)malloc(size*sizeof(signed char));
        db->Data=(Vec**)malloc(size*sizeof(Vec*));
        fread(db->Y,1,size,lab);
        for (i=0;i<size;i++){
            db->Data[i]=(Vec*)malloc(sizeof(Vec));
            db->Data[i]->Dim=dim;
            db->Data[i]->X=(float *)malloc(dim*sizeof(float));
            fread(db->Data[i]->X,sizeof(float),dim,data);
        }
        fclose(lab);
        fclose(data);
        return 0;
    }
    printf("Data file not found.\n");
    if (data) fclose(data);
    if (lab) fclose(lab);
    return -1;
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

static inline int select_alpha1(float *d_Alpha,signed char *d_Y,float *d_E,int n,float *d_cache,float *d_block_max,int *d_block_idx,float *d_max,int *d_idx,float C,int skip_a1){
    int block1d=256;
    int grid1d=(n+255)/256;
    if (skip_a1==-1){
        alpha1UpdateCache<<<grid1d,block1d>>>(d_Alpha,d_Y,d_E,n,C,d_cache);
    }
    else{
        cudaMemset(d_cache+skip_a1,0,sizeof(float));
    }
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

void SaveState(Model *M,const char *file_name,const char *B_C_lab_name){
    FILE *f=fopen(file_name,"wb+");
    FILE *fl=fopen(B_C_lab_name,"wb+");
    if (!f||!fl){
        if (f) fclose(f);
        if (fl) fclose(fl);
        printf("Failed to save.\n");
        return;
    }
    uint8_t buffer[4]={0x00,0x00,0x0D,0x02};
    fwrite(buffer,1,4,f);
    uint32_t dim=M->Data->Data[0]->Dim+2;
    uint32_t n=M->Data->Num;
    fwrite(&n,sizeof(n),1,f);
    fwrite(&dim,sizeof(dim),1,f);
    buffer[2]=0x09;
    buffer[3]=0x01;
    fwrite(buffer,1,4,fl);
    fwrite(&n,sizeof(n),1,fl);
    int i;
    for (i=0;i<n;i++){
        fwrite(M->Data->Data[i]->X,sizeof(float),dim-2,f);
        fwrite(&M->Alpha[i],sizeof(float),1,f);
        fwrite(&M->E[i],sizeof(float),1,f);
    }
    fwrite(M->Data->Y,1,n,fl);
    fwrite(&M->B,sizeof(float),1,fl);
    fwrite(&M->C,sizeof(float),1,fl);
    fclose(f);
    fclose(fl);
    return;
}
int LoadState(Model *M,const char *state_file,const char *lab_set){
    if (!M->Data) M->Data=(DataSet*)malloc(sizeof(DataSet));
    M->Data->Data=NULL;
    M->Data->Y=NULL;
    if(SetUpData(M->Data,state_file,lab_set)!=0){
        SVMFreeDataSet(M->Data);
        M->Data=NULL;
        return -1;
    }
    FILE *f=fopen(lab_set,"rb+");
    if (!f){
        return -1;
    }
    fseek(f,-(long)(2*sizeof(float)),SEEK_END);
    fread(&M->B,sizeof(float),1,f);
    fread(&M->C,sizeof(float),1,f);
    fclose(f);
    int i,n=M->Data->Num;
    M->Alpha=(float*)malloc(n*sizeof(float));
    M->E=(float*)malloc(n*sizeof(float));
    M->IS_VALID_E=(unsigned char*)malloc(n*sizeof(unsigned char));
    M->K_ij=NULL;
    for (i=0;i<n;i++){
        M->Alpha[i]=M->Data->Data[i]->X[M->Data->Data[i]->Dim-2];
        M->E[i]=M->Data->Data[i]->X[M->Data->Data[i]->Dim-1];
        M->Data->Data[i]->Dim-=2;
        if (M->Alpha[i]>SVM_Eps) M->IS_VALID_E[i]=1;
        else M->IS_VALID_E[i]=0;
    }
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
    if (a2!=-1){
        float h_alpha1,h_alpha2;
        cudaMemcpy(&h_delta2,d_delta2+a2,sizeof(float),cudaMemcpyDeviceToHost);
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
        if (d_alpha[i]>SVM_Eps){
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
    int skip_a1=-1;
    int a1;
    for (i=0;i<Cyc;i++){
        a1=select_alpha1(d_alpha,d_Y,d_E,n,d_cache,d_block_max,d_block_idx,d_max,d_idx,M->C,skip_a1);
        if (a1==-1) goto gracereturn;
        int a2=selectalpha2(d_alpha,d_E,d_K,d_Y,M->C,n,a1,d_delta1,d_delta2,d_deltab,d_block_max,d_block_idx,d_max,d_idx,&M->B);
        if (a2==-1) skip_a1=a1;
        else skip_a1=-1;
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
    if (a1==-1) return 0;
    else return -1;
}

typedef struct node{
    int idx;
    node *next;
}node;

void copyVec(Vec *src,Vec *dst){
    int v=src->Dim;
    dst->X=(float *)malloc(v*sizeof(float));
    dst->Dim=v;
    memcpy(dst->X,src->X,v*sizeof(float));
    return;
}

void extractSV(Model *src,Model *dst){
    int n=src->Data->Num;
    int cnt=0,i;
    node *SV=(node*)malloc(sizeof(node));
    node *p=SV;
    SV->next=NULL;
    for (i=0;i<n;i++){
        if (src->Alpha[i]>SVM_Eps){
            cnt++;
            p->idx=i;
            p->next=(node*)malloc(sizeof(node));
            p=p->next;
            p->next=NULL;
        }
    }
    dst->Data=(DataSet*)malloc(sizeof(DataSet));
    dst->Data->Y=(signed char*)malloc(cnt*sizeof(signed char));
    dst->Data->Data=(Vec **)malloc(cnt*sizeof(Vec*));
    dst->Data->Num=cnt;
    dst->IS_VALID_E=(unsigned char*)malloc(cnt*sizeof(unsigned char));
    dst->Alpha=(float *)malloc(cnt*sizeof(float));
    dst->E=(float *)malloc(cnt*sizeof(float));
    dst->B=src->B;
    dst->C=src->C;
    dst->K_ij=NULL;
    p=SV;
    for (i=0;i<cnt&&p!=NULL;i++){
        dst->Data->Data[i]=(Vec *)malloc(sizeof(Vec));
        copyVec(src->Data->Data[p->idx],dst->Data->Data[i]);
        dst->Data->Y[i]=src->Data->Y[p->idx];
        dst->E[i]=src->E[p->idx];
        dst->IS_VALID_E[i]=1;
        dst->Alpha[i]=src->Alpha[p->idx];
        p=p->next;
    }
    p=SV->next;
    while (p!=NULL){
        free(SV);
        SV=p;
        p=SV->next;
    }
    free(SV);
    return;
}


void dataSetSlice(DataSet *src,DataSet *dst,int beg,int end){
    int n=end-beg;
    if (end>src->Num) n=src->Num-beg;
    dst->Num=n;
    dst->Data=(Vec **)malloc(n*sizeof(Vec*));
    dst->Y=(signed char*)malloc(n*sizeof(unsigned char));
    int i;
    for (i=0;i<n;i++){
        dst->Data[i]=(Vec*)malloc(sizeof(Vec));
        copyVec(src->Data[i+beg],dst->Data[i]);
        dst->Y[i]=src->Y[i+beg];
    }
    return;
}

void mergeDataSet(DataSet *src1,DataSet *src2,DataSet *dst){
    int n1=src1->Num;
    int n2=src2->Num;
    int n=n1+n2;
    dst->Num=n;
    dst->Data=(Vec**)malloc(n*sizeof(Vec*));
    dst->Y=(signed char*)malloc(n*sizeof(signed char));
    int i;
    for (i=0;i<n1;i++){
        dst->Data[i]=(Vec*)malloc(sizeof(Vec));
        copyVec(src1->Data[i],dst->Data[i]);
        dst->Y[i]=src1->Y[i];
    }
    for (i=n1;i<n;i++){
        dst->Data[i]=(Vec*)malloc(sizeof(Vec));
        copyVec(src2->Data[i-n1],dst->Data[i]);
        dst->Y[i]=src2->Y[i-n1];
    }
    return;
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


