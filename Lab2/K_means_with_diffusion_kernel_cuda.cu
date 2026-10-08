#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<signal.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<time.h>
#define MAX(a,b) (((a)>(b))?(a):(b))
#define n 1000

static volatile sig_atomic_t stop=0;
void handle_interrupt(int sig){
    (void)sig;
    stop=1;
}
typedef struct{
    float x;
    float y;
}vector;
vector data[n];
int label[n],new_label[n],*cnt;
float K[n];
char file_name[20]="database.db";
int k=2;
float sigma[n];
int cmpfunc(const void *a,const void *b){
    if (*(float*)a>*(float*)b) return 1;
    if (*(float*)a==*(float*)b)return 0;
    return -1;
}
__global__ void linear_kernel(vector *d_data,float *d_k){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    if (i<n&&j<n){
        float dx=d_data[i].x-d_data[j].x;
        float dy=d_data[i].y-d_data[j].y;
        d_k[i*n+j]=dx*dx+dy*dy;
    }
}
__global__ void gauss_kernel(vector *d_data,float *d_K,float *sigma){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    if (i<n&&j<n){
        float dx=d_data[i].x-d_data[j].x;
        float dy=d_data[i].y-d_data[j].y;
        float dist=dx*dx+dy*dy;
        d_K[i*n+j]=__expf(-dist/(sigma[i]*sigma[j]));
    }
}
__global__ void set_summat(float *d_mat){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    if (i<n&&j<n){
        d_mat[i*n+j]=1.0f;
    }
}
__global__ void set_D(float *d_D){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    if (i<n&&j<n){
        if (i==j) d_D[i*n+j]=1.0f/d_D[i*n+j];
        else d_D[i*n+j]=0.0f;
    }
}
float K_ij[n][n];
float dist(int i,int k){
	float res=K_ij[i][i];
	int j;
	int cnt=0;
	float sigma=0;
	for (j=0;j<n;j++){
		if (label[j]==k){
			cnt++;
			sigma+=K_ij[i][j];
		}
	}
	res=res-2*sigma/(float)cnt;
	res=res+K[k]/((float)(cnt*cnt));
	return res;
}
int cmp(const void* f1,const void* f2){
    if ((*(float*)f1)>(*(float*)f2)) return 1;
    if ((*(float*)f1)<(*(float*)f2)) return -1;
    return 0;
}
int main(){
    cublasHandle_t handle;
    if (cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS){
        printf("CUBLAS初始化失败。\n");
        return -1;
    }
    signal(SIGINT,handle_interrupt);
    time_t t;
    srand((unsigned)time(&t));
    FILE *db=fopen(file_name,"r");
    int i;
    for (i=0;i<n;i++){
        fscanf(db,"%f%f",&data[i].x,&data[i].y);
    }
    fclose(db);
    int cont=1,j;
    vector *d_data=NULL;
    float *d_K=NULL;
    cudaMalloc((void **)&d_data,n*sizeof(vector));
    cudaMalloc((void **)&d_K,n*n*sizeof(float));
    cudaMemcpy(d_data,data,n*sizeof(vector),cudaMemcpyHostToDevice);
    dim3 block(16,16);
    dim3 grid((n+15)/16,(n+15)/16);
    linear_kernel<<<grid,block>>>(d_data,d_K);
    cudaMemcpy(K_ij,d_K,n*n*sizeof(float),cudaMemcpyDeviceToHost);
    for (i=0;i<n;i++){
        qsort(K_ij[i],n,sizeof(float),cmp);
        sigma[i]=sqrtf(K_ij[i][1]);
    }
    float *d_sigma=NULL;
    cudaMalloc((void **)&d_sigma,n*sizeof(float));
    cudaMemcpy(d_sigma,sigma,n*sizeof(float),cudaMemcpyHostToDevice);
    gauss_kernel<<<grid,block>>>(d_data,d_K,d_sigma);
    cudaFree(d_sigma);
    cudaFree(d_data);
    float *d_summat;
    cudaMalloc((void**)&d_summat,n*n*sizeof(float));
    set_summat<<<grid,block>>>(d_summat);
    float a=1.0f,b=0.0f;
    float *d_D;
    cudaMalloc((void **)&d_D,n*n*sizeof(float));
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&a,d_summat,n,d_K,n,&b,d_D,n);
    set_D<<<grid,block>>>(d_D);
    float *d_tmpK,*d_K0;
    cudaMalloc((void **)&d_tmpK,n*n*sizeof(float));
    cudaMalloc((void **)&d_K0,n*n*sizeof(float));
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&a,d_K,n,d_D,n,&b,d_tmpK,n);
    cudaFree(d_D);
    cudaMemcpy(d_K,d_tmpK,n*n*sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_K0,d_tmpK,n*n*sizeof(float),cudaMemcpyDeviceToDevice);
    int m,l;
    for (m=5000;m>0;m--){
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&a,d_K,n,d_K0,n,&b,d_tmpK,n);
        cudaMemcpy(d_K,d_tmpK,n*n*sizeof(float),cudaMemcpyDeviceToDevice);
    }
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,n,n,&a,d_K,n,d_K,n,&b,d_tmpK,n);
    cudaMemcpy(K_ij,d_tmpK,n*n*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_tmpK);
    cudaFree(d_K);
    cudaFree(d_K0);
    for (i=0;i<n;i++){
    	label[i]=rand()%k;
    }
    while (cont&&!stop){
	    int cntupd=0;
    	memset(K,0,sizeof(K));
    	for (l=0;l<k;l++)
	    	for (i=0;i<n;i++)
		    	for (j=0;j<n;j++)
			    	if (label[i]==l&&label[j]==l) K[l]+=K_ij[i][j];
        	for (i=0;i<n;i++){
	    	float d=2147483647;
		    for (j=0;j<k;j++){
    			float tmpd=dist(i,j);	
	    		if (tmpd<d){
		    		new_label[i]=j;
			    	d=tmpd;
    			}
	    	}
	    	if (label[i]!=new_label[i]) cntupd++;
    	}
	    memcpy(label,new_label,sizeof(label));
    	if (cntupd==0) cont=0;
    }
    db=fopen("final.db","w");
    fprintf(db,"%d %d\n",n,k);
    for (i=0;i<n;i++){
    	fprintf(db,"%.2f %.2f %d\n",data[i].x,data[i].y,label[i]);
    }
    fclose(db);
    return 0;
}
