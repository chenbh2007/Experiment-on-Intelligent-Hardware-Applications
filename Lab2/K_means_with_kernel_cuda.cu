#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<signal.h>
#include<cuda_runtime.h>
#include<time.h>

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
int label[n],new_label[n];
float K[n];
char file_name[20]="database.db";
int k=2;
#define gamma 0.1
__global__ void gauss_kernel(vector *d_data,float *d_K){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    if (i<n&&j<n){
        float dx=d_data[i].x-d_data[j].x;
        float dy=d_data[i].y-d_data[j].y;
        float dist=dx*dx+dy*dy;
        d_K[j*n+i]=__expf(-gamma*dist);
    }
}
float K_ij[n][n];
float dist(int i,int k){
	float res=1;
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
int main(int argc,char **argv){
    signal(SIGINT,handle_interrupt);
    time_t t;
    srand((unsigned)time(&t));
    FILE *db=fopen(file_name,"r");
    int i;
    for (i=0;i<n;i++){
        fscanf(db,"%f%f",&(data[i].x),&(data[i].y));
    }
    fclose(db);
    int cont=1,j;
    float *d_K=NULL;
    vector *d_data=NULL;
    cudaMalloc((void **)&d_K,n*n*sizeof(float));
    cudaMalloc((void **)&d_data,n*sizeof(vector));
    cudaMemcpy(d_data,data,n*sizeof(vector),cudaMemcpyHostToDevice);
    dim3 block(16,16);
    dim3 grid((n+15)/16,(n+15)/16);
    gauss_kernel<<<grid,block>>>(d_data,d_K);
    cudaMemcpy(K_ij,d_K,n*n*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_K);
    cudaFree(d_data);
    for (i=0;i<n;i++){
    	label[i]=rand()%k;
    }
    while (cont&&!stop){
	    int cntupd=0;
    	memset(K,0,sizeof(K));
    	int l;
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
