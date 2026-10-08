#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<signal.h>
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
vector **data;
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
float kernel(int i,int j){
    return exp(-(pow(data[i]->x-data[j]->x,2)+pow(data[i]->y-data[j]->y,2))/(2*sigma[i]*sigma[j]));
}
float K_ij[n][n],tmpK[n][n];
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
int main(){
    signal(SIGINT,handle_interrupt);
    time_t t;
    srand((unsigned)time(&t));
    data=(vector **)malloc(n*sizeof(vector*));
    FILE *db=fopen(file_name,"r");
    int i;
    for (i=0;i<n;i++){
        data[i]=(vector *)malloc(sizeof(vector));
        fscanf(db,"%f%f",&data[i]->x,&data[i]->y);
    }
    fclose(db);
    int cont=1,j;
    for (i=0;i<n;i++){
        for (j=0;j<n;j++)
            K_ij[i][j]=pow(data[i]->x-data[j]->x,2)+pow(data[i]->y-data[j]->y,2);
        qsort(K_ij[i],n,sizeof(float),cmpfunc);
        float tmp=0;
        int step=4;
        sigma[i]=sqrtf(K_ij[i][step]);
    }
    for (i=0;i<n;i++){
        float D=0;
	    for (j=0;j<n;j++){
		    K_ij[i][j]=kernel(i,j);
            D+=K_ij[i][j];
        }
        for (j=0;j<n;j++)
            K_ij[i][j]=K_ij[i][j]/D;
    }
    int m,l;
    for (m=13;m>0;m--){
        memcpy(tmpK,K_ij,sizeof(tmpK));
        for (i=0;i<n;i++)
            for (j=0;j<n;j++){
                K_ij[i][j]=0;
                for (l=0;l<n;l++)
                    K_ij[i][j]+=tmpK[i][l]*tmpK[l][j];
            }
    }
    memcpy(tmpK,K_ij,sizeof(tmpK));
    for (i=0;i<n;i++)
        for (j=0;j<n;j++){
            K_ij[i][j]=0;
            for (l=0;l<n;l++)
                K_ij[i][j]+=tmpK[i][l]*tmpK[j][l];
        }
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
    	fprintf(db,"%.2f %.2f %d\n",data[i]->x,data[i]->y,label[i]);
    }
    fclose(db);
    for (i=0;i<n;i++) free(data[i]);
    free(data);
    return 0;
}
