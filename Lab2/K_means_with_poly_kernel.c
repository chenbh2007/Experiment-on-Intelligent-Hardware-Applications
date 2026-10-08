#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<signal.h>
#include<time.h>

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
int label[10000],new_label[10000],*cnt;
float K[10000];
char file_name[20]="database.db";
int n=10000,k=2;
#define gamma -0.1
float kernel(vector *v1,vector *v2){
    return pow(v1->x*v2->x+v1->y*v2->y+1,2);
}
float K_ij[10000][10000];
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
int main(int argc,char **argv){
    signal(SIGINT,handle_interrupt);
    time_t t;
    srand((unsigned)time(&t));
    if (argc>=2){
        snprintf(file_name,sizeof(file_name),"%s",argv[1]);
    }
    if (argc>=3) n=atoi(argv[2]);
    if (argc>=4) k=atoi(argv[3]);
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
		K_ij[i][j]=kernel(data[i],data[j]);
    }
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
    	fprintf(db,"%.2f %.2f %d\n",data[i]->x,data[i]->y,label[i]);
    }
    fclose(db);
    for (i=0;i<n;i++) free(data[i]);
    free(data);
    return 0;
}
