#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<signal.h>
#include<time.h>

#define tol 1e-3
static volatile sig_atomic_t stop=0;
void handle_interrupt(int sig){
    (void)sig;
    stop=1;
}
typedef struct{
    float x;
    float y;
}vector;
vector **data,**center;
int *label,*cnt;
float *tot_x,*tot_y;
char file_name[20]="database.db";
char save_name[20]="result.db";
int n=1000,k=2;
float dist(vector *v1,vector *v2){
    return pow(v1->x-v2->x,2)+pow(v1->y-v2->y,2);
}
int main(int argc,char **argv){
    signal(SIGINT,handle_interrupt);
    time_t t;
    srand((unsigned)time(&t));
    if (argc>=2){
        snprintf(file_name,sizeof(file_name),"%s",argv[1]);
    }
    if (argc>=3){
        snprintf(save_name,sizeof(save_name),"%s",argv[2]);
    }
    if (argc>=4) n=atoi(argv[3]);
    if (argc>=5) k=atoi(argv[4]);
    label=(int*)malloc(n*sizeof(int));
    center=(vector **)malloc(k*sizeof(vector*));
    tot_x=(float *)malloc(k*sizeof(float));
    tot_y=(float *)malloc(k*sizeof(float));
    cnt=(int *)malloc(k*sizeof(int));
    data=(vector **)malloc(n*sizeof(vector*));
    FILE *db=fopen(file_name,"r");
    int i;
    float maxx=-2147483637,minx=2147482647,maxy=-2147483647,miny=2147483647;
    for (i=0;i<n;i++){
        data[i]=(vector *)malloc(sizeof(vector));
        fscanf(db,"%f%f",&data[i]->x,&data[i]->y);
        if (data[i]->x>maxx) maxx=data[i]->x;
        if (data[i]->x<minx) minx=data[i]->x;
        if (data[i]->y>maxy) maxy=data[i]->y;
        if (data[i]->y<miny) miny=data[i]->y;
    }
    for (i=0;i<k;i++){
        center[i]=(vector *)malloc(sizeof(vector));
        center[i]->x=((float)rand()/(float)RAND_MAX)*(maxx-minx)+minx;
        center[i]->y=((float)rand()/(float)RAND_MAX)*(maxy-miny)+miny;
    }
    fclose(db);
    int cont=1,j;
    while (cont&&!stop){
        memset(tot_x,0,k*sizeof(float));
        memset(tot_y,0,k*sizeof(float));
        memset(cnt,0,k*sizeof(int));
        for (i=0;i<n;i++){
            float d=2147483647;
            for (j=0;j<k;j++)
                if (dist(center[j],data[i])<d){
                    d=dist(center[j],data[i]);
                    label[i]=j;
                }
            tot_x[label[i]]+=data[i]->x;
            tot_y[label[i]]+=data[i]->y;
            cnt[label[i]]++;
        }
        float tmp_x=0,tmp_y=0;
        vector tmpv;
        cont=0;
        for (i=0;i<k;i++){
            if (cnt[i]==0){
                center[i]->x=((float)rand()/(float)RAND_MAX)*(maxx-minx)+minx;
                center[i]->y=((float)rand()/(float)RAND_MAX)*(maxy-miny)+miny;
                continue;
            }
            tmp_x=tot_x[i]/cnt[i];
            tmp_y=tot_y[i]/cnt[i];
            tmpv.x=tmp_x;
            tmpv.y=tmp_y;
            if (dist(&tmpv,center[i])>tol) cont=1;
            center[i]->x=tmp_x;
            center[i]->y=tmp_y;
        }
    }
    db=fopen(save_name,"w");
    fprintf(db,"%d\n",k);
    for (i=0;i<k;i++){
        fprintf(db,"%f %f\n",center[i]->x,center[i]->y);
        free(center[i]);
    }
    fclose(db);
    db=fopen("final.db","w");
    fprintf(db,"%d %d\n",n,k);
    for (i=0;i<n;i++){
    	fprintf(db,"%.2f %.2f %d\n",data[i]->x,data[i]->y,label[i]);
    }
    fclose(db);
    free(center);
    for (i=0;i<n;i++) free(data[i]);
    free(data);
    free(tot_x);
    free(tot_y);
    free(cnt);
    free(label);
    return 0;
}
