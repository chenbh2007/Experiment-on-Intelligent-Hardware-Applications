#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
char file_name[20]="database.db";
int n=2000;
float x=8,y=12,r=0.5;
int main(int argc,char **argv){
    time_t t;
    srand((unsigned)time(&t));
    if (argc>=2) snprintf(file_name,sizeof(file_name),"%s",argv[1]);
    if (argc>=3) x=atoi(argv[2]);
    if (argc>=4) y=atoi(argv[3]);
    if (argc>=5) r=atoi(argv[4]);
    FILE *f=fopen(file_name,"w");
    int i;
    for (i=0;i<n;i++){
        float tmpr,theta;
        int tmp=rand()%2;
        tmpr=r*sqrt(-2*log(((float)rand()/((float)RAND_MAX+1.0))))*cos(2*M_PI*((float)rand()/(float)RAND_MAX))+x*tmp+y*(1-tmp);
        theta=2*M_PI*((float)rand()/(float)RAND_MAX);
        fprintf(f,"%f %f\n",tmpr*cos(theta),tmpr*sin(theta));
    }
    fclose(f);
    return 0;
}
