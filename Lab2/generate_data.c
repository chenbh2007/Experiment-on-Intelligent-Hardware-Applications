#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
char file_name[20]="database.db";
int n=2000;
float x=2,y=2,r=1;
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
        float tmpx,tmpy;
        int tmp=rand()%2;
        tmpx=r*sqrt(-2*log(((float)rand()/((float)RAND_MAX+1.0))))*cos(2*M_PI*((float)rand()/(float)RAND_MAX))+x*tmp;
        tmpy=r*sqrt(-2*log(((float)rand()/((float)RAND_MAX+1.0))))*sin(2*M_PI*((float)rand()/(float)RAND_MAX))+y*tmp;
        fprintf(f,"%f %f\n",tmpx,tmpy);
    }
    fclose(f);
    return 0;
}
