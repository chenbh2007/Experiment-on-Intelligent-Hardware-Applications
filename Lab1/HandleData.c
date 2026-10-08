#include<stdio.h>
#include<stdlib.h>
#include<string.h>
int cnt;
int tmp[200040][6],h=0,t=0;
int main(){
    char vec[10][30]={"1 0 0 0 0 0 0 0 0 0 ",
                    "0 1 0 0 0 0 0 0 0 0 ",
                    "0 0 1 0 0 0 0 0 0 0 ",
                    "0 0 0 1 0 0 0 0 0 0 ",
                    "0 0 0 0 1 0 0 0 0 0 ",
                    "0 0 0 0 0 1 0 0 0 0 ",
                    "0 0 0 0 0 0 1 0 0 0 ",
                    "0 0 0 0 0 0 0 1 0 0 ",
                    "0 0 0 0 0 0 0 0 1 0 ",
                    "0 0 0 0 0 0 0 0 0 1 "
    };
    FILE *f1=fopen("DataSet.db","r");
    int dim;
    scanf("%d",&dim);
    int y;
    scanf("%d",&y);
    char file_name[20];
    snprintf(file_name,sizeof(file_name),"DataBase%d.db",y);
    FILE *f2=fopen(file_name,"w");
    int i;
    fprintf(f2,"%d %d\n",10000,10*dim);
    while (fscanf(f1,"%d%d",&tmp[t][0],&tmp[t][1])!=EOF){\
        if (tmp[t][0]==1) h=t;
        t++;
        if (t-h<dim+1){
            for (i=0;i<dim-(t-1-h);i++){
                fprintf(f2,"%s",vec[9]);
            }
            for (i=h;i<t-1;i++){
                fprintf(f2,"%s",vec[tmp[i][1]]);
            }
            if (tmp[t-1][1]%3==y) fprintf(f2,"1\n");
            else fprintf(f2,"-1\n");
        }
        else{
            for (i=h;i<t-1;i++){
                fprintf(f2,"%s",vec[tmp[i][1]]);
            }
            if (tmp[t-1][1]%3==y) fprintf(f2,"1\n");
            else fprintf(f2,"-1\n");
            h++;
        }
    }
    fclose(f1);
    fclose(f2);
    return 0;
}
