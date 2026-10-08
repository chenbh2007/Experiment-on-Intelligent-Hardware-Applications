#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)>(b)?(b):(a))
typedef struct{
    unsigned char r,g,b;
}pt;
pt image[1000000];
pt color[10];
void scattor(float x,float y,int c){
    int w,h;
    for (w=MAX(floor(x)-3,0);w<=x+3&&w<=999;w++)
        for (h=MAX(floor(y)-3,0);h<=y+3&&h<=999;h++)
		if(pow(w-x,2)+pow(h-y,2)<=17)
            		image[h*1000+w]=color[c];
    return;
}
char file_name[20]="final.db";
char graph_name[256]="picture.ppm";
int main(int argc,char** argv){
    srand((unsigned)time(NULL));
    if (argc>1) snprintf(graph_name,sizeof(graph_name),"%s",argv[1]);
    FILE *db=fopen(file_name,"r");
    FILE *pic=fopen(graph_name,"wb");
    memset(image,0,sizeof(image));
    int n,k;
    fscanf(db,"%d%d",&n,&k);
    int i;
    for (i=0;i<k;i++){
    	color[i].r=255*i/k;
	color[i].g=255*(1-i)/k;
	color[i].b=128;
    }
    float *x,*y;
    x=(float*)malloc(n*sizeof(float));
    y=(float*)malloc(n*sizeof(float));
    int *l=(int *)malloc(n*sizeof(int));
    float minx=2147483627,maxx=-2147483647,maxy=-2147483647,miny=2147483647;
    for (i=0;i<n;i++){
	fscanf(db,"%f%f%d",&x[i],&y[i],&l[i]);
	maxx=MAX(maxx,x[i]);
	maxy=MAX(maxy,y[i]);
	minx=MIN(minx,x[i]);
	miny=MIN(miny,y[i]);
    }
    for (i=0;i<n;i++){
    	scattor((x[i]-minx)/(maxx-minx)*1000,(y[i]-miny)/(maxy-miny)*1000,l[i]);
    }
    fclose(db);
    fprintf(pic,"P6\n1000 1000\n255\n");
    for (i=0;i<1000000;i++){
    	unsigned char rgb[3]={image[i].r,image[i].g,image[i].b};
	fwrite(rgb,1,3,pic);
    }
    fclose(pic);
    free(x);
    free(y);
    return 0;
}
