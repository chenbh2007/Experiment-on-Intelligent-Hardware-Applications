#include<iostream>
#include<sstream>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/geometry/2d.hpp>
#include<vector>
#include<cuda_runtime.h>
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)       \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)
__global__ void CalcNabla(float* d_Mx,float* d_My,int width,int height,float* d_Mres){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    if (i<height&&j<width){
        float dx=(float)d_Mx[i*width+j];
        float dy=(float)d_My[i*width+j];
        d_Mres[i*width+j]=sqrtf(dx*dx+dy*dy);
    }
}
int main(int argc,char **argv){
    if (argc<6){
        std::cout<<"Too few argument."<<std::endl;
        return -1;
    }
    cv::Mat ori=cv::imread(argv[1],cv::IMREAD_COLOR);
    size_t blurSize;
    std::stringstream ss;
    int threshold,dstWidth,dstHeight;
    ss<<argv[2]<<std::endl<<argv[3]<<std::endl<<argv[4]<<std::endl<<argv[5];
    ss>>blurSize>>threshold>>dstWidth>>dstHeight;
    ss.clear();
    std::vector<cv::Point2f> dst;
    int nRows=ori.rows,nCols=ori.cols;
    dst.push_back(cv::Point2f(0,0));
    dst.push_back(cv::Point2f(dstWidth,0));
    dst.push_back(cv::Point2f(dstWidth,dstHeight));
    dst.push_back(cv::Point2f(0,dstHeight));
    cv::Mat blur,sharpen(nRows,nCols,CV_32F,cv::Scalar::all(0)),gray,thresh;
    cv::cvtColor(ori,gray,cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray,blur,blurSize);
    cv::Mat m1,m2;
    blur.convertTo(m1,CV_32F);
    m2=m1.clone();
    cv::Mat sobelx=cv::Mat_<float>({3,3},{-1,0,1,-2,0,2,-1,0,1});
    cv::Mat sobely=cv::Mat_<float>({3,3},{-1,-2,-1,0,0,0,1,2,1});
    cv::filter2D(m1,m1,m1.depth(),sobelx);
    cv::filter2D(m2,m2,m2.depth(),sobely);
    float *d_m1,*d_m2,*d_res;
    size_t chars=nRows*nCols*sizeof(float);
    CHECK_CUDA(cudaMalloc((void**)&d_m1,chars));
    CHECK_CUDA(cudaMalloc((void**)&d_m2,chars));
    CHECK_CUDA(cudaMalloc((void**)&d_res,chars));
    CHECK_CUDA(cudaMemcpy(d_m1,m1.ptr<float>(),chars,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_m2,m2.ptr<float>(),chars,cudaMemcpyHostToDevice));
    dim3 block(16,16);
    dim3 grid((nCols+15)/16,(nRows+15)/16);
    CalcNabla<<<grid,block>>>(d_m1,d_m2,nCols,nRows,d_res);
    CHECK_CUDA(cudaMemcpy(sharpen.ptr<float>(),d_res,chars,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_m1));
    CHECK_CUDA(cudaFree(d_m2));
    CHECK_CUDA(cudaFree(d_res));
    sharpen.convertTo(sharpen,CV_8UC1);
    cv::threshold(sharpen,thresh,threshold,255,cv::THRESH_BINARY_INV);
    cv::imwrite("sharpen.png",sharpen);
    cv::imwrite("thresh.png",thresh);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> h;
    cv::findContours(thresh,contours,h,cv::RETR_LIST,cv::CHAIN_APPROX_SIMPLE);
    int cnt=0;
    for (auto it=contours.begin();it!=contours.end();++it){
        if (cv::contourArea(*it)<5000||cv::contourArea(*it)>0.8*nCols*nRows) continue;
        std::vector<cv::Point2f> rect; 
        std::vector<cv::Point2f> src;
        src.push_back(cv::Point2f(nCols,nRows));
        src.push_back(cv::Point2f(0,nRows));
        src.push_back(cv::Point2f(0,0));
        src.push_back(cv::Point2f(nCols,0));
        cv::approxPolyDP((*it),rect,0.02*cv::arcLength(*it,true),true);
        if (rect.size()!=4||!cv::isContourConvex(rect)) continue;
        cnt++;
        for (int i=0;i<4;i++){
            if (rect[i].x+rect[i].y<src[0].x+src[0].y) src[0]=rect[i];
            if (rect[i].x+rect[i].y>src[2].x+src[2].y) src[2]=rect[i];
            if (rect[i].x-rect[i].y>src[1].x-src[1].y) src[1]=rect[i];
            if (rect[i].x-rect[i].y<src[3].x-src[3].y) src[3]=rect[i];
        }
        cv::Mat trans=cv::getPerspectiveTransform(src,dst);
        cv::Mat res;
        cv::warpPerspective(ori,res,trans,cv::Size(dstWidth,dstHeight));
        std::stringstream convs;
        convs<<"result"<<cnt<<".png"<<std::endl;
        char filename[25];
        convs>>filename;
        convs.clear();
        cv::imwrite(filename,res);
    }
    return 0;
}
