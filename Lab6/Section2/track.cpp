#include<opencv2/core.hpp>
#include<opencv2/video.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<sstream>
#include<cmath>
#include<csignal>
static volatile std::sig_atomic_t is_running=1;
void handle_interruption(int signal){
    (void)signal;
    is_running=0;
    return;
}
int main(int argc,char **argv){
    std::signal(SIGINT,handle_interruption);
    if (argc<3){
        std::cout<<"Too few arguments"<<std::endl;
        return -1;
    }
    cv::VideoCapture source(argv[1]);
    if (!source.isOpened()){
        std::cout<<"Failed to load the source video."<<std::endl;
        return -1;
    }
    cv::Mat target=cv::imread(argv[2],cv::IMREAD_COLOR);
    if (target.empty()){
        std::cout<<"Failed to read the image of target."<<std::endl;
        return -1;
    }
    cv::cvtColor(target,target,cv::COLOR_BGR2HSV);
    const int histSize[]={30,32};
    float hranges[]={0,180};
    float sranges[]={0,256};
    const float *ranges[]={hranges,sranges};
    const int channels[]={0,1};
    cv::Mat targetHist;
    cv::calcHist(&target,1,channels,cv::Mat(),targetHist,2,histSize,ranges);
    cv::normalize(targetHist,targetHist,0,255,cv::NORM_MINMAX);
    cv::Mat raw,result;
    int init=1;
    cv::namedWindow("Result",cv::WINDOW_AUTOSIZE);
    cv::Rect w;
    while (is_running){
        source>>raw;
        if (raw.empty()) break;
        cv::cvtColor(raw,raw,cv::COLOR_BGR2HSV);
        cv::calcBackProject(&raw,1,channels,targetHist,result,ranges);
        if (init){
            cv::Moments m=cv::moments(result);
            if (m.m00==0) break;
            float cx=m.m10/m.m00,cy=m.m01/m.m00;
            float sigmax=sqrt(m.m20/m.m00-cx*cx),sigmay=sqrt(m.m02/m.m00-cy*cy);
            w=cv::Rect(cx-2*sigmax,cy-2*sigmay,4*sigmax,4*sigmay);
        }
        cv::RotatedRect w1=cv::CamShift(result,w,cv::TermCriteria(cv::TermCriteria::EPS,1000,(double)1.0));
        cv::cvtColor(raw,raw,cv::COLOR_HSV2BGR);
        cv::Point2f vertices[4];
        w1.points(vertices);
        for (int i=0;i<4;i++){
            cv::line(raw,vertices[i],vertices[(i+1)%4],cv::Scalar(0,(1-init)*255,255));
        }
        cv::imshow("Result",raw);
        w=w1.boundingRect();
        if (cv::waitKey(10)!=-1) init=0;
    }
    source.release();
    return 0;
}
