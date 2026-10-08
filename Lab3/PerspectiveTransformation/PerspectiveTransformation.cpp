#include<iostream>
#include<sstream>
#include<vector>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/geometry/2d.hpp>
int main(int argc,char **argv){
    if (argc<11){
        std::cout<<"Too few argument."<<std::endl;
        return -1;
    }
    std::stringstream convs;
    int ifVideo;
    std::vector<cv::Point2f> src,dst;
    convs<<argv[1];
    convs>>ifVideo;
    convs.clear();
    cv::Mat ori;
    if (ifVideo){
        cv::VideoCapture Cam(argv[2]);
        if (!Cam.isOpened()){
            std::cout<<"Cannot open Camera "<<argv[2]<<"."<<std::endl;
            return -1;
        }
        Cam>>ori;
        if (ori.empty()){
            std::cout<<"Cannot capture picture."<<std::endl;
            Cam.release();
            return -1;
        }
        Cam.release();
    }
    else{
        ori=cv::imread(argv[2],cv::IMREAD_COLOR);
        if (ori.empty()){
            std::cout<<"Cannot open "<<argv[2]<<"."<<std::endl;
            return -1;
        }
    }
    int nRows=ori.rows;
    int nCols=ori.cols;
    for (int i=0;i<4;i++){
        convs<<argv[i*2+3]<<std::endl<<argv[i*2+4]<<std::endl;
        float tmpx,tmpy;
        convs>>tmpx>>tmpy;
        src.push_back(cv::Point2f(tmpx*nCols,tmpy*nRows));
        convs.clear();
    }
    dst.push_back(cv::Point2f(0,0));
    dst.push_back(cv::Point2f(0,nRows));
    dst.push_back(cv::Point2f(nCols,nRows));
    dst.push_back(cv::Point2f(nCols,0));
    cv::Mat transMat=cv::getPerspectiveTransform(src,dst);
    cv::Mat conv;
    cv::warpPerspective(ori,conv,transMat,cv::Size(nCols,nRows));
    cv::imwrite("converted.png",conv);
    return 0;
}
