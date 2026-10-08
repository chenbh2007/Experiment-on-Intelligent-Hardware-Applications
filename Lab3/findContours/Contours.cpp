#include<iostream>
#include<vector>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
int main(int argc,char **argv){
    if (argc<2){
        std::cout<<"Too few argument."<<std::endl;
        return -1;
    }
    cv::Mat ori=cv::imread(argv[1],cv::IMREAD_COLOR);
    cv::Mat gray,thresh;
    cv::cvtColor(ori,gray,cv::COLOR_BGR2GRAY);
    std::vector<std::vector<cv::Point>> contours;
    cv::threshold(gray,thresh,127,255,cv::THRESH_BINARY);
    cv::findContours(thresh,contours,cv::RETR_LIST,cv::CHAIN_APPROX_NONE);
    cv::Mat res=ori.clone();
    cv::drawContours(res,contours,-1,cv::Scalar(255,0,255),2,cv::LINE_AA);
    cv::imwrite("result.png",res);
    return 0;
}
