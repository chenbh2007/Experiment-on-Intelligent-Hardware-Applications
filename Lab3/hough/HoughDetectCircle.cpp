#include<iostream>
#include<sstream>
#include<vector>
#include<opencv2/imgproc.hpp>
#include<opencv2/core.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgcodecs.hpp>
int main(int argc,char **argv){
    if (argc<2){
        std::cout<<"Too few argument."<<std::endl;
        return -1;
    }
    int ifVideo;
    std::stringstream convs;
    convs<<argv[1];
    convs>>ifVideo;
    cv::Mat ori;
    if (ifVideo){
        cv::VideoCapture Cam(argv[2]);
        if (!Cam.isOpened()){
            std::cout<<"Cannot open the camera "<<argv[2]<<"."<<std::endl;
            return -1;
        }
        Cam>>ori;
        if (ori.empty()){
            std::cout<<"Cannot capture the photo."<<std::endl;
            Cam.release();
            return -1;
        }
    }
    else{
        ori=cv::imread(argv[2],cv::IMREAD_COLOR);
        if (ori.empty()){
            std::cout<<"Cannot open "<<argv[2]<<"."<<std::endl;
            return -1;
        }
    }
    cv::Mat gray;
    cv::cvtColor(ori,gray,cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray,gray,5);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray,circles,cv::HOUGH_GRADIENT,1,gray.rows/16);
    for (size_t i=0;i<circles.size();i++){
        cv::Vec3i c=circles[i];
        cv::Point center(c[0],c[1]);
        cv::circle(ori,center,1,cv::Scalar(0,100,100),3,cv::LINE_AA);
        cv::circle(ori,center,c[2],cv::Scalar(255,0,255),3,cv::LINE_AA);
    }
    cv::imwrite("result.png",ori);
    return 0;
}
