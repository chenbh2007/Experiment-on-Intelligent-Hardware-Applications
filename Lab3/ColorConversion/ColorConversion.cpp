#include<iostream>
#include<vector>
#include<opencv2/core.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
int main(){
    cv::VideoCapture Cam("/dev/video0");
    if (!Cam.isOpened()){
        std::cout<<"Cannot open Camera /dev/video0."<<std::endl;
        return -1;
    }
    cv::Mat Ori;
    Cam>>Ori;
    if (Ori.empty()){
        std::cout<<"Cannot capture the picture."<<std::endl;
        return -1;
    }
    cv::Mat Conv;
    cvtColor(Ori,Conv,cv::COLOR_BGR2HSV);
    cv::imwrite("Ori.png",Ori);
    cv::imwrite("Conv.png",Conv);
    Cam.release();
    return 0;
}

