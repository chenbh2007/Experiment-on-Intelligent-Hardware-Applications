#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/objdetect.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/video.hpp>
#include<opencv2/highgui.hpp>
#include<csignal>
#include<vector>
static volatile std::sig_atomic_t is_running=1;
void handler(int sigint){
	(void)sigint;
	is_running=0;
}
#define Camera "/dev/video0"
int main(){
	std::signal(SIGINT,handler);
	cv::CascadeClassifier face("/home/HwHiAiUser/haarcascade_frontalface_default.xml");
	cv::CascadeClassifier eye("/home/HwHiAiUser/haarcascade_eye.xml");
	if (face.empty()){
		std::cerr<<"Cannot load model of face."<<std::endl;
		return -1;
	}
	if (eye.empty()){
		std::cerr<<"Cannot load model of eye."<<std::endl;
		return -1;
	}
	cv::VideoCapture source(Camera);
	if (!source.isOpened()){
		std::cerr<<"Cannot open camera."<<std::endl;
		return -1;
	}
	cv::Mat raw;
	cv::namedWindow("Result",cv::WINDOW_AUTOSIZE);
	while (is_running){
		source>>raw;
		cv::Mat gray;
		cv::cvtColor(raw,gray,cv::COLOR_BGR2GRAY);
		cv::equalizeHist(gray,gray);
		std::vector<cv::Rect> frames;
		face.detectMultiScale(
					gray,
					frames,
					1.1,
					3,
					0,
					cv::Size(30,30)
				);
		for (int i=0;i<frames.size();i++){
			cv::rectangle(raw,frames[i],cv::Scalar(0,255,255));
			cv::Mat ROI(gray,frames[i]);
			std::vector<cv::Rect> eyes;
			eye.detectMultiScale(
					ROI,
					eyes,
					1.1,
					5,
					0,
					cv::Size(5,5)
			);
			for (int j=0;j<eyes.size();j++){
				eyes[j].x+=frames[i].x;
				eyes[j].y+=frames[i].y;
				cv::rectangle(raw,eyes[j],cv::Scalar(0,0,255));
			}
		}
		cv::imshow("Result",raw);
		cv::waitKey(50);
	}
	source.release();
	return 0;
}
