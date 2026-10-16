#include<iostream>
#include<cstdio>
#include<cmath>
#include<algorithm>
#include<sstream>
#include<vector>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/objdetect.hpp>
int contour2Vec(const cv::Mat &thresh,const std::vector<cv::Point> &contour,const std::vector<cv::Point2f> &dst,float Area,std::vector<float> &vec){		
    std::vector<cv::Point> hull;
	cv::convexHull(contour,hull);
	float hullArea=cv::contourArea(hull);
	if(hullArea>0.5*Area||hullArea<0.005*Area) return -1;
	std::vector<cv::Point2f> approxPoly;
	cv::approxPolyDP(hull,approxPoly,0.02*cv::arcLength(hull,true),true);
	if (approxPoly.size()!=4||!cv::isContourConvex(approxPoly)) return -1;
	std::vector<cv::Point2f> src;
	for (int j=0;j<4;j++)
		src.push_back(approxPoly[0]);
	for (int j=1;j<4;j++){
		if (approxPoly[j].x+approxPoly[j].y<src[0].x+src[0].y) src[0]=approxPoly[j];
		if (approxPoly[j].x+approxPoly[j].y>src[2].x+src[2].y) src[2]=approxPoly[j];
		if (approxPoly[j].x-approxPoly[j].y>src[1].x-src[1].y) src[1]=approxPoly[j];
		if (approxPoly[j].x-approxPoly[j].y<src[3].x-src[3].y) src[3]=approxPoly[j];
	}
	cv::Mat res;
	cv::Mat kernel=cv::getPerspectiveTransform(src,dst);
	cv::warpPerspective(thresh,res,kernel,cv::Size(32,32));
    cv::Mat morphkernel=cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3));
    cv::morphologyEx(res,res,cv::MORPH_CLOSE,morphkernel);
    cv::Moments m=cv::moments(255-res);
    float cx=m.m10/m.m00;
    float cy=m.m01/m.m00;
    float Sx=m.mu20/m.m00;
    float Sy=m.mu02/m.m00;
    float current_spread=std::sqrt(std::max(Sx,Sy));
    float target_spread=6.0f;
    float s=target_spread/current_spread;
    kernel=cv::Mat_<float>({2,3},{s,0,16-s*cx,0,s,16-s*cy});
    cv::warpAffine(res,res,kernel,cv::Size(32,32),cv::INTER_LINEAR,cv::BORDER_CONSTANT,255);
	cv::HOGDescriptor hog(cv::Size(32,32),cv::Size(16,16),cv::Size(8,8),cv::Size(8,8),9,1,-1,cv::HOGDescriptor::L2Hys,0.2,true,cv::HOGDescriptor::DEFAULT_NLEVELS,false);
	hog.compute(res,vec);
    return 0;
}
int main(int argc,char **argv){
	FILE *f=fopen("raw.db","a");
	if (argc<3){
		std::cout<<"Too few Arguments."<<std::endl;
		return -1;
	}
	cv::Mat ori=cv::imread(argv[1]);
	int nCols=ori.cols,nRows=ori.rows;
	cv::Mat gray,thresh;
	cv::cvtColor(ori,gray,cv::COLOR_BGR2GRAY);
	cv::medianBlur(gray,gray,3);
	cv::adaptiveThreshold(gray,thresh,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY,nCols/10*2+1,0);
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(thresh,contours,cv::RETR_LIST,cv::CHAIN_APPROX_SIMPLE);
	std::vector<cv::Point2f> dst;
	dst.push_back(cv::Point2f(-2,-2));
	dst.push_back(cv::Point2f(34,-2));
	dst.push_back(cv::Point2f(34,34));
	dst.push_back(cv::Point2f(-2,34));
	cv::imwrite("thresh.jpg",thresh);
	for (int i=0;i<contours.size();i++){
        std::vector<float> vec;
        if (contour2Vec(thresh,contours[i],dst,nCols*nRows,vec)==-1) continue;
		for (int j=0;j<vec.size();j++){
			fprintf(f,"%.6f ",vec[j]);
		}
		fprintf(f,"%s\n",argv[2]);
	}
	fclose(f);
	return 0;
}
