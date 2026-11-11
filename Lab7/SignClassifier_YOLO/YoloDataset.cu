#include "YoloDataset.hpp"

#include <opencv2/opencv.hpp>
#include <cjson/cJSON.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        const cudaError_t yolo_cuda_error_ = (call);                            \
        TORCH_CHECK(                                                           \
            yolo_cuda_error_ == cudaSuccess,                                   \
            "CUDA call failed: ", #call, " -> ",                               \
            cudaGetErrorName(yolo_cuda_error_),                                \
            " (", static_cast<int>(yolo_cuda_error_), "): ",                    \
            cudaGetErrorString(yolo_cuda_error_));                              \
    } while (false)

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)>(b))?(b):(a))

std::map<std::string,int64_t> gesture_classes={{"call",0},
                                                {"dislike",1},
                                                {"fist",2},
                                                {"four",3},
                                                {"like",4},
                                                {"mute",5},
                                                {"ok",6},
                                                {"one",7},
                                                {"palm",8},
                                                {"three3",9},
                                                {"thumb_index2",10},
                                                {"peace",11},
                                                {"peace_inverted",12},
                                                {"rock",13},
                                                {"stop",14},
                                                {"stop_inverted",15},
                                                {"three",16},
                                                {"three2",17},
                                                {"two_up",18},
                                                {"two_up_inverted",19},
                                                {"three_gun",20},
                                                {"timeout",21},
                                                {"grabbing",22},
                                                {"grip",23},
                                                {"hand_heart",24},
                                                {"hand_heart2",25},
                                                {"holy",26},
                                                {"little_finger",27},
                                                {"middle_finger",28},
                                                {"point",29},
                                                {"take_picture",30},
                                                {"thumb_index",31},
                                                {"xsign",32},
                                                {"no_gesture",33}
};

__global__ void resize(float *d_pic,float *d_target,int w,int h,int target_w,int target_h,double scale){
    size_t j=blockIdx.x*blockDim.x+threadIdx.x;
    size_t i=blockIdx.y*blockDim.y+threadIdx.y;
    if (i<target_h&&j<target_w){
        size_t source_i_lower=(((i-1)*scale)>=h) ? (h-1):((i-1)*scale);
        source_i_lower=(i>0)? source_i_lower:0;
        size_t source_j_lower=(((j-1)*scale)>=w) ? (w-1):((j-1)*scale);
        source_j_lower=(j>0)? source_j_lower:0;
        size_t source_i_upper=(((i+1)*scale)>h) ? h:((i+1)*scale);
        size_t source_j_upper=(((j+1)*scale)>w) ? w:((j+1)*scale);
        d_target[i*target_w+j]=0;
        int cnt=0;
        for (size_t si=source_i_lower;si<source_i_upper;si++){
            for (size_t sj=source_j_lower;sj<source_j_upper;sj++){
                cnt++;
                d_target[i*target_w+j]+=d_pic[si*w+sj];
            }
        }
        if (cnt!=0)
            d_target[i*target_w+j]/=(double)cnt;
    }
}

YoloDataset::YoloDataset(std::string annotations_dir,std::string data_dir,size_t size0,int w,int h):width(w),height(h),maxsize(size0){
    for (const auto &annotations:std::filesystem::directory_iterator(annotations_dir)){
        if (annotations.path().extension()==".json"){
            std::ifstream log(annotations.path().string());
            if (!log.is_open()) continue;
            std::string content{std::istreambuf_iterator<char>(log),std::istreambuf_iterator<char>()};
            cJSON* tot_json=cJSON_Parse(content.c_str());
            if (!tot_json) continue;
            std::string class_name=annotations.path().filename().stem().string();
            for (const auto &pictures:std::filesystem::directory_iterator(std::filesystem::path(data_dir)/class_name)){
                if (pictures.path().extension()==".jpg"){
                    std::string pic_name=pictures.path().filename().stem().string();
                    cJSON* pic_json=cJSON_GetObjectItem(tot_json,pic_name.c_str());
                    if (!pic_json) continue; 
                    cJSON* bboxes_json=cJSON_GetObjectItem(pic_json,"bboxes");
                    cJSON* labels_json=cJSON_GetObjectItem(pic_json,"labels");
                    cJSON* united_bbox_json=cJSON_GetObjectItem(pic_json,"united_bbox");
                    cJSON* united_label_json=cJSON_GetObjectItem(pic_json,"united_label");
                    if (!bboxes_json&&!united_bbox_json) continue;
                    size_t num_bboxes=cJSON_GetArraySize(bboxes_json);
                    size_t num_ubboxes=cJSON_GetArraySize(united_bbox_json);
                    if (num_bboxes==0&&num_ubboxes==0) continue;
                    pic_paths.push_back(pictures.path().string());
                    std::vector<std::vector<float> > bboxes_tmp;
                    std::vector<int64_t> labels_tmp;
                    if (num_ubboxes==0)
                        for (int i=0;i<num_bboxes;++i){
                            cJSON *bbox_json=cJSON_GetArrayItem(bboxes_json,i);
                            std::vector<float> bbox;
                            for (int j=0;j<4;++j)
                                bbox.push_back(cJSON_GetArrayItem(bbox_json,j)->valuedouble);
                            std::string label_name=cJSON_GetArrayItem(labels_json,i)->valuestring;
                            int64_t label=gesture_classes["no_gesture"];
                            if (gesture_classes.find(label_name)!=gesture_classes.end()){
                                label=gesture_classes[label_name];
                            }
                            labels_tmp.push_back(label);
                            bboxes_tmp.push_back(bbox);
                        }
                    else{
                        for (int i=0;i<num_ubboxes;++i){
                            cJSON *united_bbox=cJSON_GetArrayItem(united_bbox_json,i);
                            std::vector<float> bbox;
                            for (int j=0;j<4;++j){
                                bbox.push_back(cJSON_GetArrayItem(united_bbox,j)->valuedouble);
                            }
                            int64_t label=gesture_classes["no_gesture"];
                            std::string label_name=cJSON_GetArrayItem(united_label_json,i)->valuestring;
                            if (gesture_classes.find(label_name)!=gesture_classes.end()){
                                label=gesture_classes[label_name];
                            }
                            labels_tmp.push_back(label);
                            bboxes_tmp.push_back(bbox);    
                        }
                    }
                    bboxes.push_back(bboxes_tmp);
                    labels.push_back(labels_tmp);
                }
            }
            cJSON_Delete(tot_json);
        }
    }
    d_buf.h_output=(float*)malloc(w*h*sizeof(float));
    CHECK_CUDA(cudaMalloc((void**)&d_buf.d_input,size0*sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_buf.d_output,w*h*sizeof(float)));
}

torch::data::Example<torch::Tensor,std::vector<torch::Tensor> > YoloDataset::get(size_t index){
    const int64_t w=width,h=height;
    cv::Mat raw=cv::imread(pic_paths[index],cv::IMREAD_COLOR);
    cv::cvtColor(raw,raw,cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> raw_channels;
    cv::split(raw,raw_channels);
    raw_channels[0].convertTo(raw_channels[0],CV_32F,1/180.0);
    raw_channels[1].convertTo(raw_channels[1],CV_32F,1/255.0);
    raw_channels[2].convertTo(raw_channels[2],CV_32F,1/255.0);
    size_t w0=raw.cols;
    size_t h0=raw.rows;
    const size_t required_pixels=w0*h0;
    double coeff=(((double)h0/h)>((double)w0/w))? ((double)h0/h):((double)w0/w);
    torch::Tensor input=torch::empty({0,h,w},torch::kFloat32);
    dim3 block(32,32);
    dim3 grid((w+31)/32,(h+31)/32);
    if (required_pixels>maxsize){
        std::cout<<"Raw size exceed limit. Expanding buffer."<<std::endl;
        CHECK_CUDA(cudaFree(d_buf.d_input));
        d_buf.d_input=nullptr;
        maxsize=0;
        CHECK_CUDA(cudaMalloc((void**)&d_buf.d_input,required_pixels*sizeof(float)));
        maxsize=required_pixels;
    }
    for (int i=0;i<3;i++){
        CHECK_CUDA(cudaMemcpy(d_buf.d_input,raw_channels[i].ptr<float>(),required_pixels*sizeof(float),cudaMemcpyHostToDevice));
        resize<<<grid,block>>>(d_buf.d_input,d_buf.d_output,w0,h0,w,h,coeff);
        // Check launch errors here; the blocking copy also reports execution errors.
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(d_buf.h_output,d_buf.d_output,w*h*sizeof(float),cudaMemcpyDeviceToHost));
        input=torch::cat({input,torch::from_blob(d_buf.h_output,{1,h,w},torch::kFloat32).clone()},0);
    }
    std::vector<torch::Tensor> target;        
    int64_t num_class=gesture_classes.size();
    torch::Tensor locatepic=torch::zeros({5,h,w},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)),
        predictpic=torch::zeros({num_class,h,w},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto locate=locatepic.accessor<float,3>();
    auto predict=predictpic.accessor<float,3>();
    for (int s=8;s<=32;s*=2){
        for (int j=0;j<bboxes[index].size();j++){
            int x1=(int)((double)w0*bboxes[index][j][0]/coeff);
            int y1=(int)((double)h0*bboxes[index][j][1]/coeff);
            int x2=(int)((double)w0*(bboxes[index][j][2]+bboxes[index][j][0])/coeff);
            int y2=(int)((double)h0*(bboxes[index][j][3]+bboxes[index][j][1])/coeff);
            int xc=(x1+x2)/2;
            int yc=(y1+y2)/2;
            for (int y=yc-s*1.5;y<yc+s*1.5;y++){
                for (int x=xc-s*1.5;x<xc+s*1.5;x++){
                    if (x>=0&&x<w&&y>=0&&y<h&&x>=x1&&x<x2&&y>=y1&&y<y2){
                        float ctr=std::sqrt((float)MIN(x-x1,x2-x)/(float)MAX(x-x1,x2-x)*(float)MIN(y-y1,y2-y)/(float)MAX(y-y1,y2-y));
                        if (ctr<locate[0][y][x]){
                            continue;
                        }
                        locate[0][y][x]=ctr;
                        locate[1][y][x]=x-x1;
                        locate[2][y][x]=x2-x;
                        locate[3][y][x]=y-y1;
                        locate[4][y][x]=y2-y;
                        for (int k=0;k<num_class;k++)
                            predict[k][y][x]=0;
                        predict[labels[index][j]][y][x]=1;
                    }
                }
            }
        }  
        target.push_back(locatepic.clone());
        target.push_back(predictpic.clone());
    }
    return {input,target};
}

torch::optional<size_t> YoloDataset::size()const{
    return pic_paths.size();
}
