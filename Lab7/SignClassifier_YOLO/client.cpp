#include<opencv2/opencv.hpp>
#include<iostream>
#include<csignal>
#include<torch/torch.h>
#include<filesystem>
#include<cmath>
#include"YOLOStructure.hpp"
#include"toml.hpp"
#include"YoloDataset.hpp"

extern std::vector<torch::Tensor> getboxes(const std::vector<torch::Tensor>&,float,int,int,bool);
extern std::vector<torch::Tensor> nms(const std::vector<torch::Tensor>&,float,int);

static volatile sig_atomic_t is_running=1;
void handler(int sig){
    (void)sig;
    is_running=0;
}

int main(int argc,char **argv){
    std::signal(SIGINT,handler);
    if (argc<2){
        std::cout<<"Too few arguments."<<std::endl;
        return -1;
    }
    auto config=toml::parse_file(argv[1]);

    int num_bottleneck=config["YOLO"]["num_bottleneck"].value_or(1);
    int base=config["YOLO"]["base"].value_or(64);
    bool enable_shortcut=config["YOLO"]["enable_shortcut"].value_or(true);
    float r_bottleneck=config["YOLO"]["r_bottleneck"].value_or(2.0f);
    float r_C2f=config["YOLO"]["r_C2f"].value_or(2.0f);
    int num_class=yolo::gesture_classes.size();
    auto Net=std::make_shared<yolo::YOLO>(3,num_class,base,base*32,r_C2f,r_bottleneck,num_bottleneck,enable_shortcut);

    std::string device_str=config["predict"]["device"].value_or("cpu");
    float conf_threshold=config["predict"]["conf_threshold"].value_or(0.6f);
    float box_threshold=config["predict"]["box_threshold"].value_or(0.25f);
    std::string camera=config["predict"]["camera"].value_or("/dev/video0");
    cv::VideoCapture cams(camera);
    if (!cams.isOpened()){
        std::cout<<"Cannot open the camera. Exit."<<std::endl;
        return -1;
    }

    torch::Device device=torch::kCPU;
    if (device_str=="cuda"){
        if (torch::cuda::is_available()){
            std::cout<<"CUDA is available."<<std::endl;
            device=torch::kCUDA;
        }
        else{
            std::cout<<"CUDA is not available. Run on CPU."<<std::endl;
        }
    }

    std::string best_state_file=config["model"]["best_state_file"].value_or("best.pt");
    if (std::filesystem::exists(best_state_file)) torch::load(Net,best_state_file,device);
    else{
        std::cout<<"State file not found."<<std::endl;
        return -1;
    }

    int w=config["data"]["width"].value_or(512);
    int h=config["data"]["height"].value_or(512);

    torch::NoGradGuard no_grad;
    Net->to(device);
    Net->eval();
    cv::namedWindow("Result",cv::WINDOW_AUTOSIZE);
    while (is_running){
        cv::Mat raw;
        cams>>raw;
        if (raw.empty()) continue;
        cv::Mat resized;
        double scale=yolo::resize(raw,resized,w,h);
        cv::cvtColor(resized,resized,cv::COLOR_BGR2HSV);
        cv::Mat channels[3];
        cv::split(resized,channels);
        channels[0].convertTo(channels[0],CV_32F,1/180.0);
        channels[1].convertTo(channels[1],CV_32F,1/255.0);
        channels[2].convertTo(channels[2],CV_32F,1/255.0);
        auto opt=torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input=torch::stack({
                torch::from_blob(channels[0].ptr<float>(),{h,w},opt),
                torch::from_blob(channels[1].ptr<float>(),{h,w},opt),
                torch::from_blob(channels[2].ptr<float>(),{h,w},opt)
                },0).unsqueeze(0);
        input=input.to(device);
        std::vector<torch::Tensor> output=Net->forward(input);
        for (int i=0;i<3;i++){
            output[i*2]=torch::cat({output[i*2].slice(1,0,1).sigmoid(),torch::softplus(output[i*2].slice(1,1,5))},1);
            output[i*2+1]=output[i*2+1].sigmoid();
        }
        std::vector<torch::Tensor> cls_boxes=getboxes(output,conf_threshold,3,8,false);
        for (int cls=0;cls<num_class;cls++){
            int num_boxes=cls_boxes[cls].size(0);
            torch::Tensor cls_dim=torch::full({num_boxes,1},static_cast<float>(cls),cls_boxes[cls].options());
            cls_boxes[cls]=torch::cat({cls_boxes[cls],cls_dim},1);
        }
        torch::Tensor tot_boxes=torch::cat(cls_boxes,0);
        std::vector<torch::Tensor> detected_boxes=nms({tot_boxes},box_threshold,100);
        int n=detected_boxes[0].size(0);
        torch::Tensor boxes=detected_boxes[0].cpu().contiguous();
        auto acc=boxes.accessor<float,2>();
        for (int i=0;i<n;i++){
            cv::Rect box(acc[i][0]/scale,acc[i][2]/scale,acc[i][1]/scale-acc[i][0]/scale,acc[i][3]/scale-acc[i][2]/scale);
            int cls=static_cast<int>(std::round(acc[i][5]));
            float conf=acc[i][4];
            cv::rectangle(raw,box,cv::Scalar(0,255*conf,255*(1-conf)));
            std::string cls_name=yolo::gesture_classes_dict[cls];
            cv::putText(raw,cls_name,cv::Point((acc[i][0]+acc[i][1])/(2*scale),(acc[i][2]+acc[i][3])/(2*scale)),cv::FONT_HERSHEY_PLAIN,0.8,cv::Scalar(255,0,0));
        }
        cv::imshow("Result",raw);
        if(cv::waitKey(10)==27) break;
    }
    cams.release();
    cv::destroyAllWindows();
    return 0;
}
