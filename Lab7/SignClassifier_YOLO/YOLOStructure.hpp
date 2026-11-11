#ifndef YOLO_net
#define YOLO_net
#include<torch/torch.h>
#include<vector>
#include<memory>
namespace yolo{
    struct Conv:torch::nn::Module{
        torch::nn::Conv2d conv{nullptr};
        torch::nn::BatchNorm2d bn{nullptr};
        Conv(int in_channel,int out_channel,int kernel_size,int s=1,int p=0){
            conv=register_module("conv",torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel,out_channel,kernel_size).stride(s).padding(p).bias(false)));
            bn=register_module("bn",torch::nn::BatchNorm2d(out_channel));
        }
        torch::Tensor forward(torch::Tensor input){
            torch::Tensor x=conv->forward(input);
            x=bn->forward(x);
            return torch::silu(x);
        }
    };
    struct bottleneck:torch::nn::Module{
        std::shared_ptr<Conv> conv1;
        std::shared_ptr<Conv> conv2;
        bool act_shortcut;
        bottleneck(int in_channel,int out_channel,int hidden_channel,int p=1,bool shortcut=true){
            conv1=register_module("conv1",std::make_shared<Conv>(in_channel,hidden_channel,1,1,0));
            conv2=register_module("conv2",std::make_shared<Conv>(hidden_channel,out_channel,3,1,p));
            if (shortcut&&in_channel==out_channel&&p==1) act_shortcut=true;
            else act_shortcut=false;
        }
        torch::Tensor forward(torch::Tensor input){
            torch::Tensor x=conv2->forward(conv1->forward(input));
            if (act_shortcut){
                return x+input;
            }
            else return x;
        }
    };
    struct C2f:torch::nn::Module{
        std::shared_ptr<Conv> conv1;
        std::shared_ptr<Conv> conv2;
        torch::nn::ModuleList bns;
        int c_mid;
        C2f(int in_channel,int middle_channel,int out_channel,int hidden_channel,int num_bottleneck,int p=1,bool shortcut=false){
            bns=register_module("bns",torch::nn::ModuleList());
            for (int i=0;i<num_bottleneck;i++)
                bns->push_back(std::make_shared<bottleneck>(middle_channel,middle_channel,hidden_channel,p,shortcut));
            c_mid=middle_channel;
            conv1=register_module("conv1",std::make_shared<Conv>(in_channel,2*middle_channel,1,1,0));
            conv2=register_module("conv2",std::make_shared<Conv>((num_bottleneck+2)*middle_channel,out_channel,1,1,0));
        }
        torch::Tensor forward(torch::Tensor input){
            torch::Tensor x=conv1->forward(input);
            std::vector<int64_t> sections={c_mid,c_mid};
            std::vector<torch::Tensor> parts=torch::split(x,sections,1);
            torch::Tensor y=parts[1];
            for (auto &bn:*bns){
                y=bn->as<bottleneck>()->forward(y);
                parts.push_back(y);
            }
            torch::Tensor output=torch::cat(parts,1);
            return conv2->forward(output);
        }
    };
    struct Backbone:torch::nn::Module{
        std::shared_ptr<Conv> cv1,cv2,cv3,cv4,cv5;
        std::shared_ptr<C2f> c2,c3,c4,c5;
        torch::nn::ModuleList SPPF;
        Backbone(int in_channel,int middle_channel_C2f,int base,int hidden_channel_bottleneck,int num_bottleneck,int p_bottleneck=1,bool shortcut_bottleneck=true){
            cv1=register_module("cv1",std::make_shared<Conv>(in_channel,base,3,2,1));
            cv2=register_module("cv2",std::make_shared<Conv>(base,base*2,3,2,1));
            cv3=register_module("cv3",std::make_shared<Conv>(base*2,base*4,3,2,1));
            cv4=register_module("cv4",std::make_shared<Conv>(base*4,base*8,3,2,1));
            cv5=register_module("cv5",std::make_shared<Conv>(base*8,base*16,3,2,1));
            c2=register_module("c2f2",std::make_shared<C2f>(base*2,middle_channel_C2f*2,base*2,hidden_channel_bottleneck*2,num_bottleneck,p_bottleneck,shortcut_bottleneck));
            c3=register_module("c2f3",std::make_shared<C2f>(base*4,middle_channel_C2f*4,base*4,hidden_channel_bottleneck*4,num_bottleneck,p_bottleneck,shortcut_bottleneck));
            c4=register_module("c2f4",std::make_shared<C2f>(base*8,middle_channel_C2f*8,base*8,hidden_channel_bottleneck*8,num_bottleneck,p_bottleneck,shortcut_bottleneck));
            c5=register_module("c2f5",std::make_shared<C2f>(base*16,middle_channel_C2f*16,base*16,hidden_channel_bottleneck*16,num_bottleneck,p_bottleneck,shortcut_bottleneck));
            SPPF=register_module("SPPF",torch::nn::ModuleList());
            for (int i=0;i<3;i++){
                SPPF->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(5).stride(1).padding(2)));
                SPPF->push_back(std::make_shared<Conv>(base*32,base*16,1,1,0));
            }
        }
        std::vector<torch::Tensor> forward(torch::Tensor input){
            torch::Tensor x=c2->forward(cv2->forward(cv1->forward(input)));
            std::vector<torch::Tensor> output;
            x=c3->forward(cv3->forward(x));
            output.push_back(x);
            x=c4->forward(cv4->forward(x));
            output.push_back(x);
            x=c5->forward(cv5->forward(x));
            torch::Tensor y=x;
            for (int i=0;i<3;i++){
                x=SPPF[i*2]->as<torch::nn::MaxPool2d>()->forward(x);
                y=torch::cat({x,y},1);
                y=SPPF[i*2+1]->as<Conv>()->forward(y);
            }
            output.push_back(y);
            return output;
        }
    };
    struct FPN:torch::nn::Module{
        std::shared_ptr<Conv> cv3,cv4,cv5;
        std::shared_ptr<C2f> c4,c3;
        FPN(int base,int out_channel,int middle_channel_C2f,int hidden_channel_bottleneck,int num_bottleneck,int p_bottleneck=1,bool shortcut_bottleneck=true){
            cv5=register_module("cv5",std::make_shared<Conv>(base*16,out_channel,1,1,0));
            cv4=register_module("cv4",std::make_shared<Conv>(base*8,out_channel,1,1,0));
            cv3=register_module("cv3",std::make_shared<Conv>(base*4,out_channel,1,1,0));
            c4=register_module("c4",std::make_shared<C2f>(out_channel*2,middle_channel_C2f,out_channel,hidden_channel_bottleneck,num_bottleneck,p_bottleneck,shortcut_bottleneck));
            c3=register_module("c3",std::make_shared<C2f>(out_channel*2,middle_channel_C2f,out_channel,hidden_channel_bottleneck,num_bottleneck,p_bottleneck,shortcut_bottleneck));
        }
        std::vector<torch::Tensor> forward(std::vector<torch::Tensor> input){
            torch::Tensor x=cv5->forward(input[2]);
            torch::Tensor y=cv4->forward(input[1]);
            torch::Tensor z=cv3->forward(input[0]);
            y=torch::cat({torch::upsample_nearest2d(x,{y.size(2),y.size(3)}),y},1);
            y=c4->forward(y);
            z=torch::cat({torch::upsample_nearest2d(y,{z.size(2),z.size(3)}),z},1);
            z=c3->forward(z);
            return {z,y,x};
        }
    };
    struct PAN:torch::nn::Module{
        std::shared_ptr<Conv> cv3,cv4;
        std::shared_ptr<C2f> c3,c4;
        PAN(int in_channel,/*out_channel=in_channel*/int middle_channel_C2f,int hidden_channel_bottleneck,int num_bottleneck,int p_bottleneck=1,bool shortcut_bottleneck=true){
            cv3=register_module("cv3",std::make_shared<Conv>(in_channel,in_channel,3,2,1));
            cv4=register_module("cv4",std::make_shared<Conv>(in_channel,in_channel,3,2,1));
            c3=register_module("c3",std::make_shared<C2f>(2*in_channel,middle_channel_C2f,in_channel,hidden_channel_bottleneck,num_bottleneck,p_bottleneck,shortcut_bottleneck));
            c4=register_module("c4",std::make_shared<C2f>(2*in_channel,middle_channel_C2f,in_channel,hidden_channel_bottleneck,num_bottleneck,p_bottleneck,shortcut_bottleneck));
        }
        std::vector<torch::Tensor> forward(std::vector<torch::Tensor> input){
            torch::Tensor x=input[0];
            torch::Tensor y=input[1];
            torch::Tensor z=input[2];
            y=torch::cat({cv3->forward(x),y},1);
            y=c3->forward(y);
            z=torch::cat({cv4->forward(y),z},1);
            z=c4->forward(z);
            return {x,y,z};
        }
    };
    struct Neck:torch::nn::Module{
        std::shared_ptr<FPN> fpn;
        std::shared_ptr<PAN> pan;
        Neck(int base,int out_channel,int middle_channel_C2f,int hidden_channel_bottleneck,int num_bottleneck,int p_bottleneck=1,bool shortcut_bottleneck=true){
            fpn=register_module("FPN",std::make_shared<FPN>(base,out_channel,middle_channel_C2f,hidden_channel_bottleneck,num_bottleneck,p_bottleneck,shortcut_bottleneck));
            pan=register_module("PAN",std::make_shared<PAN>(out_channel,middle_channel_C2f,hidden_channel_bottleneck,num_bottleneck,p_bottleneck,shortcut_bottleneck));
        }
        std::vector<torch::Tensor> forward(std::vector<torch::Tensor> input){
            return pan->forward(fpn->forward(input));
        }
    };
    struct SingleHead:torch::nn::Module{
        std::shared_ptr<Conv> in;
        std::shared_ptr<Conv> middle;
        torch::nn::Conv2d out{nullptr};
        SingleHead(int in_channel,int out_channel){
            in=register_module("in",std::make_shared<Conv>(in_channel,in_channel/2,3,1,1));
            middle=register_module("mid",std::make_shared<Conv>(in_channel/2,in_channel/2,3,1,1));
            out=register_module("out",torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel/2,out_channel,1).stride(1).padding(0)));
        }
        torch::Tensor forward(torch::Tensor input){
            return out->forward(middle->forward(in->forward(input)));
        }
    };
    struct Head:torch::nn::Module{
        std::shared_ptr<SingleHead> LocateHeadP3,LocateHeadP4,LocateHeadP5;  
        std::shared_ptr<SingleHead> ClassifyHeadP3,ClassifyHeadP4,ClassifyHeadP5;
        Head(int in_channel,int num_class){
            LocateHeadP3=register_module("LocateHeadP3",std::make_shared<SingleHead>(in_channel,5));
            LocateHeadP4=register_module("LocateHeadP4",std::make_shared<SingleHead>(in_channel,5));
            LocateHeadP5=register_module("LocateHeadP5",std::make_shared<SingleHead>(in_channel,5));
            ClassifyHeadP3=register_module("ClassifyHeadP3",std::make_shared<SingleHead>(in_channel,num_class));
            ClassifyHeadP4=register_module("ClassifyHeadP4",std::make_shared<SingleHead>(in_channel,num_class));
            ClassifyHeadP5=register_module("ClassifyHeadP5",std::make_shared<SingleHead>(in_channel,num_class));
        }
        std::vector<torch::Tensor> forward(std::vector<torch::Tensor>input){
            torch::Tensor P3=input[0];
            torch::Tensor P4=input[1];
            torch::Tensor P5=input[2];
            return {LocateHeadP3->forward(P3),ClassifyHeadP3->forward(P3),LocateHeadP4->forward(P4),ClassifyHeadP4->forward(P4),LocateHeadP5->forward(P5),ClassifyHeadP5->forward(P5)};
        }
    };
    struct YOLO:torch::nn::Module{
        std::shared_ptr<Backbone> backbone;
        std::shared_ptr<Neck> neck;
        std::shared_ptr<Head> head;
        YOLO(int in_channel,int class_num,int base,int P_channel,float r_middle_channel,float r_hidden_channel,int num_bottleneck,bool shortcut_bottleneck=true){
            int p=1;
            int middle_channel=(int)(base*r_middle_channel),hidden_channel=(int)(base*r_hidden_channel);
            backbone=register_module("Backbone",std::make_shared<Backbone>(in_channel,middle_channel,base,hidden_channel,num_bottleneck,p,shortcut_bottleneck));
            middle_channel=(int)(P_channel*r_middle_channel);
            hidden_channel=(int)(P_channel*r_hidden_channel);
            neck=register_module("Neck",std::make_shared<Neck>(base,P_channel,middle_channel,hidden_channel,num_bottleneck,p,shortcut_bottleneck));
            head=register_module("Head",std::make_shared<Head>(P_channel,class_num));
        }
        std::vector<torch::Tensor> forward(torch::Tensor input){
            return head->forward(neck->forward(backbone->forward(input)));
        }
    };
}
#endif
