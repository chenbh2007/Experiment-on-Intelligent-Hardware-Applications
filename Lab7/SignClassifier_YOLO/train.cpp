#include "YoloDataset.hpp"
#include "YOLOStructure.hpp"
#include "toml.hpp"

#include <csignal>
#include <iostream>
#include <filesystem>

static volatile sig_atomic_t is_running=1;
void handler(int sig){
    (void)sig;
    is_running=0;
}

yolo::Sample stack_batch(std::vector<yolo::Sample> samples){
    std::vector<torch::Tensor> inputs;
    std::vector<std::vector<torch::Tensor> > targets(samples.front().target.size());
    for (int k=0;k<samples.size();++k){
        inputs.push_back(samples[k].data);
        for (int l=0;l<targets.size();++l){
            targets[l].push_back(samples[k].target[l]);
        }
    }
    std::vector<torch::Tensor> target;
    for (int k=0;k<targets.size();++k)
        target.push_back(torch::stack(targets[k],0));
    return {torch::stack(inputs,0),std::move(target)};
}

torch::Tensor loss(std::vector<torch::Tensor> res,std::vector<torch::Tensor> tar,float l_cls,float l_box,float l_ctr,float alpha=0.25,float gamma=2){
    torch::nn::BCEWithLogitsLoss bceloss(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kNone));
    auto full=torch::indexing::Slice();
    torch::Tensor box_loss_tot=torch::zeros({},res[0].options().dtype(torch::kFloat32));
    torch::Tensor weight_sum=torch::zeros({},res[0].options().dtype(torch::kFloat32));
    torch::Tensor ctr_loss_tot=torch::zeros({},res[0].options().dtype(torch::kFloat32));
    torch::Tensor pos_sum=torch::zeros({},res[0].options().dtype(torch::kFloat32));
    torch::Tensor cls_loss_tot=torch::zeros({},res[0].options().dtype(torch::kFloat32));
    torch::Tensor size_sum=torch::zeros({},res[0].options().dtype(torch::kFloat32));
    for (int i=0;i<3;++i){
        torch::Tensor positive=tar[i*2+1].gt(0).any(1);
        torch::Tensor ctr_target=tar[i*2].select(1,0).index({positive});
        ctr_loss_tot+=bceloss->forward(res[i*2].select(1,0).index({positive}),ctr_target).sum();
        pos_sum+=positive.sum();
        torch::Tensor box_target=tar[i*2].permute({0,2,3,1}).index({positive});
        torch::Tensor box_res=torch::softplus(res[i*2].permute({0,2,3,1}).index({positive}));
        torch::Tensor p1_target=box_target.index({full,torch::tensor({1,3})});
        torch::Tensor p3_target=box_target.index({full,torch::tensor({2,4})});
        torch::Tensor p1_res=box_res.index({full,torch::tensor({1,3})});
        torch::Tensor p3_res=box_res.index({full,torch::tensor({2,4})});
        torch::Tensor I_area=(torch::minimum(p1_target,p1_res)+torch::minimum(p3_target,p3_res)).prod(1);
        torch::Tensor C_area=(torch::maximum(p1_target,p1_res)+torch::maximum(p3_target,p3_res)).prod(1);
        torch::Tensor U_area=(p1_target+p3_target).prod(1)+(p1_res+p3_res).prod(1)-I_area;
        torch::Tensor giou=I_area/U_area.clamp_min(1e-7)-(C_area-U_area)/C_area.clamp_min(1e-7);
        torch::Tensor weight=ctr_target.to(torch::kFloat32);
        box_loss_tot+=((1.0-giou)*weight).sum();
        weight_sum+=weight.sum();
        torch::Tensor bce=bceloss->forward(res[i*2+1],tar[i*2+1]);
        torch::Tensor pt=(-bce).exp();
        torch::Tensor alpha_t=alpha*tar[i*2+1]+(1-alpha)*(1-tar[i*2+1]);
        cls_loss_tot+=(alpha_t*(1.0-pt).pow(gamma)*bce).sum();
        size_sum+=tar[i*2].size(2)*tar[i*2].size(3)*tar[i*2].size(0);
    }
    return l_cls*cls_loss_tot/size_sum.clamp_min(1)+l_box*box_loss_tot/weight_sum.clamp_min(1e-7)+l_ctr*ctr_loss_tot/pos_sum.clamp_min(1);
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

    std::string state_file=config["model"]["state_file"].value_or("state.pt");
    std::string best_state_file=config["model"]["best_state_file"].value_or("best.pt");
    if (std::filesystem::exists(state_file)) torch::load(Net,state_file);

    std::string train_picture_dir=config["data"]["train_picture_dir"].value_or("/path/to/dataset");
    std::string train_annotations_dir=config["data"]["train_annotations_dir"].value_or("/path/to/annotations"); 
    size_t max_size=config["data"]["max_size"].value_or(512*1024);
    yolo::YoloDataset train_dataset(train_annotations_dir,train_picture_dir,max_size);

    std::string eval_picture_dir=config["data"]["eval_picture_dir"].value_or("/path/to/eval_dataset");
    std::string eval_annotations_dir=config["data"]["eval_picture_dir"].value_or("/path/to/eval_annotations");
    yolo::YoloDataset eval_dataset(eval_annotations_dir,eval_picture_dir,max_size);

    int batch_size=config["control"]["batch_size"].value_or(64);
    int opt_every_step=config["control"]["opt_every_step"].value_or(1);
    int test_every_step=config["control"]["test_every_step"].value_or(1000);
    int save_every_step=config["control"]["save_every_step"].value_or(200); 
    auto train_dataloader=torch::data::make_data_loader(
            std::move(train_dataset).map(torch::data::transforms::Collate<yolo::Sample>(stack_batch)),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(0)
            );
    auto eval_dataloader=torch::data::make_data_loader(
            std::move(train_dataset).map(torch::data::transforms::Collate<yolo::Sample>(stack_batch)),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(0)
            );

    std::string device_str=config["control"]["device"].value_or("cpu");
    torch::Device device=torch::kCPU;
    if (device_str=="cuda"){
        if (torch::cuda::is_available){
            std::cout<<"GPU is available."<<std::endl;
            device=torch::Device(torch::kCUDA);
        }
        else
            std::cout<<"GPU is not availabel. Run on CPU. "<<std::endl;
    }
    else
        std::cout<<"Run on CPU."<<std::endl;

    double eps=config["optiom"]["eps"].value_or(1e-8);
    double lr=config["optim"]["lr"].value_or(1e-4);
    double weight_decay=config["optim"]["weight_decay"].value_or(0.01);
    std::string optim_checkpoint=config["optim"]["checkpoint"].value_or("checkpoint.pt");
    torch::optim::AdamW optimizer(
            Net->parameters(),
            torch::optim::AdamWOptions(lr)
            .betas({0.9,0.999})
            .weight_decay(weight_decay)
            .eps(eps)
            );
    if (std::filesystem::exists(optim_checkpoint)) torch::load(optimizer,optim_checkpoint);

    Net->to(device);
    Net->train();
}


