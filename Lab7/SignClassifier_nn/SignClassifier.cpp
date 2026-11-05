#include<torch/torch.h>
#include<iostream>
#include<cstdint>
#include"toml.hpp"
#include<string>
#include<vector>
#include<filesystem>
#include<opencv2/opencv.hpp>
#include<csignal>
static volatile sig_atomic_t is_running=1;
void handle_interrupt(int sig){
    (void)sig;
    is_running=0;
    return;
}
struct torch::nn::Sequential net;
struct ImageDataSet:public torch::data::datasets::Dataset<ImageDataSet>{
private:
    std::vector<std::string> image_paths;
    std::vector<int64_t> labels;
    int width;
    int height;
public:
    ImageDataSet()=default;
    explicit ImageDataSet(const std::string& file_index,int width=80,int height=60):width(width),height(height){
        for (const auto& class_dir :std::filesystem::directory_iterator(file_index)){
            if (class_dir.is_directory()){
                uint8_t current_label;
                std::string class_name=class_dir.path().filename().string();
                if (class_name.compare("Fist")==0){
                    current_label=0;
                }
                else if(class_name.compare("LSign")==0){
                    current_label=1;
                }
                else if(class_name.compare("VSign")==0){
                    current_label=2;
                }
                else if(class_name.compare("Palm")==0){
                    current_label=3;
                }
                else if(class_name.compare("Pointing")==0){
                    current_label=4;
                }
                else continue;
                for (const auto& file:std::filesystem::directory_iterator(class_dir.path())){
                    if (file.path().extension()==".png"||file.path().extension()==".jpg"){
                        image_paths.push_back(file.path().string());
                        labels.push_back(current_label);
                    }
                }
            }
        }
    }
    torch::data::Example<> get(size_t index) override{
        cv::Mat img=cv::imread(image_paths[index],cv::IMREAD_COLOR);
        cv::resize(img,img,cv::Size(width,height));
        cv::cvtColor(img,img,cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> img_channels;
        cv::split(img,img_channels);
        img_channels[0].convertTo(img_channels[0],CV_32F,1.0/180.0);
        img_channels[1].convertTo(img_channels[1],CV_32F,1.0/255.0);
        img_channels[2].convertTo(img_channels[2],CV_32F,1.0/255.0);
        cv::merge(img_channels,img);
        torch::Tensor img_tensor=torch::from_blob(img.data,{height,width,3},torch::kFloat32).clone();
        img_tensor=img_tensor.permute({2,0,1});
        torch::Tensor label_tensor=torch::one_hot(torch::tensor(labels[index],torch::kInt64),5).to(torch::kFloat32);
        return {img_tensor,label_tensor};
    }
    torch::optional<size_t> size() const override{
        return image_paths.size();
    }
    void save_index(const std::string& path) const{
        std::ofstream ofs(path);
        ofs<<width<<" "<<height<<"\n";
        for (size_t i=0;i<image_paths.size();++i) ofs<<image_paths[i]<<" "<<labels[i]<<"\n";
    }
    void load_index(const std::string &path){
        std::ifstream ifs(path);
        image_paths.clear();
        labels.clear();
        std::string p;
        int64_t l;
        ifs>>width>>height;
        while (ifs>>p>>l){
            image_paths.push_back(p);
            labels.push_back(l);
        }
    }
};
template <typename DataLoaderType>
std::pair<double,double> test_model(torch::nn::Sequential &net,DataLoaderType &test_data_loader,torch::Device device){
    net->eval();
    torch::NoGradGuard no_grad;
    double total_loss=0.0;
    int cnt=0,tot=0;
    for (auto &test_batch:*test_data_loader){
        torch::Tensor test_data=test_batch.data.to(device);
        torch::Tensor test_targets=test_batch.target.to(device);
        torch::Tensor test_output=net->forward(test_data);
        torch::Tensor loss=torch::nn::functional::cross_entropy(test_output,test_targets);
        total_loss+=loss.item<double>()*test_data.size(0);
        torch::Tensor pred=test_output.argmax(1);
        torch::Tensor test_labels=test_targets.argmax(1);
        cnt+=pred.eq(test_labels).sum().item<int64_t>();
        tot+=test_data.size(0);
    }
    return std::pair(total_loss/(double)tot,(double)cnt/(double)tot);
}
int main(int argc,char **argv){
    std::signal(SIGINT,handle_interrupt);
    std::string file;
    if (argc<2){
        std::cout<<"Please enter the config file:";
        std::cin>>file;
    }
    else file=argv[1];
    auto config=toml::parse_file(file);
    std::string state_file=config["control"]["state_file"].value_or("baseline.pth");
    std::string best_state_file=config["control"]["best_state_file"].value_or("best.pth");
    std::string device_str=config["control"]["device"].value_or("cpu");
    std::string train_path=config["dataset"]["train_data_path"].value_or("database/train/");
    std::string test_path=config["dataset"]["test_data_path"].value_or("database/test/");
    std::string train_file_index=config["dataset"]["train_file_index"].value_or("train_file_index.pth");
    std::string test_file_index=config["dataset"]["test_file_index"].value_or("test_file_index.pth");
    int batch_size=config["control"]["batch_size"].value<int>().value();
    int h=config["control"]["height"].value_or(60);
    int w=config["control"]["width"].value_or(80);
    int opt_every=config["control"]["opt_every"].value_or(1);
    int save_every=config["control"]["save_every"].value<int>().value();
    int test_every=config["control"]["test_every"].value<int>().value();
    int conv_size=config["nn"]["num_conv_layers"].value<int>().value();
    int linear_size=config["nn"]["num_linear_layers"].value<int>().value();
    int conv_kernel_size=config["nn"]["conv_kernel_size"].value<int>().value();
    int pool_kernel_size=config["nn"]["pool_kernel_size"].value<int>().value();
    int linear_width=config["nn"]["linear_width"].value<int>().value();
    double eps=config["optim"]["eps"].value<double>().value();
    double lr=config["optim"]["lr"].value<double>().value();
    double weight_decay=config["optim"]["weight_decay"].value<double>().value();
    torch::Device device=torch::kCPU;
    if (device_str=="cuda"){
        if (torch::cuda::is_available()){
            device=torch::Device(torch::kCUDA);
            std::cout<<"CUDA is available. "<<std::endl;
        }
        else{
            std::cout<<"CUDA is not available. Use CPU. "<<std::endl;
        }
    }
    else{
        std::cout<<"Running on CPU. "<<std::endl;
    }
    ImageDataSet train_data,test_data;
    if (std::filesystem::exists(train_file_index)){
        train_data.load_index(train_file_index);
    }
    else{
        std::cout<<"Constructing file index..."<<std::endl;
        train_data=ImageDataSet(train_path,w,h);
        train_data.save_index(train_file_index);
        std::cout<<"File index save to "<<train_file_index<<"."<<std::endl;
    }
    if (std::filesystem::exists(test_file_index)){
        test_data.load_index(test_file_index);
    }
    else{
        std::cout<<"Constructing file index..."<<std::endl;
        test_data=ImageDataSet(test_path,w,h);
        test_data.save_index(test_file_index);
        std::cout<<"File index save to "<<test_file_index<<"."<<std::endl;
    }
    std::cout<<"Loading data base..."<<std::endl;
    auto train_data_loader=torch::data::make_data_loader(
            train_data.map(torch::data::transforms::Stack<>()),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(4)
            );
    auto test_data_loader=torch::data::make_data_loader(
            test_data.map(torch::data::transforms::Stack<>()),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(4)
            );
    std::cout<<"Loading model..."<<std::endl;
    int current_w=w;
    int current_h=h;
    int current_c=3;
    net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3,3,conv_kernel_size)));
    net->push_back(torch::nn::ReLU());
    current_w+=1-conv_kernel_size;
    current_h+=1-conv_kernel_size;
    for (int i=0;i<conv_size;i++){
        net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(current_c,3,conv_kernel_size)));
        net->push_back(torch::nn::ReLU());
        current_w+=1-conv_kernel_size;
        current_h+=1-conv_kernel_size;
        net->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(pool_kernel_size)));
        current_w/=pool_kernel_size;
        current_h/=pool_kernel_size;
    }
    net->push_back(torch::nn::Flatten());
    net->push_back(torch::nn::Linear(current_c*current_w*current_h,linear_width));
    net->push_back(torch::nn::ReLU());
    for (int i=0;i<linear_size;i++){
        net->push_back(torch::nn::Linear(linear_width,linear_width));
        net->push_back(torch::nn::ReLU());
    }
    net->push_back(torch::nn::Linear(linear_width,5));
    double best_acc=0.0;
    if (std::filesystem::exists(best_state_file)){
        torch::load(net,best_state_file);
        net->to(device);
        std::pair<double,double> res=test_model(net,test_data_loader,device);
        best_acc=res.second;
    }
    if (std::filesystem::exists(state_file)){
        torch::load(net,state_file);
    }
    torch::optim::AdamW optimizer(
        net->parameters(),
        torch::optim::AdamWOptions(lr)
        .weight_decay(weight_decay)
        .betas({0.9,0.999})
        .eps(eps)
        );
    net->to(device);
    net->train();
    while(is_running){
        int batch_index=0;
        for (auto& batch:*train_data_loader){
            if (!is_running) break;
            auto data=batch.data.to(device);
            auto targets=batch.target.to(device);
            if (batch_index%opt_every==0)optimizer.zero_grad();
            auto output=net->forward(data);
            auto loss=torch::nn::functional::cross_entropy(output,targets);
            loss.backward();
            if ((batch_index+1)%opt_every==0)optimizer.step();
            batch_index++;
            if (batch_index%save_every==0){
                torch::save(net,state_file);
            }
            if (batch_index%test_every==0){
                std::cout<<"Testing..."<<std::endl;
                std::pair<double,double> res=test_model(net,test_data_loader,device);
                std::cout<<"Test result:\n"<<"Avg loss: "<<res.first<<"\n"<<"Acc: "<<res.second*100<<"%\n";
                if (res.second>best_acc){
                    best_acc=res.second;
                    std::cout<<"A new record is made. Save best.\n";
                    torch::save(net,best_state_file);
                }
                net->train();
            }
        }
    }
    return 0;
}
