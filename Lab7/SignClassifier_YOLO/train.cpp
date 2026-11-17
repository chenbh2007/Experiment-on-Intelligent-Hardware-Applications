#include "YoloDataset.hpp"
#include "YOLOStructure.hpp"
#include "toml.hpp"

#include <csignal>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <algorithm>
#include <numeric>

static volatile sig_atomic_t is_running=1;
#ifndef TRAIN_AS_LIBRARY
void handler(int sig){
    (void)sig;
    is_running=0;
}
#endif

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

std::pair<torch::Tensor,std::vector<torch::Tensor> > loss_func(const std::vector<torch::Tensor> &res,const std::vector<torch::Tensor> &tar,float l_cls,float l_box,float l_ctr,float alpha=0.25,float gamma=2){
    torch::nn::BCEWithLogitsLoss bceloss(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kNone));
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
        torch::Tensor p1_target=box_target.slice(1,1,5,2);
        torch::Tensor p3_target=box_target.slice(1,2,5,2);
        torch::Tensor p1_res=box_res.slice(1,1,5,2);
        torch::Tensor p3_res=box_res.slice(1,2,5,2);
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
    torch::Tensor cls_loss_avg=cls_loss_tot/size_sum.clamp_min(1);
    torch::Tensor box_loss_avg=box_loss_tot/weight_sum.clamp_min(1e-7);
    torch::Tensor ctr_loss_avg=ctr_loss_tot/pos_sum.clamp_min(1);
    return {l_cls*cls_loss_avg+l_box*box_loss_avg+l_ctr*ctr_loss_avg,{cls_loss_avg,box_loss_avg,ctr_loss_avg}};
}

std::vector<torch::Tensor> getboxes(const std::vector<torch::Tensor> &output,float threshold,int p,int s0,bool is_target=false){
    std::vector<torch::Tensor> result;
    int num_class=yolo::gesture_classes.size();
    std::vector<std::vector<torch::Tensor> > boxes(num_class);
    int s=s0;
    int batchsize=output[0].size(0);
    torch::Tensor batchidx=torch::arange(batchsize,output[0].options()).view({batchsize,1,1});
    for (int i=0;i<p&&is_running;i++){
        torch::Tensor loc=output[i*2].permute({0,2,3,1});
        torch::Tensor cls_score=output[i*2+1].permute({0,2,3,1});
        torch::Tensor ctr_score=loc.slice(-1,0,1);
        torch::Tensor score=is_target ? cls_score:(cls_score*ctr_score).sqrt();
        torch::Tensor conf=score.amax(-1);
        torch::Tensor positive=conf.gt(threshold);
        int w=loc.size(2);
        int h=loc.size(1);
        torch::Tensor xs=torch::arange(w,loc.options()).view({1,1,w});
        torch::Tensor ys=torch::arange(h,loc.options()).view({1,h,1});
        std::vector<torch::Tensor> cltrb=loc.unbind(-1);
        torch::Tensor xxyyconf=torch::stack({
                (-cltrb[1]+xs)*s+batchidx*10000.0f,
                (cltrb[2]+xs)*s+batchidx*10000.0f,
                (-cltrb[3]+ys)*s+batchidx*10000.0f,
                (cltrb[4]+ys)*s+batchidx*10000.0f,
                conf},-1);
        torch::Tensor index=score.argmax(-1);
        for (int j=0;j<num_class&&is_running;j++){
            torch::Tensor cls_index=index.eq(j)&positive;
            boxes[j].push_back(xxyyconf.index({cls_index}));
        }
        s=s*2;
    }
    for (int i=0;i<num_class&&is_running;i++){
        result.push_back(torch::cat(boxes[i],0));
    }
    return result;
}

std::vector<torch::Tensor> nms(const std::vector<torch::Tensor> &boxes,float threshold,int max_det=-1){
    std::vector<torch::Tensor> result;
    int channel=boxes[0].size(1);
    int num_class=boxes.size();
    for (int i=0;i<num_class&&is_running;i++){
        if (boxes[i].size(0)==0){
            result.push_back(torch::empty({0,channel},boxes[i].options().device(torch::kCPU)));
            continue;
        }
        torch::Tensor cls_boxes=boxes[i];
        torch::Tensor cpu_boxes=cls_boxes.cpu().contiguous();
        int64_t n=cpu_boxes.size(0);
        auto acc=cpu_boxes.accessor<float,2>();
        std::vector<int64_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        int64_t limit = (max_det >= 0) ? std::min<int64_t>(n, 2000) : n;
        if (limit < n) {
            std::partial_sort(order.begin(), order.begin() + limit, order.end(), [&](int64_t a, int64_t b) {
                return acc[a][4] > acc[b][4];
            });
            order.resize(limit);
        } else {
            std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
                return acc[a][4] > acc[b][4];
            });
        }
        std::vector<int64_t> kept_idx;
        std::vector<bool> suppressed(n, false);
        for (int64_t p = 0; p < limit; ++p) {
            int64_t idx = order[p];
            if (suppressed[idx]) continue;

            kept_idx.push_back(idx);
            if (max_det >= 0 && static_cast<int>(kept_idx.size()) >= max_det) break;

            float x1_a = acc[idx][0], x2_a = acc[idx][1];
            float y1_a = acc[idx][2], y2_a = acc[idx][3];
            float area_a = std::max(0.0f, x2_a - x1_a) * std::max(0.0f, y2_a - y1_a);

            for (int64_t q = p + 1; q < limit; ++q) {
                int64_t j = order[q];
                if (suppressed[j]) continue;

                float x1_b = acc[j][0], x2_b = acc[j][1];
                float y1_b = acc[j][2], y2_b = acc[j][3];
                float iw = std::max(0.0f, std::min(x2_a, x2_b) - std::max(x1_a, x1_b));
                float ih = std::max(0.0f, std::min(y2_a, y2_b) - std::max(y1_a, y1_b));
                float inter = iw * ih;
                if (inter <= 0.0f) continue;

                float area_b = std::max(0.0f, x2_b - x1_b) * std::max(0.0f, y2_b - y1_b);
                float iou = inter / std::max(area_a + area_b - inter, 1e-7f);
                if (iou >= threshold) {
                    suppressed[j] = true;
                }
            }
        }
        torch::Tensor idx_tensor = torch::from_blob(
            kept_idx.data(), {static_cast<int64_t>(kept_idx.size())}, torch::kLong
        ).clone();
        result.push_back(cpu_boxes.index({idx_tensor}));
    }
    return result;
}

template <typename DataLoaderType,typename NetType>
std::vector<std::vector<std::vector<float> > > eval_model(DataLoaderType &dataLoader,NetType &Net,float box_threshold,torch::Device device){
    float threshold;
    Net->eval();
    torch::NoGradGuard no_grad;
    int num_class=yolo::gesture_classes.size();
    std::vector<std::vector<std::vector<float> > > result(5);
    std::vector<std::vector<float> > cls_IoU(5,std::vector<float>(num_class));
    std::vector<std::vector<int64_t> > cls_FP(5,std::vector<int64_t>(num_class)),cls_TP(5,std::vector<int64_t>(num_class)),cls_tot(5,std::vector<int64_t>(num_class));
    for (auto &batch:*dataLoader){
        torch::Tensor data=batch.data.to(device);
        int batch_size=data.size(0);
        std::vector<torch::Tensor> target;
        for (const torch::Tensor &target_it:batch.target)
            target.push_back(target_it.to(device));
        std::vector<torch::Tensor> output=Net->forward(data);
        for (int i=0;i<3;i++){
            output[i*2]=torch::cat({output[i*2].slice(1,0,1).sigmoid(),torch::softplus(output[i*2].slice(1,1,5))},1);
            output[i*2+1]=output[i*2+1].sigmoid();
        }
        std::vector<torch::Tensor> boxes_target=nms(getboxes(target,0.5f,3,8,true),0.99);
        std::vector<torch::Tensor> base_boxes=getboxes(output,0.4f,3,8);
        for (torch::Tensor &b:base_boxes){
            b=b.cpu();
        }
        for (int l=0;l<5;l++){
            threshold=0.4f+l*0.1f;
            std::vector<torch::Tensor> filtered(num_class);
            for (int c = 0; c < num_class; c++) {
            if (l == 0 || base_boxes[c].size(0) == 0) {
                filtered[c] = base_boxes[c];
            } 
            else {
                filtered[c] = base_boxes[c].index({base_boxes[c].select(-1, 4).gt(threshold)});
                }
            }
            std::vector<torch::Tensor>  boxes_output=nms(filtered,box_threshold,batch_size*10);
            if (!is_running) break;
            for (int i=0;i<boxes_target.size();i++){
                int64_t TP=0,FP=0,tot=boxes_target[i].size(0),pred_tot=boxes_output[i].size(0);
                float IoU_sum=0.0f;
                if (tot==0||pred_tot==0){
                    cls_FP[l][i]+=pred_tot;
                    cls_tot[l][i]+=tot;
                    continue;
                }
                torch::Tensor cls_boxes=boxes_output[i].unsqueeze(0);
                torch::Tensor tar_boxes=boxes_target[i].unsqueeze(1);
                torch::Tensor p1_cls=cls_boxes.slice(-1,0,4,2);
                torch::Tensor p3_cls=cls_boxes.slice(-1,1,4,2);
                torch::Tensor p1_tar=tar_boxes.slice(-1,0,4,2);
                torch::Tensor p3_tar=tar_boxes.slice(-1,1,4,2);
                torch::Tensor I_area=(torch::minimum(p3_tar,p3_cls)-torch::maximum(p1_tar,p1_cls)).clamp_min(0).prod(-1);
                torch::Tensor U_area=(p3_cls-p1_cls).prod(-1)+(p3_tar-p1_tar).prod(-1)-I_area;
                torch::Tensor IoU=I_area/U_area.clamp_min(1e-7);
                for (int j=0;j<tot&&pred_tot>0;j++){
                    torch::Tensor max_index=IoU.select(0,j).argmax(-1);
                    if (IoU.index({j,max_index}).item<float>()<0) break;
                    if (IoU.index({j,max_index}).item<float>()>0.5f){
                        TP++;
                        IoU_sum+=IoU.index({j,max_index}).item<float>();
                        IoU.select(1,max_index.item<int64_t>()).fill_(-1.0f);
                    }
                }
                FP=pred_tot-TP;
                cls_FP[l][i]+=FP;
                cls_TP[l][i]+=TP;
                cls_IoU[l][i]+=IoU_sum;
                cls_tot[l][i]+=tot;
            } 
        }
    }
    for (int l=0;l<5;l++)
        for (int i=0;i<num_class;i++){
            cls_tot[l][i]=std::max<int64_t>(cls_tot[l][i],1);
            result[l].push_back({(float)cls_TP[l][i]/(float)cls_tot[l][i],(float)cls_FP[l][i]/(float)cls_tot[l][i],cls_IoU[l][i]/(float)std::max<int64_t>(cls_TP[l][i],1)});
        }
    return result;
}

std::ofstream open_tsv(const std::filesystem::path &p,const std::string &header){
    bool need_header=!std::filesystem::exists(p)||std::filesystem::file_size(p)==0;
    std::ofstream ofs(p,std::ios::app);
    if (need_header&&ofs.is_open()){
        ofs<<header<<"\n"<<std::flush;
    }
    return ofs;
}

#ifndef TRAIN_AS_LIBRARY
int main(int argc,char **argv){
    std::signal(SIGINT,handler);
    if (argc<2){
        std::cout<<"Too few arguments."<<std::endl;
        return -1;
    }
    auto config=toml::parse_file(argv[1]);
   
    if (!torch::cuda::is_available()){
        std::cout<<"CUDA is required."<<std::endl;
        return -1;
    }
    torch::Device device=torch::kCUDA;

    
    int num_bottleneck=config["YOLO"]["num_bottleneck"].value_or(1);
    int base=config["YOLO"]["base"].value_or(64);
    bool enable_shortcut=config["YOLO"]["enable_shortcut"].value_or(true);
    float r_bottleneck=config["YOLO"]["r_bottleneck"].value_or(2.0f);
    float r_C2f=config["YOLO"]["r_C2f"].value_or(2.0f);
    int num_class=yolo::gesture_classes.size();
    auto Net=std::make_shared<yolo::YOLO>(3,num_class,base,base*32,r_C2f,r_bottleneck,num_bottleneck,enable_shortcut);

    std::string state_file=config["model"]["state_file"].value_or("state.pt");
    std::string best_state_file=config["model"]["best_state_file"].value_or("best.pt");
    if (!std::filesystem::path(state_file).parent_path().empty()) std::filesystem::create_directories(std::filesystem::path(state_file).parent_path());
    if (!std::filesystem::path(best_state_file).parent_path().empty()) std::filesystem::create_directories(std::filesystem::path(best_state_file).parent_path());
    if (std::filesystem::exists(state_file)) torch::load(Net,state_file);
    Net->to(device);

    std::string train_picture_dir=config["data"]["train_picture_dir"].value_or("/path/to/dataset");
    std::string train_annotations_dir=config["data"]["train_annotations_dir"].value_or("/path/to/annotations"); 
    int w=config["data"]["width"].value_or(512);
    int h=config["data"]["height"].value_or(512);
    yolo::YoloDataset train_dataset(train_annotations_dir,train_picture_dir,w,h);

    std::string eval_picture_dir=config["data"]["eval_picture_dir"].value_or("/path/to/eval_dataset");
    std::string eval_annotations_dir=config["data"]["eval_annotations_dir"].value_or("/path/to/eval_annotations");
    yolo::YoloDataset eval_dataset(eval_annotations_dir,eval_picture_dir,w,h);

    int batch_size=config["control"]["batch_size"].value_or(64);
    int opt_every_step=config["control"]["opt_every_step"].value_or(1);
    int test_every_step=config["control"]["test_every_step"].value_or(1000);
    int save_every_step=config["control"]["save_every_step"].value_or(200); 
    int num_worker=config["control"]["workers"].value_or(4);
    auto train_dataloader=torch::data::make_data_loader(
            std::move(train_dataset).map(torch::data::transforms::Collate<yolo::Sample>(stack_batch)),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_worker)
            );
    auto eval_dataloader=torch::data::make_data_loader(
            std::move(eval_dataset).map(torch::data::transforms::Collate<yolo::Sample>(stack_batch)),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_worker)
            );

    float l_cls=config["loss"]["l_cls"].value_or(0.5f);
    float l_box=config["loss"]["l_box"].value_or(1.0f);
    float l_ctr=config["loss"]["l_ctr"].value_or(0.8f);
    float alpha=config["loss"]["alpha"].value_or(0.25f);
    float gamma=config["loss"]["gamma"].value_or(2.0f);
    float box_threshold=config["loss"]["box_threshold"].value_or(0.5f);
 
    double eps=config["optim"]["eps"].value_or(1e-8);
    double lr=config["optim"]["lr"].value_or(1e-4);
    double weight_decay=config["optim"]["weight_decay"].value_or(0.01);
    std::string optim_checkpoint=config["optim"]["checkpoint"].value_or("checkpoint.pt");
    if (!std::filesystem::path(optim_checkpoint).parent_path().empty())
        std::filesystem::create_directories(std::filesystem::path(optim_checkpoint).parent_path());
    torch::optim::AdamW optimizer(
            Net->parameters(),
            torch::optim::AdamWOptions(lr)
            .betas({0.9,0.999})
            .weight_decay(weight_decay)
            .eps(eps)
            );
    if (std::filesystem::exists(optim_checkpoint)) torch::load(optimizer,optim_checkpoint);

    int log_every_step=config["log"]["log_every_step"].value_or(10);
    std::string log_dir=config["log"]["log_dir"].value_or("/path/to/log");
    auto log_path=std::filesystem::path(log_dir);
    if (!std::filesystem::exists(log_dir)){
        std::filesystem::create_directories(log_dir);
    }
    
    std::ofstream train_log=open_tsv(log_path/"train_log.tsv","epoch\tstep\tloss_tot\tloss_cls\tloss_box\tloss_ctr");
    std::ofstream eval_log=open_tsv(log_path/"eval_log.tsv","epoch\tstep\tthreshold\tbox_threshold\tcls\tprecision\terr_rate\tavg_IoU");

    auto best_record_path=std::filesystem::path(log_dir)/"best_record";
    float best_fitness=-1.0f;
    if (std::filesystem::exists(best_record_path)){
        std::ifstream ifs(best_record_path);
        ifs>>best_fitness;
    }

    Net->train();
    int epoch=1;
    int64_t step=0;
    if (std::filesystem::exists(log_path/"epoch_step")){
        std::ifstream ifs(log_path/"epoch_step");
        ifs>>epoch>>step;
    }
    while (is_running){
        for (auto &batch:*train_dataloader){
            if (!is_running) break;
            step++;
            torch::Tensor data=batch.data.to(device);
            std::vector<torch::Tensor> target=batch.target;
            for (int i=0;i<target.size();i++){
                target[i]=target[i].to(device);
            }
            std::vector<torch::Tensor> output=Net->forward(data);
            auto loss_all=loss_func(output,target,l_cls,l_box,l_ctr,alpha,gamma);
            torch::Tensor loss=loss_all.first;
            (loss/opt_every_step).backward();
            std::vector<torch::Tensor> losses=loss_all.second;
            if (step%opt_every_step==0){
                optimizer.step();
                optimizer.zero_grad();
            }
            if (step%log_every_step==0){
                train_log<<epoch<<"\t"
                        <<step<<"\t"
                        <<std::fixed<<std::setprecision(6)
                        <<loss.item<float>()<<"\t"
                        <<losses[0].item<float>()<<"\t"
                        <<losses[1].item<float>()<<"\t"
                        <<losses[2].item<float>()<<"\n"<<std::flush;
                std::cout<<"."<<std::flush;
            }
            if (step%save_every_step==0){
                std::cout<<"Round "<<epoch<<" step "<<step<<" completed. Saving..."<<std::endl;
                torch::save(Net,state_file);
                torch::save(optimizer,optim_checkpoint);
                std::ofstream ofs(log_path/"epoch_step",std::ios::out|std::ios::trunc);
                ofs<<epoch<<" "<<step;
            }
            if (step%test_every_step==0){
                auto score=eval_model(eval_dataloader,Net,box_threshold,device);
                for (int l=0;l<5&&is_running;l++){
                    for (int cls=0;cls<num_class;cls++){
                        eval_log<<epoch<<"\t"
                        <<step<<"\t"<<std::fixed<<std::setprecision(6)
                        <<0.4f+0.1f*l<<"\t"
                        <<box_threshold<<"\t"
                        <<yolo::gesture_classes_dict[cls]<<"\t"
                        <<score[l][cls][0]<<"\t"
                        <<score[l][cls][1]<<"\t"
                        <<score[l][cls][2]<<"\n"<<std::flush;
                    }
                }
                
                if (!is_running) break;

                float cur_best_fitness = -1.0f;
                int best_l = 0;

                for (int l = 0; l < 5; ++l) {
                    double f1_sum = 0.0;
                    double iou_sum = 0.0;
                    int valid_iou_cls = 0;

                    for (int cls = 0; cls < num_class; ++cls) {
                        float r = score[l][cls][0];   
                        float f = score[l][cls][1];  
                        float iou = score[l][cls][2];

                        float f1 = (r > 0.0f) ? (2.0f * r / (1.0f + r + f)) : 0.0f;
                        f1_sum += f1;
                        if (r > 0.0f) {
                            iou_sum += iou;
                            valid_iou_cls++;
                        }
                    }

                    float macro_f1 = static_cast<float>(f1_sum / num_class);
                    float mean_iou = (valid_iou_cls > 0) ? static_cast<float>(iou_sum / valid_iou_cls) : 0.0f;
                    float fitness = 0.7f * macro_f1 + 0.3f * mean_iou;

                    if (fitness > cur_best_fitness) {
                        cur_best_fitness = fitness;
                        best_l = l;
                    }
                }

                std::cout << "\n[Eval step " << step << "] best_thresh=" << (0.4f + 0.1f * best_l)
                          << " fitness=" << cur_best_fitness << " (hist_best=" << best_fitness << ")" << std::endl;

                if (cur_best_fitness > best_fitness) {
                    best_fitness = cur_best_fitness;
                    torch::save(Net, best_state_file);

                    auto tmp_path = best_record_path;
                    tmp_path += ".tmp";
                    {
                        std::ofstream ofs(tmp_path, std::ios::out | std::ios::trunc);
                        ofs << std::fixed << std::setprecision(6)
                            << best_fitness << "\t" << epoch << "\t" << step << "\t" << (0.4f + 0.1f * best_l) << "\n";
                        for (int cls = 0; cls < num_class; ++cls) {
                            ofs << yolo::gesture_classes_dict[cls] << "\t"
                                << score[best_l][cls][0] << "\t"
                                << score[best_l][cls][1] << "\t"
                                << score[best_l][cls][2] << "\n";
                        }
                    }
                    std::filesystem::rename(tmp_path, best_record_path);
                    std::cout << "New best model saved to " << best_state_file << std::endl;
                }

                Net->train();   
            }
        }
        epoch++;
    }
    return 0;
}
#endif

