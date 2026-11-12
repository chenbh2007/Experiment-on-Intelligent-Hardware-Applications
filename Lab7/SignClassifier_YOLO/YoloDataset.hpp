#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
namespace yolo{
using Sample=torch::data::Example<torch::Tensor,std::vector<torch::Tensor> >;

extern std::map<std::string,int64_t> gesture_classes;

typedef struct{
    float *d_input;
    float *d_output;
    float *h_output;
}Gpuwindow;

class YoloDataset:public torch::data::datasets::Dataset<YoloDataset,Sample>{
    private:
    std::vector<std::string> pic_paths;
    std::vector<std::vector<int64_t> > labels;
    std::vector<std::vector<std::vector<float> > > bboxes;
    Gpuwindow d_buf{};
    size_t maxsize;
    int width,height;
    public:
    YoloDataset()=default;
    explicit YoloDataset(std::string annotations_dir,std::string data_dir,size_t size0,int w=512,int h=512);
    torch::data::Example<torch::Tensor,std::vector<torch::Tensor> > get(size_t index) override;
    torch::optional<size_t> size()const override;
};
}
