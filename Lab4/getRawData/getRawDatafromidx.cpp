#include<iostream>
#include<fstream>
#include<cstdint>
#include<byteswap.h>
#include<vector>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/objdetect.hpp>
std::uint32_t chars2uint32(char *cache){
    std::uint32_t val;
    std::memcpy(&val,cache,4);
    return bswap_32(val);
}
int main(int argc,char **argv){
    if (argc<3){
        std::cout<<"Too few arguments."<<std::endl;
        return -1;
    }
    std::ifstream images(argv[1],std::ios::binary);
    char magicnumberimg[4];
    char cache[4];
    images.read(magicnumberimg,4);
    if (magicnumberimg[2]!=0x08||magicnumberimg[3]!=3){
        std::cout<<"Format doesn't fit."<<std::endl;
    }
    std::uint32_t l,w,h;
    images.read(cache,4);
    l=chars2uint32(cache);
    images.read(cache,4);
    w=chars2uint32(cache);
    images.read(cache,4);
    h=chars2uint32(cache);
    std::ofstream rawdb(argv[2],std::ios::binary);
    char buffer[4]={0x00,0x00,0x0D,0x02};
    rawdb.write(buffer,sizeof(buffer));
    rawdb.write((char*)&l,sizeof(l));
    cv::HOGDescriptor hog(cv::Size(28,28),cv::Size(14,14),cv::Size(7,7),cv::Size(7,7),9,1,-1,cv::HOGDescriptor::L2Hys,0.2,true,cv::HOGDescriptor::DEFAULT_NLEVELS,false);
    for (int i=0;i<l;i++){
        cv::Mat M(w,h,CV_8U);
        unsigned char *p;
        p=M.ptr<unsigned char>(0);
        for (int j=0;j<w*h;j++){
            char pixel;
            images.read(&pixel,1);
            p[j]=(unsigned char)pixel;
        } 
        std::vector<float> vec;
        hog.compute(M,vec);
        if (i==0){
            std::uint32_t dim=vec.size();
            rawdb.write((char*)&dim,sizeof(dim));
        }
        rawdb.write((char*)vec.data(),vec.size()*sizeof(float));
    }
    images.close();
    rawdb.close();
    return 0;
}
