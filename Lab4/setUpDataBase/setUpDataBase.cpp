#include<vector>
#include<iostream>
#include<fstream>
#include<byteswap.h>
#include<string>
#include<cstdint>
#include<sstream>
#include<cstdio>
int main(int argc,char **argv){
    if (argc<4){
        std::cout<<"Too few Arguments."<<std::endl;
        return -1;
    }
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::ifstream rawlab(argv[2],std::ios::binary);
    std::stringstream ss;
    ss<<argv[1];
    std::uint32_t y,n;
    std::uint8_t buffer[4];
    ss>>y;
    ss.clear();
    std::ofstream newlab(argv[3],std::ios::binary);
    rawlab.read((char*)buffer,4);
    if (buffer[2]!=0x08||buffer[3]!=0x01){
        std::cout<<"Raw label doesn't fit. Abort."<<std::endl;
        rawlab.close();
        newlab.close();
        return 0;
    }
    rawlab.read((char*)&n,sizeof(uint32_t));
    n=bswap_32(n);
    char cache[4]={0x00,0x00,0x09,0x01};
    newlab.write(cache,sizeof(cache));
    newlab.write((char*)&n,sizeof(n));
    for (;n>0;n--){
        std::uint8_t l;
        rawlab.read((char*)&l,sizeof(uint8_t));
        std::int8_t nl;
        if (l==y){
            nl=1;
        }
        else nl=-1;
        newlab.write((char*)&nl,sizeof(int8_t));
    }
    rawlab.close();
    newlab.close();
    return 0;
}
