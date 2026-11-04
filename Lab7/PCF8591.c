#include"PCF8591.h"
#include<stdio.h>
uint8_t AD_input(int fd,uint8_t chip_addr,uint8_t channel,uint8_t input_method){
    struct i2c_msg msgs[2];
    msgs[0].addr=chip_addr;
    msgs[0].flags=0;
    if (channel<0||channel>3){
        perror("Invalid channel number.");
        return 0x00;
    }
    uint8_t control_buf=0x40|channel;
    msgs[0].buf=&control_buf;
    msgs[0].len=1;
    msgs[1].addr=chip_addr;
    msgs[1].flags=I2C_M_RD;
    msgs[1].len=1;
    uint8_t res;
    msgs[1].buf=&res;
    struct i2c_rdwr_ioctl_data package={.msgs=msgs,.nmsgs=2};
    if (ioctl(fd,I2C_RDWR,&package)<0){
        perror("Ioctl failed");
        return 0x00;
    }
    return res;
}
int DA_output(int fd,uint8_t chip_addr,uint8_t *raw,int len){
    struct i2c_msg msgs={.addr=chip_addr,.flags=0,.len=len,.buf=raw};
    struct i2c_rdwr_ioctl_data package={.msgs=&msgs,.nmsgs=1};
    return ioctl(fd,I2C_RDWR,&package);
}
