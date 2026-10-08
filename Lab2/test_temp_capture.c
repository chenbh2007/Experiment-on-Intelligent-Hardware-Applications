#include<stdio.h>
#include<fcntl.h>
#include<unistd.h>
#include<stdint.h>
#include<sys/ioctl.h>
#include<linux/i2c-dev.h>
#include<linux/i2c.h>
#include"BMP280.h"
#include<time.h>
#include<signal.h>

#define I2C_BUS "/dev/i2c-7"
#define chip_addr 0x76

static volatile sig_atomic_t stop=0;
void handle_sigint(int sig){
    (void)sig;
    stop=1;
}

int main(){
    signal(SIGINT,handle_sigint);
    int i2c=open(I2C_BUS,O_RDWR);
    if (i2c<0){
        perror("打开总线失败");
        return -1;
    }
    uint8_t id;
    bmp280_read_burst(i2c,chip_addr,bmp_id_addr,&id,1);
    printf("bmp280 chip ID: 0x%02X\n",id);
    if (id!=0x58){
        printf("设备校验失败!\n");
        close(i2c);
        return -1;
    }
    int32_t dig_t[3],dig_p[9];
    bmp280_write_reg(i2c,chip_addr,bmp_ctrl_meas_addr,bmp_default_ctrl_meas);
    bmp280_write_reg(i2c,chip_addr,bmp_config_addr,bmp_default_config);
    bmp280_get_dig(i2c,chip_addr,dig_t,dig_p);
    bmp_data data;
    while (!stop){
        if(bmp280_read_burst(i2c,chip_addr,bmp_press_addr,data.raw_tp,6)<0){
            perror("获取数据失败");
            close(i2c);
            return -1;
        };
        bmp280_get_temp_press(&data,dig_t,dig_p);
        printf("温度(C):%.2f  气压(hPa):%.2f\n",(float)data.temp/100,data.press);
        sleep(1);
    }
    close(i2c);
    printf("总线已释放。\n");
    return 0;
}
