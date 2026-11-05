#ifndef PCF8591_H
#define PCF8591_H
#include<fcntl.h>
#include<unistd.h>
#include<sys/ioctl.h>
#include<linux/i2c.h>
#include<linux/i2c-dev.h>
#include<stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
uint8_t AD_input(int fd,uint8_t chip_address,uint8_t channel,uint8_t read_method);
int DA_output(int fd,uint8_t chip_address,uint8_t raw);
#ifdef __cplusplus
}
#endif
#endif
