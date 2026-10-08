#ifndef _BMP280_H_
#define _BMP280_H_

#define bmp_id_addr 0xD0
#define bmp_reset_addr 0xE0
#define bmp_status_addr 0xF3
#define bmp_ctrl_meas_addr 0xF4
#define bmp_config_addr 0xF5
#define bmp_calib_addr 0x88
#define bmp_press_addr 0xF7
#define bmp_temp_addr 0xFA
#define bmp_default_ctrl_meas 0x27
#define bmp_default_config 0x28

#define bmp_t_sb_0_5 (0x00<<5)
#define bmp_t_sb_62_5 (0x01<<5)
#define bmp_t_sb_125 (0x02<<5)
#define bmp_t_sb_250 (0x03<<5)
#define bmp_t_sb_500 (0x04<<5)
#define bmp_t_sb_1000 (0x05<<5)
#define bmp_t_sb_2000 (0x06<<5)
#define bmp_t_sb_4000 (0x07<<5)
#define bmp_press_len 3
#define bmp_temp_len 3
#define bmp_calib_len 24

#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>

typedef struct{
    int32_t temp;
    float press;
    uint8_t raw_tp[6];
}bmp_data;
int bmp280_write_reg(int fd,uint8_t bmp_addr,uint8_t reg,uint8_t value);
int bmp280_read_burst(int fd,uint8_t bmp_addr,uint8_t start_reg,uint8_t *buf,uint16_t len);
int bmp280_get_dig(int fd,uint8_t bmp_addr,int32_t *dig_t,int32_t *dig_p);
int bmp280_get_temp_press(bmp_data *raw_data,int32_t *dig_t,int32_t *dig_p);

#endif
