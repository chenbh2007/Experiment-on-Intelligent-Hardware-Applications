#include"BMP280.h"
int bmp280_write_reg(int fd,uint8_t bmp_addr,uint8_t reg,uint8_t value){
    uint8_t out_buf[2]={reg,value};
    struct i2c_msg msg={
        .addr=bmp_addr,
        .flags=0,
        .len=2,
        .buf=out_buf
    };
    struct i2c_rdwr_ioctl_data package={.msgs=&msg, .nmsgs=1};
    return ioctl(fd,I2C_RDWR,&package);
}
int bmp280_read_burst(int fd,uint8_t bmp_addr,uint8_t start_reg,uint8_t *buf,uint16_t len){
    struct i2c_msg msg[2];
    msg[0].addr=bmp_addr;
    msg[0].flags=0;
    msg[0].len=1;
    msg[0].buf=&start_reg;
    msg[1].addr=bmp_addr;
    msg[1].flags=I2C_M_RD;
    msg[1].len=len;
    msg[1].buf=buf;
    struct i2c_rdwr_ioctl_data package={.msgs=msg,.nmsgs=2};
    return ioctl(fd,I2C_RDWR,&package);
}
int bmp280_get_dig(int fd,uint8_t bmp_addr,int32_t *dig_t,int32_t *dig_p){
    uint8_t dig[24];
    int res=bmp280_read_burst(fd,bmp_addr,bmp_calib_addr,dig,24);
    uint8_t i;
    dig_t[0]=(int32_t)(uint16_t)((dig[1]<<8)|(dig[0]));
    int16_t tmp;
    for (i=1;i<3;i++){
        tmp=(int16_t)((dig[i*2+1]<<8)|dig[i*2]);
        dig_t[i]=(int32_t)tmp;
    }
    dig_p[0]=(int32_t)(uint16_t)((dig[7]<<8)|dig[6]);
    for (i=1;i<9;i++){
        tmp=(int16_t)((dig[(i+3)*2+1]<<8)|dig[(i+3)*2]);
        dig_p[i]=(int32_t)tmp;
    }
    return res;
}
int bmp280_get_temp_press(bmp_data *raw_data,int32_t *dig_t,int32_t *dig_p){
    int32_t var1,var2,t_fine,T;
    int32_t adc_t,adc_p;
    uint8_t *raw=raw_data->raw_tp;
    adc_p=(int32_t)((raw[0]<<12)|(raw[1]<<4)|(raw[2]>>4));
    adc_t=(int32_t)((raw[3]<<12)|(raw[4]<<4)|(raw[5]>>4));
    var1=((((adc_t>>3)-((int32_t)dig_t[0]<<1))*(int32_t)dig_t[1]))>>11;
    var2=(((((adc_t>>4)-((int32_t)dig_t[0]))*((adc_t>>4)-((int32_t)dig_t[0])))>>12)*((int32_t)dig_t[2]))>>14;
    t_fine=var1+var2;
    T=(t_fine*5+128)>>8;
    raw_data->temp=T;
    int64_t v1,v2,p;
    v1=((int64_t)t_fine)-128000;
    v2=v1*v1*(int64_t)dig_p[5];
    v2=v2+((v1*(int64_t)dig_p[4])<<17);
    v2=v2+(((int64_t)dig_p[3])<<35);
    v1=((v1*v1*(int64_t)dig_p[2])>>8)+((v1*(int64_t)dig_p[1])<<12);
    v1=(((((int64_t)1)<<47)+v1)*((int64_t)dig_p[0]))>>33;
    if(v1==0) return -1;
    p=1048576-adc_p;
    p=(((p<<31)-v2)*3125)/v1;
    v1=(((int64_t)dig_p[8])*(p>>13)*(p>>13))>>25;
    v2=(((int64_t)dig_p[7])*p)>>19;
    p=((p+v1+v2)>>8)+(((int64_t)dig_p[6])<<4);
    raw_data->press=(float)p/25600.0;
    return 0;
}
