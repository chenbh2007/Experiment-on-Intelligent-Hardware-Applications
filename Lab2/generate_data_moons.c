#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double rand_u(void) {
    return ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
}

static inline double rand_gauss(double mu, double sigma) {
    double u1 = rand_u();
    double u2 = rand_u();
    return mu + sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2) * sigma;
}

int main(int argc, char **argv) {
    srand((unsigned int)time(NULL));

    char filename[64] = "database.db";
    int n = 1000;
    double radius = 1.5;
    double noise = 0.10;

    if (argc >= 2) snprintf(filename, sizeof(filename), "%s", argv[1]);
    if (argc >= 3) n = atoi(argv[2]);

    FILE *f = fopen(filename, "w");
    if (!f) {
        perror("无法创建数据文件");
        return -1;
    }

    int half = n / 2;
    // 上月牙 (上半圆弧)
    for (int i = 0; i < half; i++) {
        double theta = rand_u() * M_PI; // [0, pi]
        double x = radius * cos(theta) + rand_gauss(0.0, noise);
        double y = radius * sin(theta) + rand_gauss(0.0, noise);
        fprintf(f, "%.4f %.4f\n", x, y);
    }
    // 下月牙 (下凹半圆弧，向右平移 radius，向下错位)
    for (int i = 0; i < n - half; i++) {
        double theta = rand_u() * M_PI;
        double x = radius - radius * cos(theta) + rand_gauss(0.0, noise);
        double y = -radius * sin(theta) - 0.35 + rand_gauss(0.0, noise);
        fprintf(f, "%.4f %.4f\n", x, y);
    }

    fclose(f);
    printf("成功生成交错双月牙数据 -> %s (样本数: %d)\n", filename, n);
    return 0;
}
