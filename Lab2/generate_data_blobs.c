#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 保证随机浮点数落在 (0, 1) 开区间，防止 log(0)
static inline double rand_u(void) {
    return ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
}

// Box-Muller 变换生成高斯噪声
static inline double rand_gauss(double mu, double sigma) {
    double u1 = rand_u();
    double u2 = rand_u();
    return mu + sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2) * sigma;
}

int main(int argc, char **argv) {
    srand((unsigned int)time(NULL));

    char filename[64] = "database.db";
    int n = 1000;
    if (argc >= 2) snprintf(filename, sizeof(filename), "%s", argv[1]);
    if (argc >= 3) n = atoi(argv[2]);

    FILE *f = fopen(filename, "w");
    if (!f) {
        perror("无法创建数据文件");
        return -1;
    }

    int half = n / 2;
    // 簇 1：以 (-2.0, -1.5) 为中心的正态分布
    for (int i = 0; i < half; i++) {
        double x = rand_gauss(-2.0, 0.55);
        double y = rand_gauss(-1.5, 0.55);
        fprintf(f, "%.4f %.4f\n", x, y);
    }
    // 簇 2：以 (2.0, 1.5) 为中心的正态分布
    for (int i = 0; i < n - half; i++) {
        double x = rand_gauss(2.0, 0.55);
        double y = rand_gauss(1.5, 0.55);
        fprintf(f, "%.4f %.4f\n", x, y);
    }

    fclose(f);
    printf("成功生成双高斯斑块数据 -> %s (样本数: %d)\n", filename, n);
    return 0;
}
