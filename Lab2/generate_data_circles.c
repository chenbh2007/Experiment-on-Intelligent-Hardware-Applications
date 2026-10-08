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
    double r1 = 1.5;   // 内环中心半径
    double r2 = 2.4;   // 外环中心半径
    double noise = 0.12;

    if (argc >= 2) snprintf(filename, sizeof(filename), "%s", argv[1]);
    if (argc >= 3) n = atoi(argv[2]);
    if (argc >= 4) r1 = atof(argv[3]);
    if (argc >= 5) r2 = atof(argv[4]);

    FILE *f = fopen(filename, "w");
    if (!f) {
        perror("无法创建数据文件");
        return -1;
    }

    int half = n / 2;
    // 内环生成
    for (int i = 0; i < half; i++) {
        double theta = rand_u() * 2.0 * M_PI;
        double r = r1 + rand_gauss(0.0, noise);
        fprintf(f, "%.4f %.4f\n", r * cos(theta), r * sin(theta));
    }
    // 外环生成
    for (int i = 0; i < n - half; i++) {
        double theta = rand_u() * 2.0 * M_PI;
        double r = r2 + rand_gauss(0.0, noise);
        fprintf(f, "%.4f %.4f\n", r * cos(theta), r * sin(theta));
    }

    fclose(f);
    printf("成功生成同心双圆环数据 -> %s (样本数: %d, R1=%.2f, R2=%.2f)\n", filename, n, r1, r2);
    return 0;
}
