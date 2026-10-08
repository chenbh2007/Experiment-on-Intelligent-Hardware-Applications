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
    double angle = M_PI / 4.0; // 倾斜 45 度角
    double cos_a = cos(angle), sin_a = sin(angle);

    if (argc >= 2) snprintf(filename, sizeof(filename), "%s", argv[1]);
    if (argc >= 3) n = atoi(argv[2]);

    FILE *f = fopen(filename, "w");
    if (!f) {
        perror("无法创建数据文件");
        return -1;
    }

    int half = n / 2;
    for (int i = 0; i < n; i++) {
        int cluster = (i < half) ? 0 : 1;
        // 沿长轴均匀延伸 [-3.0, 3.0]，法向受紧窄高斯扰动
        double u = (rand_u() - 0.5) * 6.0;
        double v = rand_gauss(0.0, 0.25) + (cluster == 0 ? -1.2 : 1.2);

        // 旋转变换至斜向坐标系
        double x = u * cos_a - v * sin_a;
        double y = u * sin_a + v * cos_a;
        fprintf(f, "%.4f %.4f\n", x, y);
    }

    fclose(f);
    printf("成功生成倾斜各向异性条带数据 -> %s (样本数: %d)\n", filename, n);
    return 0;
}
