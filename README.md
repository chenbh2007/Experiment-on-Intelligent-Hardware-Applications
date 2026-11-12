# 人工智能实验

本仓库收录人工智能课程的实验代码，涵盖嵌入式 I/O、聚类、计算机视觉、手势识别和语音识别。课程任务与实验要求见[实验指导书](ailab.pdf)。

## 实验目录

| 实验 | 内容 | 主要依赖或设备 |
| --- | --- | --- |
| [Lab1](Lab1/) | 猜拳客户端、对局数据处理、马尔可夫预测和 SVM 训练与测试 | C 编译器、cJSON；在线对战需连接课程服务器 |
| [Lab2](Lab2/) | BMP280 温度采集、二维样本生成、K-means 与核 K-means、PPM 绘图 | C 编译器；传感器实验需 Linux I²C 和 BMP280，GPU 版本需 CUDA 和 cuBLAS |
| [Lab3](Lab3/) | 颜色转换、透视变换、轮廓提取、霍夫圆检测与图像自动矫正 | C++17、CMake、OpenCV 5；自动矫正子项目还需 CUDA |
| [Lab4](Lab4/) | MNIST 手写数字识别，包括数据转换、SVM 训练与评测 | C++17；默认 CMake 构建需 CMake 3.22+、OpenCV 4 和 CUDA；目录也提供 CPU 版训练与评测源码 |
| Lab5 | Prolog 动物识别专家系统与舵机联动 | 本仓库暂无源码，实验报告位于外层工作区的 `report/Lab5/` |
| [Lab6](Lab6/) | Haar 级联人脸与人眼检测、人脸隐私保护、CAMShift 目标跟踪 | C++17、CMake、带图形界面支持的 OpenCV；实时检测需摄像头和级联模型 XML |
| [Lab7](Lab7/) | PCF8591 驱动、CNN 手势分类、实时手势控制 LED，以及自定义 YOLO 检测实验 | CMake、LibTorch、OpenCV；LED 控制需摄像头与 I²C 设备，YOLO 实验使用 CUDA 和 cJSON |
| [Lab8](Lab8/) | 基于 MFCC 与 GMM-HMM 的 0–9 孤立数字语音识别，支持录音、训练、评估和预测 | Python，依赖见 [requirements.txt](Lab8/requirements.txt)；麦克风模式另需 PyAudio 和 PortAudio |

## 快速运行

以下各组命令均从仓库根目录开始执行。如果从外层实验工作区进入，先执行 `cd src`。

### Lab2 · CPU K-means

只需 GCC。示例生成 1,000 个二维样本，将其分为 2 簇，再绘制结果。

```bash
cd Lab2
gcc generate_data_blobs.c -o generate_data.out -lm
gcc K_means.c -o K_means.out -lm
gcc generate_graph.c -o generate_graph.out -lm
./generate_data.out
./K_means.out
./generate_graph.out result.ppm
```

绘图程序读取聚类生成的 `final.db`，输出 `result.ppm`。同目录的 `test.sh` 可用指定的 CUDA 源文件对 blobs、circles、moons 和 bars 数据批量运行聚类，例如 `bash test.sh K_means_with_diffusion_kernel_cuda.cu diffusion_cuda`。

### Lab3 · 摄像头颜色转换

```bash
cd Lab3/ColorConversion
cmake -S . -B build
cmake --build build
./build/ColorConversion
```

程序从 `/dev/video0` 采集一帧，在当前目录生成 `Ori.png` 和 `Conv.png`。Lab3 和 Lab6 的其他 CMake 子项目也可在各自目录构建。Lab6 的人脸相关示例会从源码中指定的路径加载 Haar 级联 XML，运行前需确认该路径和摄像头设备与本机一致。

### Lab7 · CNN 手势分类

在 `Lab7/SignClassifier_nn/` 下构建 `SignClassifier`。可通过 `CMAKE_PREFIX_PATH` 指定本机 LibTorch 的安装目录。

```bash
cd Lab7/SignClassifier_nn
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build --target SignClassifier
```

以 [example_config.toml](Lab7/SignClassifier_nn/example_config.toml) 为模板创建本地 TOML 配置，填入训练集、测试集、索引文件和模型保存路径，再执行 `./build/SignClassifier config.toml`。模板默认使用 CUDA，使用 CPU 时将 `device` 改为 `"cpu"`。

### Lab8 · 数字语音识别

```bash
cd Lab8
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python digit_recognition.py train --data-dir /path/to/dataset
python digit_recognition.py predict /path/to/sample.wav
```

`--data-dir` 接受带 `groups.csv` 或 `meta.json` 的录音目录，也支持按 `序号_数字.wav` 命名的样本。带 `groups.csv` 的 ZIP 数据可通过 `train --data-zip` 指定。默认模型文件为 `digit_hmm.pkl`。安装 PyAudio 后可用 `python digit_recognition.py record` 录制样本，也可省略 `predict` 的 WAV 参数来使用麦克风。

## 运行所需文件

各实验独立构建。运行前请检查相应源码或配置中的摄像头、I²C 设备、数据集及模型路径。部分录音、图片、模型和生成文件需要按实验自行准备。
