# SignClassifier_YOLO

使用 C++20、LibTorch 和 OpenCV 实现的手势目标检测工程，包含训练、评估、摄像头推理及 TensorBoard 日志可视化。网络包含 C2f、SPPF、FPN/PAN 和三个尺度的检测头，检测步长为 8、16、32；分类使用 Focal Loss，边界框使用中心度加权的 GIoU Loss，中心度使用 BCEWithLogitsLoss。

当前支持的 34 个手势类别及编号定义在 `YoloDataset.cpp` 中。类别数固定由该映射决定，仅提供少量类别的数据不会自动缩减模型输出类别数。

## 工程文件

以下清单仅包含 Git 跟踪的工程文件。后文的数据目录、依赖安装位置和构建输出均为使用示例，需要自行准备或生成，不表示仓库已经提供这些文件。

| 文件 | 用途 |
| --- | --- |
| `CMakeLists.txt` | 构建公共静态库 `yolo_common`、训练程序 `train` 及推理程序 `client` |
| `YOLOStructure.hpp` | 网络结构 |
| `YoloDataset.hpp`、`YoloDataset.cpp` | 类别映射、标注解析、图像预处理及训练目标生成 |
| `train.cpp` | 训练、损失计算、评估、解码及 NMS |
| `client.cpp` | 视频采集、模型推理及检测结果显示 |
| `example.toml` | 训练配置示例 |
| `toml.hpp` | 随工程提供的 toml++ 3.4.0 单头文件，无需另行安装 |
| `watch_tb.py` | 将训练程序的 TSV 日志同步到 TensorBoard |

## 依赖

| 依赖 | 要求与用途 |
| --- | --- |
| C++ 编译器 | 支持 C++20；Linux 使用与所选 CUDA Toolkit、LibTorch 兼容的 GCC，Windows 使用 MSVC x64 |
| CMake | **3.24 或更新版本** |
| LibTorch | 可使用独立 C++ 发行包，或 Python PyTorch 安装中提供的 C++ 库；训练必须选择 CUDA 版本，Linux 应注意编译器 ABI 兼容性 |
| NVIDIA GPU、驱动及 CUDA Toolkit | 训练需要 CUDA GPU；驱动、Toolkit、LibTorch 的 CUDA 版本须相互兼容 |
| OpenCV | **4.x**，必需模块为 `core`、`imgproc`、`imgcodecs`；构建 `client` 还需 `highgui`、`videoio` |
| cJSON | 提供 `<cjson/cJSON.h>`、CMake 配置包和 `cjson` 链接目标；注意不是 json-c |
| Python、PyTorch、TensorBoard | `watch_tb.py` 需要 Python PyTorch 和 TensorBoard；使用 Python PyTorch 提供 C++ 库时也需要相应 Python 环境；脚本要求 Python 3.10+，并满足所安装 PyTorch 的版本要求 |

`train.cpp` 会检查 CUDA 可用性，不能直接在 CPU 上训练。`client` 支持 CPU 或 CUDA 推理；只使用 CPU 推理时可以选择 CPU 版 LibTorch，无需安装 CUDA Toolkit。工程没有自定义 CUDA 源文件，但 CUDA 版 LibTorch 的 CMake 配置仍可能需要 Toolkit 和 `nvcc`。

## Linux 安装与构建

以下以 Ubuntu/Debian 系列的 x86_64 环境为例，命令在本工程目录执行。

### 1. 安装基础工具、OpenCV 和 cJSON

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build curl unzip \
    libopencv-dev libcjson-dev python3 python3-venv
cmake --version
g++ --version
```

确认 CMake 至少为 3.24。若发行版提供的版本较旧，可在独立环境中安装新版构建工具：

```bash
python3 -m venv "$HOME/.venvs/yolo-tools"
source "$HOME/.venvs/yolo-tools/bin/activate"
python -m pip install --upgrade pip cmake ninja
cmake --version
```

### 2. 安装 NVIDIA 驱动、CUDA Toolkit 和 LibTorch

1. 按 [NVIDIA Linux 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)安装适用于本机 GPU、系统和编译器的驱动及 CUDA Toolkit。
2. 在 [PyTorch 下载页面](https://pytorch.org/get-started/locally/)选择 LibTorch、Linux、C++ 和匹配的 CUDA 版本，复制下载链接。使用包含依赖的发行包，并根据编译器选择相容的 ABI。安装方法可参考 [LibTorch 官方文档](https://docs.pytorch.org/cppdocs/installing.html)。
3. 将以下 URL 替换成实际链接，下载并解压到用户目录：

```bash
LIBTORCH_URL='替换为所选 Linux CUDA LibTorch ZIP 的下载链接'
mkdir -p "$HOME/deps"
curl --fail --location "$LIBTORCH_URL" --output /tmp/libtorch.zip
unzip /tmp/libtorch.zip -d "$HOME/deps"
export LIBTORCH_ROOT="$HOME/deps/libtorch"
```

如果当前 Python 环境已经安装匹配的 CUDA 版 PyTorch，可以跳过 ZIP 下载与解压，直接使用该安装中的 C++ 库。若尚未安装，可在 [PyTorch 下载页面](https://pytorch.org/get-started/locally/)选择 Linux、Pip、Python 和所需 CUDA 版本，执行页面给出的安装命令。激活对应 Python 环境后设置库根目录：

```bash
export LIBTORCH_ROOT="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent)')"
```

若 Toolkit 安装在 `/usr/local/cuda`，可设置以下环境变量；其他位置请替换路径：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$LIBTORCH_ROOT/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvidia-smi
nvcc --version
```

`nvidia-smi` 用于检查驱动和 GPU，`nvcc --version` 用于检查 Toolkit。两者显示的 CUDA 版本含义不同，应按所选 LibTorch 发行包的要求配置环境。

### 3. 配置与编译

```bash
cmake -S . -B build-linux -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_ROOT"
cmake --build build-linux --parallel 2
```

使用 Python PyTorch 时，`CMAKE_PREFIX_PATH` 可以直接从当前环境获取，配置命令改为：

```bash
cmake -S . -B build-linux -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-linux --parallel 2
```

该方法见 [LibTorch 官方安装说明](https://docs.pytorch.org/cppdocs/installing.html)。查询路径与配置构建时应使用同一个 Python 环境；训练还需确认 `python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"` 显示所需 CUDA 版本及 `True`。

LibTorch 头文件和预编译头较占内存，可按机器内存调整并行数。默认构建训练程序；找到 OpenCV 的 `highgui` 和 `videoio` 时还会构建推理程序，否则 CMake 会提示跳过 `client`。

## Windows 安装与构建

以下使用 Windows x64、Visual Studio 2022 和 PowerShell。编译器版本还应符合所选 CUDA Toolkit、LibTorch 的支持范围。

### 1. 安装编译工具和 CUDA

1. 从 [Visual Studio 下载页面](https://visualstudio.microsoft.com/downloads/)安装 Visual Studio 2022 或对应 Build Tools，勾选“使用 C++ 的桌面开发”、MSVC v143 和 Windows SDK。若默认页面提供其他版本，可从旧版本入口获取 2022。
2. 安装 [CMake](https://cmake.org/download/) 3.24+ 和 [Git for Windows](https://git-scm.com/download/win)，并加入 PATH。
3. 训练时按 [NVIDIA Windows 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)安装驱动及兼容的 CUDA Toolkit。运行 `nvidia-smi` 和 `nvcc --version` 检查安装。
4. 从 [PyTorch 下载页面](https://pytorch.org/get-started/locally/)下载 Windows x64、CUDA、Release、包含依赖的 LibTorch ZIP，解压到例如 `C:\deps\libtorch`。

也可使用 Python 环境中安装的 PyTorch，跳过独立 LibTorch ZIP。若尚未安装，先安装 [Python](https://www.python.org/downloads/)，再在 PyTorch 下载页面选择 Windows、Pip、Python 和所需 CUDA 版本，执行对应安装命令。

使用 MSVC 构建。Windows 的 Debug 和 Release LibTorch ABI 不兼容，下文统一使用 Release；详见 [LibTorch 官方安装说明](https://docs.pytorch.org/cppdocs/installing.html)。

### 2. 使用 vcpkg 安装 OpenCV 和 cJSON

在 PowerShell 中执行；安装位置可以自行调整：

```powershell
New-Item -ItemType Directory -Force C:\deps | Out-Null
git clone https://github.com/microsoft/vcpkg.git C:\deps\vcpkg
& C:\deps\vcpkg\bootstrap-vcpkg.bat
& C:\deps\vcpkg\vcpkg.exe install "opencv4[core,jpeg,highgui,win32ui,dshow,msmf]:x64-windows" cjson:x64-windows
```

这里显式选择 JPEG、窗口显示和 Windows 视频采集相关功能。vcpkg 的 cJSON 包提供工程需要的头文件布局及 CMake 配置。可参考 [vcpkg 安装与 CMake 集成指南](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started)、[OpenCV 包说明](https://vcpkg.io/en/package/opencv4)和 [cJSON 打包配置](https://github.com/microsoft/vcpkg/blob/master/ports/cjson/portfile.cmake)。

### 3. 配置与编译

进入本工程目录，在 PowerShell 中执行：

```powershell
$env:LIBTORCH_ROOT = "C:\deps\libtorch"
$env:VCPKG_ROOT = "C:\deps\vcpkg"

cmake -S . -B build-windows -G "Visual Studio 17 2022" -A x64 `
    "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
    "-DVCPKG_TARGET_TRIPLET=x64-windows" `
    "-DCMAKE_PREFIX_PATH=$env:LIBTORCH_ROOT" `
    "-DCMAKE_CXX_FLAGS=/bigobj /utf-8"
cmake --build build-windows --config Release --parallel 2
```

若使用 Python PyTorch，激活对应环境后，将上述配置与编译命令替换为：

```powershell
$env:LIBTORCH_ROOT = python -c "from pathlib import Path; import torch; print(Path(torch.__file__).parent)"
$env:VCPKG_ROOT = "C:\deps\vcpkg"
$torchCmakePrefixPath = python -c "import torch; print(torch.utils.cmake_prefix_path)"

cmake -S . -B build-windows -G "Visual Studio 17 2022" -A x64 `
    "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
    "-DVCPKG_TARGET_TRIPLET=x64-windows" `
    "-DCMAKE_PREFIX_PATH=$torchCmakePrefixPath" `
    "-DCMAKE_CXX_FLAGS=/bigobj /utf-8"
cmake --build build-windows --config Release --parallel 2
```

训练时同样应确认该 Python 环境安装的是 CUDA 版 PyTorch，且 `python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"` 显示所需 CUDA 版本及 `True`。

运行前，在同一个 PowerShell 会话中加入动态库目录：

```powershell
$env:PATH = "$env:LIBTORCH_ROOT\lib;$env:VCPKG_ROOT\installed\x64-windows\bin;$env:PATH"
```

CUDA 安装程序通常会配置 Toolkit 的 PATH；若未配置，还需加入实际 Toolkit 的 `bin` 目录。当前 `CMakeLists.txt` 没有为 LibTorch 设置自动复制 DLL 的步骤，因此运行时需要保持上述环境可用。

## 训练数据目录与标注

### 1. 数据目录结构

训练集和评估集分别由“标注根目录”和“图片根目录”组成。以下为需要自行准备的目录示例：

```text
dataset/
├── train/
│   ├── annotations/
│   │   ├── mute.json
│   │   └── palm.json
│   └── images/
│       ├── mute/
│       │   └── sample_mute.jpg
│       └── palm/
│           └── sample_palm.jpg
└── eval/
    ├── annotations/
    │   ├── mute.json
    │   └── palm.json
    └── images/
        ├── mute/
        │   └── eval_mute.jpg
        └── palm/
            └── eval_palm.jpg
```

加载规则由 `YoloDataset.cpp` 实现：

- 标注目录直接包含 `.json` 文件，不递归搜索。
- 每个 JSON 的文件名去掉扩展名后，必须对应图片根目录下的同名子目录，例如 `mute.json` 对应 `images/mute/`。
- 图片目录只直接扫描小写 `.jpg` 扩展名；图片文件名去掉扩展名后，必须是对应 JSON 顶层的键。
- 标注提供每个框的类别，目录名用于分组查找图片，不能代替逐框标签。没有匹配标注或没有目标框的图片会被跳过；缺少同名图片目录时，该 JSON 会被跳过。
- 正式训练应使用独立的训练集、评估集，避免同一图片同时出现在两者中。

### 2. 本工程实际读取的 JSON

当前读取的是 **HaGRID 风格、以图片名为键的 JSON**。例如，上述训练集 `mute.json` 的最小内容可为：

```json
{
  "sample_mute": {
    "bboxes": [[0.10, 0.20, 0.30, 0.40]],
    "labels": ["mute"]
  }
}
```

`bboxes` 的每一项是相对于原图尺寸归一化的 `[x, y, width, height]`：`x`、`y` 是左上角，后两项是框宽和框高，不是右下角坐标。上述框覆盖原图水平方向的 10%～40%、垂直方向的 20%～60%。

`labels` 与 `bboxes` 按位置一一对应。类别字符串应与 `YoloDataset.cpp` 中的映射一致；未知类别会被归入 `no_gesture`。还支持同样结构的 `united_bbox` 与 `united_label` 数组；存在非空 `united_bbox` 时优先使用这一组，忽略普通 `bboxes`。应保证框数组和标签数组长度一致、每个框恰有四个数值、坐标有效，当前解析器没有完整的格式校验。

### 3. COCO JSON 简介与区别

[COCO JSON](https://cocodataset.org/#format-data) 是目标检测、分割等任务常用的标注组织格式。目标检测的主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `images` | 图片记录，包括 `id`、`file_name`、`width`、`height` |
| `categories` | 类别记录，通过 `id` 对应 `name` |
| `annotations` | 每个目标的记录，通过 `image_id`、`category_id` 关联图片和类别 |
| `bbox` | `[x, y, width, height]`，左上角及宽高，使用原图像素坐标 |
| `area`、`iscrowd` | 目标面积及是否为群体标注；分割任务还可提供 `segmentation` |

下面是仅包含检测框的简化示例：

```json
{
  "images": [
    {"id": 1, "file_name": "sample_mute.jpg", "width": 1000, "height": 500}
  ],
  "categories": [
    {"id": 1, "name": "mute"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 300, 200],
      "area": 60000,
      "iscrowd": 0
    }
  ]
}
```

**当前工程不能直接读取上述标准 COCO JSON。** 使用 COCO 数据时，需要自行转换为上一节的格式，并按本工程目录规则组织图片：按 `image_id` 汇总目标，通过 `category_id` 查找类别名，把像素框转换为 `[x/W, y/H, width/W, height/H]`，再用图片文件名去掉扩展名作为 JSON 键。多目标图片应保留同一图片的全部框与标签；COCO 类别 ID 不必连续，也不能直接当作本工程内部类别编号。上述 COCO 示例转换后，框恰为上一节的 `[0.10, 0.20, 0.30, 0.40]`。

## 配置与运行训练

编辑 Git 跟踪的 `example.toml`，将 `[data]` 中四个路径改为实际数据位置。对应前面的目录示例：

```toml
[data]
train_annotations_dir = "./dataset/train/annotations"
train_picture_dir = "./dataset/train/images"
eval_annotations_dir = "./dataset/eval/annotations"
eval_picture_dir = "./dataset/eval/images"
width = 512
height = 512
```

这是对原有 `[data]` 表的修改示例，不要在同一 TOML 中重复声明该表。相对路径均相对于**程序启动时的工作目录**，而非配置文件所在目录；Windows 路径建议使用正斜杠，例如 `C:/datasets/gestures/train/images`。

| 配置项 | 作用 |
| --- | --- |
| `[YOLO]` | 网络宽度、瓶颈层数及通道缩放；`P_channel` 当前固定为 `base * 32` |
| `[model] state_file / best_state_file` | 常规模型存档与最佳评估模型的保存位置 |
| `[data] width / height` | 输入尺寸，默认均为 512，建议使用 32 的倍数以匹配多尺度特征拼接 |
| `[control] batch_size / workers` | 批大小及数据加载线程数；示例中 `workers` 为 4 |
| `[control] opt_every_step` | 梯度累积的步数 |
| `[control] save_every_step / test_every_step` | 保存和评估间隔，单位为训练 step |
| `[loss]` | 分类、框、中心度损失权重，Focal Loss 参数和 NMS IoU 阈值 |
| `[optim]` | AdamW 超参数及优化器存档位置 |
| `[log]` | 日志写入间隔和目录 |

批大小、累积步数及各类间隔应为正整数。

从本工程目录启动：

```bash
# Linux
./build-linux/train example.toml
```

```powershell
# Windows
.\build-windows\Release\train.exe example.toml
```

程序会按配置创建模型、优化器存档和日志的父目录。若指定的模型、优化器存档以及日志中的进度记录存在，会分别读取以继续训练；新实验应使用一组新的输出路径。训练循环没有固定 epoch 上限，使用 `Ctrl+C` 请求退出；退出分支不会额外保存，应以最近一次定期保存为恢复点。

## 推理与日志可视化

### 视频推理

确保已经生成最佳模型，且 `[YOLO]` 网络参数与训练一致。按需修改 `example.toml` 中已有的 `[predict]` 表：

```toml
[predict]
device = "cpu" # 也可设为 "cuda"，不可用时回退到 CPU
conf_threshold = 0.6
box_threshold = 0.25
camera = "/dev/video0"
```

`client` 从 `[model] best_state_file` 加载模型，需要 OpenCV 窗口环境及可用的视频输入。Linux 可使用实际摄像头设备路径；Windows 可将 `camera` 改为 OpenCV 后端能够打开的视频文件路径或视频流 URL。**当前代码使用字符串版 `cv::VideoCapture`，配置 `camera = "0"` 不会选择编号为 0 的摄像头；Windows 本机摄像头索引需先修改 `client.cpp`，改用整数版接口。** 相关接口见 [OpenCV VideoCapture 文档](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)。

```bash
# Linux
./build-linux/client example.toml
```

```powershell
# Windows
.\build-windows\Release\client.exe example.toml
```

按 `Esc` 关闭推理窗口。

### TensorBoard（可选）

使用独立 LibTorch 发行包构建、运行 C++ 程序时不需要 Python PyTorch。使用 `watch_tb.py` 时，先准备 Python 3.10+ 环境，安装 Python PyTorch 与 TensorBoard；日志转换可使用单独环境中的 CPU 版 PyTorch，无需与 C++ LibTorch 共用安装目录。

Linux：

```bash
python3 -m venv "$HOME/.venvs/yolo-tb"
source "$HOME/.venvs/yolo-tb/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install tensorboard
```

Windows：先安装 [Python](https://www.python.org/downloads/)，再执行：

```powershell
py -3 -m venv "$env:USERPROFILE\venvs\yolo-tb"
$tbPython = "$env:USERPROFILE\venvs\yolo-tb\Scripts\python.exe"
& $tbPython -m pip install --upgrade pip
& $tbPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& $tbPython -m pip install tensorboard
```

在已安装上述依赖的 Python 环境中，分别打开两个终端运行；路径应与 `[log] log_dir` 一致：

```bash
python watch_tb.py --log-dir ./runs/mute_test/logs
python -m tensorboard.main --logdir ./runs/mute_test/logs/tb
```

Windows 若未激活环境，将上述 `python` 替换成 `& $tbPython`，并在每个终端设置该变量。浏览器访问 `http://localhost:6006`。给脚本加上 `--once` 可仅转换现有日志，加上 `--per-class` 可记录逐类别曲线。

评估日志的 `precision` 列实际保存的是 `TP / 真实目标数`，即召回率；`err_rate` 为 `FP / 真实目标数`，并非通常定义的误分类率。`watch_tb.py` 据此计算 precision、F1 等指标，避免直接按列名误读结果。
