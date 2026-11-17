cmake_minimum_required(VERSION 3.24)
project(SignClassifier_YOLO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Help CMake/Torch locate nvcc when /usr/local/cuda/bin is not in PATH (e.g. colab ssh).
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

find_package(Torch REQUIRED)
find_package(CUDAToolkit QUIET)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs OPTIONAL_COMPONENTS highgui videoio)
find_package(cJSON CONFIG REQUIRED)

# Shared core library (YoloDataset + common headers/dependencies + precompiled headers)
add_library(yolo_common STATIC
    YoloDataset.cpp
    YoloDataset.hpp
    YOLOStructure.hpp
    toml.hpp
)

target_include_directories(yolo_common PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo_common PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    cjson
)

if(TARGET CUDA::cudart)
    target_link_libraries(yolo_common PUBLIC CUDA::cudart)
endif()

# Preserve the ABI/compiler options required by the selected LibTorch build.
if(TORCH_CXX_FLAGS)
    separate_arguments(YOLO_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(yolo_common PUBLIC ${YOLO_TORCH_CXX_FLAGS})
endif()

target_precompile_headers(yolo_common PUBLIC
    <torch/torch.h>
    <opencv2/opencv.hpp>
    "toml.hpp"
)

# Training executable
add_executable(train train.cpp)
target_link_libraries(train PRIVATE yolo_common)
target_precompile_headers(train REUSE_FROM yolo_common)

# Inference client executable (uses TRAIN_AS_LIBRARY to skip train.cpp's main/handler)
if(OpenCV_highgui_FOUND AND OpenCV_videoio_FOUND)
    add_executable(client
        client.cpp
        train.cpp
    )
    target_compile_definitions(client PRIVATE TRAIN_AS_LIBRARY)
    target_link_libraries(client PRIVATE yolo_common)
    target_precompile_headers(client REUSE_FROM yolo_common)
else()
    message(STATUS "OpenCV highgui/videoio not found; skipping 'client' target.")
endif()
