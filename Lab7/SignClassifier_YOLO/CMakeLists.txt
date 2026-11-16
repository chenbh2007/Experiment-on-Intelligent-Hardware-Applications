cmake_minimum_required(VERSION 3.24)
project(SignClassifier_YOLO LANGUAGES CXX CUDA)

# Recent LibTorch releases require C++20 in both C++ and CUDA sources.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# CUDA-enabled LibTorch uses TORCH_CUDA_ARCH_LIST (8.0 for an A100).
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(cJSON CONFIG REQUIRED)

# Build the split sources; train.cu retains the original single-file version.
add_executable(train
    train.cpp
    YoloDataset.cu
    YoloDataset.hpp
    YOLOStructure.hpp
    toml.hpp
)

target_include_directories(train PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(train PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    cjson
    CUDA::cudart
)

# Preserve the ABI/compiler options required by the selected LibTorch build.
if(TORCH_CXX_FLAGS)
    separate_arguments(YOLO_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(train PRIVATE ${YOLO_TORCH_CXX_FLAGS})
endif()

# Set on the target after find_package(Torch), since Torch resets CMAKE_CUDA_ARCHITECTURES to OFF.
set_target_properties(train PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES native
)
