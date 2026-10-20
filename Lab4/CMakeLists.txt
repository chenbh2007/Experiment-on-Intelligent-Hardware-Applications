cmake_minimum_required(VERSION 3.22)
project(numberProc LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV 4 REQUIRED)
add_executable(picToVec getRawData/picToVec.cpp)
target_include_directories(picToVec PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(picToVec PRIVATE ${OpenCV_LIBS})

add_executable(idx2db getRawData/getRawDatafromidx.cpp)
target_include_directories(idx2db PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(idx2db PRIVATE ${OpenCV_LIBS})

add_executable(raw2db setUpDataBase/setUpDataBase.cpp)
add_executable(train train.cpp SVM/cuSVM.cu)
target_include_directories(train PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(train PRIVATE CUDA::cudart m)

add_executable(test predict.cpp SVM/SVM.c)
target_include_directories(test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test PRIVATE m)
