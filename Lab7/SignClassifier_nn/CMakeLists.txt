cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(signclassifer)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(SignClassifier SignClassifier.cpp toml.hpp)
target_link_libraries(SignClassifier PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
