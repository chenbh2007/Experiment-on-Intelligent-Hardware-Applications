cmake_minimum_required(VERSION 2.8)
project(faceDetect)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(faceDetect facedetect.cpp)
target_link_libraries(faceDetect PRIVATE ${OpenCV_LIBS} m)

add_executable(privacyProtect privacyprotect.cpp)
target_link_libraries(privacyProtect PRIVATE ${OpenCV_LIBS})
