cmake_minimum_required(VERSION 2.8)
project(objectTrack)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(track track.cpp)
target_link_libraries(track PRIVATE ${OpenCV_LIBS} m)
