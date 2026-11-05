cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(signclassifer)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/PCF8591.c")
    set(PCF8591_SRC "${CMAKE_CURRENT_SOURCE_DIR}/PCF8591.c")
else()
    set(PCF8591_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../PCF8591.c")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SignClassifier.cpp")
    add_executable(SignClassifier SignClassifier.cpp toml.hpp)
    target_link_libraries(SignClassifier PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
endif()

add_executable(GestureControlLED GestureControlLED.cpp ${PCF8591_SRC} toml.hpp)
target_link_libraries(GestureControlLED PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
