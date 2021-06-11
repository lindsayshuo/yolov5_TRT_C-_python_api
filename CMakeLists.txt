cmake_minimum_required(VERSION 2.6)

project(yolov5)

# add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
# include_directories(/usr/local/cuda/include)
# link_directories(/usr/local/cuda/lib64)
include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1/lib64)

# tensorrt
# include_directories(/usr/include/x86_64-linux-gnu/)
# link_directories(/usr/lib/x86_64-linux-gnu/)
#include_directories(/home/sz/TensorRT-7.2.3.4/include)
#link_directories(/home/sz/TensorRT-7.2.3.4/lib)
include_directories(/home/lindsay/TensorRT-7.2.2.3/include)
link_directories(/home/lindsay/TensorRT-7.2.2.3/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yolov5_trt SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/yolov5_lib.cpp)

find_package(OpenCV)
# find_package(OpenCV 4.4.0  REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(yolov5_trt nvinfer cudart ${OpenCV_LIBS})

# add_executable(yolov5 ${PROJECT_SOURCE_DIR}/yolov5.cpp)
add_executable(yolov5 ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/yolov5.cpp)
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 yolov5_trt)
target_link_libraries(yolov5 ${OpenCV_LIBS})

message(${OpenCV_LIBS})

add_definitions(-O2 -pthread)

