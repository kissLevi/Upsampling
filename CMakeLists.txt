cmake_minimum_required(VERSION 3.15)
project(upsampling)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

add_executable(upsampling main.cpp)

# Additional include directories
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(upsampling ${OpenCV_LIBS}) # replace binary_name