cmake_minimum_required(VERSION 3.23)
project(GrabCut)

set(CMAKE_CXX_STANDARD 17)

# find_library(OpenCV)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(maxflow)

aux_source_directory(maxflow maxflow)
aux_source_directory(. source_list)

add_executable(GrabCut ${source_list} ${maxflow})
target_link_libraries(GrabCut ${OpenCV_LIBS})