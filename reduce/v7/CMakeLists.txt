cmake_minimum_required(VERSION 3.10)
project(reduce_v7 CUDA CXX)

find_package(CUDAToolkit REQUIRED)
add_executable(reduce_v7 main.cu)
target_link_libraries(reduce_v7 PRIVATE CUDA::cudart)
