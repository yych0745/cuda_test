cmake_minimum_required(VERSION 3.10)
project(reduce_v4 CUDA CXX)

find_package(CUDAToolkit REQUIRED)
add_executable(reduce_v4 main.cu)
target_link_libraries(reduce_v4 PRIVATE CUDA::cudart)
