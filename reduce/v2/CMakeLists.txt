cmake_minimum_required(VERSION 3.10)
project(reduce_v1 CUDA CXX)

find_package(CUDAToolkit REQUIRED)
add_executable(reduce_v2 main.cu)
target_link_libraries(reduce_v2 PRIVATE CUDA::cudart)
