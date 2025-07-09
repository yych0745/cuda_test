cmake_minimum_required(VERSION 3.10)
project(sgeem_v1 CUDA CXX)

find_package(CUDAToolkit REQUIRED)
add_executable(sgeem_v1 main.cu)
target_link_libraries(sgeem_v1 PRIVATE CUDA::cudart)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sgeem_v1 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()