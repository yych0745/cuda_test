cmake_minimum_required(VERSION 3.10)
project(reduce_v1 CUDA CXX)

find_package(CUDAToolkit REQUIRED)
add_executable(reduce_v1 main.cu)
target_link_libraries(reduce_v1 PRIVATE CUDA::cudart)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(reduce_v1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(reduce_v1 PROPERTIES $<$<COMPILE_LANGUAGE:CUDA>:-G>)
<<<<<<< HEAD
endif()
=======
endif()
>>>>>>> f27810a66702dd9d383d9eba840e9761313baa90
