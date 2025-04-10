
cmake_minimum_required(VERSION 3.18)
project(FireSimulator LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(SOURCES
    main.cpp
    fire_kernel.cu
)

add_executable(fire_simulator ${SOURCES})
set_source_files_properties(fire_kernel.cu PROPERTIES LANGUAGE CUDA)
