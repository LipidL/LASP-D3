cmake_minimum_required(VERSION 3.18)

# Set the project name and enable CUDA
project(MyProject LANGUAGES CXX CUDA)

# Require C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add your source files here
add_executable(MyExecutable src/main.cpp)

# Link CUDA to the target
target_link_libraries(MyExecutable PRIVATE CUDA::cudart)

# Set the compiler for C++ and CUDA
set(CMAKE_CXX_COMPILER cl)
set(CMAKE_CUDA_COMPILER nvcc)

# Specify compile options for CUDA
target_compile_options(MyExecutable PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# Specify compile options for C++
target_compile_options(MyExecutable PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
)