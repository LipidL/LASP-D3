cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR)

project(d3_project LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Source files
set(SOURCES src/d3.cu)
set(HEADERS src/constants.h)

# Create executable
add_executable(d3 ${SOURCES} ${HEADERS})

# Set CUDA compilation options
target_compile_options(d3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xptxas -O2
    -lineinfo
>)

# Link math library
target_link_libraries(d3 PRIVATE m)

# Set CUDA properties
set_target_properties(d3 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Create a static library target
add_library(d3_lib STATIC src/d3.cu src/d3.h ${HEADERS})

# Define BUILD_LIBRARY for the library target
target_compile_definitions(d3_lib PRIVATE BUILD_LIBRARY)

# Set CUDA compilation options for the library
target_compile_options(d3_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xptxas -O2
    -lineinfo
>)

# Set output name and properties
set_target_properties(d3_lib PROPERTIES
    OUTPUT_NAME d3
    CUDA_SEPARABLE_COMPILATION ON
)

# Link with the static CUDA runtime library
target_link_libraries(d3_lib PRIVATE
    /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcudart_static.a
    ${CMAKE_DL_LIBS}  # Required for static CUDA runtime
    rt                # Required for static CUDA runtime
)