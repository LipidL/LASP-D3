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

# Generate object file
add_library(d3_obj OBJECT ${SOURCES} ${HEADERS})

# Apply same CUDA options to object file
target_compile_options(d3_obj PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xptxas -O2
    -lineinfo
>)

# Set same CUDA properties for object file
set_target_properties(d3_obj PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link math library for object file
target_link_libraries(d3_obj PRIVATE m)

# Custom command to create standalone .o file
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/d3.o
    COMMAND ${CMAKE_CUDA_COMPILER} -c ${CMAKE_CURRENT_SOURCE_DIR}/src/d3.cu -o ${CMAKE_CURRENT_BINARY_DIR}/d3.o
        -Xptxas -O2 -lineinfo -DBUILD_LIBRARY
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/d3.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/constants.h
    COMMENT "Generating standalone d3.o file"
)

# Custom target to ensure .o file is built
add_custom_target(d3_o ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/d3.o)