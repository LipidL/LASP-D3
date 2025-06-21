cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR)

project(d3_project LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)  # Compatible with CUDA 11.8
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Option to use static constants instead of runtime loading
option(USE_STATIC_CONSTANTS "Use statically compiled constants instead of runtime loading" ON)

# Configure build with static constants if enabled
if(USE_STATIC_CONSTANTS)
    add_compile_definitions(BUILD_WITH_STATIC_CONSTANTS)
endif()

# Source files
set(SOURCES 
    src/d3.cu
    src/d3_buffer.cu
    src/d3_kernel.cu
)

set(HEADERS 
    src/constants.h
    src/constants_include.h
    src/d3_buffer.cuh
    src/d3_internal.h
    src/d3_types.h
    src/d3_kernel.cuh
    src/d3.h
)

# Add custom command to generate constants header at build time
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/gen_constants.h
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/data/generate_constants.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data/generate_constants.py ${CMAKE_CURRENT_SOURCE_DIR}/data/dftd3_params.npz
    COMMENT "Generating constants header file"
)

# Add the generated header to the build dependencies
add_custom_target(generate_constants ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gen_constants.h)

# Create executable
add_executable(d3 ${SOURCES} ${HEADERS})
add_dependencies(d3 generate_constants)

# Set CUDA compilation options
target_compile_options(d3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xptxas -O3
    -lineinfo
>)

# Set define for static constants if enabled
if(USE_STATIC_CONSTANTS)
    target_compile_definitions(d3 PRIVATE BUILD_WITH_STATIC_CONSTANTS)
endif()

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

# Set define for static constants if enabled
if(USE_STATIC_CONSTANTS)
    target_compile_definitions(d3_obj PRIVATE BUILD_WITH_STATIC_CONSTANTS)
endif()

# Set same CUDA properties for object file
set_target_properties(d3_obj PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link math library for object file
target_link_libraries(d3_obj PRIVATE m)

# Custom command to create standalone .o file
# Generate output object filenames based on source files
set(OUTPUT_OBJECTS "")
set(COMPILE_COMMANDS "")
foreach(SOURCE_FILE ${SOURCES})
    get_filename_component(FILE_NAME ${SOURCE_FILE} NAME_WE)
    list(APPEND OUTPUT_OBJECTS ${CMAKE_CURRENT_BINARY_DIR}/${FILE_NAME}.o)
    list(APPEND COMPILE_COMMANDS 
        COMMAND ${CMAKE_CUDA_COMPILER} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${CMAKE_CURRENT_BINARY_DIR}/${FILE_NAME}.o
        -Xptxas -O2 -lineinfo -DBUILD_LIBRARY $<$<BOOL:${USE_STATIC_CONSTANTS}>:-DBUILD_WITH_STATIC_CONSTANTS>)
endforeach()

add_custom_command(
    OUTPUT ${OUTPUT_OBJECTS}
    ${COMPILE_COMMANDS}
    DEPENDS ${SOURCES} ${HEADERS}
            $<$<BOOL:${USE_STATIC_CONSTANTS}>:${CMAKE_CURRENT_SOURCE_DIR}/src/gen_constants.h>
    COMMENT "Generating standalone object files"
)

# Custom target to ensure .o file is built
add_custom_target(d3_o ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/d3.o)
if(USE_STATIC_CONSTANTS)
    add_dependencies(d3_o generate_constants)
endif()

# Create dynamic shared library
add_library(d3_shared SHARED $<TARGET_OBJECTS:d3_obj>)

# Add dependency on generated constants if static constants are enabled
if(USE_STATIC_CONSTANTS)
    add_dependencies(d3_shared generate_constants)
endif()

# Set library properties
set_target_properties(d3_shared PROPERTIES
    OUTPUT_NAME "d3"
    VERSION 1.0.0
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Link with required libraries
target_link_libraries(d3_shared PRIVATE 
    ${CUDA_RUNTIME_LIBRARY}
    m
)

# Enable Fortran language
enable_language(Fortran)

# Find CUDA runtime location
find_library(CUDA_RUNTIME_LIBRARY cudart
    HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

# Create Fortran interface library
add_library(d3_fortran OBJECT src/fortran_d3.f90)

# Set properties for the Fortran interface
set_target_properties(d3_fortran PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Add d3.o as dependency
# Custom command to create standalone Fortran .o file
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/d3_fortran.o
    COMMAND ${CMAKE_Fortran_COMPILER} -c ${CMAKE_CURRENT_SOURCE_DIR}/src/fortran_d3.f90 
            -o ${CMAKE_CURRENT_BINARY_DIR}/d3_fortran.o
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/fortran_d3.f90 ${CMAKE_CURRENT_BINARY_DIR}/d3.o ${CMAKE_CURRENT_BINARY_DIR}/d3_buffer.o
    COMMENT "Generating Fortran interface d3_fortran.o file"
)

# Custom target to ensure Fortran .o file is built
add_custom_target(d3_fortran_o ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/d3_fortran.o)

# Create benchmark executable
add_executable(d3_cpp src/main.cpp)

# Add d3.o as dependency
add_dependencies(d3_cpp d3_o)

# Add static constants dependency if enabled
if(USE_STATIC_CONSTANTS)
    add_dependencies(d3_cpp generate_constants)
endif()

# Link with the d3.o and d3_buffer.o files and required libraries
target_link_libraries(d3_cpp PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/d3.o
    ${CMAKE_CURRENT_BINARY_DIR}/d3_buffer.o
    ${CMAKE_CURRENT_BINARY_DIR}/d3_kernel.o
    ${CUDA_RUNTIME_LIBRARY}
    m stdc++)