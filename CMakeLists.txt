cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR)

project(d3_project LANGUAGES CXX CUDA Fortran)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)  # Compatible with CUDA 11.8
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Source files
set(SOURCES 
    src/d3.cu
    src/d3_buffer.cu
    src/d3_kernel.cu
)

set(HEADERS 
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
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/data/generate_constants.py ${CMAKE_CURRENT_SOURCE_DIR}/data/parameters.f90
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data/generate_constants.py ${CMAKE_CURRENT_SOURCE_DIR}/data/parameters.f90
    COMMENT "Generating constants header file"
)

# Add the generated header to the build dependencies
add_custom_target(generate_constants ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gen_constants.h)

# Generate object file
add_library(d3_obj OBJECT ${SOURCES} ${HEADERS})

# Apply CUDA options to object file
target_compile_options(d3_obj PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -O2
    -lineinfo
    --maxrregcount=64
>)

add_dependencies(d3_obj generate_constants)

# Set CUDA properties for object file
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
        -O2 -lineinfo --maxrregcount=64 -DBUILD_LIBRARY)
endforeach()

add_custom_command(
    OUTPUT ${OUTPUT_OBJECTS}
    ${COMPILE_COMMANDS}
    DEPENDS ${SOURCES} ${HEADERS}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/gen_constants.h
    COMMENT "Generating standalone object files"
)

# Custom target to ensure .o file is built
add_custom_target(d3_o ALL DEPENDS ${OUTPUT_OBJECTS})
add_dependencies(d3_o generate_constants)

# Create static library from object files
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libd3.a
    COMMAND ${CMAKE_AR} rcs ${CMAKE_CURRENT_BINARY_DIR}/libd3.a ${OUTPUT_OBJECTS}
    DEPENDS ${OUTPUT_OBJECTS}
    COMMENT "Creating static library archive libd3.a"
)

# Custom target to ensure static library is built
add_custom_target(d3_static ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libd3.a)
add_dependencies(d3_static d3_o)

# Install the static library
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libd3.a
        DESTINATION lib
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)

# Create dynamic shared library
add_library(d3_shared SHARED $<TARGET_OBJECTS:d3_obj>)

# Add dependency on generated constants if static constants are enabled
add_dependencies(d3_shared generate_constants)

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

# Find CUDA runtime location
find_library(CUDA_RUNTIME_LIBRARY cudart
    HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

# Create Fortran executable
add_executable(d3_fortran src/fortran_d3.f90)

# Link with static library and required libraries
target_link_libraries(d3_fortran PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/libd3.a
    ${CUDA_RUNTIME_LIBRARY}
    m
    stdc++
)

# Add dependency on static library
add_dependencies(d3_fortran d3_static)

# Create benchmark executable
add_executable(d3_cpp src/main.cpp)

# Add static library as dependency
add_dependencies(d3_cpp d3_static)

# Add static constants dependency if enabled
if(USE_STATIC_CONSTANTS)
    add_dependencies(d3_cpp generate_constants)
endif()

# Link with the static library and required libraries
target_link_libraries(d3_cpp PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/libd3.a
    ${CUDA_RUNTIME_LIBRARY}
    m 
    stdc++
)

# Include FetchContent for downloading GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY http://10.158.134.245:3000/luohz/googletest.git
    GIT_TAG v1.17.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
include(CTest)
enable_testing()

# Add the test executable
add_executable(d3_test test/d3_test.cpp)

# Link against GoogleTest
target_link_libraries(d3_test PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/libd3.a
    GTest::gtest_main
    ${CUDA_RUNTIME_LIBRARY}
    m
)

# Register the test
include(GoogleTest)
gtest_discover_tests(d3_test)