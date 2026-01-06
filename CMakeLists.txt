cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR)

project(d3_project LANGUAGES CXX CUDA Fortran)

# Option to use extended parameter space
option(USE_EXTENDED_PARAMETERS "Use extended parameter space (103 elements) instead of original (95 elements)" OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)  # Compatible with CUDA 11.8
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA runtime location
find_library(CUDA_RUNTIME_LIBRARY cudart
    HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

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

# Create static library
add_library(d3_static STATIC ${SOURCES} ${HEADERS})

# Apply CUDA options
target_compile_options(d3_static PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -O2
    -lineinfo
    --maxrregcount=64
>)

# Add compile definition for extended parameters if enabled
if(USE_EXTENDED_PARAMETERS)
    target_compile_definitions(d3_static PRIVATE USE_EXTENDED_PARAMETERS)
endif()

# Set CUDA properties for static library
set_target_properties(d3_static PROPERTIES
    OUTPUT_NAME "d3"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(d3_static PUBLIC ${CUDA_RUNTIME_LIBRARY} m)

# Create shared library
add_library(d3_shared SHARED ${SOURCES} ${HEADERS})

# Apply CUDA options
target_compile_options(d3_shared PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -O2
    -lineinfo
    --maxrregcount=64
>)

# Add compile definition for extended parameters if enabled
if(USE_EXTENDED_PARAMETERS)
    target_compile_definitions(d3_shared PRIVATE USE_EXTENDED_PARAMETERS)
endif()

# Set CUDA properties for shared library
set_target_properties(d3_shared PROPERTIES
    OUTPUT_NAME "d3"
    VERSION 1.0.0
    SOVERSION 1
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(d3_shared PUBLIC ${CUDA_RUNTIME_LIBRARY} m)

# Install libraries
install(TARGETS d3_static d3_shared
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

# Create Fortran executable
add_executable(d3_fortran src/fortran_d3.f90)
target_link_libraries(d3_fortran PRIVATE d3_static stdc++)

# Create benchmark executable
add_executable(d3_cpp src/main.cpp)
target_link_libraries(d3_cpp PRIVATE d3_static stdc++)

# Include FetchContent for downloading GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
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
target_link_libraries(d3_test PRIVATE d3_static GTest::gtest_main)

# Register the test
include(GoogleTest)
gtest_discover_tests(d3_test)