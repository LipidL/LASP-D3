# DFT-D3 Calculator for Periodic Systems

This project implements a calculator for computing DFT-D3 dispersion corrections in periodic systems with CUDA acceleration.

## Overview

DFT-D3 is a semi-empirical dispersion correction method developed by Grimme et al. that accounts for van der Waals interactions in DFT calculations.
This implementation supports:

- Calculation of DFT-D3 dispersion energy
- Calculation of dispersion forces on atoms
- Calculation of dispersion contribution to stress tensor
- Full periodic boundary condition support
- GPU acceleration with CUDA

This tool can be used as a standalone calculator or integrated with other computational chemistry software.

## Parameters Management

The project offers two methods for handling the large parameter arrays required for DFT-D3 calculations:

1. **Runtime Loading**: Parameters are loaded at runtime from a binary file (`params.bin`).
   - Advantage: Keeps source code clean and language server responsive.
   - Disadvantage: Performance overhead from loading parameters at runtime and requires distributing the params.bin file.

2. **Compile-time Constants**: Parameters are embedded directly into the compiled binary.
   - Advantage: Faster runtime performance, no need to distribute params.bin separately.
   - Disadvantage: Large constants in source code (but generated automatically, so language server issues are avoided).

## Quick start

### Prerequirements

- cmake
    Starting from version 0.5.0, this project uses `cmake` for compilation and deployment.
    Due to the use of C++ 20 features in CUDA code, we require `cmake` newer than 3.25.2.
- cuda compiler
    Since this is a CUDA program, it requires a CUDA compiler, most commonly, `nvcc`, to compile the project
- C++ compiler
    The C++ code in this project uses C++ 20 features, so your C++ compiler should support C++ 20 standard.

### Build

You can quickly build this project using cmake by running the following commands:

#### Basic Build

```powershell
cd path/to/dft_d3 # navigate to project's root path
mkdir -p build; cd build # create build directory and navigate to it
cmake .. # configure the project
cmake --build . # build both executable and library
```

If you prefer building specific targets, you can run `cmake --build . --target d3` or `cmake --build . --target d3_lib` for building executable or library separately.

#### Build with Different Parameter Management Options

##### Using Compile-time Constants (Recommended for Performance)

This approach generates a header file with embedded constants during the build process, avoiding the need for runtime loading:

```powershell
cd path/to/dft_d3
mkdir -p build; cd build
cmake -DUSE_STATIC_CONSTANTS=ON .. # ON by default
cmake --build .
```

##### Using Runtime Parameter Loading

If you prefer to load parameters at runtime:

```powershell
cd path/to/dft_d3
mkdir -p build; cd build
cmake -DUSE_STATIC_CONSTANTS=OFF ..
cmake --build .
```

Make sure the `params.bin` file is available in the working directory when using runtime loading.

### Use

The executable can be executed directly as long as you have CUDA runtime and a supported GPU.
running it, it will give the D3 energy, force and stress of an acetone molecule.

To use this project as a library, include the header file `src/d3.h`, and add `libd3.a` to your compilation list.
`libd3.a` only includes project-specific code and doesn't include the CUDA runtime.
Therefore, you need your own CUDA runtime when building your project using libd3.
