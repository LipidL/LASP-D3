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

#### Build using extended constants

The DFT-D3 implementation has multiple version of constants.
For example, torch-dftd(https://github.com/pfnet-research/torch-dftd) and simple-dftd3(https://github.com/dftd3/simple-dftd3) use different parameters.
`torch-dftd` uses 5-reference system for C6 evaluation, in consistent with the algorithm of Grimme et al.
`simple-dftd3`, on the other hand, uses an extended 7-references system for C6 evaluation.

We provide support for both reference types.
By default, we use the original 5-reference system.
User can use extended 7-reference system by specifying `-DUSE_EXTENDED_PARAMETERS=ON` when configuring.

### Use

The executable can be executed directly as long as you have CUDA runtime and a supported GPU.
running it, it will give the D3 energy, force and stress of an acetone molecule.

To use this project as a library, include the header file `src/d3.h`, and add `libd3.a` to your compilation list.
`libd3.a` only includes project-specific code and doesn't include the CUDA runtime.
Therefore, you need your own CUDA runtime when building your project using libd3.
