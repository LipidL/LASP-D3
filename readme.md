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

## Quick start

### Prerequirements

- cmake
    Starting from version 0.5.0, this project uses `cmake` for compilation and deployment.
    Due to the use of C++ 17 features in CUDA code, we require `cmake` newer than 3.25.2.
- cuda compiler
    Since this is a CUDA program, it requires a CUDA compiler, most commonly, `nvcc`, to compile the project
- C++ compiler
    The C++ code in this project uses C++ 17 features, so your C++ compiler should support C++ 17 standard.

### Build

You can quickly build this project using cmake by running the following commands:

#### Basic Build

```powershell
cd path/to/dft_d3 # navigate to project's root path
mkdir -p build; cd build # create build directory and navigate to it
cmake .. # configure the project
cmake --build . # build both executable and library
ctest # run unit tests for validation
```

#### Build Targets

This project provides multiple targets, including a static library(`d3_static`), a dynamic library(`d3_dynamic`), a C++ executable(`d3_cpp`) and a Fortran executable(`d3_fortran`).
The static and dynamic library follows C-style API and can be used by most programming languages.
The C++ executable reads structures from `.arc` file specified by command line arguement and calculates the dispersion energy, force and stress.
The Fortran executable is an example executable that computes the dispersion energy, force and stress of a small molecule.

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

This project provides both static library (`libd3.a`) and dynamic library (`libd3.so`).
The apis are defined in `src/d3.h` with C-style APIs for maximum compatibility.
Example code for Fortran is provided at `src/fortran_d3.f90` 
and for Python is provided at `src/d3_cffi.py` for an object-oriented encapsulation and `src/test_d3_cffi.py` for usage.

This project also provides a C++ erxecutable (`d3_cpp`) to calculate dispersion energy, force and stress from a given `.arc` file.
