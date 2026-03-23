# Example: Integrating LASP-D3 with MACE and ASE

This directory contains example code for integrating LASP-D3 with the MACE model and running a simple MD simulation using the ASE engine.
The main script is `ase_mace_laspd3.py`.

## Requirements

- Python 3.14
- mace-torch 0.3.15
- ase 3.28.0

## Models and Input Files

Download your MACE model and set the `PATH_TO_MODEL` constant in the script.
Locate your input structure file and set the `PATH_TO_STRUCTURES` constant accordingly.

## Python Interface

The Python interface is defined in `d3_cffi.py`.
A symbolic link to it has been created in this directory — ensure `ase_mace_laspd3.py` can resolve this path before running.

## Running the Example

Run the script with:

```bash
python ase_mace_laspd3.py
```

This will perform a simple MD simulation using the MACE model with LASP-D3 dispersion correction via the ASE engine.
