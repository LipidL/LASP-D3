"""
Test script for the D3Calculator from d3_cffi module.

This script demonstrates how to use the D3 dispersion correction 
calculator with a simple example of a water molecule.
"""
import numpy as np
from d3_cffi import D3Calculator

def main():
    """
    Main function to test the D3 calculator
    """
    print("Testing D3 dispersion calculator...")
    
    # Create a simple system: water molecule (H2O)
    # Elements: H=1, O=8
    elements = [8, 1, 1]  # O, H, H
    
    # Coordinates in Angstroms (simple geometry, not optimized)
    coords = np.array([
        [0.0, 0.0, 0.0],  # O at origin
        [0.0, 0.8, 0.6],  # H1
        [0.0, -0.8, 0.6]  # H2
    ], dtype=np.float32)
    
    # Create a periodic cell (box size in Angstroms)
    cell = np.eye(3, dtype=np.float32) * 10.0  # 10Å cubic box
    
    # Initialize the D3 calculator
    print("Initializing D3Calculator...")
    d3calc = D3Calculator(
        elements=elements,
        max_length=5,
        cutoff_radius=46.475800,
        cn_cutoff_radius=46.475800,
        max_neighbors=10000
    )
    
    # Set the atoms and cell
    print("Setting atoms and cell...")
    d3calc.set_atoms(coords, elements)
    d3calc.set_cell(cell)
    
    # Compute dispersion energy, forces and stress
    print("Computing dispersion energy...")
    energy, forces, stress = d3calc.compute()
    
    # Print the results
    print("\nResults:")
    print(f"Dispersion energy: {energy:.6f} eV")
    print("\nForces (eV/Å):")
    for i, element in enumerate(elements):
        element_symbol = "O" if element == 8 else "H"
        print(f"Atom {i} ({element_symbol}): {forces[i][0]:.6f}, {forces[i][1]:.6f}, {forces[i][2]:.6f}")
    
    print("\nStress tensor (eV/Å³):")
    for i in range(3):
        print(f"  {stress[i][0]:.6f}  {stress[i][1]:.6f}  {stress[i][2]:.6f}")
    
    # Clear the calculator
    print("\nClearing D3 calculator...")
    d3calc.clear()

if __name__ == "__main__":
    main()
