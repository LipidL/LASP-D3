# filepath: d3_python.py
import numpy as np
from d3_cffi import D3Calculator

def test_methane_dispersion():
    """Test dispersion calculation for a simple methane molecule."""
    print("Testing D3 dispersion calculation for methane...")
    
    # Define elements and max number of atoms for the calculator
    # Methane: 1 carbon (atomic number 6) and 4 hydrogens (atomic number 1)
    elements = [6, 1, 1, 1, 1]
    max_atoms = len(elements)
    
    # Create the D3Calculator with appropriate parameters
    calculator = D3Calculator(
        elements=elements,
        max_length=max_atoms,
        cutoff_radius=30.0,
        cn_cutoff_radius=40.0,
        max_neighbors=100
    )
    
    # Define methane coordinates (in Angstroms)
    # Carbon at center, hydrogens at tetrahedral positions
    coords = np.array([
        [0.000,  0.000,  0.000],  # C
        [0.629,  0.629,  0.629],  # H
        [-0.629, -0.629,  0.629],  # H
        [-0.629,  0.629, -0.629],  # H
        [0.629, -0.629, -0.629]   # H
    ], dtype=np.float32)
    
    # Set atoms in the calculator
    calculator.set_atoms(coords, elements)
    
    # Define a large periodic box (non-periodic calculation)
    cell = np.eye(3, dtype=np.float32) * 50.0  # 50 Å cubic box
    calculator.set_cell(cell)
    
    # Patch the class to add get_num_atoms method
    D3Calculator.get_num_atoms = lambda self: len(elements)
    
    # Compute dispersion energy, forces, and stress
    energy, forces, stress = calculator.compute()
    
    # Print results
    print(f"Dispersion energy: {energy:.6f} eV")
    print("Forces (eV/Å):")
    for i, force in enumerate(forces):
        element = "C" if elements[i] == 6 else "H"
        print(f"  Atom {i} ({element}): [{force[0]:.6f}, {force[1]:.6f}, {force[2]:.6f}]")
    
    print("Stress tensor (eV/Å³):")
    print(stress)
    
    # Basic validation
    print("\nBasic validation:")
    print(f"  Energy is finite: {np.isfinite(energy)}")
    print(f"  Forces sum to zero (conservation): {np.allclose(np.sum(forces, axis=0), 0.0, atol=1e-5)}")
    
    # Clean up
    calculator.clear()
    
    return energy, forces, stress

if __name__ == "__main__":
    # Initialize parameters if needed
    # This might be needed only once at the beginning
    from d3_cffi import lib
    lib.init_params()
    
    # Run the test
    test_methane_dispersion()