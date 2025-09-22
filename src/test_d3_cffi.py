"""
Test script for the D3Calculator from d3_cffi module with MD

This script tests the D3Calculator class from the d3_cffi module by performing
dispersion energy calculations on a set of randomly generated atomic coordinates.
"""
import numpy as np
from d3_cffi import D3Calculator
import time
import re
import sys

def read_file(filename: str) -> str:
    with open(filename) as f:
        file = f.read()
    return file


def abc2cell(abc: np.ndarray) -> np.ndarray:
    a, b, c, alpha, beta, gamma = abc
    alpha_rad, beta_rad, gamma_rad = np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)
    cell = np.zeros((3, 3))
    cell[0, 0] = a
    cell[0, 1] = b * np.cos(gamma_rad)
    cell[1, 1] = b * np.sin(gamma_rad)
    cell[0, 2] = c * np.cos(beta_rad)
    cell[1, 2] = (c * np.cos(alpha_rad) - c * np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
    cell[2, 2] = np.sqrt(c ** 2 - cell[0, 2] ** 2 - cell[1, 2] ** 2)
    cell = cell.T  # Transpose to match standard crystallographic cell definition
    return cell

ELE2INT = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16,
    'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24,
    'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32,
    'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
    'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56,
    'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72,
    'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88,
    'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96,
    'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104,
    'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
    'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

def read_arc(context: str):
    struct_id, energy = 0, 0
    struct = None
    for line in context.split("\n"):
        if "Energy" in line:
            energy = float(line.split()[3])
        elif "PBC " in line and "ON" not in line:
            struct = dict()
            struct["energy"] = energy
            struct["abc"] = np.array([float(c) for c in line.split()[1:7]])
            struct["cell"] = abc2cell(struct["abc"])
            struct["elem"], struct["z"], struct['pos'], struct['charge'] = [], [], [], []
        elif struct and "abc" in struct and not line.startswith("end"):
            line = line.split()
            match = re.match(r'[A-Z][a-z]?', line[0])
            if match:
                elem = match.group(0)
                struct["elem"].append(elem)
                struct["z"].append(ELE2INT[elem])
                struct["pos"].append([float(xyz)  for xyz in line[1:4]])
                struct["charge"].append(float(line[8]))
            else:
                continue  # Skip this line if no valid element symbol is found
        elif struct and line.startswith("end"):
            struct["id"] = struct_id
            struct_id += 1
            return struct
        else:
            continue
class MD_system:
    def __init__(self, elements:list[int]):
        self.elements = elements
        self.calculator = D3Calculator(elements, max_length=len(elements))
    
    def set_atoms(self, coords:np.ndarray, elements:list[int]):
        self.calculator.set_atoms(coords, elements)

    def set_cell(self, cell:np.ndarray):
        # check cell shape
        if cell.shape != (3,3):
            raise ValueError("Cell must be a 3x3 matrix")
        self.calculator.set_cell(cell)
    
    def compute(self):
        energy, forces, stress = self.calculator.compute()
        self.calculator.clear()
        
        # Check if energy, forces, or stress contain NaN or inf values
        if not np.isfinite(energy) or np.any(~np.isfinite(forces)) or np.any(~np.isfinite(stress)):
            raise ValueError("Computation resulted in non-finite values (NaN or inf)")
        
        return energy, forces, stress
    
    def forward(self, coords, step_size=0.01):
        """
        Randomly modify atom positions slightly.
        
        Args:
            coords (np.ndarray): Current atom coordinates with shape (n_atoms, 3)
            step_size (float): Maximum magnitude of random displacement
            
        Returns:
            np.ndarray: New coordinates after random displacement
        """
        # Generate random displacements
        random_displacement = np.random.uniform(-step_size, step_size, size=coords.shape)
        
        # Apply the displacement to get new coordinates
        new_coords = coords + random_displacement
        
        # Update the calculator with new coordinates
        self.calculator.set_atoms(new_coords, self.elements)
        
        return new_coords

if __name__ == "__main__":
    # Define elements and initial coordinates
    # Check for correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python test_d3_cffi.py <filename> <number of steps>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        num_steps = int(sys.argv[2])
    except ValueError:
        print("Error: Number of steps must be an integer")
        sys.exit(1)

    # Read the file
    content = read_file(filename)
    struct = read_arc(content)

    if struct is None:
        print(f"Error: Could not read structure from {filename}")
        sys.exit(1)

    # Convert elements to integers and prepare coordinates
    elements = struct["z"]
    coords = np.array(struct["pos"])
    cell = struct["cell"]

    # Initialize MD system
    md_system = MD_system(elements)
    md_system.set_atoms(coords, elements)
    md_system.set_cell(cell)

    # Run fake MD simulation
    print(f"Running {num_steps} steps of fake MD simulation for {len(elements)} atoms")
    print(f"Initial structure: {filename}")

    energies = []
    start_time = time.time()

    try:
        for step in range(num_steps):
            # Forward propagate the system (update coordinates)
            coords = md_system.forward(coords)
            
            # Compute energy, forces, and stress
            energy, forces, stress = md_system.compute()
            energies.append(energy)
            
            # Print progress
            if (step + 1) % 10 == 0 or step == 0:
                print(f"Step {step+1}/{num_steps}, Energy: {energy:.6f}")
        
        elapsed = time.time() - start_time
        avg_energy = np.mean(energies)
        std_energy = np.std(energies)
        
        # Print summary
        print("\nSimulation complete!")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Average energy: {avg_energy:.6f}")
        print(f"Energy std dev: {std_energy:.6f}")
        print(f"Steps per second: {num_steps/elapsed:.2f}")

    except Exception as e:
        print(f"Error during simulation: {e}")