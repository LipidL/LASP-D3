"""
Test script for the D3Calculator from d3_cffi module with multithreading.

This script demonstrates how to use the D3 dispersion correction 
calculator with multiple threads for parallel computation.
"""
import numpy as np
from d3_cffi import D3Calculator
import concurrent.futures
import threading
import time

def calculate_dispersion(system_id, elements, coords, cell):
    """
    Calculate dispersion for a molecular system in a separate thread.
    Each thread must use its own D3Calculator instance.
    """
    thread_name = threading.current_thread().name
    print(f"Thread {thread_name} processing system {system_id}")
    
    # Create a dedicated calculator for this thread
    d3calc = D3Calculator(
        elements=elements,
        max_length=5,
        cutoff_radius=46.475800,
        cn_cutoff_radius=46.475800,
        damping_type=0,
        functional_type=0
    )
    
    # Set the atoms and cell
    d3calc.set_atoms(coords, elements)
    d3calc.set_cell(cell)
    
    # Compute dispersion energy, forces and stress
    try:
        energy, forces, stress = d3calc.compute()
    except Exception as e:
        raise RuntimeError(f"Error in thread {thread_name} for system {system_id}: {e}")
    
    # Clean up
    d3calc.clear()
    
    return {
        "system_id": system_id,
        "energy": energy,
        "forces": forces
    }

def test_multithreading():
    """Test D3Calculator with multiple threads"""
    print("\nTesting D3 dispersion calculator with multithreading...")
    
    # Create several test systems
    systems = [
        # Water molecule
        {
            "id": 1,
            "elements": [8, 1, 1],
            "coords": np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.8, 0.6],
                [0.0, -0.8, 0.6]
            ], dtype=np.float32),
            "cell": np.eye(3, dtype=np.float32) * 10.0
        },
        # Methane
        {
            "id": 2,
            "elements": [6, 1, 1, 1, 1],
            "coords": np.array([
                [0.0, 0.0, 0.0],
                [0.6, 0.6, 0.6],
                [-0.6, -0.6, 0.6],
                [0.6, -0.6, -0.6],
                [-0.6, 0.6, -0.6]
            ], dtype=np.float32),
            "cell": np.eye(3, dtype=np.float32) * 10.0
        }
    ]
    
    # Run calculations in parallel
    start_time = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all calculation tasks
        futures = [
            executor.submit(
                calculate_dispersion,
                system["id"], 
                system["elements"],
                system["coords"],
                system["cell"]
            ) for system in systems
        ]
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    # Print results
    print(f"\nMultithreaded calculations completed in {time.time() - start_time:.4f} seconds")
    for result in sorted(results, key=lambda x: x["system_id"]):
        print(f"System {result['system_id']} energy: {result['energy']:.6f} eV")

def main():
    # Original single-threaded example
    # ... (existing code) ...
    
    # Add the multithreading test
    test_multithreading()

if __name__ == "__main__":
    main()
