import cffi
import numpy as np
import os

ffi = cffi.FFI()

# Define the C API
ffi.cdef("""
typedef void D3Handle_t;

void init_params();

void compute_dispersion_energy(
    float atoms[][3],
    uint16_t *elements,
    uint64_t num_atoms,
    float cell[3][3],
    float cutoff_radius,
    float CN_cutoff_radius,
    uint64_t max_neighbors,
    float *energy,
    float *force,
    float *stress    
);

D3Handle_t *init_d3_handle( 
    uint16_t *elements,
    uint64_t max_length, 
    float cutoff_radius,
    float coordination_number_cutoff,
    uint64_t max_neigbors
);

void set_atoms(D3Handle_t *handle, float *coords, uint16_t *elements, uint64_t length);
void set_cell(D3Handle_t *handle, float cell[3][3]);
void free_d3_handle(D3Handle_t *handle);
void clear_d3_handle(D3Handle_t *handle);
void compute_dispersion_energy_from_handle(
    D3Handle_t *handle,
    float *energy,
    float *force,
    float *stress
);
""")

# Load the compiled library
lib_path = os.path.join(os.path.dirname(__file__), "..", "build", "libd3.so")
lib = ffi.dlopen(lib_path)

class D3Calculator:
    def __init__(self, elements: list[int], max_length: int, cutoff_radius: float = 46.475800, cn_cutoff_radius: float = 46.475800, max_neighbors: int = 10000):
        """Initialize a D3 calculator object"""
        # Convert elements to the right type
        elements_array = np.array(elements, dtype=np.uint16)
        elements_ptr = ffi.cast("uint16_t*", elements_array.ctypes.data)

        # Initialize parameters
        lib.init_params()  # type: ignore
        
        # Initialize the handle
        self.handle = lib.init_d3_handle( # type: ignore
            elements_ptr, 
            max_length,
            cutoff_radius,
            cn_cutoff_radius,
            max_neighbors
        )

        self.num_atoms = max_length

    def set_atoms(self, coords: np.ndarray, elements: list[int]):
        """Set atom coordinates and elements"""
        # Ensure arrays are contiguous and correct type
        coords_array = np.ascontiguousarray(coords, dtype=np.float32)
        elements_array = np.ascontiguousarray(elements, dtype=np.uint16)
        
        # Get pointers
        coords_ptr = ffi.cast("float*", coords_array.ctypes.data)
        elements_ptr = ffi.cast("uint16_t*", elements_array.ctypes.data)
        
        # Call the C function
        lib.set_atoms(self.handle, coords_ptr, elements_ptr, len(elements_array)) # type: ignore
        self.num_atoms = len(elements_array)

    def set_cell(self, cell: np.ndarray):
        """Set periodic cell parameters"""
        # Ensure array is contiguous and correct type
        cell_array = np.ascontiguousarray(cell, dtype=np.float32)
        cell_ptr = ffi.cast("float (*)[3]", cell_array.ctypes.data)
        
        # Call the C function
        lib.set_cell(self.handle, cell_ptr) # type: ignore
    
    def compute(self):
        """Compute dispersion energy, forces, and stress"""
        # Create output arrays
        num_atoms = self.num_atoms
        if num_atoms <= 0:
            raise ValueError("Number of atoms must be greater than zero.")
        energy = ffi.new("float*")
        forces = np.zeros((num_atoms, 3), dtype=np.float32)
        stress = np.zeros(9, dtype=np.float32)
        
        # Get pointers
        forces_ptr = ffi.cast("float*", forces.ctypes.data)
        stress_ptr = ffi.cast("float*", stress.ctypes.data)
        
        # Call the C function
        lib.compute_dispersion_energy_from_handle( # type: ignore
            self.handle,
            energy,
            forces_ptr,
            stress_ptr
        )
        
        return float(energy[0]), forces, stress.reshape(3, 3)
    
    def clear(self):
        """Clear the D3 handle data"""
        lib.clear_d3_handle(self.handle) # type: ignore
    
    def __del__(self):
        """Clean up when the object is deleted"""
        if hasattr(self, 'handle'):
            lib.free_d3_handle(self.handle) # type: ignore