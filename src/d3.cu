#include "constants.h"
#include <stdlib.h>

// macros for debugging
#ifdef DEBUG
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", \
            __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
#else
#define CHECK_CUDA(call) call
#endif

#define debug(...) fprintf(stderr, __VA_ARGS__)

// global parameters
// cuda kernel launch parameters
#define BLOCK_SIZE 256
#define MAX_BLOCKS 65535
#define MAX_THREADS 1024
#define MAX_THREADS_PER_BLOCK 1024

// constants used in the simulation
// these constants are from Grimme, S., Antony, J., Ehrlich, S. & Krieg, H. The Journal of Chemical Physics 132, 154104 (2010).
#define K1 16.0f
#define K2 1.33333f
#define K3 4.0f
#define ALPHA_N(N) (N + 8.0f)

typedef struct device_data {
    size_t num_atoms;
    size_t num_elements;
    size_t *atom_types; // array of atom types, length: num_atoms. the entries is not the atomic number, but the index of the corresponding entry in constants.
    atom_t *atoms; // array of atom data
    d3_constant_t *constants; // constants for the simulation
    real_t *coordination_numbers; // array of coordination numbers, length: num_atoms. This field is initialized to zero to store the results produced during the simulation.
    result_t *results; // results of the simulation
} device_data_t;

/**
 * @note this kernel should be launched with a 1D grid of blocks, each block containing a 1D array of threads.
 * @note the proper grid and block sizes should be calculated based on the number of atoms in the system.
 */
__device__ void compute_coordination_number(device_data_t *data) {
    // compute the energy of the system using the D3 potential
    // total atomic interactions
    size_t num_atoms = data->num_atoms;
    uint64_t total_interactions = (num_atoms * (num_atoms - 1)) / 2;
    // identify thread number
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_threads = blockDim.x * gridDim.x;
    // identify the atom pair this thread is responsible for
    if (thread_id >= total_interactions) {
        // we hope this will never happen, but if it does, we need to return
        return; // thread is out of bounds
    }

    // calculate all cordination numbers
    // all atom pairs compose a triangular matrix with size num_atoms * num_atoms.
    // each thread is responsible for several pair of atoms, i.e. one entry in the triangular matrix. 
    // thread with thread_id is responsible for atom pairs [thread_id, thread_id+num_threads, thread_id+2*num_threads, ...]
    for (size_t pair_index = thread_id; pair_index < total_interactions; pair_index += num_threads) {
        // "row" index i is computed by solving the inequality:
        // R(i) <= pair_index < R(i+1)
        // where R(i) is the total number of pairs of atoms in the triangular matrix before row i.
        // a closed form of i can be obtained by solving equation:
        // i^2 + (2N-1)i - 2 * pair_index = 0
        float discriminant = (2.0f * num_atoms - 1.0f) * (2.0f * num_atoms - 1.0f) - 8.0f * pair_index;
        size_t atom_1_index = (size_t)floorf((2.0f * num_atoms - 1.0f - sqrtf(discriminant)) / 2.0f);
        size_t row_start = atom_1_index * (2 * num_atoms - atom_1_index - 1) / 2;
        size_t atom_2_index = pair_index - row_start + atom_1_index + 1;
        // find the proper index in the atom_types array
        // this index is further used to access entries in data.constants
        size_t atom_1_type = data->atom_types[atom_1_index];
        size_t atom_2_type = data->atom_types[atom_2_index];
        atom_t atom_1 = data->atoms[atom_1_index];
        atom_t atom_2 = data->atoms[atom_2_index];
        // compute the distance between the two atoms
        real_t distance = sqrtf(powf(atom_1.x - atom_2.x, 2) + powf(atom_1.y - atom_2.y, 2) + powf(atom_1.z - atom_2.z, 2));
        real_t covalent_radii_1 = data->constants->rcov[atom_1_type];
        real_t covalent_radii_2 = data->constants->rcov[atom_2_type];
        // eq 15 in Grimme et al. 2010
        // $CN^A = \sum_{B \neq A}^{N} \sqrt{1}{1+exp(-k_1(k_2(R_{A,cov}+R_{B,cov})/r_{AB}-1))}$
        real_t coordination_number = 1/(1+expf(-K1*(K2*(covalent_radii_1 + covalent_radii_2)/distance - 1))); 
        // increment the data.coordination_number array for both atoms
        atomicAdd(&data->coordination_numbers[atom_1_index], coordination_number); // increment the coordination number for atom 1
        atomicAdd(&data->coordination_numbers[atom_2_index], coordination_number); // increment the coordination number for atom 2
    }
    __syncthreads(); // synchronize threads in the block
    // now the coordination numbers are stored in the data.coordination_numbers array
}