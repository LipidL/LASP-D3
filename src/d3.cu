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

typedef struct device_data {
    size_t num_atoms;
    size_t num_elements;
    atom_t *atoms; // array of atom data
    d3_constant_t *constants; // constants for the simulation
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
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // identify the atom pair this thread is responsible for
    if (thread_id >= total_interactions) {
        // we hope this will never happen, but if it does, we need to return
        return; // thread is out of bounds
    }
    float discriminant = (2.0f * num_atoms - 1.0f) * (2.0f * num_atoms - 1.0f) - 8.0f * thread_id;
    size_t atom_1_index = (size_t)floorf((2.0f * num_atoms - 1.0f - sqrtf(discriminant)) / 2.0f);
    size_t row_start = atom_1_index * (2 * num_atoms - atom_1_index - 1) / 2;
    size_t atom_2_index = thread_id - row_start + atom_1_index + 1;
}