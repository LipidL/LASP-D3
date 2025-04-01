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
// parameters specified for PBE0 functional
// obtained from Grimme et al. 2010, Table SI1
#define S6 1.0f
#define S8 0.926f
#define SR_6 1.326f
#define SR_8 1.0f

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
__device__ void compute_dispersion_energy_kernel(device_data_t *data) {
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
    for (size_t pair_index = thread_id; pair_index < total_interactions; pair_index += num_threads) {
        // calculate the C6AB value for each atom pair
        // determine atom indices, same as the previous loop
        float discriminant = (2.0f * num_atoms - 1.0f) * (2.0f * num_atoms - 1.0f) - 8.0f * pair_index;
        size_t atom_1_index = (size_t)floorf((2.0f * num_atoms - 1.0f - sqrtf(discriminant)) / 2.0f);
        size_t row_start = atom_1_index * (2 * num_atoms - atom_1_index - 1) / 2;
        size_t atom_2_index = pair_index - row_start + atom_1_index + 1;
        // extract the coordination number to local memory
        real_t coordination_number_1 = data->coordination_numbers[atom_1_index];
        real_t coordination_number_2 = data->coordination_numbers[atom_2_index];
        size_t atom_1_type = data->atom_types[atom_1_index];
        size_t atom_2_type = data->atom_types[atom_2_index];
        atom_t atom_1 = data->atoms[atom_1_index];
        atom_t atom_2 = data->atoms[atom_2_index];
        real_t distance = sqrtf(powf(atom_1.x - atom_2.x, 2) + powf(atom_1.y - atom_2.y, 2) + powf(atom_1.z - atom_2.z, 2));
        // calculate the coordination number based dispersion coefficient
        // formula: C_6^{AB} = Z/W
        // where: Z = \sum_{i,j}C_{6,ref}^{A,B}L_{i,j}
        //      W = \sum_{i,j}L_{i,j}
        //      L_{i,j} = \exp(-k_3((CN^A-CN^A_{ref,i})^2 + (CN^B-CN^B_{ref,j})^2))
        real_t Z = 0.0f;
        real_t W = 0.0f;
        for (size_t i = 0; i < NUM_C6AB_ENTRIES; ++i) {
            for (size_t j = 0; j < NUM_C6AB_ENTRIES; ++j) {
                // these entries could be -1.0f if the entries are not valid, but at least one entry should be valid
                real_t c6_ref = data->constants->c6ab_ref->get(atom_1_type, atom_2_type, i, j, 0);
                real_t coordination_number_ref_1 = data->constants->c6ab_ref->get(atom_1_type, atom_2_type, i, j, 1);
                real_t coordination_number_ref_2 = data->constants->c6ab_ref->get(atom_1_type, atom_2_type, i, j, 2);
                // because of the presence of invalid entries, the L_ij_ref cannot be calculated directly
                real_t L_ij_ref = expf(-K3 * (powf(coordination_number_1 - coordination_number_ref_1, 2) + powf(coordination_number_2 - coordination_number_ref_2, 2)));
                // since we need the value $\frac{\sum_{i,j}C_{6,ref}^{A,B}L_{i,j}}{\sum_{i,j}L_{i,j}}$
                // we can set invalid L_ij to 0.0f and perform the summation in the same loop
                // invalid entry: have -1.0f in c6_ref, coordination_number_ref_1 and coordination_number_ref_2
                // se check coordination_number_ref_1 here.
                real_t L_ij = ((coordination_number_ref_1 - (-1.0f) <= 1e-3f) ? 0.0f : L_ij_ref); // conditional move, no branching, fast!
                Z += c6_ref * L_ij;
                W += L_ij;
            }
        }
        real_t c6_ab = (W > 0.0f) ? Z / W : 0.0f; // avoid division by zero
        // calculate c8_ab, which is obtained by $C_8^{AB} = 3C_6^{AB}\sqrt{Q^AQ^B}$
        // $\sqrt{Q}$ is precomputed and stored in data.constants.r2r4
        real_t r2r4_1 = data->constants->r2r4[atom_1_type];
        real_t r2r4_2 = data->constants->r2r4[atom_2_type];
        real_t c8_ab = 3.0f * c6_ab * r2r4_1 * r2r4_2; // the value in r2r4 is already squared
        // acqauire the cutoff radius between the two atoms
        real_t cutoff_radius = data->constants->r0ab[atom_1_type][atom_2_type];
        // calculate the dampling function
        // see Grimme et al. 2010, eq 4
        real_t f_dn_6 = 1/(1+6.0f*powf(distance/(SR_6*cutoff_radius), ALPHA_N(6.0f)));
        real_t f_dn_8 = 1/(1+6.0f*powf(distance/(SR_8*cutoff_radius), ALPHA_N(8.0f)));
        // calculate the dispersion energy
        // see Grimme et al. 2010, eq 3
        real_t dispersion_energy_6 = S6*(c6_ab/powf(distance, 6.0f))*f_dn_6;
        real_t dispersion_energy_8 = S8*(c8_ab/powf(distance, 8.0f))*f_dn_8;
        // the total dispersion energy is the sum of the two contributions
        real_t dispersion_energy = dispersion_energy_6 + dispersion_energy_8;
        // store the result in the results array
        atomicAdd(&data->results[atom_1_index].energy, dispersion_energy); // increment the energy for atom 1
        atomicAdd(&data->results[atom_2_index].energy, dispersion_energy); // increment the energy for atom 2
    }
}