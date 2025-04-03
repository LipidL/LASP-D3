#include "constants.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define DEBUG

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
#define debug(...) fprintf(stderr, __VA_ARGS__)
#else
#define CHECK_CUDA(call) call
#define debug(...)
#endif

/* global parameters */
/* cuda kernel launch parameters */ 
#define BLOCK_SIZE 256
#define MAX_BLOCKS 65535
#define MAX_ELEMENTS 118

/* 
constants used in the simulation
these constants are from Grimme, S., Antony, J., Ehrlich, S. & Krieg, H. The Journal of Chemical Physics 132, 154104 (2010).
*/
#define K1 16.0f
#define K2 1.33333f
#define K3 4.0f
#define ALPHA_N(N) (N + 8.0f)

/*
parameters specified for PBE0 functional
obtained from Grimme et al. 2010, Table SI1
*/
#define S6 1.0f
#define S8 0.926f
#define SR_6 1.326f
#define SR_8 1.0f

typedef struct device_data {
    size_t num_atoms;
    size_t num_elements;
    /*
    to construct it, sort the elements by their atomic number, and then assign the index of the element in the sorted array to the atom_types array.
    */
    size_t *atom_types; /* array of atom types, length: num_atoms. the entries is not the atomic number, but the index of the corresponding entry in constants. */
    atom_t *atoms; // array of atom data
    d3_constant_t *constants; // constants for the simulation
    real_t *coordination_numbers; // array of coordination numbers, length: num_atoms. This field is initialized to zero to store the results produced during the simulation.
    result_t *results; // results of the simulation
} device_data_t;

/**
 * @note this kernel should be launched with a 1D grid of blocks, each block containing a 1D array of threads.
 * @note the proper grid and block sizes should be calculated based on the number of atoms in the system.
 */
__global__ void compute_dispersion_energy_kernel(device_data_t *data) {
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
        printf("thread_id: %llu, total_interactions: %llu\n", thread_id, total_interactions);
        return; // thread is out of bounds
    }

    // calculate all cordination numbers
    // all atom pairs compose a triangular matrix with size num_atoms * num_atoms.
    // each thread is responsible for several pair of atoms, i.e. one entry in the triangular matrix. 
    // thread with thread_id is responsible for atom pairs [thread_id, thread_id+num_threads, thread_id+2*num_threads, ...]
    for (size_t pair_index = thread_id; pair_index < total_interactions; pair_index += num_threads) {
        // "row" index i is computed by solving the inequality:
        // R(i) <= pair_index < R(i+1)
        // where R(i) is the total number of pairs of atoms in the triangular matrix before row i, i.e. i*(i-1)/2.
        // expanding the inequality gives us:
        // i^2 - i - 2*pair_index <= 0
        // consider the equality:
        // i^2 - i - 2*pair_index = 0
        // the positive solution is:
        // i' = (1 + sqrt(1 + 8*pair_index)) / 2
        // i' is a float that lies between i and i+1, so i is acquired by flooring i'.
        // the column index can then be acquired by solving the equation
        // pair_index = i*(i-1)/2 + j, where j is the column index.
        // this gives us:
        // j = pair_index - i*(i-1)/2
        size_t atom_1_index_candidate = (size_t)floorf((1.0f + sqrtf(1.0f + 8.0f * pair_index)) / 2.0f); // this candidate could be i or i+1 due to floating point error
        size_t atom_1_index = (atom_1_index_candidate * (atom_1_index_candidate-1)/ 2 <= pair_index) ? atom_1_index_candidate : atom_1_index_candidate - 1; // this is the corrent index, validated by data/test.c for 5-5000 atoms.
        assert(atom_1_index < num_atoms); // make sure the index is in bounds
        assert(atom_1_index*(atom_1_index-1) <= 2*pair_index); // make sure the index is in bounds
        size_t atom_2_index = pair_index - atom_1_index * (atom_1_index - 1) / 2; // column index

        assert(atom_1_index != atom_2_index); // make sure the indices are not equal
        assert(atom_1_index < num_atoms && atom_2_index < num_atoms); // make sure the indices are in bounds
        // the atom_1_index is the index of the first atom in the pair, and atom_2_index is the index of the second atom in the pair

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
        size_t atom_1_index_candidate = (size_t)floorf((1.0f + sqrtf(1.0f + 8.0f * pair_index)) / 2.0f); // this candidate could be i or i+1 due to floating point error
        size_t atom_1_index = (atom_1_index_candidate * (atom_1_index_candidate-1)/ 2 <= pair_index) ? atom_1_index_candidate : atom_1_index_candidate - 1; // this is the corrent index, validated by data/test.c for 5-5000 atoms.
        assert(atom_1_index < num_atoms); // make sure the index is in bounds
        assert(atom_1_index*(atom_1_index-1) <= 2*pair_index); // make sure the index is in bounds
        size_t atom_2_index = pair_index - atom_1_index * (atom_1_index - 1) / 2; // column index

        assert(atom_1_index != atom_2_index); // make sure the indices are not equal
        assert(atom_1_index < num_atoms && atom_2_index < num_atoms); // make sure the indices are in bounds
        // the atom_1_index is the index of the first atom in the pair, and atom_2_index is the index of the second atom in the pair
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
        for (size_t i = 0; i < NUM_REF_C6; ++i) {
            for (size_t j = 0; j < NUM_REF_C6; ++j) {
                // these entries could be -1.0f if the entries are not valid, but at least one entry should be valid
                size_t stride_1 = data->num_elements * NUM_REF_C6 * NUM_REF_C6  * NUM_C6AB_ENTRIES;
                size_t stride_2 = NUM_REF_C6 * NUM_REF_C6  * NUM_C6AB_ENTRIES;
                size_t stride_3 = NUM_REF_C6  * NUM_C6AB_ENTRIES;
                size_t stride_4 = NUM_C6AB_ENTRIES;
                size_t index = atom_1_type * stride_1 + atom_2_type * stride_2 + i * stride_3 + j * stride_4; // gpu might give wrong value, I don't know why yet...
                real_t c6_ref = data->constants->c6ab_ref->data[index + 0];
                real_t coordination_number_ref_1 = data->constants->c6ab_ref->data[index + 1];
                real_t coordination_number_ref_2 = data->constants->c6ab_ref->data[index + 2];
                // because of the presence of invalid entries, the L_ij cannot be calculated directly
                real_t L_ij_ref = expf(-K3 * (powf(coordination_number_1 - coordination_number_ref_1, 2) + powf(coordination_number_2 - coordination_number_ref_2, 2))) * 1e5f;// scale it to avoid floating point error
                // since we need the value $\frac{\sum_{i,j}C_{6,ref}^{A,B}L_{i,j}}{\sum_{i,j}L_{i,j}}$
                // we can set invalid L_ij to 0.0f and perform the summation in the same loop
                // invalid entry: have -1.0f in c6_ref, coordination_number_ref_1 and coordination_number_ref_2
                // we check coordination_number_ref_1 here.
                real_t L_ij = ((coordination_number_ref_1 - (-1.0f) <= 1e-5f) ? 0.0f : L_ij_ref); // conditional move, no branching, fast!
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
        real_t f_dn_6 = 1/(1+6.0f*powf(distance/(SR_6*cutoff_radius), -ALPHA_N(6.0f)));
        real_t f_dn_8 = 1/(1+6.0f*powf(distance/(SR_8*cutoff_radius), -ALPHA_N(8.0f)));
        // calculate the dispersion energy
        // see Grimme et al. 2010, eq 3
        real_t dispersion_energy_6 = S6*(c6_ab/powf(distance, 6.0f))*f_dn_6;
        real_t dispersion_energy_8 = S8*(c8_ab/powf(distance, 8.0f))*f_dn_8;
        // the total dispersion energy is the sum of the two contributions
        real_t dispersion_energy = dispersion_energy_6 + dispersion_energy_8;
        // store the result in the results array
        // if dispersion_energy is NaN, print some debug information
        if (isnan(dispersion_energy) && atom_1_index == 1) {
            printf("Error: dispersion energy is NaN for atom %llu and %llu\n", atom_1_index, atom_2_index);
            printf("atom 1: %llu %f %f %f\n", atom_1_type, atom_1.x, atom_1.y, atom_1.z);
            printf("atom 2: %llu %f %f %f\n", atom_2_type, atom_2.x, atom_2.y, atom_2.z);
            printf("distance: %f\n", distance);
            printf("c6_ab: %f\n", c6_ab);
            printf("c8_ab: %f\n", c8_ab);
            printf("cutoff_radius: %f\n", cutoff_radius);
            printf("coordination_number_1: %f\n", coordination_number_1);
            printf("coordination_number_2: %f\n", coordination_number_2);
            assert(0);
        }
        atomicAdd(&data->results[atom_1_index].energy, dispersion_energy); // increment the energy for atom 1
        atomicAdd(&data->results[atom_2_index].energy, dispersion_energy); // increment the energy for atom 2
    }
}

/**
 * @brief the function is used to find the index of an element in an array. if the element is not found, it inserts the element in the array and returns the index of the element.
 * @param elements the array of elements.
 * @param length the length of the array.
 * @param element the element to find.
 * @return the index of the element in the array
 * @note this function hypothesizes that the array is sorted in ascending order.
 */
__host__ int32_t find(size_t *elements, size_t length, size_t element){
    // use a binary search to find the element in the array
    size_t left = 0;
    size_t right = length - 1;
    while (left <= right) {
        size_t mid = (left + right) / 2;
        if (elements[mid] == element) {
            return mid; // found the element
        } else if (elements[mid] < element) {
            left = mid + 1; // search in the right half
        } else {
            right = mid - 1; // search in the left half
        }
    }
    // not found, return -1
    return -1;
}

/**
 * @brief the function is used to compute the dispersion energy of the system using the D3 potential.
 * 
 * @param atoms the array of atoms in the system. The array is of size num_atoms * 4, where the first entry is atomic number and the last 3 entries are the coordinates of the atom.
 * @param length the number of atoms in the system.
 * @note the function is not thread safe, and should be called from a single thread.
 */
__host__ void compute_dispersion_energy(real_t atoms[][4], size_t length) {
    // allocate memory for device_data_t
    debug("starting compute_dispersion_energy...\n");
    device_data_t h_data;
    h_data.num_atoms = length;
    // allocate memory for atoms
    atom_t *h_atoms = (atom_t *)malloc(length * sizeof(atom_t));
    if (h_atoms == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for atoms\n");
        exit(EXIT_FAILURE);
    }
    size_t elements_presence[MAX_ELEMENTS + 1] = {0}; // bucket sort :)
    size_t maximum_atomic_number = 0;
    for (size_t i = 0; i < length; ++i) {
        h_atoms[i].element = (size_t)atoms[i][0];
        h_atoms[i].x = atoms[i][1];
        h_atoms[i].y = atoms[i][2];
        h_atoms[i].z = atoms[i][3];
        // the element cannot exceed MAX_ELEMENTS
        assert(h_atoms[i].element <= MAX_ELEMENTS); // if you toggle this assertion, check if the length exceeds actual number of atoms
        elements_presence[h_atoms[i].element] = 1; // mark the element as present in the array
        maximum_atomic_number = (h_atoms[i].element > maximum_atomic_number) ? h_atoms[i].element : maximum_atomic_number;
    }
    debug("maximum atomic number: %zu\n", maximum_atomic_number);
    // h_atoms is ready, now construct d_atoms
    atom_t *d_atoms;
    CHECK_CUDA(cudaMalloc((void **)&d_atoms, length * sizeof(atom_t)));
    CHECK_CUDA(cudaMemcpy(d_atoms, h_atoms, length * sizeof(atom_t), cudaMemcpyHostToDevice));
    h_data.atoms = d_atoms;
    size_t *sorted_elements = (size_t *)malloc((maximum_atomic_number + 1) * sizeof(size_t));
    if (sorted_elements == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for sorted elements\n");
        free(h_atoms);
        exit(EXIT_FAILURE);
    }
    // sort the elements in ascending order
    size_t num_elements = 0; // the number of elements in the sorted array
    for (size_t i = 0; i <= maximum_atomic_number; ++i) {
        if (elements_presence[i] == 1) {
            sorted_elements[num_elements] = i;
            debug("sorted_elements[%zu] = %zu\n", num_elements, i);
            num_elements++;
        }
    }
    h_data.num_elements = num_elements; // set the number of elements in the device data
    // assign the atom types
    size_t *atom_types = (size_t *)malloc(length * sizeof(size_t));
    if (atom_types == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for atom types\n");
        free(h_atoms);
        free(sorted_elements);
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < length; ++i) {
        int32_t find_result = find(sorted_elements, num_elements, h_atoms[i].element);
        if (find_result == -1) {
            fprintf(stderr, "Error: failed to find the element in the array\n");
            free(h_atoms);
            free(sorted_elements);
            free(atom_types);
            exit(EXIT_FAILURE);
        } else {
            atom_types[i] = find_result; // assign the index of the element in the sorted array to the atom_types array
        }
    }
    // now the atom_types array is ready, copy it to device memory
    size_t *d_atom_types;
    CHECK_CUDA(cudaMalloc((void **)&d_atom_types, length * sizeof(size_t)));
    CHECK_CUDA(cudaMemcpy(d_atom_types, atom_types, length * sizeof(size_t), cudaMemcpyHostToDevice));
    h_data.atom_types = d_atom_types;
    // allocate memory for results and coordination_numbers
    result_t *d_results;
    CHECK_CUDA(cudaMalloc((void **)&d_results, length * sizeof(result_t)));
    CHECK_CUDA(cudaMemset(d_results, 0, length * sizeof(result_t))); // initialize the results array to zero
    h_data.results = d_results;
    real_t *d_coordination_numbers;
    CHECK_CUDA(cudaMalloc((void **)&d_coordination_numbers, length * sizeof(real_t)));
    CHECK_CUDA(cudaMemset(d_coordination_numbers, 0, length * sizeof(real_t))); // initialize the coordination numbers array to zero
    h_data.coordination_numbers = d_coordination_numbers;
    // initialize constants
    printf("sorted_elements is at %p\n", sorted_elements);
    h_data.constants = d3_constant_init(num_elements, sorted_elements); // initialize the constants
    // initialize the device data
    device_data_t *d_data;
    CHECK_CUDA(cudaMalloc((void **)&d_data, sizeof(device_data_t)));
    CHECK_CUDA(cudaMemcpy(d_data, &h_data, sizeof(device_data_t), cudaMemcpyHostToDevice));

    // launch the kernel
    int num_multiprocessors;
    cudaDeviceGetAttribute(&num_multiprocessors, cudaDevAttrMultiProcessorCount, 0);
    size_t num_pairs = length * (length - 1) / 2; // number of pairs of atoms
    size_t num_blocks = num_multiprocessors;
    compute_dispersion_energy_kernel<<<1, BLOCK_SIZE>>>(d_data);
    CHECK_CUDA(cudaDeviceSynchronize()); // synchronize the device to ensure all threads are finished
    result_t *h_results = (result_t *)malloc(length * sizeof(result_t));
    if (h_results == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for results\n");
        free(h_atoms);
        free(sorted_elements);
        free(atom_types);
        CHECK_CUDA(cudaFree(d_atoms));
        CHECK_CUDA(cudaFree(d_atom_types));
        CHECK_CUDA(cudaFree(d_results));
        CHECK_CUDA(cudaFree(d_coordination_numbers));
        CHECK_CUDA(cudaFree(d_data));
        exit(EXIT_FAILURE);
    }
    // copy the results back to host memory
    CHECK_CUDA(cudaMemcpy(h_results, d_results, length * sizeof(result_t), cudaMemcpyDeviceToHost));
    real_t total_energy = 0.0f;
    for (size_t i = 0; i < length; ++i) {
        // print the results
        // printf("Atom %zu: energy = %f\n", i, h_results[i].energy);
        // accumulate the total energy
        total_energy += h_results[i].energy;
    }
    printf("Total energy = %f\n", total_energy);
    // free the device memory
    CHECK_CUDA(cudaFree(d_atoms));
    CHECK_CUDA(cudaFree(d_atom_types));
    CHECK_CUDA(cudaFree(d_results));
    CHECK_CUDA(cudaFree(d_coordination_numbers));
    CHECK_CUDA(cudaFree(d_data));
    // free the host memory
    free(h_atoms);
}

#define TOTAL_ATOMS 100
int main()
{
    // example usage of the compute_dispersion_energy function
    real_t atoms[TOTAL_ATOMS][4];
    real_t angstron_to_bohr = 1/0.529f; // angstron to bohr conversion factor
    // fill the atoms array with Po element
    for (size_t i = 0; i < TOTAL_ATOMS/100; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            for(size_t k = 0; k < 10; ++k) {
                atoms[i*100+j*10+k][0] = 84; // atomic number of Po
                atoms[i*100+j*10+k][1] = (real_t)i*3.352*angstron_to_bohr; // x coordinate
                atoms[i*100+j*10+k][2] = (real_t)j*3.352*angstron_to_bohr; // y coordinate
                atoms[i*100+j*10+k][3] = (real_t)k*3.352*angstron_to_bohr; // z coordinate
            }
        }
    }
    debug("Computing dispersion energy for %zu atoms...\n", sizeof(atoms)/sizeof(atoms[0]));
    // initialize parameters
    init_params();
    debug("c6ab_ref is at %p\n", c6ab_ref);
    debug("c6ab between C andC: %f\n", c6ab_ref[6][6][0][0][0]);
    debug("r0ab between C and C: %f\n", r0ab[6][6]);
    debug("rcov of C: %f\n", rcov[6]);
    debug("r2r4 of C: %f\n", r2r4[6]);
    debug("Computing dispersion energy...\n");

    compute_dispersion_energy(atoms, TOTAL_ATOMS);
    return 0;
}