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
#define debug(...) printf(__VA_ARGS__)
#else
#define CHECK_CUDA(call) call
#define debug(...)
#endif

/* global parameters */
/* cuda kernel launch parameters */ 
#define MAX_BLOCK_SIZE 512 // number of threads per block
#define GRID_SIZE 65536 // number of blocks per grid
#define MAX_ELEMENTS 118
#define MAX_NEIGHBORS 1000 // the maximum number of neighbors, dependent on the cutoff choice

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
#define S8 0.722f
#define SR_6 1.217f
#define SR_8 1.0f

typedef struct neighbor {
    size_t index; // index of the neighbor atom
    atom_t atom; // atom data of the neighbor atom
    real_t distance; // distance to the neighbor atom
} neighbor_t;

typedef struct device_data {
    size_t num_atoms;
    size_t num_elements;
    /*
    to construct it, sort the elements by their atomic number, and then assign the index of the element in the sorted array to the atom_types array.
    */
    size_t *atom_types; // array of atom types, length: num_atoms. the entries is not the atomic number, but the index of the corresponding entry in constants.
    atom_t *atoms; // array of atom data
    d3_constant_t *constants; // constants for the simulation
    real_t cell[3][3]; // cell matrix, specify the three vectors of the cell
    size_t max_cell_bias[3]; // the maximum bias of the cell in each direction, this must be an odd number (because of symmetry)
    size_t *num_neighbors; // array of number of neighbors for each atom, length: num_atoms.
    neighbor_t *neighbors; // array of neighbors, size: num_atoms * MAX_NEIGHBORS.
    real_t coordination_number_cutoff; // the cutof radius for CN computation
    real_t cutoff_radius; // the cutoff radius for the dispersion energy calculation
    /* some intermediate variables, not initialized but used during computation*/
    real_t *coordination_numbers; // array of coordination numbers, length: num_atoms.
    result_t *results; // results of the simulation
} device_data_t;

/**
 * @brief this kernel is used to compute the coordination number of each atom in the system.
 * @note this kernel should be launched with a 1D grid of blocks, each block containing a 1D array of threads.
 * @note the number of blocks equals the number of atoms in the system.
 * @note and the number of threads in each block equals to num_atoms * total_cell_bias.
 * @note the total_cell_bias should be precomputed at host
 */
__global__ void coordination_number_kernel(device_data_t *data) {
    size_t atom_1_index = blockIdx.x; // each block is responsible for one central atom
    size_t total_cell_bias = data->max_cell_bias[0] * data->max_cell_bias[1] * data->max_cell_bias[2]; // total number of cell bias
    size_t num_threads = blockDim.x; // total number of threads in the block
    size_t thread_index = threadIdx.x; // the linear index of thread in current block
    /* a chunk of shared memory block is used to store neighbor indicies 
     each thread can have 1 entry */
    __shared__ size_t neighbor_flags[MAX_BLOCK_SIZE]; // shared memory for neighbor indices
    /* initiate this block, each thread is responsible for a few entries */
    for (size_t i = thread_index; i < MAX_BLOCK_SIZE; i += num_threads) {
        neighbor_flags[i] = 0; // initialize the neighbor flags to false
    }
    __syncthreads(); // synchronize threads in the block
    int64_t x_bias = (thread_index % data->max_cell_bias[0]) - (data->max_cell_bias[0]/2); // x bias
    int64_t y_bias = (thread_index / data->max_cell_bias[0] % data->max_cell_bias[1]) - (data->max_cell_bias[1]/2); // y bias
    int64_t z_bias = (thread_index / (data->max_cell_bias[0] * data->max_cell_bias[1]) % data->max_cell_bias[2]) - (data->max_cell_bias[2]/2); // z bias
    /* if number of threads exceed num_atoms*total_cell_bias, then some thread should be idle */
    if (num_threads > data->num_atoms * total_cell_bias) {
        /* threads with index larger than num_atoms*total_cell_bias should be idle */
        if (thread_index >= data->num_atoms * total_cell_bias) {
            return; // thread is out of bounds
        }
    }
    size_t atom_2_index = thread_index / total_cell_bias; // each thread is responsible for one atom pair
    if (atom_2_index == atom_1_index) {
        return; // skip the central atom
    }
    assert(atom_2_index < data->num_atoms); // make sure the index is in bounds
    size_t atom_1_type = data->atom_types[atom_1_index]; // type of the central atom
    size_t atom_2_type = data->atom_types[atom_2_index]; // type of the surrounding atom
    atom_t atom_1 = data->atoms[atom_1_index]; // central atom
    atom_t atom_2 = data->atoms[atom_2_index]; // surrounding atom
    /* translate atom_2 due to periodic boundaries */
    atom_2.x += x_bias * data->cell[0][0] + y_bias * data->cell[1][0] + z_bias * data->cell[2][0]; // translate in x direction
    atom_2.y += x_bias * data->cell[0][1] + y_bias * data->cell[1][1] + z_bias * data->cell[2][1]; // translate in y direction
    atom_2.z += x_bias * data->cell[0][2] + y_bias * data->cell[1][2] + z_bias * data->cell[2][2]; // translate in z direction
    /* calculate the distance between the two atoms */
    real_t distance = sqrtf(powf(atom_1.x - atom_2.x, 2) + powf(atom_1.y - atom_2.y, 2) + powf(atom_1.z - atom_2.z, 2));
    /* if the distance is within cutoff range, update neighbor_flags */
    if (distance <= data->coordination_number_cutoff) {
        neighbor_flags[thread_index] = 1; // mark the atom as a neighbor
    }
    __syncthreads();
    /* now we need to convert entries in neighbor_flags to indicies in neighbors.
     algorithm: calculate prefix sum of each entry */
    // Perform exclusive prefix sum on neighbor_flags
    if (thread_index == 0 || (atom_1_index == 0 && thread_index == blockDim.x - 1)) {
        size_t sum = 0;
        for (size_t i = 0; i < num_threads; i++) {
            size_t temp = neighbor_flags[i];
            neighbor_flags[i] = sum;
            sum += temp;
        }
    }
    __syncthreads(); // Make sure all threads see the updated neighbor_flags
    /* now the indicies in neighbor_flags is the position to write in neighbors */
    if (distance <= data->coordination_number_cutoff) {
        /* if the distance is within cutoff range, update neighbors */
        atomicAdd((unsigned long long int*)&data->num_neighbors[atom_1_index], (unsigned long long int)1); // increment the number of neighbors for atom 1
        size_t neighbor_index = neighbor_flags[thread_index]; // index of the neighbor in the neighbors array
        neighbor_t *neighbors = &data->neighbors[atom_1_index * MAX_NEIGHBORS]; // pointer to the neighbors array for the central atom
        assert(neighbor_index < MAX_NEIGHBORS); // make sure the index is in bounds
        neighbors[neighbor_index].index = atom_2_index; // set the index of the neighbor atom
        neighbors[neighbor_index].distance = distance; // set the distance to the neighbor atom
        neighbors[neighbor_index].atom = atom_2; // set the atom data of the neighbor atom
        /* compute the coordination number and add to the CN of atom 1 and atom 2 */
        real_t covalent_radii_1 = data->constants->rcov[atom_1_type]; // covalent radii of atom 1
        real_t covalent_radii_2 = data->constants->rcov[atom_2_type]; // covalent radii of atom 2
        /* eq 15 in Grimme et al. 2010
        $CN^A = \sum_{B \neq A}^{N} \sqrt{1}{1+exp(-k_1(k_2(R_{A,cov}+R_{B,cov})/r_{AB}-1))}$ */
        real_t coordination_number = 1.0f/(1.0f+expf(-K1*((covalent_radii_1 + covalent_radii_2)/distance - 1.0f))); // the covalent radii in input table have already taken K2 coefficient into onsideration
        // increment the data.coordination_number array for both atoms
        atomicAdd(&data->coordination_numbers[atom_1_index], coordination_number); // increment the coordination number for atom 1
        atomicAdd(&data->coordination_numbers[atom_2_index], coordination_number); // increment the coordination number for atom 2
    }
    /* now the coordination number and neighbors are completed, 
     but every coordination number have been computed for two times.
     the division should be performed only once.
     the next kernel will do the work because of inter-block synchronization
     REMENBER TO DO THAT!!! :) */
    return; // return from the kernel
}


/**
 * @brief this kernel is used to adjust the coordination number of each atom in the system.
 * @note this kernel should be launched with a 1D grid of blocks, each block containing a 1D array of threads.
 * @note the number of blocks equals the number of atoms in the system.
 * 
 * ugly little kernel, but it's the only way for cross-block synchronization :(
 */
__global__ void adjust_coordination_number_kernel(device_data_t *data) {
    size_t atom_index = threadIdx.x; // each block is responsible for one atom
    data->coordination_numbers[atom_index] /= 2.0f; // divide the coordination number by 2
}

/**
 * @brief this kernel is used to compute the energy and force of each atom in the system.
 * @note this kernel should be launched with a 1D grid of blocks, each block containing a 2D array of threads.
 * @note the number of blocks should be equal to the number of atoms in the system.
 * @note the dimention of threads in each block should be equal to the maximum number of neighbors
 */
__global__ void energy_force_kernel(device_data_t *data) {
    size_t central_atom_index = blockIdx.x; // each block is responsible for one central atom
    size_t atom_i_index = threadIdx.x;
    size_t atom_j_index = threadIdx.y; // each thread in block handles one pair of atoms
    if (atom_i_index >= data->num_neighbors[central_atom_index] || atom_j_index >= data->num_neighbors[central_atom_index]) {
        return; // thread is out of bounds
    }
    /* work distribution:
     thread where atom_i_index != atom_j_index handle the derivative $\partial C_n^{i,j}/partial r_m$ 
     thread where atom_i_index == atom_j_index handle the rest two derivatives and energy calculation */
    if (atom_i_index == atom_j_index) {
        /* this thread is responsible for the rest two derivatives and energy*/
        /* here we implement energy part first */
        /* compute the energy between central atom and atom_i*/
        size_t atom_1_index = central_atom_index;
        size_t atom_2_index = data->neighbors[central_atom_index * MAX_NEIGHBORS + atom_j_index].index; // index of the second atom in the pair
        real_t coordination_number_1 = data->coordination_numbers[atom_1_index];
        real_t coordination_number_2 = data->coordination_numbers[atom_2_index];
        printf("coordination number of atom %llu: %f\n", atom_1_index, coordination_number_1);
        printf("coordination number of atom %llu: %f\n", atom_2_index, coordination_number_2);
        size_t atom_1_type = data->atom_types[atom_1_index];
        size_t atom_2_type = data->atom_types[atom_2_index];
        atom_t atom_1 = data->atoms[atom_1_index]; // central atom
        atom_t atom_2 = data->neighbors[central_atom_index * MAX_NEIGHBORS + atom_j_index].atom; // surrounding atom
        if (atom_2.element == 0) {
            return; // skip the atom if it is not valid
        }
        real_t distance = data->neighbors[central_atom_index * MAX_NEIGHBORS + atom_j_index].distance; // distance to the neighbor atom
        /* calculate the coordination number based on dispersion coefficient
         formula: $C_6^{ij} = Z/W$ 
         where $Z = \sum_{a,b}C_{6,ref}^{i,j}L_{a,b}$
            $W = \sum_{a,b}L_{a,b}$
            $L_{a,b} = \exp(-k3((CN^A-CN^A_{ref,a})^2 + (CN^B-CN^B_{ref,b})^2))$*/
        real_t Z = 0.0f;
        real_t W = 0.0f;
        for (size_t i = 0; i < NUM_REF_C6; ++i) {
            for (size_t j = 0; j < NUM_REF_C6; ++j) {
                /* find the C6ref */
                size_t stride_1 = data->num_elements * NUM_REF_C6 * NUM_REF_C6  * NUM_C6AB_ENTRIES;
                size_t stride_2 = NUM_REF_C6 * NUM_REF_C6  * NUM_C6AB_ENTRIES;
                size_t stride_3 = NUM_REF_C6  * NUM_C6AB_ENTRIES;
                size_t stride_4 = NUM_C6AB_ENTRIES;
                size_t index = atom_1_type * stride_1 + atom_2_type * stride_2 + i * stride_3 + j * stride_4;
                real_t c6_ref = data->constants->c6ab_ref->data[index + 0];
                /* these entries could be -1.0f if they are not valid, but at least one should be valid*/
                real_t coordination_number_ref_1 = data->constants->c6ab_ref->data[index + 1];
                real_t coordination_number_ref_2 = data->constants->c6ab_ref->data[index + 2];
                /* because they could be invalid, L_ij cannot be used directly */
                real_t L_ij_candidate = expf(-K3 * (powf(coordination_number_1 - coordination_number_ref_1, 2) + powf(coordination_number_2 - coordination_number_ref_2, 2))) * 1e5f; // scale it to avoid floating point error
                /* since we need the value $\frac{\sum_{i,j}C_{6,ref}^{A,B}L_{i,j}}{\sum_{i,j}L_{i,j}}$
                 we can set invalid L_ij to 0.0f and perform the summation in the same loop
                 invalid entry: have -1.0f in c6_ref, coordination_number_ref_1 and coordination_number_ref_2
                 we check coordination_number_ref_1 here. */
                real_t L_ij = ((coordination_number_ref_1 - (-1.0f) <= 1e-5f) ? 0.0f : L_ij_candidate); // conditional move, no branching, fast!
                Z += c6_ref * L_ij; // accumulate the value of Z
                W += L_ij; // accumulate the value of W
            }
        }
        real_t c6_ab = (W > 0.0f) ? Z / W : 0.0f; // avoid division by zero
        /* calculate c8_ab by $C_8^{AB} = 3C_6^{AB}\sqrt{Q^AQ^B}$*/
        real_t r2r4_1 = data->constants->r2r4[atom_1_type];
        real_t r2r4_2 = data->constants->r2r4[atom_2_type];
        real_t c8_ab = 3.0f * c6_ab * r2r4_1 * r2r4_2; // the value in r2r4 is already squared
        /* acquire the cutoff radius between the two atoms */
        real_t cutoff_radius = data->constants->r0ab[atom_1_type][atom_2_type];
        /* calculate the dampling function as Grimme et al. 2010, eq4 */
        real_t f_dn_6 = 1/(1+6.0f*powf(distance/(SR_6*cutoff_radius), -ALPHA_N(6.0f)));
        real_t f_dn_8 = 1/(1+6.0f*powf(distance/(SR_8*cutoff_radius), -ALPHA_N(8.0f)));
        /* calculate the dispersion enegry as Grimme et al. 2010, eq3 */
        real_t dispersion_energy_6 = S6*(c6_ab/powf(distance, 6.0f))*f_dn_6;
        real_t dispersion_energy_8 = S8*(c8_ab/powf(distance, 8.0f))*f_dn_8;
        // printf("dispersion energy between atoms (%llu,%llu): %f\n",atom_1_index,atom_2_index,dispersion_energy_6 + dispersion_energy_8);
        real_t dispersion_energy = dispersion_energy_6 + dispersion_energy_8;
        /* add the energy back to results */
        atomicAdd(&data->results[atom_1_index].energy, dispersion_energy);
        atomicAdd(&data->results[atom_2_index].energy, dispersion_energy);
        return;
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
__host__ void compute_dispersion_energy(
    real_t atoms[][4], 
    size_t length, 
    real_t cell[3][3],
    real_t cutoff_radius,
    real_t coordination_number_cutoff) {
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
    /* allocate memory for data.neighbors and data.num_neighbors */
    size_t *d_num_neighbors;
    CHECK_CUDA(cudaMalloc((void **)&d_num_neighbors, length * sizeof(size_t)));
    CHECK_CUDA(cudaMemset(d_num_neighbors, 0, length * sizeof(size_t)));
    h_data.num_neighbors = d_num_neighbors; // set the number of neighbors in the device data
    neighbor_t *d_neighbors;
    CHECK_CUDA(cudaMalloc((void **)&d_neighbors, length * MAX_NEIGHBORS * sizeof(neighbor_t)));
    CHECK_CUDA(cudaMemset(d_neighbors, 0, length * MAX_NEIGHBORS * sizeof(neighbor_t)));
    h_data.neighbors = d_neighbors; // set the neighbors in the device data
    /* initiate cell info and cutoff parameters */
    size_t total_cell_bias = 1;
    for(size_t i = 0; i < 3; ++i) {
        real_t length = 0.0f;
        for(size_t j = 0; j < 3; ++j) {
            h_data.cell[i][j] = cell[i][j]; // set the cell info in the device data
            length += cell[i][j] * cell[i][j]; // calculate the length of the cell vector
        }
        /* calculate max_cell_bias */
        h_data.max_cell_bias[i] = (size_t)ceilf(cutoff_radius / sqrtf(length))*2+1; // set the max cell bias in the device data
        printf("max_cell_bias[%zu] = %zu\n", i, h_data.max_cell_bias[i]);
        total_cell_bias *= h_data.max_cell_bias[i]; // calculate the total cell bias
    }
    h_data.cutoff_radius = cutoff_radius; // set the cutoff radius in the device data
    h_data.coordination_number_cutoff = coordination_number_cutoff; // set the coordination number cutoff in the device data

    // initialize the device data
    device_data_t *d_data;
    CHECK_CUDA(cudaMalloc((void **)&d_data, sizeof(device_data_t)));
    CHECK_CUDA(cudaMemcpy(d_data, &h_data, sizeof(device_data_t), cudaMemcpyHostToDevice));

    // launch the kernel
    coordination_number_kernel<<<length, total_cell_bias*length>>>(d_data); // launch the kernel to compute the coordination numbers
    CHECK_CUDA(cudaDeviceSynchronize()); // synchronize the device to ensure all threads are finished
    adjust_coordination_number_kernel<<<1, length>>>(d_data); // launch the kernel to adjust the coordination numbers
    CHECK_CUDA(cudaDeviceSynchronize()); // synchronize the device to ensure all threads are finished
    dim3 block_size(10, 10); // 10x10 threads per block
    dim3 grid_size(length);  // one block per atom
    energy_force_kernel<<<grid_size, block_size>>>(d_data);
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
        printf("Atom %zu: energy = %f\n", i, h_results[i].energy);
        // accumulate the total energy
        total_energy += h_results[i].energy;
    }
    total_energy /= 4.0f; /* the energy of between two atoms is added to both of the atoms. So the totoal energy should be divided by 2 after adding all atomic energy */
    printf("Total energy = %.9f\n", total_energy);
    // free the device memory
    CHECK_CUDA(cudaFree(d_atoms));
    CHECK_CUDA(cudaFree(d_atom_types));
    CHECK_CUDA(cudaFree(d_results));
    CHECK_CUDA(cudaFree(d_coordination_numbers));
    CHECK_CUDA(cudaFree(d_data));
    // free the host memory
    free(h_atoms);
}

int main()
{
    // example usage of the compute_dispersion_energy function
    real_t atoms[10][4] = {
        {6, 5.137f, 5.551f, 10.1047f},
        {6, 4.5168f, 6.1365f, 11.36043f},
        {6, 6.1936f, 4.4752f, 10.2703f},
        {8, 4.78716f, 5.9358f, 8.99372f},
        {1, 6.7474f, 4.3475f, 9.3339f},
        {1, 5.69748f, 3.5214f, 10.5181f},
        {1, 6.88699f, 4.7006f, 11.0939f},
        {1, 4.85788f, 5.6442f, 12.2774f},
        {1, 3.42038, 6.0677, 11.29354},
        {1, 4.7677f, 7.20752f, 11.4098f}
    };
    real_t angstron_to_bohr = 1/0.529f; // angstron to bohr conversion factor
    for(size_t i = 0; i < 10; ++i) {
        atoms[i][1] *= angstron_to_bohr; // convert to bohr
        atoms[i][2] *= angstron_to_bohr; // convert to bohr
        atoms[i][3] *= angstron_to_bohr; // convert to bohr
    }
    // fill the atoms array with Po element
    debug("Computing dispersion energy for %zu atoms...\n", sizeof(atoms)/sizeof(atoms[0]));
    // initialize parameters
    init_params();
    debug("Computing dispersion energy...\n");
    real_t cell[3][3] = {
        {100.0f, 0.0f, 0.0f},
        {0.0f, 100.0f, 0.0f},
        {0.0f, 0.0f, 100.0f}
    };
    real_t cutoff_radius = 50.0f; // cutoff radius in bohr
    compute_dispersion_energy(atoms, 10, cell, cutoff_radius, cutoff_radius);
    return 0;
}