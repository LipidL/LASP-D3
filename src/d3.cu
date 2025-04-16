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
#define MAX_NEIGHBORS 5000 // the maximum number of neighbors, dependent on the cutoff choice

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
    uint64_t index; // index of the neighbor atom
    atom_t atom; // atom data of the neighbor atom
    real_t distance; // distance to the neighbor atom
} neighbor_t;

typedef struct device_data {
    uint64_t num_atoms;
    uint64_t num_elements;
    /*
    to construct it, sort the elements by their atomic number, and then assign the index of the element in the sorted array to the atom_types array.
    */
    uint64_t *atom_types; // array of atom types, length: num_atoms. the entries is not the atomic number, but the index of the corresponding entry in constants.
    atom_t *atoms; // array of atom data
    d3_constant_t *constants; // constants for the simulation
    real_t cell[3][3]; // cell matrix, specify the three vectors of the cell
    uint64_t max_cell_bias[3]; // the maximum bias of the cell in each direction, this must be an odd number (because of symmetry)
    // uint64_t *num_neighbors; // array of number of neighbors for each atom, length: num_atoms.
    neighbor_t *neighbors; // array of neighbors, size: num_atoms * MAX_NEIGHBORS.
    neighbor_t *CN_neighbors; //array of neighbors within CN cutoff, size: num_atoms * MAX_NEIGHBORS
    real_t coordination_number_cutoff; // the cutof radius for CN computation
    real_t cutoff; // the cutoff radius for the dispersion energy calculation
    /* some intermediate variables, not initialized but used during computation*/
    real_t *coordination_numbers; // array of coordination numbers, length: num_atoms.
    uint64_t *num_neighbors; // array of maximum number of neighbors for each atom, length: num_atoms.
    uint64_t *num_CN_neighbors;
    uint64_t max_num_neighbor; // the maximum nimber of neighbors for the system
    uint64_t max_num_CN_neighbor;
    real_t *dCNi_datom_i; // array of dCN_i/dr_i, used in force computation, length: 3*num_atoms.
    result_t *results; // results of the simulation
} device_data_t;

/**
 * @brief this kernel is used to compute the coordination number of each atom in the system.
 * @note this kernel should be launched with a 1D grid of blocks, each block containing a 1D array of threads.
 * @note the number of blocks equals the number of atoms in the system.
 * @note and the number of threads in each block equals to num_atoms
 * @note the total_cell_bias should be precomputed at host
 */
__global__ void coordination_number_kernel(device_data_t *data) {
    uint64_t atom_1_index = blockIdx.x; // each block is responsible for one central atom
    uint64_t total_cell_bias = data->max_cell_bias[0] * data->max_cell_bias[1] * data->max_cell_bias[2]; // total number of cell bias
    uint64_t num_threads = blockDim.x; // total number of threads in the block
    uint64_t thread_index = threadIdx.x; // the linear index of thread in current block
    uint64_t atom_2_index = thread_index;
    /* a chunk of shared memory block is used to store neighbor indicies 
     each thread can have 1 entry */
    __shared__ uint64_t neighbor_flags[MAX_BLOCK_SIZE]; // shared memory for neighbor indices
    __shared__ uint64_t CN_neighbor_flags[MAX_BLOCK_SIZE]; // shared memory for CN neighbor indices
    /* initiate this block, each thread is responsible for a few entries */
    for (uint64_t i = thread_index; i < MAX_BLOCK_SIZE; i += num_threads) {
        neighbor_flags[i] = 0; // initialize the neighbor flags to false
        CN_neighbor_flags[i] = 0; // initialize the CN neighbor flags to false
    }
    __syncthreads(); // synchronize threads in the block
    /* loop ovber three dimentions to find neighbor of atom 1 */
    neighbor_t neighbors[MAX_NEIGHBORS]; // array of neighbors for the central atom
    neighbor_t CN_neighbors[MAX_NEIGHBORS]; // array of neighbors for the CN calculation
    uint64_t neighbors_index = 0; // index of the neighbor in the neighbors array
    uint64_t CN_neighbors_index = 0; // index of the neighbor in the CN neighbors array
    uint64_t atom_1_type = data->atom_types[atom_1_index]; // type of the central atom
    uint64_t atom_2_type = data->atom_types[atom_2_index]; // type of the surrounding atom
    atom_t atom_1 = data->atoms[atom_1_index]; // central atom
    for(uint64_t bias_index = 0; bias_index < total_cell_bias; ++bias_index) {
        /* each thread is responsible for one atom pair, so the number of threads should be equal to num_atoms * total_cell_bias */
        int64_t x_bias = (bias_index % data->max_cell_bias[0]) - (data->max_cell_bias[0]/2); // x bias
        int64_t y_bias = (bias_index / data->max_cell_bias[0] % data->max_cell_bias[1]) - (data->max_cell_bias[1]/2); // y bias
        int64_t z_bias = (bias_index / (data->max_cell_bias[0] * data->max_cell_bias[1]) % data->max_cell_bias[2]) - (data->max_cell_bias[2]/2); // z bias
        assert(atom_2_index < data->num_atoms); // make sure the index is in bounds

        atom_t atom_2 = data->atoms[atom_2_index]; // surrounding atom
        /* translate atom_2 due to periodic boundaries */
        atom_2.x += x_bias * data->cell[0][0] + y_bias * data->cell[1][0] + z_bias * data->cell[2][0]; // translate in x direction
        atom_2.y += x_bias * data->cell[0][1] + y_bias * data->cell[1][1] + z_bias * data->cell[2][1]; // translate in y direction
        atom_2.z += x_bias * data->cell[0][2] + y_bias * data->cell[1][2] + z_bias * data->cell[2][2]; // translate in z direction
        /* calculate the distance between the two atoms */
        real_t distance = sqrtf(powf(atom_1.x - atom_2.x, 2) + powf(atom_1.y - atom_2.y, 2) + powf(atom_1.z - atom_2.z, 2));
        /* if the distance is within cutoff range, update neighbor_flags */
        if (distance <= data->coordination_number_cutoff && distance > 0.0f) {
            CN_neighbor_flags[thread_index] += 1; // mark the atom as a neighbor
            if (CN_neighbors_index < MAX_NEIGHBORS) {
                CN_neighbors[CN_neighbors_index].index = atom_2_index; // set the index of the neighbor atom
                CN_neighbors[CN_neighbors_index].distance = distance; // set the distance to the neighbor atom
                CN_neighbors[CN_neighbors_index].atom = atom_2; // set the atom data of the neighbor atom
                CN_neighbors_index++; // increment the index of the neighbor
            } else {
                printf("Warning: too many neighbors for atom %llu\n", atom_1_index); // too many neighbors, skip this one
            }
        }
        if (distance <= data->cutoff && distance > 0.0f) {
            neighbor_flags[thread_index] += 1; // mark the atom as a neighbor for CN calculation
            if (neighbors_index < MAX_NEIGHBORS) {
                neighbors[neighbors_index].index = atom_2_index; // set the index of the neighbor atom
                neighbors[neighbors_index].distance = distance; // set the distance to the neighbor atom
                neighbors[neighbors_index].atom = atom_2; // set the atom data of the neighbor atom
                neighbors_index++; // increment the index of the neighbor
            } else {
                printf("Warning: too many neighbors for atom %llu\n", atom_1_index); // too many neighbors, skip this one
            }
        }
    }
    __syncthreads();
    /* now we need to convert entries in neighbor_flags to indicies in neighbors.
     algorithm: calculate prefix sum of each entry */
    // Perform exclusive prefix sum on neighbor_flags
    if (thread_index == 0) {
        uint64_t sum = 0;
        uint64_t CN_sum = 0;
        for (uint64_t i = 0; i < num_threads; i++) {
            uint64_t temp = neighbor_flags[i];
            neighbor_flags[i] = sum;
            sum += temp;
            uint64_t CN_temp = CN_neighbor_flags[i];
            CN_neighbor_flags[i] = CN_sum;
            CN_sum += CN_temp;
        }
        data->num_neighbors[atom_1_index] = sum; // set the maximum number of neighbors for the central atom
        data->num_CN_neighbors[atom_1_index] = CN_sum; // set the maximum number of CN neighbors for the central atom
    }
    __syncthreads(); // Make sure all threads see the updated neighbor_flags
    /* now the indicies in neighbor_flags is the position to write in neighbors */
    for(uint64_t i = 0; i < CN_neighbors_index; ++i) {
        real_t distance = CN_neighbors[i].distance; // distance to the neighbor atom
        assert(distance <= data->coordination_number_cutoff); // make sure the distance is in bounds
        /* if the distance is within cutoff range, update neighbors */
        uint64_t neighbor_index = CN_neighbor_flags[thread_index]; // index of the neighbor in the neighbors array
        neighbor_t *data_neighbors = &data->CN_neighbors[atom_1_index * MAX_NEIGHBORS]; // pointer to the neighbors array for the central atom
        assert(neighbor_index + i < MAX_NEIGHBORS); // make sure the index is in bounds
        assert(data_neighbors[neighbor_index+i].index == 0); // make sure the index is not already set
        assert(distance <= data->coordination_number_cutoff); // make sure the distance is in bounds
        data_neighbors[neighbor_index+i].index = CN_neighbors[i].index;
        data_neighbors[neighbor_index+i].distance = distance;
        data_neighbors[neighbor_index+i].atom = CN_neighbors[i].atom;
        /* compute the coordination number and add to the CN of atom 1 and atom 2 */
        real_t covalent_radii_1 = data->constants->rcov[atom_1_type];
        real_t covalent_radii_2 = data->constants->rcov[atom_2_type];
        /* eq 15 in Grimme et al. 2010
        $CN^A = \sum_{B \neq A}^{N} \sqrt{1}{1+exp(-k_1(k_2(R_{A,cov}+R_{B,cov})/r_{AB}-1))}$ */
        real_t exp = expf(-K1*((covalent_radii_1 + covalent_radii_2)/distance - 1.0f)); // $\exp(-k_1*(\frac{R_A+R_b}{r_{ab}}-1))$
        real_t coordination_number = 1.0f/(1.0f+exp); // the covalent radii in input table have already taken K2 coefficient into onsideration
        real_t dCN_datom = powf(1.0f+exp,-2.0f)*(-K1)*exp*(covalent_radii_1 + covalent_radii_2)*powf(distance, -3.0f); // dCN_ij/dr_i * 1/r_m
        // increment the data.coordination_number array for both atoms
        atomicAdd(&data->coordination_numbers[atom_1_index], coordination_number); // increment the coordination number for atom 1
        atomicAdd(&data->coordination_numbers[atom_2_index], coordination_number); // increment the coordination number for atom 2
        /* increment the data.dCNi_datomi entry for both atoms*/
        atomicAdd(&data->dCNi_datom_i[3*atom_1_index+0], dCN_datom * (atom_1.x - CN_neighbors[i].atom.x) / 2.0f);
        atomicAdd(&data->dCNi_datom_i[3*atom_1_index+1], dCN_datom * (atom_1.y - CN_neighbors[i].atom.y) / 2.0f);
        atomicAdd(&data->dCNi_datom_i[3*atom_1_index+2], dCN_datom * (atom_1.z - CN_neighbors[i].atom.z) / 2.0f);
        atomicAdd(&data->dCNi_datom_i[3*atom_2_index+0], dCN_datom * (CN_neighbors[i].atom.x - atom_1.x) / 2.0f);
        atomicAdd(&data->dCNi_datom_i[3*atom_2_index+1], dCN_datom * (CN_neighbors[i].atom.y - atom_1.y) / 2.0f);
        atomicAdd(&data->dCNi_datom_i[3*atom_2_index+2], dCN_datom * (CN_neighbors[i].atom.z - atom_1.z) / 2.0f);
    }
    for (uint64_t i = 0; i < neighbors_index; ++i) {
        real_t distance = neighbors[i].distance; // distance to the neighbor atom
        assert(distance <= data->cutoff);
        /* if the distance is within cutoff range, update neighbors */
        uint64_t neighbor_index = neighbor_flags[thread_index]; // index of the neighbor in the neighbors array
        neighbor_t *data_neighbors = &data->neighbors[atom_1_index * MAX_NEIGHBORS]; // pointer to the neighbors array for the central atom
        assert(neighbor_index + i < MAX_NEIGHBORS); // make sure the index is in bounds
        assert(data_neighbors[neighbor_index+i].index == 0); // make sure the index is not already set
        assert(distance <= data->cutoff); // make sure the distance is in bounds
        data_neighbors[neighbor_index+i].index = neighbors[i].index;
        data_neighbors[neighbor_index+i].distance = distance;
        data_neighbors[neighbor_index+i].atom = neighbors[i].atom;
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
    uint64_t atom_index = threadIdx.x; // each block is responsible for one atom
    data->coordination_numbers[atom_index] /= 2.0f; // divide the coordination number by 2
    printf("coordination number of atom %llu: %.13f\n", atom_index, data->coordination_numbers[atom_index]);
    // Find the maximum number of neighbors
    __shared__ uint64_t max_neighbors;
    if (atom_index == 0) {
        max_neighbors = 0;
    }
    __syncthreads();
    atomicMax((unsigned long long *)&max_neighbors, (unsigned long long)data->num_neighbors[atom_index]);
    __syncthreads();

    if (atom_index == 0) {
        // Store the result for later use
        data->max_num_neighbor = max_neighbors;
    }
}

/**
 * @brief this kernel is used to compute the two-body interactions between atoms in the system.
 * @brief i.e the energy and two-atom part of force
 * @note this kernel should be launched with a 1D grid of blocks, each block containining a 1D array of threads.
 * @note the number of blocks should be equal to the number of atoms in the system.
 * @note the number of threads in each block can be any value.
 */
__global__ void two_body_kernel(device_data_t *data){
    uint64_t atom_1_index = blockIdx.x; // each block is responsible for one central atom
    atom_t atom_1 = data->atoms[atom_1_index]; // central atom
    for (uint64_t neighbor_index = threadIdx.x; neighbor_index < data->num_neighbors[atom_1_index]; neighbor_index += blockDim.x) {
        /* each thread is responsible for one atom pair, so the number of threads should be equal to num_atoms * total_cell_bias */
        uint64_t atom_2_index = data->neighbors[atom_1_index * MAX_NEIGHBORS + neighbor_index].index; // index of the second atom in the pair
        atom_t atom_2 = data->neighbors[atom_1_index * MAX_NEIGHBORS + neighbor_index].atom; // surrounding atom
        real_t distance = data->neighbors[atom_1_index * MAX_NEIGHBORS + neighbor_index].distance; // distance to the neighbor atom
        // printf("atom_1: %llu, atom_2: %llu, distance: %f\n", atom_1_index, atom_2_index, distance);
        if (atom_2.element == 0) {
            return; // skip the atom if it is not valid
        }
        real_t coordination_number_1 = data->coordination_numbers[atom_1_index];
        real_t coordination_number_2 = data->coordination_numbers[atom_2_index];
        uint64_t atom_1_type = data->atom_types[atom_1_index];
        uint64_t atom_2_type = data->atom_types[atom_2_index];
        /* calculate the coordination number based on dispersion coefficient
            formula: $C_6^{ij} = Z/W$ 
            where $Z = \sum_{a,b}C_{6,ref}^{i,j}L_{a,b}$
            $W = \sum_{a,b}L_{a,b}$
            $L_{a,b} = \exp(-k3((CN^A-CN^A_{ref,a})^2 + (CN^B-CN^B_{ref,b})^2))$*/
        real_t Z = 0.0f;
        real_t W = 0.0f;
        real_t c_ref_L_ij = 0.0f;
        real_t c_ref_dL_ij_1 = 0.0f;
        real_t c_ref_dL_ij_2 = 0.0f;
        real_t dL_ij_1 = 0.0f;
        real_t dL_ij_2 = 0.0f;
        for (uint64_t i = 0; i < NUM_REF_C6; ++i) {
            for (uint64_t j = 0; j < NUM_REF_C6; ++j) {
                /* find the C6ref */
                uint64_t stride_1 = data->num_elements * NUM_REF_C6 * NUM_REF_C6  * NUM_C6AB_ENTRIES;
                uint64_t stride_2 = NUM_REF_C6 * NUM_REF_C6  * NUM_C6AB_ENTRIES;
                uint64_t stride_3 = NUM_REF_C6  * NUM_C6AB_ENTRIES;
                uint64_t stride_4 = NUM_C6AB_ENTRIES;
                uint64_t index = atom_1_type * stride_1 + atom_2_type * stride_2 + i * stride_3 + j * stride_4;
                real_t c6_ref = data->constants->c6ab_ref->data[index + 0];
                /* these entries could be -1.0f if they are not valid, but at least one should be valid*/
                real_t coordination_number_ref_1 = data->constants->c6ab_ref->data[index + 1];
                real_t coordination_number_ref_2 = data->constants->c6ab_ref->data[index + 2];
                /* because they could be invalid, L_ij cannot be used directly */
                real_t L_ij_candidate = expf(-K3 * (powf(coordination_number_1 - coordination_number_ref_1, 2) + powf(coordination_number_2 - coordination_number_ref_2, 2)));
                real_t dL_ij_candidate_1 = -2.0f * K3 * (coordination_number_1 - coordination_number_ref_1) * L_ij_candidate; // dL_ij/dCN_1
                real_t dL_ij_candidate_2 = -2.0f * K3 * (coordination_number_2 - coordination_number_ref_2) * L_ij_candidate; // dL_ij/dCN_2
                /* since we need the value $\frac{\sum_{i,j}C_{6,ref}^{A,B}L_{i,j}}{\sum_{i,j}L_{i,j}}$
                    we can set invalid L_ij to 0.0f and perform the summation in the same loop
                    invalid entry: have -1.0f in c6_ref, coordination_number_ref_1 and coordination_number_ref_2
                    we check coordination_number_ref_1 here. */
                if (coordination_number_ref_1 - (-1.0f) > 1e-5f) {
                    c_ref_L_ij += c6_ref * L_ij_candidate;
                    c_ref_dL_ij_1 += c6_ref * dL_ij_candidate_1;
                    c_ref_dL_ij_2 += c6_ref * dL_ij_candidate_2;
                    dL_ij_1 += dL_ij_candidate_1;
                    dL_ij_2 += dL_ij_candidate_2; // accumulate the value of dL_ij
                }
                real_t L_ij = ((coordination_number_ref_1 - (-1.0f) <= 1e-5f) ? 0.0f : L_ij_candidate); // conditional move, no branching, fast!
                Z += c6_ref * L_ij; // accumulate the value of Z
                W += L_ij; // accumulate the value of W
            }
        }
        real_t L_ij = W;
        real_t dC_ab_dCN_1 = (L_ij > 0.0f) ? (c_ref_dL_ij_1*L_ij - c_ref_L_ij * dL_ij_1) / powf(L_ij,2.0f) : 0.0f; // avoid division by zero
        real_t dC_ab_dCN_2 = (L_ij > 0.0f) ? (c_ref_dL_ij_2*L_ij - c_ref_L_ij * dL_ij_2) / powf(L_ij,2.0f) : 0.0f; // avoid division by zero
        /* add dC12/dr1 */
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
        /* increment the dCab/dr_a and dC_ab/dr_b to force of atom a and b */
        real_t force_1 = 0.0f;
        real_t *dCN1_datom_1 = &data->dCNi_datom_i[3*atom_1_index]; // dCN_1/dr_1
        force_1 += S6 * f_dn_6 * powf(distance, -6.0f) * dC_ab_dCN_1; // force_6
        force_1 += S8 * f_dn_8 * powf(distance, -8.0f) * dC_ab_dCN_1; // force_8
        atomicAdd(&data->results[atom_1_index].force[0], force_1 * dCN1_datom_1[0]);
        atomicAdd(&data->results[atom_1_index].force[1], force_1 * dCN1_datom_1[1]);
        atomicAdd(&data->results[atom_1_index].force[2], force_1 * dCN1_datom_1[2]);
        real_t force_2 = 0.0f;
        real_t *dCN2_datom_2 = &data->dCNi_datom_i[3*atom_2_index]; // dCN_2/dr_2
        force_2 += S6 * f_dn_6 * powf(distance, -6.0f) * dC_ab_dCN_2; // force_8
        force_2 += S8 * f_dn_8 * powf(distance, -8.0f) * dC_ab_dCN_2; // force_8
        atomicAdd(&data->results[atom_2_index].force[0], force_2 * dCN2_datom_2[0]);
        atomicAdd(&data->results[atom_2_index].force[1], force_2 * dCN2_datom_2[1]);
        atomicAdd(&data->results[atom_2_index].force[2], force_2 * dCN2_datom_2[2]);
        /* the first entry of two-body force
         $F_a = S_n C_n^{ab} f_{d,n}(r_{ab}) \frac{\partial}{\partial r_a} r_{ab}^{-n}$
         $F_a = S_n C_n^{ab} f_{d,n}(r_{ab}) * (-n)r_{ab}^{-n-2} * \uparrow{r_{ab}}$ */
        real_t force = 0.0f;
        force += S6 * c6_ab * f_dn_6 * (-6.0f) * powf(distance, -8.0f); // force_6
        force += S8 * c8_ab * f_dn_8 * (-8.0f) * powf(distance, -10.0f); // force_8
        /* the second entry of two-body force 
         $F_a = S_n C_n^{ab} r_{ab}^{-n} \frac{\partial}{\partial r_a} f_{d,n}(r_{ab})$
         $F_a = S_n C_n^{ab} r_{ab}^{-n} -f_{d,n}^2 * (6*(-\alpha_n)*(r_{ab}/{S_{r,n}R_0^{AB}})^(-\alpha_n - 1) * 1/(S_{r,n}R_0^{AB}})) / r_ab \uparrow{r_{ab}}$*/
        force += S6 * c6_ab * powf(distance, -6.0f) * (-f_dn_6 * f_dn_6) * (6.0f * (-ALPHA_N(6.0f))* powf(distance/(SR_6*cutoff_radius), -ALPHA_N(6.0f) - 1.0f) / (SR_6*cutoff_radius)) / distance; // force_6
        force += S8 * c8_ab * powf(distance, -8.0f) * (-f_dn_8 * f_dn_8) * (6.0f * (-ALPHA_N(8.0f))* powf(distance/(SR_8*cutoff_radius), -ALPHA_N(8.0f) - 1.0f) / (SR_8*cutoff_radius)) / distance; // force_8
        atomicAdd(&data->results[atom_1_index].force[0], force * (atom_1.x - atom_2.x)/2.0f);
        atomicAdd(&data->results[atom_1_index].force[1], force * (atom_1.y - atom_2.y)/2.0f);
        atomicAdd(&data->results[atom_1_index].force[2], force * (atom_1.z - atom_2.z)/2.0f);
        atomicAdd(&data->results[atom_2_index].force[0], -force * (atom_1.x - atom_2.x)/2.0f);
        atomicAdd(&data->results[atom_2_index].force[1], -force * (atom_1.y - atom_2.y)/2.0f);
        atomicAdd(&data->results[atom_2_index].force[2], -force * (atom_1.z - atom_2.z)/2.0f);
    }

}

/**
 * @brief this kernel is used to compute the three-body force between atoms in the system.
 * @brief i.e. the dC(ij)/dr_m
 * @note this kernel should be launched with a 1D grid of blocks, each block containining a 2D array of threads.
 * @note the dimention of block is the number of atoms in the system.
 * @note the number of blocks can be any value
 */
__global__ void three_body_kernel(device_data_t *data){
    /* this kernel calculates the force entry dC_{ij}/dr_m
     the diffrensial is not 0 if atom m (central atom) has an impact on C_{ij}
     so atom m should be within data->CN_cutoff with at least one of atoms i and j
     and atom i and j should be within data->cutoff so that they have a C_{ij} */
    uint64_t central_atom_index = blockIdx.x; // each block is responsible for one central atom
    atom_t central_atom = data->atoms[central_atom_index]; // central atom
    uint64_t central_atom_type = data->atom_types[central_atom_index]; // type of the central atom
    /* atom 1 should be within CN_cutoff with central_atom, and atom 2 should be within cutoff with atom1*/
    for(uint64_t thread_idx_x = threadIdx.x; thread_idx_x < data->num_CN_neighbors[central_atom_index]; thread_idx_x += blockDim.x) {
        uint64_t atom_1_index = data->CN_neighbors[central_atom_index * MAX_NEIGHBORS + thread_idx_x].index;
        uint64_t atom_1_type = data->atom_types[atom_1_index];
        atom_t atom_1 = data->CN_neighbors[central_atom_index * MAX_NEIGHBORS + thread_idx_x].atom;
        real_t coordination_number_1 = data->coordination_numbers[atom_1_index];
        real_t distance_mi = data->CN_neighbors[central_atom_index * MAX_NEIGHBORS + thread_idx_x].distance; // distance to the neighbor atom
        for(uint64_t thread_idx_y = threadIdx.y; thread_idx_y < data->num_neighbors[atom_1_index]; thread_idx_y += blockDim.y) {
            /* each thread is responsible for one atom pair, so the number of threads should be equal to num_atoms * total_cell_bias */
            /* dC_{ij}/dr_m = dC_{ij}/dCN_i * dCN_i/dr_m + dC_{ij}/dCN_i * dCN_i/dr_m
             dCN_i/dr_m is not 0 only if atoms i and m are CN_neighbors
             so here we only calculate dC_{1,2}/dCN_1 * dCN_1/dr_m because atom1 is found in m's CN-neighbor list
             the dC_{1,2}/dCN_2 * dCN_2/dr_m is calculated in next thread if atom2 is within m's CN-neighbor list,
             or don't need to be calculated if atom2 is not within m's CN-neighbor list */
            uint64_t atom_2_index = data->neighbors[atom_1_index * MAX_NEIGHBORS + thread_idx_y].index;
            uint64_t atom_2_type = data->atom_types[atom_2_index];
            // printf("central atom: %llu, atom 1: %llu, atom 2: %llu\n", central_atom_index, atom_1_index, atom_2_index);
            real_t coordination_number_2 = data->coordination_numbers[atom_2_index];
            real_t distance_ij = data->neighbors[atom_1_index * MAX_NEIGHBORS + thread_idx_y].distance; // distance to the neighbor atom
            /* calculate dC_{ij}/dCN_i */
            real_t L_ij = 0.0f;
            real_t dL_ij = 0.0f;
            real_t c_ref_dL_ij = 0.0f;
            real_t c_ref_L_ij = 0.0f;
            for (uint64_t i = 0; i < NUM_REF_C6; ++i) {
                for (uint64_t j = 0; j < NUM_REF_C6; ++j) {
                    /* find the C6ref */
                    uint64_t stride_1 = data->num_elements * NUM_REF_C6 * NUM_REF_C6  * NUM_C6AB_ENTRIES;
                    uint64_t stride_2 = NUM_REF_C6 * NUM_REF_C6  * NUM_C6AB_ENTRIES;
                    uint64_t stride_3 = NUM_REF_C6  * NUM_C6AB_ENTRIES;
                    uint64_t stride_4 = NUM_C6AB_ENTRIES;
                    uint64_t index = atom_1_type * stride_1 + atom_2_type * stride_2 + i * stride_3 + j * stride_4;
                    real_t c6_ref = data->constants->c6ab_ref->data[index + 0];
                    /* these entries could be -1.0f if they are not valid, but at least one should be valid*/
                    real_t coordination_number_ref_1 = data->constants->c6ab_ref->data[index + 1];
                    real_t coordination_number_ref_2 = data->constants->c6ab_ref->data[index + 2];
                    /* because they could be invalid, L_ij cannot be used directly */
                    real_t L_ij_candidate = expf(-K3 * (powf(coordination_number_1 - coordination_number_ref_1, 2) + powf(coordination_number_2 - coordination_number_ref_2, 2))); // scale it to avoid floating point error
                    real_t dL_ij_candidate = -2.0f * K3 * (coordination_number_1 - coordination_number_ref_1) * L_ij_candidate; // dL_ij/dCN_1
                    /* since we need the value $\frac{\sum_{i,j}C_{6,ref}^{A,B}L_{i,j}}{\sum_{i,j}L_{i,j}}$
                        we can set invalid L_ij to 0.0f and perform the summation in the same loop
                        invalid entry: have -1.0f in c6_ref, coordination_number_ref_1 and coordination_number_ref_2
                        we check coordination_number_ref_1 here. */
                    if (coordination_number_ref_1 - (-1.0f) > 1e-5f) {
                        c_ref_L_ij += c6_ref * L_ij_candidate;
                        c_ref_dL_ij += c6_ref * dL_ij_candidate;
                        dL_ij += dL_ij_candidate;
                        L_ij += L_ij_candidate; // accumulate the value of L_ij
                    }
                }
            }
            real_t dC_ab_dCN_1 = (L_ij > 0.0f) ? (c_ref_dL_ij*L_ij - c_ref_L_ij * dL_ij) / powf(L_ij,2.0f) : 0.0f; // avoid division by zero
            real_t covalent_radii_central = data->constants->rcov[central_atom_type];
            real_t covalent_radii_1 = data->constants->rcov[atom_1_type];
            real_t exp = expf(-K1*((covalent_radii_central + covalent_radii_1)/distance_mi - 1.0f)); // $\exp(-k_1*(\frac{R_A+R_b}{r_{ab}}-1))$
            real_t dCN_1_datom_m = powf(1+exp,-2.0f)*(-K1)*exp*(covalent_radii_central + covalent_radii_1)*powf(distance_mi, -3.0f); // dCN_i/dr_m * 1/r_m
            real_t cutoff_radius = data->constants->r0ab[atom_1_type][atom_2_type];
            real_t r2r4_1 = data->constants->r2r4[atom_1_type];
            real_t r2r4_2 = data->constants->r2r4[atom_2_type];
            real_t f_dn_6 = 1/(1+6.0f*powf(distance_ij/(SR_6*cutoff_radius), -ALPHA_N(6.0f)));
            real_t f_dn_8 = 1/(1+6.0f*powf(distance_ij/(SR_8*cutoff_radius), -ALPHA_N(8.0f)));
            real_t force = 0.0f;
            /* F_{a,n} = S_n * f_{d,n}  * r_{AB}^{-n} * (dC_n^{AB}/dCNi) * (dCNi/dr_m) */
            force += S6 * f_dn_6 * powf(distance_ij, -6.0f) * dC_ab_dCN_1 * dCN_1_datom_m; // force_6
            /* c8_ab = 3.0f * c6_ab * r2r4_1 * r2r4_2 and all value except c6_ab are constant
             so (dC_8^{AB}/dCNi) = (dC_6^{AB}/dCNi) *3 * r2r4_1 * r2r4*2 */
            force += S8 * f_dn_8 * powf(distance_ij, -8.0f) * dC_ab_dCN_1 * dCN_1_datom_m * 3.0f * r2r4_1 * r2r4_2; // force_8
            /* add the force to central atom */
            atomicAdd(&data->results[central_atom_index].force[0], force * (central_atom.x - atom_1.x));
            atomicAdd(&data->results[central_atom_index].force[1], force * (central_atom.y - atom_1.y));
            atomicAdd(&data->results[central_atom_index].force[2], force * (central_atom.z - atom_1.z));
        }
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
__host__ int32_t find(uint64_t *elements, uint64_t length, uint64_t element){
    // use a binary search to find the element in the array
    uint64_t left = 0;
    uint64_t right = length - 1;
    while (left <= right) {
        uint64_t mid = (left + right) / 2;
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
    uint64_t length, 
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
    uint64_t elements_presence[MAX_ELEMENTS + 1] = {0}; // bucket sort :)
    uint64_t maximum_atomic_number = 0;
    for (uint64_t i = 0; i < length; ++i) {
        h_atoms[i].element = (uint64_t)atoms[i][0];
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
    uint64_t *sorted_elements = (uint64_t *)malloc((maximum_atomic_number + 1) * sizeof(uint64_t));
    if (sorted_elements == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for sorted elements\n");
        free(h_atoms);
        exit(EXIT_FAILURE);
    }
    // sort the elements in ascending order
    uint64_t num_elements = 0; // the number of elements in the sorted array
    for (uint64_t i = 0; i <= maximum_atomic_number; ++i) {
        if (elements_presence[i] == 1) {
            sorted_elements[num_elements] = i;
            debug("sorted_elements[%zu] = %zu\n", num_elements, i);
            num_elements++;
        }
    }
    h_data.num_elements = num_elements; // set the number of elements in the device data
    // assign the atom types
    uint64_t *atom_types = (uint64_t *)malloc(length * sizeof(uint64_t));
    if (atom_types == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for atom types\n");
        free(h_atoms);
        free(sorted_elements);
        exit(EXIT_FAILURE);
    }
    for (uint64_t i = 0; i < length; ++i) {
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
    uint64_t *d_atom_types;
    CHECK_CUDA(cudaMalloc((void **)&d_atom_types, length * sizeof(uint64_t)));
    CHECK_CUDA(cudaMemcpy(d_atom_types, atom_types, length * sizeof(uint64_t), cudaMemcpyHostToDevice));
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
    /* allocate memory for data.neighbors */
    neighbor_t *d_neighbors;
    CHECK_CUDA(cudaMalloc((void **)&d_neighbors, length * MAX_NEIGHBORS * sizeof(neighbor_t)));
    CHECK_CUDA(cudaMemset(d_neighbors, 0, length * MAX_NEIGHBORS * sizeof(neighbor_t)));
    h_data.neighbors = d_neighbors; // set the neighbors in the device data
    uint64_t *d_num_neighbors;
    CHECK_CUDA(cudaMalloc((void **)&d_num_neighbors, length * sizeof(uint64_t)));
    CHECK_CUDA(cudaMemset(d_num_neighbors, 0, length * sizeof(uint64_t))); // initialize the max number of neighbors array to zero
    h_data.num_neighbors = d_num_neighbors; // set the max number of neighbors in the device data
    h_data.max_num_neighbor = 0; // set the maximum number of neighbors to zero
    neighbor_t *d_CN_neighbors;
    CHECK_CUDA(cudaMalloc((void **)&d_CN_neighbors, length * MAX_NEIGHBORS * sizeof(neighbor_t)));
    CHECK_CUDA(cudaMemset(d_CN_neighbors, 0, length * MAX_NEIGHBORS * sizeof(neighbor_t))); // initialize the CN neighbors array to zero
    h_data.CN_neighbors = d_CN_neighbors; // set the CN neighbors in the device data
    uint64_t *d_num_CN_neighbors;
    CHECK_CUDA(cudaMalloc((void **)&d_num_CN_neighbors, length * sizeof(uint64_t)));
    CHECK_CUDA(cudaMemset(d_num_CN_neighbors, 0, length * sizeof(uint64_t))); // initialize the max number of CN neighbors array to zero
    real_t *d_dCNi_datom_i;
    CHECK_CUDA(cudaMalloc((void **)&d_dCNi_datom_i,3 *  length * sizeof(real_t)));
    CHECK_CUDA(cudaMemset(d_dCNi_datom_i, 0, 3 * length * sizeof(real_t))); // initialize the dCNi/datom_i array to zero
    h_data.dCNi_datom_i = d_dCNi_datom_i; // set the dCNi/datom_i in the device data
    h_data.num_CN_neighbors = d_num_CN_neighbors; // set the max number of CN neighbors in the device data
    h_data.max_num_CN_neighbor = 0; // set the maximum number of CN neighbors to zero
    /* initiate cell info and cutoff parameters */
    uint64_t total_cell_bias = 1;
    for(uint64_t i = 0; i < 3; ++i) {
        real_t length = 0.0f;
        for(uint64_t j = 0; j < 3; ++j) {
            h_data.cell[i][j] = cell[i][j]; // set the cell info in the device data
            length += cell[i][j] * cell[i][j]; // calculate the length of the cell vector
        }
        /* calculate max_cell_bias */
        h_data.max_cell_bias[i] = (uint64_t)ceilf(cutoff_radius / sqrtf(length))*2+1; // set the max cell bias in the device data
        printf("max_cell_bias[%zu] = %zu\n", i, h_data.max_cell_bias[i]);
        total_cell_bias *= h_data.max_cell_bias[i]; // calculate the total cell bias
    }
    h_data.cutoff = cutoff_radius; // set the cutoff radius in the device data
    h_data.coordination_number_cutoff = coordination_number_cutoff; // set the coordination number cutoff in the device data

    // initialize the device data
    device_data_t *d_data;
    CHECK_CUDA(cudaMalloc((void **)&d_data, sizeof(device_data_t)));
    CHECK_CUDA(cudaMemcpy(d_data, &h_data, sizeof(device_data_t), cudaMemcpyHostToDevice));

    // launch the kernel
    printf("launching coordination_number_kernel, size: %zu, %zu\n", length, length);
    coordination_number_kernel<<<length, length>>>(d_data); // launch the kernel to compute the coordination numbers
    CHECK_CUDA(cudaDeviceSynchronize()); // synchronize the device to ensure all threads are finished
    printf("launching adjust_coordination_number_kernel, size: %zu\n", length);
    adjust_coordination_number_kernel<<<1, length>>>(d_data); // launch the kernel to adjust the coordination numbers
    CHECK_CUDA(cudaDeviceSynchronize()); // synchronize the device to ensure all threads are finished
    printf("launching two_body_kernel, size: %zu, %zu\n", length, 512);
    two_body_kernel<<<length, 512>>>(d_data);
    CHECK_CUDA(cudaDeviceSynchronize()); // synchronize the device to ensure all threads are finished
    printf("launching three_body_kernel, size: %zu, %zu\n", length, 512);
    three_body_kernel<<<length, dim3(8, 8)>>>(d_data); // launch the kernel to compute the three body forces
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
    for (uint64_t i = 0; i < length; ++i) {
        // print the results
        printf("Atom %zu: energy: %f, force = (%.13f, %.13f, %.13f)\n", i, h_results[i].energy, h_results[i].force[0], h_results[i].force[1], h_results[i].force[2]);
        // accumulate the total energy
        total_energy += h_results[i].energy;
    }
    total_energy /= 4.0f; /* the energy of between two atoms is added to both of the atoms. So the totoal energy should be divided by 2 after adding all atomic energy */
    printf("Total energy = %.13f\n", total_energy);
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
    for(uint64_t i = 0; i < 10; ++i) {
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
        {20.0f, 0.0f, 0.0f},
        {0.0f, 20.0f, 0.0f},
        {0.0f, 0.0f, 20.0f}
    };
    for (uint64_t i = 0; i < 3; ++i) {
        for (uint64_t j = 0; j < 3; ++j) {
            cell[i][j] *= angstron_to_bohr; // convert to bohr
        }
    }
    real_t CN_cutoff_radius = 40.0f; // cutoff radius in bohr
    real_t cutoff_radius = 94.8683f;
    compute_dispersion_energy(atoms, 10, cell, cutoff_radius, CN_cutoff_radius);
    return 0;
}