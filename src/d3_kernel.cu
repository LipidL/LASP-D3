#include "d3_kernel.cuh"

__device__ real_t calculate_cell_volume(const real_t cell[3][3]) {
    // Calculate the volume of the cell using the determinant of the matrix formed by the cell vectors
    return cell[0][0] * (cell[1][1] * cell[2][2] - cell[1][2] * cell[2][1]) -
           cell[0][1] * (cell[1][0] * cell[2][2] - cell[1][2] * cell[2][0]) +
           cell[0][2] * (cell[1][0] * cell[2][1] - cell[1][1] * cell[2][0]);
}

/**
 * @brief this kernel is used to compute the coordination number of each atom in the system.
 * @note this kernel should be launched with a 1D grid of blocks, each block containing a 1D array of threads.
 * @note the number of blocks equals the number of atoms in the system.
 * @note and the number of threads in each block shouldn't exceed the number of atoms in the system
 * @note it's best to use a division of the number of atoms in the system as the number of threads in each block.
 * @note the number of threads in each block must not exceed MAX_BLOCK_SIZE.
 * @note the total_cell_bias should be precomputed at host
 */
__global__ void coordination_number_kernel(device_data_t *data) {
    uint64_t atom_1_index = blockIdx.x; // each block is responsible for one central atom
    uint64_t atom_1_type = data->atom_types[atom_1_index]; // type of the central atom
    atom_t atom_1 = data->atoms[atom_1_index]; // central atom
    uint64_t total_cell_bias = data->max_cell_bias[0] * data->max_cell_bias[1] * data->max_cell_bias[2]; // total number of cell bias
    /* a chunk of shared memory block is used to store neighbor indicies 
     each thread can have 1 entry */
    __shared__ uint64_t neighbor_flags[MAX_BLOCK_SIZE]; // shared memory for neighbor indices
    __shared__ uint64_t CN_neighbor_flags[MAX_BLOCK_SIZE]; // shared memory for CN neighbor indices
    /* initiate this block, each thread is responsible for a few entries */
    
    neighbor_flags[threadIdx.x] = 0; // initialize the neighbor flags to false
    CN_neighbor_flags[threadIdx.x] = 0; // initialize the CN neighbor flags to false
    __syncthreads(); // synchronize threads in the block
    
    /* local variables to reduce global memory access */
    neighbor_t neighbors[MAX_LOCAL_NEIGHBORS]; // array of neighbors for the central atom
    neighbor_t CN_neighbors[MAX_LOCAL_NEIGHBORS]; // array of neighbors for the CN calculation
    uint64_t neighbors_index = 0; // index of the neighbor in the neighbors array
    uint64_t CN_neighbors_index = 0; // index of the neighbor in the CN neighbors array
    const uint64_t mcb0 = data->max_cell_bias[0]; // maximum cell bias in x direction
    const uint64_t mcb1 = data->max_cell_bias[1]; // maximum cell bias in y direction
    const uint64_t mcb2 = data->max_cell_bias[2]; // maximum cell bias in z direction
    const real_t cell[3][3] = {
        {data->cell[0][0], data->cell[0][1], data->cell[0][2]},
        {data->cell[1][0], data->cell[1][1], data->cell[1][2]},
        {data->cell[2][0], data->cell[2][1], data->cell[2][2]}
    };
    const real_t CN_cutoff = data->coordination_number_cutoff;
    const real_t cutoff = data->cutoff;
    /* print some debug information */
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        debug("Coordination number kernel launched with %llu atoms, cell: \n", data->num_atoms);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                debug("%f ", cell[i][j]);
            }
            debug("\n");
        }
        debug("max_cell_bias: %llu %llu %llu\n", mcb0, mcb1, mcb2);
        debug("CN_cutoff: %f, cutoff: %f\n", CN_cutoff, cutoff);
    }
    /* distribute workload to threads */
    uint64_t start_index; // = threadIdx.x * workload_per_thread; // start index for this thread
    uint64_t end_index; // = min(start_index + workload_per_thread, data->num_atoms); // end index for this thread
    uint64_t start_bias_index;
    uint64_t end_bias_index;
    if (data->num_atoms >= blockDim.x) {
        /* if the number of atoms exceed number of threads, each thread process an atom, going through all possible bias indicies */
        uint64_t workload_per_thread = (data->num_atoms + blockDim.x - 1) / blockDim.x; // number of atoms per thread
        start_index = threadIdx.x * workload_per_thread; // start index for this thread
        end_index = min(start_index + workload_per_thread, data->num_atoms); // end index for this thread
        start_bias_index = 0;
        end_bias_index = total_cell_bias; // each thread is responsible for all cell biases
    } else {
        /* If the number of atoms is smaller than number of threads, multiple threads process one atom */
        /* determine the element assigned and rank within that element's threads */
        uint64_t threads_per_element_base = blockDim.x / data->num_atoms; // number of threads per atom
        uint64_t threads_per_element_remainder = blockDim.x % data->num_atoms; // remainder threads
        uint64_t num_elements_getting_extra_thread = threads_per_element_remainder; // number of threads getting an extra thread
        uint64_t threads_in_larger_groups_total = num_elements_getting_extra_thread * (threads_per_element_base + 1); // total number of threads in larger groups
        uint64_t current_assigned_element_id;
        uint64_t threads_working_on_my_element;
        uint64_t rank_in_element_thread_group;
        if (threadIdx.x < threads_in_larger_groups_total) {
            /* this thread is in a larger group */
            threads_working_on_my_element = threads_per_element_base + 1; // number of threads working on my element would be one more than the base
            current_assigned_element_id = threadIdx.x / threads_working_on_my_element; // which element this thread is assigned to
            rank_in_element_thread_group = threadIdx.x % threads_working_on_my_element; // the rank of this thread within the element's threads
        } else {
            /* this thread falls in later groups that have 'base' threads */
           threads_working_on_my_element = threads_per_element_base; // number of threads working on my element would be the base
            uint64_t thraeds_already_assigned_to_larger_groups = threads_in_larger_groups_total; // number of threads already assigned to larger groups
            uint64_t threadIdx_relative_to_smaller_groups = threadIdx.x - thraeds_already_assigned_to_larger_groups; // relative thread index in smaller groups
            current_assigned_element_id = threadIdx_relative_to_smaller_groups / threads_working_on_my_element + num_elements_getting_extra_thread; // which element this thread is assigned to
            rank_in_element_thread_group = threadIdx_relative_to_smaller_groups % threads_working_on_my_element; // the rank of this thread within the element's threads
        }
        start_index = current_assigned_element_id; // start index for this thread
        end_index = min(current_assigned_element_id + 1, data->num_atoms); // end index for this thread
        uint64_t bias_per_thread = total_cell_bias / threads_working_on_my_element; // number of biases per thread
        start_bias_index = rank_in_element_thread_group * bias_per_thread; // start bias index for this thread
        end_bias_index = min(start_bias_index + bias_per_thread, total_cell_bias); // end bias index for this thread
    }
    for(uint64_t atom_2_index = start_index; atom_2_index < end_index; ++atom_2_index) {
        atom_t atom_2_original = data->atoms[atom_2_index]; // surrounding atom
        for(uint64_t bias_index = start_bias_index; bias_index < end_bias_index; ++bias_index) {
            /* each thread is responsible for one atom pair, so the number of threads should be equal to num_atoms * total_cell_bias */
            int64_t x_bias = (bias_index % mcb0) - (mcb0/2); // x bias
            int64_t y_bias = ((bias_index / mcb0) % mcb1) - (mcb1/2); // y bias
            int64_t z_bias = (bias_index / (mcb0 * mcb1) % mcb2) - (mcb2/2); // z bias
            assert_(atom_2_index < data->num_atoms); // make sure the index is in bounds

            atom_t atom_2 = atom_2_original;
            /* translate atom_2 due to periodic boundaries */
            atom_2.x += x_bias * cell[0][0] + y_bias * cell[1][0] + z_bias * cell[2][0]; // translate in x direction
            atom_2.y += x_bias * cell[0][1] + y_bias * cell[1][1] + z_bias * cell[2][1]; // translate in y direction
            atom_2.z += x_bias * cell[0][2] + y_bias * cell[1][2] + z_bias * cell[2][2]; // translate in z direction
            /* calculate the distance between the two atoms */
            real_t distance = sqrtf(powf(atom_1.x - atom_2.x, 2) + powf(atom_1.y - atom_2.y, 2) + powf(atom_1.z - atom_2.z, 2));
            /* if the distance is within cutoff range, update neighbor_flags */
            if (distance <= CN_cutoff && distance > 0.0f) {
                if (CN_neighbors_index < MAX_LOCAL_NEIGHBORS) {
                    CN_neighbors[CN_neighbors_index].index = atom_2_index; // set the index of the neighbor atom
                    CN_neighbors[CN_neighbors_index].distance = distance; // set the distance to the neighbor atom
                    CN_neighbors[CN_neighbors_index].atom = atom_2; // set the atom data of the neighbor atom
                    CN_neighbors_index++; // increment the index of the neighbor
                } else {
                    printf("Warning: too many CN neighbors for atom %llu and atom %llu\n", atom_1_index, atom_2_index); // too many neighbors, skip this one
                }
            }
            if (distance <= cutoff && distance > 0.0f) {
                if (neighbors_index < MAX_LOCAL_NEIGHBORS) {
                    neighbors[neighbors_index].index = atom_2_index; // set the index of the neighbor atom
                    neighbors[neighbors_index].distance = distance; // set the distance to the neighbor atom
                    neighbors[neighbors_index].atom = atom_2; // set the atom data of the neighbor atom
                    neighbors_index++; // increment the index of the neighbor
                } else {
                    printf("Warning: too many neighbors for atom %llu and atom %llu\n", atom_1_index, atom_2_index); // too many neighbors, skip this one
                }
            }
        }
    }
    CN_neighbor_flags[threadIdx.x] = CN_neighbors_index; // mark the atom as a neighbor
    neighbor_flags[threadIdx.x] = neighbors_index; // mark the atom as a neighbor
    __syncthreads();
    /* now we need to convert entries in neighbor_flags to indicies in neighbors. */
    if (threadIdx.x == 0) {
        uint64_t sum = 0;
        uint64_t CN_sum = 0;
        for (uint64_t i = 0; i < blockDim.x; i++) {
            uint64_t temp = neighbor_flags[i];
            neighbor_flags[i] = sum;
            sum += temp;
            uint64_t CN_temp = CN_neighbor_flags[i];
            CN_neighbor_flags[i] = CN_sum;
            CN_sum += CN_temp;
        }
        sum = min(sum, data->max_neighbors); // make sure the sum doesn't exceed the maximum number of neighbors
        CN_sum = min(CN_sum, data->max_neighbors); // make sure the sum doesn't exceed the maximum number of CN neighbors
        data->num_neighbors[atom_1_index] = sum; // set the maximum number of neighbors for the central atom
        data->num_CN_neighbors[atom_1_index] = CN_sum; // set the maximum number of CN neighbors for the central atom
    }
    __syncthreads(); // Make sure all threads see the updated neighbor_flags
    /* now the indicies in neighbor_flags is the position to write in neighbors */
    real_t local_coordination_number = 0.0f; // local coordination number for the central atom
    for(uint64_t i = 0; i < CN_neighbors_index; ++i) {
        real_t distance = CN_neighbors[i].distance; // distance to the neighbor atom
        assert_(distance <= CN_cutoff); // make sure the distance is in bounds
        /* if the distance is within cutoff range, update neighbors */
        uint64_t neighbor_index = CN_neighbor_flags[threadIdx.x]; // index of the neighbor in the neighbors array
        uint64_t atom_2_index = CN_neighbors[i].index; // index of the second atom in the pair
        uint64_t atom_2_type = data->atom_types[atom_2_index]; // type of the surrounding atom
        neighbor_t *data_neighbors = &data->CN_neighbors[atom_1_index * data->max_neighbors]; // pointer to the neighbors array for the central atom
        if (neighbor_index + i >= data->max_neighbors) {
            data->status = data->status | COMPUTE_NEIGHBOR_LIST_OVERFLOW; // set the status to indicate that the maximum number of neighbors is exceeded
            break;
        }
        assert_(data_neighbors[neighbor_index+i].index == 0); // make sure the index is not already set
        assert_(data_neighbors[neighbor_index+i].distance == 0); // make sure the distance is not already set
        data_neighbors[neighbor_index+i] = CN_neighbors[i];
        /* compute the coordination number and add to the CN of atom 1 and atom 2 */
        real_t covalent_radii_1 = data->rcov[atom_1_type];
        real_t covalent_radii_2 = data->rcov[atom_2_type];
        /* eq 15 in Grimme et al. 2010
        $CN^A = \sum_{B \neq A}^{N} \sqrt{1}{1+exp(-k_1(k_2(R_{A,cov}+R_{B,cov})/r_{AB}-1))}$ */
        real_t exp = expf(-K1*((covalent_radii_1 + covalent_radii_2)/distance - 1.0f)); // $\exp(-k_1*(\frac{R_A+R_b}{r_{ab}}-1))$
        real_t tanh_value = tanhf(CN_cutoff - distance); // $\tanh(CN_cutoff - r_{ab})$
        real_t smooth_cutoff = powf(tanh_value, 3); // $\tanh^3(CN_cutoff- r_{ab})})$, this is a smooth cutoff function added in LASP code.
        real_t d_smooth_cutoff_dr = 3.0f * powf(tanh_value, 2) * (1.0f - powf(tanh_value,2)) * (-1.0f); // derivative of the smooth cutoff function with respect to distance       
        real_t coordination_number = 1.0f/(1.0f+exp) * smooth_cutoff; // the covalent radii in input table have already taken K2 coefficient into onsideration
        real_t dCN_datom = powf(1.0f+exp,-2.0f)*(-K1)*exp*(covalent_radii_1 + covalent_radii_2)*powf(distance, -3.0f) * smooth_cutoff + d_smooth_cutoff_dr * 1.0f/(1.0f+exp) / distance; // dCN_ij/dr_ij * 1/r_ij
        data_neighbors[neighbor_index+i].dCN_dr = dCN_datom; // set the dCN/dr for the neighbor atom
        // increment the data.coordination_number array for both atoms
        local_coordination_number += coordination_number; // add the coordination number to the local coordination number
    }
    /* now the coordination number of the central atom is stored in local_coordination_number, add it back to global memory. */
    atomicAdd(&data->coordination_numbers[atom_1_index], local_coordination_number); // accumulate the coordination number for the central atom
    for (uint64_t i = 0; i < neighbors_index; ++i) {
        assert_(neighbors[i].distance <= cutoff);
        /* if the distance is within cutoff range, update neighbors */
        uint64_t neighbor_index = neighbor_flags[threadIdx.x]; // index of the neighbor in the neighbors array
        if (neighbor_index + i >= data->max_neighbors) {
            data->status = data->status | COMPUTE_NEIGHBOR_LIST_OVERFLOW; // set the status to indicate that the maximum number of neighbors is exceeded
            break;
        }
        neighbor_t *data_neighbors = &data->neighbors[atom_1_index * data->max_neighbors]; // pointer to the neighbors array for the central atom
        assert_(neighbor_index + i < data->max_neighbors); // make sure the index is in bounds
        assert_(data_neighbors[neighbor_index+i].index == 0); // make sure the index is not already set
        data_neighbors[neighbor_index+i] = neighbors[i];
    }
    return; // return from the kernel
}
__global__ void print_coordination_number_kernel(device_data_t *data) {
    if (threadIdx.x == 0) {
        printf("Coordination numbers:\n");
        for (uint64_t i = 0; i < data->num_atoms; ++i) {
            printf("Atom %llu, element: %d: %f\n", i, data->atoms[i].element, data->coordination_numbers[i]);
        }
    }
} // print coordination number kernel

/**
 * @brief this kernel is used to compute the two-body interactions between atoms in the system.
 * @brief i.e the energy and two-atom part of force
 * @note this kernel should be launched with a 1D grid of blocks, each block containining a 1D array of threads.
 * @note the number of blocks should be equal to the number of atoms in the system.
 * @note the number of threads in each block can be any value, but it's better to set a value smaller than the number of neighbors.
 */
__global__ void two_body_kernel(device_data_t *data) {
    extern __shared__ real_t dE_dCN_cache[]; // shared memory for dE/dCN cache
    __shared__ real_t energy_cache;
    if (threadIdx.x == 0) {
        energy_cache = 0.0f; // initialize the energy cache to 0
    }
    __syncthreads(); // synchronize threads in the block
    real_t dE_dCN = 0.0f; // derivative of energy with respect to coordination number
    uint64_t atom_1_index = blockIdx.x; // each block is responsible for one central atom
    atom_t atom_1 = data->atoms[atom_1_index]; // central atom
    /* local variables to reduce calls to `atomicAdd` */
    real_t local_energy = 0.0f; // local energy for the central atom
    real_t local_force_central[3] = {0.0f, 0.0f, 0.0f}; // local force for the central atom
    real_t local_stress[9] = {0.0f}; // local stress matrix
    assert_(data->num_neighbors[atom_1_index] <= data->max_neighbors); // make sure the number of neighbors is in bounds
    if (atom_1_index != data->num_atoms - 1) {
        assert_(data->neighbors[atom_1_index * data->max_neighbors + data->num_neighbors[atom_1_index]].index == 0); // make sure the last entry is empty
    }
    uint64_t num_neighbors = data->num_neighbors[atom_1_index]; // number of neighbors for the central atom
    uint64_t workload_per_thread = (num_neighbors + blockDim.x - 1) / blockDim.x; // number of neighbors per thread
    uint64_t start_index = threadIdx.x * workload_per_thread; // start index for the thread
    uint64_t end_index = min((threadIdx.x + 1) * workload_per_thread, num_neighbors); // end index for the thread
    real_t energy_conpensate = 0.0f; // energy compensation to improve numerical stability
    real_t dE_dCN_conpensate = 0.0f; // derivative of energy with respect to coordination number compensation
    real_t force_conpensate[3] = {0.0f, 0.0f, 0.0f}; // force compensation to improve numerical stability
    real_t stress_conpensate[9] = {0.0f}; // stress compensation to improve numerical stability
    /* calculate cell volume */
    real_t cell_volume = calculate_cell_volume(data->cell);
    for (uint64_t neighbor_index = start_index; neighbor_index < end_index; ++neighbor_index) {
        neighbor_t neighbor = data->neighbors[atom_1_index * data->max_neighbors + neighbor_index]; // neighbor atom
        uint64_t atom_2_index = neighbor.index; // index of the second atom in the pair
        atom_t atom_2 = neighbor.atom; // surrounding atom
        real_t distance = neighbor.distance; // distance to the neighbor atom
        real_t coordination_number_1 = data->coordination_numbers[atom_1_index];
        real_t coordination_number_2 = data->coordination_numbers[atom_2_index];
        uint64_t atom_1_type = data->atom_types[atom_1_index];
        uint64_t atom_2_type = data->atom_types[atom_2_index];
        /* calculate the coordination number based on dispersion coefficient
            formula: $C_6^{ij} = Z/W$ 
            where $Z = \sum_{a,b}C_{6,ref}^{i,j}L_{a,b}$
            $W = \sum_{a,b}L_{a,b}$
            $L_{a,b} = \exp(-k3((CN^A-CN^A_{ref,a})^2 + (CN^B-CN^B_{ref,b})^2))$ */
        real_t exponent_args[NUM_REF_C6 * NUM_REF_C6] = {0.0f}; // array to store the exponent arguments for L_ij
        real_t CN_ref_1[NUM_REF_C6 * NUM_REF_C6] = {0.0f}; // array to store the CNref values for atom_1
        real_t c6ab_ref[NUM_REF_C6 * NUM_REF_C6] = {0.0f}; // array to store the C6ab_ref values
        uint16_t valid_term_count = 0; // count of valid terms in the C6ref arrays
        real_t max_exponent_arg = -FLT_MAX; // maximum exponent argument for L_ij
        for (uint64_t i = 0; i < NUM_REF_C6; ++i) {
            for (uint64_t j = 0; j < NUM_REF_C6; ++j) {
                /* find the C6ref */
                uint64_t stride_1 = data->num_elements * NUM_REF_C6 * NUM_REF_C6  * NUM_C6AB_ENTRIES;
                uint64_t stride_2 = NUM_REF_C6 * NUM_REF_C6  * NUM_C6AB_ENTRIES;
                uint64_t stride_3 = NUM_REF_C6  * NUM_C6AB_ENTRIES;
                uint64_t stride_4 = NUM_C6AB_ENTRIES;
                uint64_t index = atom_1_type * stride_1 + atom_2_type * stride_2 + i * stride_3 + j * stride_4;
                real_t c6_ref = data->c6_ab_ref[index + 0];
                /* these entries could be -1.0f if they are not valid, but at least one should be valid*/
                real_t coordination_number_ref_1 = data->c6_ab_ref[index + 1];
                real_t coordination_number_ref_2 = data->c6_ab_ref[index + 2];
                if (coordination_number_ref_1 - (-1.0f) > 1e-5f && coordination_number_ref_2 - (-1.0f) > 1e-5f) {
                    /* if both coordination numbers are valid, we can use them */
                    exponent_args[valid_term_count] = -K3 * (powf(coordination_number_1 - coordination_number_ref_1, 2) + powf(coordination_number_2 - coordination_number_ref_2, 2));
                    CN_ref_1[valid_term_count] = coordination_number_ref_1; // store the C6ref for atom_1
                    c6ab_ref[valid_term_count] = c6_ref; // store the C6ab_ref
                    max_exponent_arg = (max_exponent_arg > exponent_args[valid_term_count]) ? max_exponent_arg : exponent_args[valid_term_count]; // update the maximum exponent argument
                    valid_term_count++;
                }
            }
        }
        real_t Z = 0.0f;
        real_t W = 0.0f;
        real_t c_ref_L_ij = 0.0f;
        real_t c_ref_dL_ij_1 = 0.0f;
        real_t dL_ij_1 = 0.0f;

        if (valid_term_count > 0) {
            for (uint16_t i = 0; i < valid_term_count; ++i) {
                /* calculate L_ij for the valid terms */
                real_t L_ij = expf(exponent_args[i] - max_exponent_arg); // normalize the exponent argument
                real_t dL_ij_1_part = -2.0f * K3 * (coordination_number_1 - CN_ref_1[i]) * L_ij; // dL_ij/dCN_1
                Z += c6ab_ref[i] * L_ij; // accumulate the value of Z
                W += L_ij; // accumulate the value of W
                c_ref_L_ij += c6ab_ref[i] * L_ij; // accumulate the value of c_ref_L_ij
                c_ref_dL_ij_1 += c6ab_ref[i] * dL_ij_1_part; // accumulate the value of c_ref_dL_ij_1
                dL_ij_1 += dL_ij_1_part; // accumulate the value of dL_ij_1
            }
        } else {
            printf("Warning: no valid C6ab terms found for atom pair (%llu, %llu)\n", atom_1_index, atom_2_index);
        }

        real_t L_ij = W;
        real_t dC6ab_dCN_1 = (L_ij * L_ij > 0.0f) ? (c_ref_dL_ij_1*L_ij - c_ref_L_ij * dL_ij_1) / powf(L_ij,2.0f) : 0.0f; // avoid division by zero
        if (isnan(dC6ab_dCN_1) || isinf(dC6ab_dCN_1)) {
            printf("Error: dC6ab/dCN_1 is NaN or Inf for atom pair (%llu, %llu)\n", atom_1_index, atom_2_index);
            printf("Z: %f, W: %f, c_ref_L_ij: %f, c_ref_dL_ij_1: %f, dL_ij_1: %f\n", Z, W, c_ref_L_ij, c_ref_dL_ij_1, dL_ij_1);
            dC6ab_dCN_1 = 0.0f; // reset to 0.0f if it's NaN or Inf
        }

        real_t c6_ab = (W > 0.0f) ? Z / W : 0.0f; // avoid division by zero
        if (isnan(c6_ab) || isinf(c6_ab)) {
            printf("Error: C6_ab is NaN or Inf for atom pair (%llu, %llu)\n", atom_1_index, atom_2_index);
            printf("Z: %f, W: %f, c_ref_L_ij: %f, c_ref_dL_ij_1: %f, dL_ij_1: %f\n", Z, W, c_ref_L_ij, c_ref_dL_ij_1, dL_ij_1);
            c6_ab = 0.0f; // reset to 0.0f if it's NaN or Inf
        }
        /* calculate c8_ab by $C_8^{AB} = 3C_6^{AB}\sqrt{Q^AQ^B}$*/
        real_t r2r4_1 = data->r2r4[atom_1_type];
        real_t r2r4_2 = data->r2r4[atom_2_type];
        real_t c8_ab = 3.0f * c6_ab * r2r4_1 * r2r4_2; // the value in r2r4 is already squared
        real_t dC8ab_dCN_1 = 3.0f * dC6ab_dCN_1 * r2r4_1 * r2r4_2; // dC8ab/dCN_1
        /* acquire the cutoff radius between the two atoms */
        real_t cutoff_radius = data->r0ab[atom_1_type*data->num_elements + atom_2_type];
        /* calculate distance^6 and distance^8 using fast power algorithm*/
        real_t distance_2 = distance * distance; // distance^2
        real_t distance_3 = distance_2 * distance; // distance^3
        real_t distance_4 = distance_2 * distance_2; // distance^4
        real_t distance_6 = distance_3 * distance_3; // distance^6
        real_t distance_8 = distance_4 * distance_4; // distance^8
        real_t distance_10 = distance_6 * distance_4; // distance^10
        /* calculate the dampling function as Grimme et al. 2010, eq4 */
        real_t f_dn_6 = 1/(1+6.0f*powf(distance/(SR_6*cutoff_radius), -ALPHA_N(6.0f)));
        real_t f_dn_8 = 1/(1+6.0f*powf(distance/(SR_8*cutoff_radius), -ALPHA_N(8.0f)));
        /* calculate the dispersion enegry as Grimme et al. 2010, eq3 */
        real_t dispersion_energy_6 = S6*(c6_ab/distance_6)*f_dn_6;
        real_t dispersion_energy_8 = S8*(c8_ab/distance_8)*f_dn_8;
        // printf("dispersion energy between atoms (%llu,%llu): %f\n",atom_1_index,atom_2_index,dispersion_energy_6 + dispersion_energy_8);
        real_t dispersion_energy = dispersion_energy_6 + dispersion_energy_8;
        dispersion_energy /= 2.0f; // divide by 2 because each pair is counted twice (once for each atom)
        /* add the energy back to results, use Kahan summation for higher accuracy */
        {
            real_t y = dispersion_energy - energy_conpensate; // calculate the difference
            real_t t = local_energy + y; // add the difference to the local energy
            energy_conpensate = (t - local_energy) - y; // calculate the compensation for the next iteration
            local_energy = t; // update the local energy
        }
        
        /* accumulate dEij/dCNi to local variable, use Kagan summation for better accuracy */
        {
            // real_t y = S6 * (f_dn_6 / distance_6) * dC6ab_dCN_1 + S8 * (f_dn_8 / distance_8) * dC8ab_dCN_1 - dE_dCN_conpensate; // calculate the difference
            real_t y = ((S6 * f_dn_6 * distance_2 * dC6ab_dCN_1 + S8 * f_dn_8 * dC8ab_dCN_1) / distance_8) - dE_dCN_conpensate; // calculate the difference
            real_t t = dE_dCN + y; // add the difference to the local dE/dCN
            dE_dCN_conpensate = (t - dE_dCN) - y; // calculate the compensation for the next iteration
            dE_dCN = t; // update the local dE/dCN
        }
       
        /* the first entry of two-body force
         $F_a = S_n C_n^{ab} f_{d,n}(r_{ab}) \frac{\partial}{\partial r_a} r_{ab}^{-n}$
         $F_a = S_n C_n^{ab} f_{d,n}(r_{ab}) * (-n)r_{ab}^{-n-2} * \uparrow{r_{ab}}$ */
        real_t force = 0.0f; // dE/dr * 1/r
        force += S6 * c6_ab * f_dn_6 * (-6.0f) / distance_8; // dE_6/dr * 1/r
        force += S8 * c8_ab * f_dn_8 * (-8.0f) / distance_10; // dE_8/dr * 1/r
        /* the second entry of two-body force 
         $F_a = S_n C_n^{ab} r_{ab}^{-n} \frac{\partial}{\partial r_a} f_{d,n}(r_{ab})$
         $F_a = S_n C_n^{ab} r_{ab}^{-n} -f_{d,n}^2 * (6*(-\alpha_n)*(r_{ab}/{S_{r,n}R_0^{AB}})^(-\alpha_n - 1) * 1/(S_{r,n}R_0^{AB}})) / r_ab \uparrow{r_{ab}}$*/
        force += S6 * c6_ab / distance_6 * (-f_dn_6 * f_dn_6) * (6.0f * (-ALPHA_N(6.0f))* powf(distance/(SR_6*cutoff_radius), -ALPHA_N(6.0f) - 1.0f) / (SR_6*cutoff_radius)) / distance; // dE_6/dr * 1/r
        force += S8 * c8_ab / distance_8 * (-f_dn_8 * f_dn_8) * (6.0f * (-ALPHA_N(8.0f))* powf(distance/(SR_8*cutoff_radius), -ALPHA_N(8.0f) - 1.0f) / (SR_8*cutoff_radius)) / distance; // dE_8/dr * 1/r
        /* accumulate force for the central atom, use Kahan summation for better accuracy */
        {
            real_t y0 = force * (atom_1.x - atom_2.x) - force_conpensate[0]; // calculate the difference for x component
            real_t t0 = local_force_central[0] + y0; // add the difference to the local force
            force_conpensate[0] = (t0 - local_force_central[0]) - y0; // calculate the compensation for the next iteration
            local_force_central[0] = t0; // update the local force for x component

            real_t y1 = force * (atom_1.y - atom_2.y) - force_conpensate[1]; // calculate the difference for y component
            real_t t1 = local_force_central[1] + y1; // add the difference to the local force
            force_conpensate[1] = (t1 - local_force_central[1]) - y1; // calculate the compensation for the next iteration
            local_force_central[1] = t1; // update the local force for y component

            real_t y2 = force * (atom_1.z - atom_2.z) - force_conpensate[2]; // calculate the difference for z component
            real_t t2 = local_force_central[2] + y2; // add the difference to the local force
            force_conpensate[2] = (t2 - local_force_central[2]) - y2; // calculate the compensation for the next iteration
            local_force_central[2] = t2; // update the local force for z component
        }

        /* accumulate stress to local matrix instead of directly using atomicAdd, divide by 2 becuase the same entry will be calculated twice (when atom_2 is the central atom), use Kahan summation for better accuracy */
        {
            real_t y0 = -1.0f * (atom_1.x - atom_2.x) * force * (atom_1.x - atom_2.x)/2.0f / cell_volume - stress_conpensate[0]; // calculate the difference for stress_xx
            real_t t0 = local_stress[0*3+0] + y0; // add the difference to the local stress
            stress_conpensate[0] = (t0 - local_stress[0*3+0]) - y0; // calculate the compensation for the next iteration
            local_stress[0*3+0] = t0; // update the local stress for stress_xx

            real_t y1 = -1.0f * (atom_1.x - atom_2.x) * force * (atom_1.y - atom_2.y)/2.0f / cell_volume - stress_conpensate[1]; // calculate the difference for stress_xy
            real_t t1 = local_stress[0*3+1] + y1; // add the difference to the local stress
            stress_conpensate[1] = (t1 - local_stress[0*3+1]) - y1; // calculate the compensation for the next iteration
            local_stress[0*3+1] = t1; // update the local stress for stress_xy

            real_t y2 = -1.0f * (atom_1.x - atom_2.x) * force * (atom_1.z - atom_2.z)/2.0f / cell_volume - stress_conpensate[2]; // calculate the difference for stress_xz
            real_t t2 = local_stress[0*3+2] + y2; // add the difference to the local stress
            stress_conpensate[2] = (t2 - local_stress[0*3+2]) - y2; // calculate the compensation for the next iteration
            local_stress[0*3+2] = t2; // update the local stress for stress_xz

            real_t y3 = -1.0f * (atom_1.y - atom_2.y) * force * (atom_1.x - atom_2.x)/2.0f / cell_volume - stress_conpensate[3]; // calculate the difference for stress_yx
            real_t t3 = local_stress[1*3+0] + y3; // add the difference to the local stress
            stress_conpensate[3] = (t3 - local_stress[1*3+0]) - y3; // calculate the compensation for the next iteration
            local_stress[1*3+0] = t3; // update the local stress for stress_yx

            real_t y4 = -1.0f * (atom_1.y - atom_2.y) * force * (atom_1.y - atom_2.y)/2.0f / cell_volume - stress_conpensate[4]; // calculate the difference for stress_yy
            real_t t4 = local_stress[1*3+1] + y4; // add the difference to the local stress
            stress_conpensate[4] = (t4 - local_stress[1*3+1]) - y4; // calculate the compensation for the next iteration
            local_stress[1*3+1] = t4; // update the local stress for stress_yy

            real_t y5 = -1.0f * (atom_1.y - atom_2.y) * force * (atom_1.z - atom_2.z)/2.0f / cell_volume - stress_conpensate[5]; // calculate the difference for stress_yz
            real_t t5 = local_stress[1*3+2] + y5; // add the difference to the local stress
            stress_conpensate[5] = (t5 - local_stress[1*3+2]) - y5; // calculate the compensation for the next iteration
            local_stress[1*3+2] = t5; // update the local stress for stress_yz

            real_t y6 = -1.0f * (atom_1.z - atom_2.z) * force * (atom_1.x - atom_2.x)/2.0f / cell_volume - stress_conpensate[6]; // calculate the difference for stress_zx
            real_t t6 = local_stress[2*3+0] + y6; // add the difference to the local stress
            stress_conpensate[6] = (t6 - local_stress[2*3+0]) - y6; // calculate the compensation for the next iteration
            local_stress[2*3+0] = t6; // update the local stress for stress_zx

            real_t y7 = -1.0f * (atom_1.z - atom_2.z) * force * (atom_1.y - atom_2.y)/2.0f / cell_volume - stress_conpensate[7]; // calculate the difference for stress_zy
            real_t t7 = local_stress[2*3+1] + y7; // add the difference to the local stress
            stress_conpensate[7] = (t7 - local_stress[2*3+1]) - y7; // calculate the compensation for the next iteration
            local_stress[2*3+1] = t7; // update the local stress for stress_zy

            real_t y8 = -1.0f * (atom_1.z - atom_2.z) * force * (atom_1.z - atom_2.z)/2.0f / cell_volume - stress_conpensate[8]; // calculate the difference for stress_zz
            real_t t8 = local_stress[2*3+2] + y8; // add the difference to the local stress
            stress_conpensate[8] = (t8 - local_stress[2*3+2]) - y8; // calculate the compensation for the next iteration
            local_stress[2*3+2] = t8; // update the local stress for stress_zz
        }
    }

    dE_dCN_cache[threadIdx.x] = dE_dCN; // store the value in shared memory
    /* accumulate energy to shared cache */
    atomicAdd(&energy_cache, local_energy);

    /* accumulate force for the central atom */
    atomicAdd(&data->forces[atom_1_index*3+0], local_force_central[0]);
    atomicAdd(&data->forces[atom_1_index*3+1], local_force_central[1]);
    atomicAdd(&data->forces[atom_1_index*3+2], local_force_central[2]);

    /* accumulate local stress to global stress */
    for (uint64_t i = 0; i < 3; ++i) {
        for (uint64_t j = 0; j < 3; ++j) {
            atomicAdd(&data->stress[i*3+j], local_stress[i*3+j]);
        }
    }
    __syncthreads(); // make sure all threads see the updated dE_dCN_cache
    /* accumulate dE/dCN and energy_cache in shared memory */
    if (threadIdx.x == 0) {
        // accumulate energy from shared memory
        atomicAdd(data->energy, energy_cache);
        real_t dE_dCN_sum = 0.0f;
        for (uint64_t i = 0; i < blockDim.x; ++i) {
            dE_dCN_sum += dE_dCN_cache[i];
        }
        data->dE_dCN[atom_1_index] = dE_dCN_sum; // store the value in global memory
    }
}

/**
 * @brief this kernel is used to compute the three-body interactions between atoms in the system.
 * @brief i.e $\frac{\partial E_{ij}}{\partial r_{ik}}$ where $i$ is the central atom, $j$ is the first neighbor and $k$ is the second neighbor.
 * 
 * @note this kernel should be launched with a 1D grid of blocks, each block containining a 1D array of threads.
 * @note the number of blocks should be equal to the number of atoms in the system.
 * @note the number of threads in each block can be any value, 512 would be a good choice, but it could be smaller if your memory is limited.
 */
__global__ void three_body_kernel(device_data_t *data) {
    real_t cell_volume = calculate_cell_volume(data->cell);
    uint64_t atom_1_index = blockIdx.x; // each block is responsible for one central atom
    atom_t atom_1 = data->atoms[atom_1_index]; // central atom
    real_t dE_dCN = data->dE_dCN[atom_1_index]; // derivative of energy with respect to coordination number
    uint64_t num_neighbors = data->num_CN_neighbors[atom_1_index]; // number of neighbors for the central atom
    assert_(num_neighbors <= data->max_neighbors);
    uint64_t workload_per_thread = (num_neighbors + blockDim.x - 1) / blockDim.x; // number of neighbors per thread
    uint64_t start_index = threadIdx.x * workload_per_thread; // start index for the thread
    if (start_index > num_neighbors) {
        return; // if the start index is out of bounds, return
    }
    uint64_t end_index = min((threadIdx.x + 1) * workload_per_thread, num_neighbors); // end index for the thread
    /* when finding neighbors, each tread is responsible for a neighbor atom and they loop over supercell indicies.
    Therefore, the neighbors list is organized in a atom_index priored way.
    A typical layout is: neighbors[a]: {1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,...} where 1,2,3 represent atom index and different entries have different coordination due to supercell
    Therefore, we can cache the force and only use `atomicAdd` after accumulating all forces */
    real_t force_neighbor_a[3] = {0.0f, 0.0f, 0.0f}; // force cache for the neighbor atom during the following loop
    real_t force_central[3] = {0.0f, 0.0f, 0.0f}; // force cache for the central atom during the following loop
    real_t force_compensate_central[3] = {0.0f, 0.0f, 0.0f}; // force compensation for the central atom to improve numerical stability
    real_t stress[9] = {0.0f}; // stress cache for the central atom during the following loop
    real_t stress_compensate[9] = {0.0f}; // stress compensation for the central atom to improve numerical stability
    uint64_t current_neighbor_a_index = data->CN_neighbors[atom_1_index * data->max_neighbors + start_index].index; // index of the first neighbor atom
    for (uint64_t neighbor_index = start_index; neighbor_index < end_index; ++neighbor_index) {
        /* calculate dEi/drik = dEi/dCNi * dCNi/drik */
        neighbor_t neighbor_data = data->CN_neighbors[atom_1_index * data->max_neighbors + neighbor_index]; // neighbor atom
        uint64_t atom_k_index = neighbor_data.index; // index of the neighbor atom
        real_t dCN_dr = neighbor_data.dCN_dr; // dCN/dr for the neighbor atom
        atom_t atom_k = neighbor_data.atom; // surrounding atom
        if (atom_k_index != current_neighbor_a_index) {
            /* if the index is different, we need to accumulate force of current neighbor to cache */
            atomicAdd(&data->forces[current_neighbor_a_index * 3 + 0], force_neighbor_a[0]); // accumulate the force for the neighbor atom
            atomicAdd(&data->forces[current_neighbor_a_index * 3 + 1], force_neighbor_a[1]); // accumulate the force for the neighbor atom
            atomicAdd(&data->forces[current_neighbor_a_index * 3 + 2], force_neighbor_a[2]); // accumulate the force for the neighbor atom
            /* reset the force cache */
            current_neighbor_a_index = atom_k_index; // update the index
            force_neighbor_a[0] = 0.0f; // reset the force cache
            force_neighbor_a[1] = 0.0f; // reset the force cache
            force_neighbor_a[2] = 0.0f; // reset the force cache
        }
        real_t dE_drik = dE_dCN * dCN_dr; // dE/drik * 1/rik
        // if (neighbor_data.distance < 5.0f)
            // printf("atom (%lld,%lld), distance: %f, dE_drik: %f, dE_dCN: %f, dCN_dr: %f\n",atom_1_index, neighbor_data.index, neighbor_data.distance,dE_drik, dE_dCN, dCN_dr);
        /* accumulate force for the central atom */
        // force_central[0] += dE_drik * (atom_1.x - atom_k.x);
        // force_central[1] += dE_drik * (atom_1.y - atom_k.y);
        // force_central[2] += dE_drik * (atom_1.z - atom_k.z);
        /* use Kahan summation to improve numerical stability */
        {
            real_t y0 = dE_drik * (atom_1.x - atom_k.x) - force_compensate_central[0]; // calculate the difference for x component
            real_t t0 = force_central[0] + y0; // add the difference to the local force
            force_compensate_central[0] = (t0 - force_central[0]) - y0; // calculate the compensation for the next iteration
            force_central[0] = t0; // update the local force for x component

            real_t y1 = dE_drik * (atom_1.y - atom_k.y) - force_compensate_central[1]; // calculate the difference for y component
            real_t t1 = force_central[1] + y1; // add the difference to the local force
            force_compensate_central[1] = (t1 - force_central[1]) - y1; // calculate the compensation for the next iteration
            force_central[1] = t1; // update the local force for y component

            real_t y2 = dE_drik * (atom_1.z - atom_k.z) - force_compensate_central[2]; // calculate the difference for z component
            real_t t2 = force_central[2] + y2; // add the difference to the local force
            force_compensate_central[2] = (t2 - force_central[2]) - y2; // calculate the compensation for the next iteration
            force_central[2] = t2; // update the local force for z component
        }
        /* accumulate force for the neighbor atom */
        force_neighbor_a[0] += -dE_drik * (atom_1.x - atom_k.x);
        force_neighbor_a[1] += -dE_drik * (atom_1.y - atom_k.y);
        force_neighbor_a[2] += -dE_drik * (atom_1.z - atom_k.z);
        /* accumulate stress */
        // stress[0 * 3 + 0] += -1.0f * (atom_1.x - atom_k.x) * dE_drik * (atom_1.x - atom_k.x); // stress_xx
        // stress[0 * 3 + 1] += -1.0f * (atom_1.x - atom_k.x) * dE_drik * (atom_1.y - atom_k.y); // stress_xy
        // stress[0 * 3 + 2] += -1.0f * (atom_1.x - atom_k.x) * dE_drik * (atom_1.z - atom_k.z); // stress_xz
        // stress[1 * 3 + 0] += -1.0f * (atom_1.y - atom_k.y) * dE_drik * (atom_1.x - atom_k.x); // stress_yx
        // stress[1 * 3 + 1] += -1.0f * (atom_1.y - atom_k.y) * dE_drik * (atom_1.y - atom_k.y); // stress_yy
        // stress[1 * 3 + 2] += -1.0f * (atom_1.y - atom_k.y) * dE_drik * (atom_1.z - atom_k.z); // stress_yz
        // stress[2 * 3 + 0] += -1.0f * (atom_1.z - atom_k.z) * dE_drik * (atom_1.x - atom_k.x); // stress_zx
        // stress[2 * 3 + 1] += -1.0f * (atom_1.z - atom_k.z) * dE_drik * (atom_1.y - atom_k.y); // stress_zy
        // stress[2 * 3 + 2] += -1.0f * (atom_1.z - atom_k.z) * dE_drik * (atom_1.z - atom_k.z); // stress_zz
        /* use Kahan summation to improve numerical stability */
        {
            real_t y0 = -1.0f * (atom_1.x - atom_k.x) * dE_drik * (atom_1.x - atom_k.x) - stress_compensate[0]; // calculate the difference for stress_xx
            real_t t0 = stress[0 * 3 + 0] + y0; // add the difference to the local stress
            stress_compensate[0] = (t0 - stress[0 * 3 + 0]) - y0; // calculate the compensation for the next iteration
            stress[0 * 3 + 0] = t0; // update the local stress for stress_xx

            real_t y1 = -1.0f * (atom_1.x - atom_k.x) * dE_drik * (atom_1.y - atom_k.y) - stress_compensate[1]; // calculate the difference for stress_xy
            real_t t1 = stress[0 * 3 + 1] + y1; // add the difference to the local stress
            stress_compensate[1] = (t1 - stress[0 * 3 + 1]) - y1; // calculate the compensation for the next iteration
            stress[0 * 3 + 1] = t1; // update the local stress for stress_xy

            real_t y2 = -1.0f * (atom_1.x - atom_k.x) * dE_drik * (atom_1.z - atom_k.z) - stress_compensate[2]; // calculate the difference for stress_xz
            real_t t2 = stress[0 * 3 + 2] + y2; // add the difference to the local stress
            stress_compensate[2] = (t2 - stress[0 * 3 + 2]) - y2; // calculate the compensation for the next iteration
            stress[0 * 3 + 2] = t2; // update the local stress for stress_xz

            real_t y3 = -1.0f * (atom_1.y - atom_k.y) * dE_drik * (atom_1.x - atom_k.x) - stress_compensate[3]; // calculate the difference for stress_yx
            real_t t3 = stress[1 * 3 + 0] + y3; // add the difference to the local stress
            stress_compensate[3] = (t3 - stress[1 * 3 + 0]) - y3; // calculate the compensation for the next iteration
            stress[1 * 3 + 0] = t3; // update the local stress for stress_yx

            real_t y4 = -1.0f * (atom_1.y - atom_k.y) * dE_drik * (atom_1.y - atom_k.y) - stress_compensate[4]; // calculate the difference for stress_yy
            real_t t4 = stress[1 * 3 + 1] + y4; // add the difference to the local stress
            stress_compensate[4] = (t4 - stress[1 * 3 + 1]) - y4; // calculate the compensation for the next iteration
            stress[1 * 3 + 1] = t4; // update the local stress for stress_yy

            real_t y5 = -1.0f * (atom_1.y - atom_k.y) * dE_drik * (atom_1.z - atom_k.z) - stress_compensate[5]; // calculate the difference for stress_yz
            real_t t5 = stress[1 * 3 + 2] + y5; // add the difference to the local stress
            stress_compensate[5] = (t5 - stress[1 * 3 + 2]) - y5; // calculate the compensation for the next iteration
            stress[1 * 3 + 2] = t5; // update the local stress for stress_yz

            real_t y6 = -1.0f * (atom_1.z - atom_k.z) * dE_drik * (atom_1.x - atom_k.x) - stress_compensate[6]; // calculate the difference for stress_zx
            real_t t6 = stress[2 * 3 + 0] + y6; // add the difference to the local stress
            stress_compensate[6] = (t6 - stress[2 * 3 + 0]) - y6; // calculate the compensation for the next iteration
            stress[2 * 3 + 0] = t6; // update the local stress for stress_zx

            real_t y7 = -1.0f * (atom_1.z - atom_k.z) * dE_drik * (atom_1.y - atom_k.y) - stress_compensate[7]; // calculate the difference for stress_zy
            real_t t7 = stress[2 * 3 + 1] + y7; // add the difference to the local stress
            stress_compensate[7] = (t7 - stress[2 * 3 + 1]) - y7; // calculate the compensation for the next iteration
            stress[2 * 3 + 1] = t7; // update the local stress for stress_zy

            real_t y8 = -1.0f * (atom_1.z - atom_k.z) * dE_drik * (atom_1.z - atom_k.z) - stress_compensate[8]; // calculate the difference for stress_zz
            real_t t8 = stress[2 * 3 + 2] + y8; // add the difference to the local stress
            stress_compensate[8] = (t8 - stress[2 * 3 + 2]) - y8; // calculate the compensation for the next iteration
            stress[2 * 3 + 2] = t8; // update the local stress for stress_zz
        }
    }
    /* accumulate force of last neighboring atom to local memory */
    atomicAdd(&data->forces[current_neighbor_a_index * 3 + 0], force_neighbor_a[0]); // accumulate the force for the central atom
    atomicAdd(&data->forces[current_neighbor_a_index * 3 + 1], force_neighbor_a[1]); // accumulate the force for the central atom
    atomicAdd(&data->forces[current_neighbor_a_index * 3 + 2], force_neighbor_a[2]); // accumulate the force for the central atom
    /* accumulate force of central atom */
    atomicAdd(&data->forces[atom_1_index * 3 + 0], force_central[0]); // accumulate the force for the central atom
    atomicAdd(&data->forces[atom_1_index * 3 + 1], force_central[1]); // accumulate the force for the central atom
    atomicAdd(&data->forces[atom_1_index * 3 + 2], force_central[2]); // accumulate the force for the central atom
    /* accumulate stress to local matrix instead of directly using atomicAdd */
    for (uint64_t i = 0; i < 3; ++i) {
        for (uint64_t j = 0; j < 3; ++j) {
            atomicAdd(&data->stress[i * 3 + j], stress[i * 3 + j] / cell_volume); // accumulate the stress for the central atom
        }
    }
}