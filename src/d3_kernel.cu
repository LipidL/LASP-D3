#include <assert.h>

#include "d3_kernel.cuh"
#include "d3_types.h"

__inline__ __device__ real_t calculate_cell_volume(const real_t cell[3][3]) {
    // Calculate the volume of the cell using the determinant of the matrix
    // formed by the cell vectors
    return cell[0][0] * (cell[1][1] * cell[2][2] - cell[1][2] * cell[2][1]) -
           cell[0][1] * (cell[1][0] * cell[2][2] - cell[1][2] * cell[2][0]) +
           cell[0][2] * (cell[1][0] * cell[2][1] - cell[1][1] * cell[2][0]);
}

__inline__ __device__ real_t warpReduceSum(real_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * @brief blockReduceSum for two_body_kernel. the entries that need to be added
 * are: dE_dCN, energy, force, stress
 */
__inline__ __device__ void blockReduceTwoBodyKernel(
    real_t dE_dCN, real_t energy, real_t force[3], real_t stress[9],
    real_t* dE_dCN_sum, real_t* energy_sum, real_t force_central_sum[3],
    real_t stress_central_sum[9]) {
    static __shared__ real_t shared_dE_dCN[32];
    static __shared__ real_t shared_energy[32];
    static __shared__ real_t shared_force[32 * 3];   // 3 components for force
    static __shared__ real_t shared_stress[32 * 9];  // 9 components for stress

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    dE_dCN = warpReduceSum(dE_dCN);  // reduce dE_dCN within the warp
    energy = warpReduceSum(energy);  // reduce energy within the warp
    for (int i = 0; i < 3; ++i) {
        force[i] =
            warpReduceSum(force[i]);  // reduce force components within the warp
    }
    for (int i = 0; i < 9; ++i) {
        stress[i] = warpReduceSum(
            stress[i]);  // reduce stress components within the warp
    }

    if (lane == 0) {
        shared_dE_dCN[warp_id] = dE_dCN;  // store dE_dCN in shared memory
        shared_energy[warp_id] = energy;  // store energy in shared memory
        for (int i = 0; i < 3; ++i) {
            shared_force[warp_id * 3 + i] =
                force[i];  // store force components in shared memory
        }
        for (int i = 0; i < 9; ++i) {
            shared_stress[warp_id * 9 + i] =
                stress[i];  // store stress components in shared memory
        }
    }
    __syncthreads();  // synchronize threads in the block
    if (warp_id == 0) {
        dE_dCN = (threadIdx.x < blockDim.x / warpSize)
                     ? shared_dE_dCN[lane]
                     : 0.0f;  // only the first warp writes to shared memory
        energy = (threadIdx.x < blockDim.x / warpSize)
                     ? shared_energy[lane]
                     : 0.0f;  // only the first warp writes to shared memory
        for (int i = 0; i < 3; ++i) {
            force[i] =
                (threadIdx.x < blockDim.x / warpSize)
                    ? shared_force[lane * 3 + i]
                    : 0.0f;  // only the first warp writes to shared memory
        }
        for (int i = 0; i < 9; ++i) {
            stress[i] =
                (threadIdx.x < blockDim.x / warpSize)
                    ? shared_stress[lane * 9 + i]
                    : 0.0f;  // only the first warp writes to shared memory
        }
        *dE_dCN_sum = warpReduceSum(dE_dCN);
        *energy_sum = warpReduceSum(energy);
        for (int i = 0; i < 3; ++i) {
            force_central_sum[i] = warpReduceSum(force[i]);
        }
        for (int i = 0; i < 9; ++i) {
            stress_central_sum[i] = warpReduceSum(stress[i]);
        }
    }
}

/**
 * @brief blockReduceSum for three_body_kernel. the entries that need to be
 * added are: force, stress
 */
__inline__ __device__ void blockReduceSumThreeBodyKernel(
    real_t force[3], real_t stress[9], real_t* force_central_sum,
    real_t* stress_central_sum) {
    static __shared__ real_t shared_force[32 * 3];   // 3 components for force
    static __shared__ real_t shared_stress[32 * 9];  // 9 components for stress
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    for (int i = 0; i < 3; ++i) {
        force[i] =
            warpReduceSum(force[i]);  // reduce force components within the warp
    }
    for (int i = 0; i < 9; ++i) {
        stress[i] = warpReduceSum(
            stress[i]);  // reduce stress components within the warp
    }
    if (lane == 0) {
        for (int i = 0; i < 3; ++i) {
            shared_force[warp_id * 3 + i] =
                force[i];  // store force components in shared memory
        }
        for (int i = 0; i < 9; ++i) {
            shared_stress[warp_id * 9 + i] =
                stress[i];  // store stress components in shared memory
        }
    }
    __syncthreads();  // synchronize threads in the block
    if (warp_id == 0) {
        for (int i = 0; i < 3; ++i) {
            force[i] =
                (threadIdx.x < blockDim.x / warpSize)
                    ? shared_force[lane * 3 + i]
                    : 0.0f;  // only the first warp writes to shared memory
        }
        for (int i = 0; i < 9; ++i) {
            stress[i] =
                (threadIdx.x < blockDim.x / warpSize)
                    ? shared_stress[lane * 9 + i]
                    : 0.0f;  // only the first warp writes to shared memory
        }
        for (int i = 0; i < 3; ++i) {
            force_central_sum[i] =
                warpReduceSum(force[i]);  // final reduction across warps
        }
        for (int i = 0; i < 9; ++i) {
            stress_central_sum[i] =
                warpReduceSum(stress[i]);  // final reduction across warps
        }
    }
}

/**
 * @brief Damping function for DFT-D3 calculations.
 * This function computes the damping factors and their derivatives
 * @param distance Distance between atoms
 * @param cutoff_radius Cutoff radius for damping
 * @param param_1 First parameter for damping. when using zero damping, it is SR_6; when using BJ damping, it is a1
 * @param param_2 Second parameter for damping. when using zero damping, it is SR_8; when using BJ damping, it is a2
 */
template <DampingType damping_type>
__device__ void damping(real_t distance, real_t cutoff_radius,
                        real_t param_1, real_t param_2, // parameters used for damping calculation.
                                                        // when using zero damping, they are SR_6 and SR_8, respectively
                                                        // when using BJ damping, they are a1 and a2, respectively
                        real_t* damping_6, real_t* damping_8,
                        real_t* d_damping_6, real_t* d_damping_8) {
    if constexpr (damping_type == ZERO_DAMPING) {
        const real_t relative_distance = distance / (cutoff_radius);
        // fast powering
        const real_t relative_distance_2 =
            relative_distance * relative_distance;
        const real_t relative_distance_4 =
            relative_distance_2 * relative_distance_2;
        const real_t relative_distance_8 =
            relative_distance_4 * relative_distance_4;
        const real_t relative_distance_14 =
            relative_distance_8 * relative_distance_4 * relative_distance_2;
        const real_t relative_distance_15 =
            relative_distance_14 * relative_distance;
        const real_t relative_distance_16 =
            relative_distance_8 * relative_distance_8;
        const real_t relative_distance_17 =
            relative_distance_16 * relative_distance;
        // calculate damping
        const real_t f_dn_6 = 1 / (1 + 6.0f * (1 / relative_distance_14) *
                                           powf(param_1, 14.0f));  // alpha_n = 14
        const real_t f_dn_8 = 1 / (1 + 6.0f * (1 / relative_distance_16) *
                                           powf(param_2, 16.0f));  // alpha_n = 16
        const real_t d_f_dn_6 = 6.0f * 14.0f * f_dn_6 * f_dn_6 *
                                (1 / relative_distance_15) *
                                (1 / (param_1 * cutoff_radius));
        const real_t d_f_dn_8 = 6.0f * 16.0f * f_dn_8 * f_dn_8 *
                                (1 / relative_distance_17) *
                                (1 / (param_2 * cutoff_radius));
        // write the result back
        *damping_6 = f_dn_6;
        *damping_8 = f_dn_8;
        *d_damping_6 = d_f_dn_6;
        *d_damping_8 = d_f_dn_8;
    } else if constexpr (damping_type == BJ_DAMPING) {
        real_t add_entry = param_1 * cutoff_radius + param_2; // $a_1 R_0^{AB} + a_2$
        // fast powering
        real_t add_entry_2 = add_entry * add_entry;
        real_t add_entry_4 = add_entry_2 * add_entry_2;
        real_t add_entry_6 = add_entry_4 * add_entry_2;
        real_t add_entry_8 = add_entry_4 * add_entry_4;
        real_t distance_2 = distance * distance;
        real_t distance_4 = distance_2 * distance_2;
        real_t distance_5 = distance_4 * distance;
        real_t distance_6 = distance_4 * distance_2;
        real_t distance_7 = distance_6 * distance;
        real_t distance_8 = distance_4 * distance_4;
        // calculate damping
        real_t f_dn_6 = distance_6 / (distance_6 + add_entry_6);
        real_t f_dn_8 = distance_8 / (distance_8 + add_entry_8);
        real_t d_f_dn_6 = 6 * add_entry_6 * distance_5 / ((distance_6 + add_entry_6) *
                                                          (distance_6 + add_entry_6));
        real_t d_f_dn_8 = 8 * add_entry_8 * distance_7 / ((distance_8 + add_entry_8) *
                                                          (distance_8 + add_entry_8));
        // write the result back
        *damping_6 = f_dn_6;
        *damping_8 = f_dn_8;
        *d_damping_6 = d_f_dn_6;
        *d_damping_8 = d_f_dn_8;
    }
}

/**
 * @brief this kernel is used to compute the coordination number of each atom in
 * the system.
 * @note this kernel should be launched with a 1D grid of blocks, each block
 * containing a 1D array of threads.
 * the number of blocks equals the number of atoms in the system.
 * and the number of threads in each block shouldn't exceed the number of
 * atoms in the system
 * it's best to use a division of the number of atoms in the system as the
 * number of threads in each block.
 * the number of threads in each block must not exceed MAX_BLOCK_SIZE.
 * the total_cell_bias should be precomputed at host
 */
__global__ void coordination_number_kernel(device_data_t* data) {
    const uint64_t atom_1_index = blockIdx.x;  // each block is responsible for one central atom
    const uint64_t atom_1_type = data->atom_types[atom_1_index];         // type of the central atom
    const atom_t atom_1 = data->atoms[atom_1_index];  // central atom
    const uint64_t total_cell_bias = data->max_cell_bias[0] * data->max_cell_bias[1] * data->max_cell_bias[2];  // total number of cell bias
    const uint64_t mcb0 = data->max_cell_bias[0];  // maximum cell bias in x direction
    const uint64_t mcb1 = data->max_cell_bias[1];  // maximum cell bias in y direction
    const uint64_t mcb2 = data->max_cell_bias[2];  // maximum cell bias in z direction
    const real_t cell[3][3] = {
        {data->cell[0][0], data->cell[0][1], data->cell[0][2]},
        {data->cell[1][0], data->cell[1][1], data->cell[1][2]},
        {data->cell[2][0], data->cell[2][1], data->cell[2][2]}}; // cell matrix
    const real_t CN_cutoff = data->coordination_number_cutoff; // cutoff radius of coordination number
    // print some debug information
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        debug("Coordination number kernel launched with %llu atoms, cell: \n",
              data->num_atoms);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                debug("%f ", cell[i][j]);
            }
            debug("\n");
        }
        debug("max_cell_bias: %llu %llu %llu\n", mcb0, mcb1, mcb2);
    }
    // distribute workload to threads
    uint64_t start_index;  // start atom index for this thread
    uint64_t end_index;    // end atom index for this thread
    uint64_t start_bias_index;  // start bias index for this thread
    uint64_t end_bias_index;    // end bias index for this thread
    if (data->num_atoms >= blockDim.x) {
        /* if the number of atoms exceed number of threads, each thread process
         * a few atoms, going through all possible bias indicies */
        uint64_t workload_per_thread = (data->num_atoms + blockDim.x - 1) / blockDim.x;  // number of atoms per thread
        start_index = threadIdx.x * workload_per_thread;
        end_index = min(start_index + workload_per_thread, data->num_atoms); // the last thread might process fewer atoms
        start_bias_index = 0;
        end_bias_index = total_cell_bias;  // each thread is responsible for all cell biases
    } else {
        /* If the number of atoms is smaller than number of threads, multiple
         * threads process one atom.
         * Divide the threads into groups, each group processes one atom.
         * The first few groups will have one extra thread to guarantee that
         * every thread is assigned to partially equal workload.
         */
        uint64_t threads_per_atom_base = blockDim.x / data->num_atoms;  // base number of threads per atom
        uint64_t threads_per_atom_remainder = blockDim.x % data->num_atoms; // remainder threads
        uint64_t num_atoms_getting_extra_thread = threads_per_atom_remainder;   // number of atoms getting an extra thread
        uint64_t threads_in_larger_groups_total = num_atoms_getting_extra_thread * (threads_per_atom_base + 1);  // total number of threads in larger groups
        uint64_t current_assigned_atom_id;  // which atom this thread is assigned to
        uint64_t threads_working_on_my_atom;    // number of threads working on my atom
        uint64_t rank_in_element_thread_group;  // the rank of this thread within the atom's threads
        if (threadIdx.x < threads_in_larger_groups_total) {
            // this thread is in a larger group
            threads_working_on_my_atom = threads_per_atom_base + 1;
            current_assigned_atom_id = threadIdx.x / threads_working_on_my_atom;
            rank_in_element_thread_group = threadIdx.x % threads_working_on_my_atom;
        } else {
            // this thread falls in later groups that have 'base' threads
            threads_working_on_my_atom = threads_per_atom_base;
            uint64_t threads_already_assigned_to_larger_groups = threads_in_larger_groups_total; // number of threads already assigned to larger groups
            uint64_t threadIdx_relative_to_smaller_groups = threadIdx.x - threads_already_assigned_to_larger_groups; // relative thread index in smaller groups
            current_assigned_atom_id = 
                threadIdx_relative_to_smaller_groups / threads_working_on_my_atom + num_atoms_getting_extra_thread;
            rank_in_element_thread_group = threadIdx_relative_to_smaller_groups % threads_working_on_my_atom;
        }
        start_index = current_assigned_atom_id;
        end_index = min(current_assigned_atom_id + 1, data->num_atoms);
        uint64_t bias_per_thread = total_cell_bias / threads_working_on_my_atom;  // number of biases per thread
        start_bias_index = rank_in_element_thread_group * bias_per_thread;
        end_bias_index = min(start_bias_index + bias_per_thread, total_cell_bias);
    }
    real_t covalent_radii_1 = data->rcov[atom_1_type];  // covalent radii of the central atom
    real_t local_coordination_number = 0.0f;    // local coordination number for the central atom
    for (uint64_t atom_2_index = start_index; atom_2_index < end_index; ++atom_2_index) {
        atom_t atom_2_original = data->atoms[atom_2_index];  // surrounding atom without supercell translation
        real_t covalent_radii_2 = data->rcov[data->atom_types[atom_2_index]];  // covalent radii of the surrounding atom
        for (uint64_t bias_index = start_bias_index; bias_index < end_bias_index; ++bias_index) {
            // iterate over the bias indices
            int64_t x_bias = (bias_index % mcb0) - (mcb0 / 2);  // x bias
            int64_t y_bias = ((bias_index / mcb0) % mcb1) - (mcb1 / 2);  // y bias
            int64_t z_bias = (bias_index / (mcb0 * mcb1) % mcb2) - (mcb2 / 2);  // z bias
            assert_(atom_2_index < data->num_atoms);  // make sure the index is in bounds

            atom_t atom_2 = atom_2_original;
            // translate atom_2 due to periodic boundaries
            atom_2.x += x_bias * cell[0][0] + y_bias * cell[1][0] +
                        z_bias * cell[2][0];
            atom_2.y += x_bias * cell[0][1] + y_bias * cell[1][1] +
                        z_bias * cell[2][1];
            atom_2.z += x_bias * cell[0][2] + y_bias * cell[1][2] +
                        z_bias * cell[2][2];
            // calculate the distance between the two atoms
            real_t distance = sqrtf(powf(atom_1.x - atom_2.x, 2) +
                                    powf(atom_1.y - atom_2.y, 2) +
                                    powf(atom_1.z - atom_2.z, 2));
            // if the distance is within cutoff range, calculate the coordination number
            if (distance <= CN_cutoff && distance > 0.0f) {
                real_t exp = expf(-K1 * ((covalent_radii_1 + covalent_radii_2) / distance - 1.0f));  // $\exp(-k_1*(\frac{R_A+R_b}{r_{ab}}-1))$
                real_t tanh_value = tanhf(CN_cutoff - distance);  // $\tanh(CN_cutoff - r_{ab})$
                real_t smooth_cutoff = powf(tanh_value, 3);  // $\tanh^3(CN_cutoff- r_{ab}))$, this is a smooth cutoff function added in LASP code.

                // the covalent radii table have already taken K2 coefficient into consideration
                real_t coordination_number = 1.0f / (1.0f + exp) * smooth_cutoff;
                local_coordination_number += coordination_number;  // accumulate the coordination number for the central atom
            }
        }
    }
    atomicAdd(&data->coordination_numbers[atom_1_index], local_coordination_number);  // accumulate the coordination number for the central atom
    return;
}
/**
 * @brief this kernel is used to print the coordination numbers of all atoms
 * in the system. It's for debug use.
 * @note this kernel should be launched with a single block and a single
 * thread.
 */
__global__ void print_coordination_number_kernel(device_data_t* data) {
    if (threadIdx.x == 0) {
        printf("Coordination numbers:\n");
        for (uint64_t i = 0; i < data->num_atoms; ++i) {
            printf("Atom %llu, element: %d: %f\n", i, data->atoms[i].element,
                   data->coordination_numbers[i]);
        }
    }
}

/**
 * @brief this kernel is used to compute the two-body interactions between atoms
 * in the system, i.e. the energy and two-atom part of force
 * @note this kernel should be launched with a 1D grid of blocks, each block
 * containing a 1D array of threads.
 * the number of blocks should be equal to the number of atoms in the
 * system.
 * the number of threads in each block can be any value.
 */
__global__ void two_body_kernel(device_data_t* data) {
    // load parameters from device data
    const real_t s6 = data->functional_params.s6;
    const real_t s8 = data->functional_params.s8;
    const real_t sr_6 = data->functional_params.sr6;
    const real_t sr_8 = data->functional_params.sr8;
    const real_t a1 = data->functional_params.a1;
    const real_t a2 = data->functional_params.a2;
    // load the central atom data
    const uint64_t atom_1_index = blockIdx.x;  // index of the central atom
    const uint64_t atom_1_type = data->atom_types[atom_1_index];  // type of the central atom
    const atom_t atom_1 = data->atoms[atom_1_index];  // central atom
    const real_t coordination_number_1 = data->coordination_numbers[atom_1_index];  // coordination number of the central atom

    // batch variables for energy, force and stress accumulation to reduce numeric error
    real_t batch_energy = 0.0f;  // temporary batch energy
    real_t batch_force[3] = {0.0f};   // temporary batch force vector
    real_t batch_stress[9] = {0.0f};  // temporary batch stress matrix
    real_t batch_dE_dCN = 0.0f;  // temporary dE/dCN
    // compensation variables to perform Kahan addition in batched variables
    real_t batch_energy_compensate = 0.0f;
    real_t batch_dE_dCN_compensate = 0.0f;
    real_t batch_force_compensate[3] = {0.0f};
    real_t batch_stress_compensate[9] = {0.0f};
    const uint32_t batch_size = 1024;   // batch storage will be updated to local storage after this many accumulations
    uint32_t batch_count = 0;   // number of accumulations in the current batch

    // local variables for central atom
    real_t local_energy = 0.0f; // local energy
    real_t local_dE_dCN = 0.0f;  // local dE/dCN
    real_t local_force[3] = {0.0f};   // local force for the central atom
    real_t local_stress[9] = {0.0f};  // local stress matrix
    // compensation variables for local variables
    real_t local_energy_compensate = 0.0f;
    real_t local_dE_dCN_compensate = 0.0f;
    real_t local_force_compensate[3] = {0.0f};
    real_t local_stress_compensate[9] = {0.0f};

    const real_t cell_volume = calculate_cell_volume(data->cell);
    const uint64_t total_cell_bias = data->max_cell_bias[0] * data->max_cell_bias[1] * data->max_cell_bias[2];  // total number of cell biases
    const uint64_t mcb0 = data->max_cell_bias[0];  // maximum cell bias in x direction
    const uint64_t mcb1 = data->max_cell_bias[1];  // maximum cell bias in y direction
    const uint64_t mcb2 = data->max_cell_bias[2];  // maximum cell bias in z direction
    const real_t cell[3][3] = {
        {data->cell[0][0], data->cell[0][1], data->cell[0][2]},
        {data->cell[1][0], data->cell[1][1], data->cell[1][2]},
        {data->cell[2][0], data->cell[2][1], data->cell[2][2]}}; // local cell matrix
    const real_t cutoff = data->cutoff; // cutoff radius for two-body interactions

    // print some debug information
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        debug("Coordination number kernel launched with %llu atoms, cell: \n",
              data->num_atoms);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                debug("%f ", cell[i][j]);
            }
            debug("\n");
        }
        debug("max_cell_bias: %llu %llu %llu\n", mcb0, mcb1, mcb2);
    }
    // distribute workload to threads
    uint64_t start_index;  // start index for this thread
    uint64_t end_index;    // end index for this thread
    uint64_t start_bias_index;  // start bias index for this thread
    uint64_t end_bias_index;    // end bias index for this thread
    if (data->num_atoms >= blockDim.x) {
        /* if the number of atoms exceed number of threads, each thread process
         * an atom, going through all possible bias indicies */
        const uint64_t workload_per_thread = (data->num_atoms + blockDim.x - 1) / blockDim.x;  // number of atoms per thread
        start_index = threadIdx.x * workload_per_thread;
        end_index = min(start_index + workload_per_thread,
                        data->num_atoms);  // end index for this thread
        start_bias_index = 0;
        end_bias_index =
            total_cell_bias;  // each thread is responsible for all cell biases
    } else {
        /* If the number of atoms is smaller than number of threads, multiple
         * threads process one atom */
        /* determine the element assigned and rank within that element's threads
         */
        const uint64_t threads_per_element_base =
            blockDim.x / data->num_atoms;  // number of threads per atom
        const uint64_t threads_per_element_remainder =
            blockDim.x % data->num_atoms;  // remainder threads
        const uint64_t num_elements_getting_extra_thread =
            threads_per_element_remainder;  // number of threads getting an
                                            // extra thread
        const uint64_t threads_in_larger_groups_total =
            num_elements_getting_extra_thread *
            (threads_per_element_base +
             1);  // total number of threads in larger groups
        uint64_t current_assigned_element_id;
        uint64_t threads_working_on_my_element;
        uint64_t rank_in_element_thread_group;
        if (threadIdx.x < threads_in_larger_groups_total) {
            /* this thread is in a larger group */
            threads_working_on_my_element =
                threads_per_element_base +
                1;  // number of threads working on my element would be one more
                    // than the base
            current_assigned_element_id =
                threadIdx.x /
                threads_working_on_my_element;  // which element this thread is
                                                // assigned to
            rank_in_element_thread_group =
                threadIdx.x %
                threads_working_on_my_element;  // the rank of this thread
                                                // within the element's threads
        } else {
            /* this thread falls in later groups that have 'base' threads */
            threads_working_on_my_element =
                threads_per_element_base;  // number of threads working on my
                                           // element would be the base
            const uint64_t thraeds_already_assigned_to_larger_groups =
                threads_in_larger_groups_total;  // number of threads already
                                                 // assigned to larger groups
            const uint64_t threadIdx_relative_to_smaller_groups =
                threadIdx.x -
                thraeds_already_assigned_to_larger_groups;  // relative thread
                                                            // index in smaller
                                                            // groups
            current_assigned_element_id =
                threadIdx_relative_to_smaller_groups /
                    threads_working_on_my_element +
                num_elements_getting_extra_thread;  // which element this thread
                                                    // is assigned to
            rank_in_element_thread_group =
                threadIdx_relative_to_smaller_groups %
                threads_working_on_my_element;  // the rank of this thread
                                                // within the element's threads
        }
        start_index =
            current_assigned_element_id;  // start index for this thread
        end_index = min(current_assigned_element_id + 1,
                        data->num_atoms);  // end index for this thread
        const uint64_t bias_per_thread =
            total_cell_bias /
            threads_working_on_my_element;  // number of biases per thread
        start_bias_index = rank_in_element_thread_group *
                           bias_per_thread;  // start bias index for this thread
        end_bias_index =
            min(start_bias_index + bias_per_thread,
                total_cell_bias);  // end bias index for this thread
    }
    for (uint64_t atom_2_index = start_index; atom_2_index < end_index;
         ++atom_2_index) {
        const atom_t atom_2_original =
            data->atoms[atom_2_index];  // surrounding atom
        const uint64_t atom_2_type =
            data->atom_types[atom_2_index];  // type of the surrounding atom
        const real_t coordination_number_2 =
            data->coordination_numbers[atom_2_index];  // coordination number of
                                                       // the surrounding atom
        /* calculate the coordination number based on dispersion coefficient
            formula: $C_6^{ij} = Z/W$
            where $Z = \sum_{a,b}C_{6,ref}^{i,j}L_{a,b}$
            $W = \sum_{a,b}L_{a,b}$
            $L_{a,b} = \exp(-k3((CN^A-CN^A_{ref,a})^2 + (CN^B-CN^B_{ref,b})^2))$
         */
        real_t exponent_args[NUM_REF_C6 * NUM_REF_C6] = {
            0.0f};  // array to store the exponent arguments for L_ij
        real_t CN_ref_1[NUM_REF_C6 * NUM_REF_C6] = {
            0.0f};  // array to store the CNref values for atom_1
        real_t c6ab_ref[NUM_REF_C6 * NUM_REF_C6] = {
            0.0f};  // array to store the C6ab_ref values
        uint16_t valid_term_count =
            0;  // count of valid terms in the C6ref arrays
        real_t max_exponent_arg =
            -FLT_MAX;  // maximum exponent argument for L_ij
        for (uint64_t i = 0; i < NUM_REF_C6; ++i) {
            for (uint64_t j = 0; j < NUM_REF_C6; ++j) {
                /* find the C6ref */
                uint64_t stride_1 = data->num_elements * NUM_REF_C6 *
                                    NUM_REF_C6 * NUM_C6AB_ENTRIES;
                uint64_t stride_2 = NUM_REF_C6 * NUM_REF_C6 * NUM_C6AB_ENTRIES;
                uint64_t stride_3 = NUM_REF_C6 * NUM_C6AB_ENTRIES;
                uint64_t stride_4 = NUM_C6AB_ENTRIES;
                uint64_t index = atom_1_type * stride_1 +
                                 atom_2_type * stride_2 + i * stride_3 +
                                 j * stride_4;
                real_t c6_ref = data->c6_ab_ref[index + 0];
                /* these entries could be -1.0f if they are not valid, but at
                 * least one should be valid*/
                real_t coordination_number_ref_1 = data->c6_ab_ref[index + 1];
                real_t coordination_number_ref_2 = data->c6_ab_ref[index + 2];
                if (coordination_number_ref_1 - (-1.0f) > 1e-5f &&
                    coordination_number_ref_2 - (-1.0f) > 1e-5f) {
                    /* if both coordination numbers are valid, we can use them
                     */
                    exponent_args[valid_term_count] =
                        -K3 *
                        (powf(coordination_number_1 - coordination_number_ref_1,
                              2) +
                         powf(coordination_number_2 - coordination_number_ref_2,
                              2));
                    CN_ref_1[valid_term_count] =
                        coordination_number_ref_1;        // store the C6ref for
                                                          // atom_1
                    c6ab_ref[valid_term_count] = c6_ref;  // store the C6ab_ref
                    max_exponent_arg =
                        (max_exponent_arg > exponent_args[valid_term_count])
                            ? max_exponent_arg
                            : exponent_args
                                  [valid_term_count];  // update the maximum
                                                       // exponent argument
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
                real_t L_ij =
                    expf(exponent_args[i] -
                         max_exponent_arg);  // normalize the exponent argument
                real_t dL_ij_1_part = -2.0f * K3 *
                                      (coordination_number_1 - CN_ref_1[i]) *
                                      L_ij;  // dL_ij/dCN_1
                Z += c6ab_ref[i] * L_ij;     // accumulate the value of Z
                W += L_ij;                   // accumulate the value of W
                c_ref_L_ij +=
                    c6ab_ref[i] * L_ij;  // accumulate the value of c_ref_L_ij
                c_ref_dL_ij_1 +=
                    c6ab_ref[i] *
                    dL_ij_1_part;  // accumulate the value of c_ref_dL_ij_1
                dL_ij_1 += dL_ij_1_part;  // accumulate the value of dL_ij_1
            }
        } else {
            printf(
                "Warning: no valid C6ab terms found for atom pair (%llu, "
                "%llu)\n",
                atom_1_index, atom_2_index);
        }

        const real_t L_ij = W;
        real_t dC6ab_dCN_1 =
            (L_ij * L_ij > 0.0f)
                ? (c_ref_dL_ij_1 * L_ij - c_ref_L_ij * dL_ij_1) /
                      powf(L_ij, 2.0f)
                : 0.0f;  // avoid division by zero
        if (isnan(dC6ab_dCN_1) || isinf(dC6ab_dCN_1)) {
            printf(
                "Error: dC6ab/dCN_1 is NaN or Inf for atom pair (%llu, %llu)\n",
                atom_1_index, atom_2_index);
            printf(
                "Z: %f, W: %f, c_ref_L_ij: %f, c_ref_dL_ij_1: %f, dL_ij_1: "
                "%f\n",
                Z, W, c_ref_L_ij, c_ref_dL_ij_1, dL_ij_1);
            dC6ab_dCN_1 = 0.0f;  // reset to 0.0f if it's NaN or Inf
        }

        real_t c6_ab = (W > 0.0f) ? Z / W : 0.0f;  // avoid division by zero
        if (isnan(c6_ab) || isinf(c6_ab)) {
            printf("Error: C6_ab is NaN or Inf for atom pair (%llu, %llu)\n",
                   atom_1_index, atom_2_index);
            printf(
                "Z: %f, W: %f, c_ref_L_ij: %f, c_ref_dL_ij_1: %f, dL_ij_1: "
                "%f\n",
                Z, W, c_ref_L_ij, c_ref_dL_ij_1, dL_ij_1);
            c6_ab = 0.0f;  // reset to 0.0f if it's NaN or Inf
        }
        /* calculate c8_ab by $C_8^{AB} = 3C_6^{AB}\sqrt{Q^AQ^B}$*/
        const real_t r2r4_1 = data->r2r4[atom_1_type];
        const real_t r2r4_2 = data->r2r4[atom_2_type];
        const real_t c8_ab = 3.0f * c6_ab * r2r4_1 *
                             r2r4_2;  // the value in r2r4 is already squared
        const real_t dC8ab_dCN_1 =
            3.0f * dC6ab_dCN_1 * r2r4_1 * r2r4_2;  // dC8ab/dCN_1
        /* acquire the cutoff radius between the two atoms */
        const real_t cutoff_radius =
            data->r0ab[atom_1_type * data->num_elements + atom_2_type];
        for (uint64_t bias_index = start_bias_index;
             bias_index < end_bias_index; ++bias_index) {
            /* each thread is responsible for one atom pair, so the number of
             * threads should be equal to num_atoms * total_cell_bias */
            const int64_t x_bias = (bias_index % mcb0) - (mcb0 / 2);  // x bias
            const int64_t y_bias =
                ((bias_index / mcb0) % mcb1) - (mcb1 / 2);  // y bias
            const int64_t z_bias =
                (bias_index / (mcb0 * mcb1) % mcb2) - (mcb2 / 2);  // z bias
            assert_(atom_2_index <
                    data->num_atoms);  // make sure the index is in bounds

            atom_t atom_2 = atom_2_original;
            /* translate atom_2 due to periodic boundaries */
            atom_2.x += x_bias * cell[0][0] + y_bias * cell[1][0] +
                        z_bias * cell[2][0];  // translate in x direction
            atom_2.y += x_bias * cell[0][1] + y_bias * cell[1][1] +
                        z_bias * cell[2][1];  // translate in y direction
            atom_2.z += x_bias * cell[0][2] + y_bias * cell[1][2] +
                        z_bias * cell[2][2];  // translate in z direction
            /* calculate the distance between the two atoms */
            const real_t distance = sqrtf(powf(atom_1.x - atom_2.x, 2) +
                                          powf(atom_1.y - atom_2.y, 2) +
                                          powf(atom_1.z - atom_2.z, 2));
            if (distance <= cutoff && distance > 0.0f) {
                /* calculate distance^6 and distance^8 using fast power
                 * algorithm*/
                const real_t distance_2 = distance * distance;    // distance^2
                const real_t distance_3 = distance_2 * distance;  // distance^3
                const real_t distance_4 =
                    distance_2 * distance_2;  // distance^4
                const real_t distance_6 =
                    distance_3 * distance_3;  // distance^6
                const real_t distance_8 =
                    distance_4 * distance_4;  // distance^8
                const real_t distance_10 =
                    distance_6 * distance_4;  // distance^10
                /* calculate the dampling function as Grimme et al. 2010, eq4 */
                real_t f_dn_6, f_dn_8, d_f_dn_6, d_f_dn_8;
                switch (data->damping_type) {
                    case ZERO_DAMPING:
                        damping<ZERO_DAMPING>(distance, cutoff_radius, sr_6, sr_8,
                                            &f_dn_6, &f_dn_8,
                                            &d_f_dn_6, &d_f_dn_8);
                        break;
                    case BJ_DAMPING:
                        damping<BJ_DAMPING>(distance, cutoff_radius, a1, a2,
                                            &f_dn_6, &f_dn_8, &d_f_dn_6,
                                            &d_f_dn_8);
                        break;
                }
                /* calculate the dispersion enegry as Grimme et al. 2010, eq3 */
                const real_t dispersion_energy_6 =
                    s6 * (c6_ab / distance_6) * f_dn_6;
                const real_t dispersion_energy_8 =
                    s8 * (c8_ab / distance_8) * f_dn_8;
                // printf("dispersion energy between atoms (%llu,%llu):
                // %f\n",atom_1_index,atom_2_index,dispersion_energy_6 +
                // dispersion_energy_8);
                const real_t dispersion_energy =
                    (dispersion_energy_6 + dispersion_energy_8) /
                    2.0f;  // divide by 2 because each atom pair is counted
                           // twice
                /* add the energy back to results, use Kahan summation for
                 * higher accuracy */
                {
                    const real_t y =
                        dispersion_energy -
                        batch_energy_compensate;  // calculate the difference
                    const real_t t =
                        batch_energy +
                        y;  // add the difference to the local energy
                    batch_energy_compensate =
                        (t - batch_energy) -
                        y;  // calculate the compensation for the next iteration
                    batch_energy = t;  // update the local energy
                }

                /* accumulate dEij/dCNi to local variable, use Kagan summation
                 * for better accuracy */
                {
                    // real_t y = s6 * (f_dn_6 / distance_6) * dC6ab_dCN_1 + s8
                    // * (f_dn_8 / distance_8) * dC8ab_dCN_1 -
                    // dE_dCN_conpensate; // calculate the difference
                    const real_t y =
                        ((s6 * f_dn_6 * distance_2 * dC6ab_dCN_1 +
                          s8 * f_dn_8 * dC8ab_dCN_1) /
                         distance_8) -
                        batch_dE_dCN_compensate;  // calculate the difference
                    const real_t t =
                        batch_dE_dCN +
                        y;  // add the difference to the local dE/dCN
                    batch_dE_dCN_compensate =
                        (t - batch_dE_dCN) -
                        y;  // calculate the compensation for the next iteration
                    batch_dE_dCN = t;  // update the local dE/dCN
                }

                /* the first entry of two-body force
                $F_a = S_n C_n^{ab} f_{d,n}(r_{ab}) \frac{\partial}{\partial
                r_a} r_{ab}^{-n}$ $F_a = S_n C_n^{ab} f_{d,n}(r_{ab}) *
                (-n)r_{ab}^{-n-2} * \uparrow{r_{ab}}$ */
                real_t force = 0.0f;  // dE/dr * 1/r
                force += s6 * c6_ab * f_dn_6 * (-6.0f) /
                         distance_8;  // dE_6/dr * 1/r
                force += s8 * c8_ab * f_dn_8 * (-8.0f) /
                         distance_10;  // dE_8/dr * 1/r
                /* the second entry of two-body force
                $F_a = S_n C_n^{ab} r_{ab}^{-n} \frac{\partial}{\partial r_a}
                f_{d,n}(r_{ab})$ $F_a = S_n C_n^{ab} r_{ab}^{-n} -f_{d,n}^2 *
                (6*(-\alpha_n)*(r_{ab}/{S_{r,n}R_0^{AB}})^(-\alpha_n - 1) *
                1/(S_{r,n}R_0^{AB}})) / r_ab \uparrow{r_{ab}}$*/
                force += s6 * c6_ab / distance_6 * d_f_dn_6 /
                         distance;  // dE_6/dr * 1/r
                force += s8 * c8_ab / distance_8 * d_f_dn_8 /
                         distance;  // dE_8/dr * 1/r
                /* accumulate force for the central atom, use Kahan summation
                 * for better accuracy */
                {
                    const real_t y0 =
                        force * (atom_1.x - atom_2.x) -
                        batch_force_compensate[0];  // calculate the difference
                                                    // for x component
                    const real_t t0 =
                        batch_force[0] +
                        y0;  // add the difference to the local force
                    batch_force_compensate[0] =
                        (t0 - batch_force[0]) -
                        y0;  // calculate the compensation for the next
                             // iteration
                    batch_force[0] =
                        t0;  // update the local force for x component

                    const real_t y1 =
                        force * (atom_1.y - atom_2.y) -
                        batch_force_compensate[1];  // calculate the difference
                                                    // for y component
                    const real_t t1 =
                        batch_force[1] +
                        y1;  // add the difference to the local force
                    batch_force_compensate[1] =
                        (t1 - batch_force[1]) -
                        y1;  // calculate the compensation for the next
                             // iteration
                    batch_force[1] =
                        t1;  // update the local force for y component

                    const real_t y2 =
                        force * (atom_1.z - atom_2.z) -
                        batch_force_compensate[2];  // calculate the difference
                                                    // for z component
                    const real_t t2 =
                        batch_force[2] +
                        y2;  // add the difference to the local force
                    batch_force_compensate[2] =
                        (t2 - batch_force[2]) -
                        y2;  // calculate the compensation for the next
                             // iteration
                    batch_force[2] =
                        t2;  // update the local force for z component
                }

                /* accumulate stress to local matrix instead of directly using
                 * atomicAdd, divide by 2 becuase the same entry will be
                 * calculated twice (when atom_2 is the central atom), use Kahan
                 * summation for better accuracy */
                {
                    const real_t y0 =
                        -1.0f * (atom_1.x - atom_2.x) * force *
                            (atom_1.x - atom_2.x) / 2.0f / cell_volume -
                        batch_stress_compensate[0];  // calculate the difference
                                                     // for stress_xx
                    const real_t t0 =
                        batch_stress[0 * 3 + 0] +
                        y0;  // add the difference to the local stress
                    batch_stress_compensate[0] =
                        (t0 - batch_stress[0 * 3 + 0]) -
                        y0;  // calculate the compensation for the next
                             // iteration
                    batch_stress[0 * 3 + 0] =
                        t0;  // update the local stress for stress_xx

                    const real_t y1 =
                        -1.0f * (atom_1.x - atom_2.x) * force *
                            (atom_1.y - atom_2.y) / 2.0f / cell_volume -
                        batch_stress_compensate[1];  // calculate the difference
                                                     // for stress_xy
                    const real_t t1 =
                        batch_stress[0 * 3 + 1] +
                        y1;  // add the difference to the local stress
                    batch_stress_compensate[1] =
                        (t1 - batch_stress[0 * 3 + 1]) -
                        y1;  // calculate the compensation for the next
                             // iteration
                    batch_stress[0 * 3 + 1] =
                        t1;  // update the local stress for stress_xy

                    const real_t y2 =
                        -1.0f * (atom_1.x - atom_2.x) * force *
                            (atom_1.z - atom_2.z) / 2.0f / cell_volume -
                        batch_stress_compensate[2];  // calculate the difference
                                                     // for stress_xz
                    const real_t t2 =
                        batch_stress[0 * 3 + 2] +
                        y2;  // add the difference to the local stress
                    batch_stress_compensate[2] =
                        (t2 - batch_stress[0 * 3 + 2]) -
                        y2;  // calculate the compensation for the next
                             // iteration
                    batch_stress[0 * 3 + 2] =
                        t2;  // update the local stress for stress_xz

                    const real_t y3 =
                        -1.0f * (atom_1.y - atom_2.y) * force *
                            (atom_1.x - atom_2.x) / 2.0f / cell_volume -
                        batch_stress_compensate[3];  // calculate the difference
                                                     // for stress_yx
                    const real_t t3 =
                        batch_stress[1 * 3 + 0] +
                        y3;  // add the difference to the local stress
                    batch_stress_compensate[3] =
                        (t3 - batch_stress[1 * 3 + 0]) -
                        y3;  // calculate the compensation for the next
                             // iteration
                    batch_stress[1 * 3 + 0] =
                        t3;  // update the local stress for stress_yx

                    const real_t y4 =
                        -1.0f * (atom_1.y - atom_2.y) * force *
                            (atom_1.y - atom_2.y) / 2.0f / cell_volume -
                        batch_stress_compensate[4];  // calculate the difference
                                                     // for stress_yy
                    const real_t t4 =
                        batch_stress[1 * 3 + 1] +
                        y4;  // add the difference to the local stress
                    batch_stress_compensate[4] =
                        (t4 - batch_stress[1 * 3 + 1]) -
                        y4;  // calculate the compensation for the next
                             // iteration
                    batch_stress[1 * 3 + 1] =
                        t4;  // update the local stress for stress_yy

                    const real_t y5 =
                        -1.0f * (atom_1.y - atom_2.y) * force *
                            (atom_1.z - atom_2.z) / 2.0f / cell_volume -
                        batch_stress_compensate[5];  // calculate the difference
                                                     // for stress_yz
                    const real_t t5 =
                        batch_stress[1 * 3 + 2] +
                        y5;  // add the difference to the local stress
                    batch_stress_compensate[5] =
                        (t5 - batch_stress[1 * 3 + 2]) -
                        y5;  // calculate the compensation for the next
                             // iteration
                    batch_stress[1 * 3 + 2] =
                        t5;  // update the local stress for stress_yz

                    const real_t y6 =
                        -1.0f * (atom_1.z - atom_2.z) * force *
                            (atom_1.x - atom_2.x) / 2.0f / cell_volume -
                        batch_stress_compensate[6];  // calculate the difference
                                                     // for stress_zx
                    const real_t t6 =
                        batch_stress[2 * 3 + 0] +
                        y6;  // add the difference to the local stress
                    batch_stress_compensate[6] =
                        (t6 - batch_stress[2 * 3 + 0]) -
                        y6;  // calculate the compensation for the next
                             // iteration
                    batch_stress[2 * 3 + 0] =
                        t6;  // update the local stress for stress_zx

                    const real_t y7 =
                        -1.0f * (atom_1.z - atom_2.z) * force *
                            (atom_1.y - atom_2.y) / 2.0f / cell_volume -
                        batch_stress_compensate[7];  // calculate the difference
                                                     // for stress_zy
                    const real_t t7 =
                        batch_stress[2 * 3 + 1] +
                        y7;  // add the difference to the local stress
                    batch_stress_compensate[7] =
                        (t7 - batch_stress[2 * 3 + 1]) -
                        y7;  // calculate the compensation for the next
                             // iteration
                    batch_stress[2 * 3 + 1] =
                        t7;  // update the local stress for stress_zy

                    const real_t y8 =
                        -1.0f * (atom_1.z - atom_2.z) * force *
                            (atom_1.z - atom_2.z) / 2.0f / cell_volume -
                        batch_stress_compensate[8];  // calculate the difference
                                                     // for stress_zz
                    const real_t t8 =
                        batch_stress[2 * 3 + 2] +
                        y8;  // add the difference to the local stress
                    batch_stress_compensate[8] =
                        (t8 - batch_stress[2 * 3 + 2]) -
                        y8;  // calculate the compensation for the next
                             // iteration
                    batch_stress[2 * 3 + 2] =
                        t8;  // update the local stress for stress_zz
                }
                batch_count++;  // increment the count of interactions for the
                                // central atom
                if (batch_count >= batch_size) {
                    /* accumulate the batch variables to local variables */
                    {
                        const real_t y =
                            batch_dE_dCN -
                            local_dE_dCN_compensate;  // calculate the
                                                      // difference
                        const real_t t =
                            local_dE_dCN +
                            y;  // add the difference to the local dE/dCN
                        local_dE_dCN_compensate =
                            (t - local_dE_dCN) -
                            y;  // calculate the compensation for the next
                                // iteration
                        local_dE_dCN = t;  // update the local dE/dCN
                    }
                    {
                        const real_t y =
                            batch_energy -
                            local_energy_compensate;  // calculate the
                                                      // difference
                        const real_t t =
                            local_energy +
                            y;  // add the difference to the local energy
                        local_energy_compensate =
                            (t - local_energy) -
                            y;  // calculate the compensation for the next
                                // iteration
                        local_energy = t;  // update the local energy
                    }
                    {
                        for (uint16_t i = 0; i < 3; ++i) {
                            const real_t y =
                                batch_force[i] -
                                local_force_compensate[i];  // calculate the
                                                            // difference for
                                                            // each component
                            const real_t t =
                                local_force[i] +
                                y;  // add the difference to the local force
                            local_force_compensate[i] =
                                (t - local_force[i]) -
                                y;  // calculate the compensation for the next
                                    // iteration
                            local_force[i] =
                                t;  // update the local force for each component
                        }
                    }
                    {
                        for (uint16_t i = 0; i < 9; ++i) {
                            const real_t y =
                                batch_stress[i] -
                                local_stress_compensate[i];  // calculate the
                                                             // difference for
                                                             // each component
                            const real_t t =
                                local_stress[i] +
                                y;  // add the difference to the local stress
                            local_stress_compensate[i] =
                                (t - local_stress[i]) -
                                y;  // calculate the compensation for the next
                                    // iteration
                            local_stress[i] = t;  // update the local stress for
                                                  // each component
                        }
                    }
                    /* reset the batch variables */
                    batch_dE_dCN = 0.0f;  // reset the batch dE/dCN
                    batch_energy = 0.0f;  // reset the batch energy
                    for (uint16_t i = 0; i < 3; ++i) {
                        batch_force[i] =
                            0.0f;  // reset the batch force for each component
                    }
                    for (uint16_t i = 0; i < 9; ++i) {
                        batch_stress[i] =
                            0.0f;  // reset the batch stress for each component
                    }
                    batch_count = 0;  // reset the batch count
                }
            }
        }
    }
    /* accumulate the remaining batch variables */
    {
        const real_t y =
            batch_dE_dCN - local_dE_dCN_compensate;  // calculate the difference
        const real_t t =
            local_dE_dCN + y;  // add the difference to the local dE/dCN
        local_dE_dCN_compensate =
            (t - local_dE_dCN) -
            y;             // calculate the compensation for the next iteration
        local_dE_dCN = t;  // update the local dE/dCN
    }
    {
        const real_t y =
            batch_energy - local_energy_compensate;  // calculate the difference
        const real_t t =
            local_energy + y;  // add the difference to the local energy
        local_energy_compensate =
            (t - local_energy) -
            y;             // calculate the compensation for the next iteration
        local_energy = t;  // update the local energy
    }
    {
        for (uint16_t i = 0; i < 3; ++i) {
            const real_t y =
                batch_force[i] -
                local_force_compensate[i];  // calculate the difference for each
                                            // component
            const real_t t =
                local_force[i] + y;  // add the difference to the local force
            local_force_compensate[i] =
                (t - local_force[i]) -
                y;  // calculate the compensation for the next iteration
            local_force[i] = t;  // update the local force for each component
        }
    }
    {
        for (uint16_t i = 0; i < 9; ++i) {
            const real_t y =
                batch_stress[i] -
                local_stress_compensate[i];  // calculate the difference for
                                             // each component
            const real_t t =
                local_stress[i] + y;  // add the difference to the local stress
            local_stress_compensate[i] =
                (t - local_stress[i]) -
                y;  // calculate the compensation for the next iteration
            local_stress[i] = t;  // update the local stress for each component
        }
    }

    real_t dE_dCN_sum = 0;  // reduce dE/dCN across the block
    real_t energy_sum = 0;  // reduce energy across the block
    real_t force_central_sum[3] = {
        0.0f, 0.0f, 0.0f};  // force cache for the central atom across the block
    real_t stress_sum[9] = {
        0.0f};  // stress cache for the central atom across the block
    /* accumulate the results across the block */
    blockReduceTwoBodyKernel(local_dE_dCN, local_energy, local_force,
                             local_stress, &dE_dCN_sum, &energy_sum,
                             force_central_sum, stress_sum);

    if (threadIdx.x == 0) {
        /* only the first thread in the block will write the results back to
         * global memory */
        data->dE_dCN[atom_1_index] =
            dE_dCN_sum;  // accumulate dE/dCN for the central atom
        // atomicAdd(data->energy, energy_sum); // accumulate energy for the
        // central atom
        data->energy[atom_1_index] =
            energy_sum;  // store the energy for the central atom
        /* accumulate force for the central atom
            there is only one thread accessing this memory, so no atomic
           operation is needed :) */
        data->forces[atom_1_index * 3 + 0] = force_central_sum[0];
        data->forces[atom_1_index * 3 + 1] = force_central_sum[1];
        data->forces[atom_1_index * 3 + 2] = force_central_sum[2];
/* accumulate stress for the central atom */
#pragma unroll
        for (uint16_t i = 0; i < 9; ++i) {
            atomicAdd(&data->stress[i], stress_sum[i]);
        }
    }
}

/**
 * @brief this kernel is used to compute the three-body interactions between
 * atoms in the system.
 * @brief i.e $\frac{\partial E_{ij}}{\partial r_{ik}}$ where $i$ is the central
 * atom, $j$ is the first neighbor and $k$ is the second neighbor.
 *
 * @note this kernel should be launched with a 1D grid of blocks, each block
 * containining a 1D array of threads.
 * @note the number of blocks should be equal to the number of atoms in the
 * system.
 * @note the number of threads in each block can be any value, 512 would be a
 * good choice, but it could be smaller if your memory is limited.
 */
__global__ void three_body_kernel(device_data_t* data) {
    real_t cell_volume = calculate_cell_volume(data->cell);
    uint64_t atom_1_index =
        blockIdx.x;  // each block is responsible for one central atom
    atom_t atom_1 = data->atoms[atom_1_index];  // central atom
    uint64_t atom_1_type =
        data->atom_types[atom_1_index];  // type of the central atom
    real_t covalent_radii_1 =
        data->rcov[atom_1_type];  // covalent radius of the central atom
    real_t dE_dCN =
        data->dE_dCN[atom_1_index];  // derivative of energy with respect to
                                     // coordination number

    /* when finding neighbors, each tread is responsible for a neighbor atom and
    they loop over supercell indicies. Therefore, the neighbors list is
    organized in a atom_index priored way. A typical layout is: neighbors[a]:
    {1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,...} where 1,2,3 represent atom index and
    different entries have different coordination due to supercell Therefore, we
    can cache the force and only use `atomicAdd` after accumulating all forces
  */
    real_t force_central[3] = {
        0.0f, 0.0f,
        0.0f};  // force cache for the central atom during the following loop
    real_t force_compensate_central[3] = {
        0.0f, 0.0f, 0.0f};  // force compensation for the central atom to
                            // improve numerical stability
    real_t stress[9] = {
        0.0f};  // stress cache for the central atom during the following loop
    real_t stress_compensate[9] = {
        0.0f};  // stress compensation for the central atom to improve numerical
                // stability

    uint64_t total_cell_bias =
        data->max_cell_bias[0] * data->max_cell_bias[1] *
        data->max_cell_bias[2];  // total number of cell biases
    const uint64_t mcb0 =
        data->max_cell_bias[0];  // maximum cell bias in x direction
    const uint64_t mcb1 =
        data->max_cell_bias[1];  // maximum cell bias in y direction
    const uint64_t mcb2 =
        data->max_cell_bias[2];  // maximum cell bias in z direction
    const real_t cell[3][3] = {
        {data->cell[0][0], data->cell[0][1], data->cell[0][2]},
        {data->cell[1][0], data->cell[1][1], data->cell[1][2]},
        {data->cell[2][0], data->cell[2][1], data->cell[2][2]}};
    const real_t CN_cutoff = data->coordination_number_cutoff;
    /* print some debug information */
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        debug("Coordination number kernel launched with %llu atoms, cell: \n",
              data->num_atoms);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                debug("%f ", cell[i][j]);
            }
            debug("\n");
        }
        debug("max_cell_bias: %llu %llu %llu\n", mcb0, mcb1, mcb2);
    }
    uint64_t start_index;
    uint64_t end_index;
    uint64_t start_bias_index;
    uint64_t end_bias_index;
    if (data->num_atoms >= blockDim.x) {
        /* if the number of atoms exceed number of threads, each thread process
         * an atom, going through all possible bias indicies */
        uint64_t workload_per_thread =
            (data->num_atoms + blockDim.x - 1) /
            blockDim.x;  // number of atoms per thread
        start_index =
            threadIdx.x * workload_per_thread;  // start index for this thread
        end_index = min(start_index + workload_per_thread,
                        data->num_atoms);  // end index for this thread
        start_bias_index = 0;
        end_bias_index =
            total_cell_bias;  // each thread is responsible for all cell biases
    } else {
        /* If the number of atoms is smaller than number of threads, multiple
         * threads process one atom */
        /* determine the element assigned and rank within that element's threads
         */
        uint64_t threads_per_element_base =
            blockDim.x / data->num_atoms;  // number of threads per atom
        uint64_t threads_per_element_remainder =
            blockDim.x % data->num_atoms;  // remainder threads
        uint64_t num_elements_getting_extra_thread =
            threads_per_element_remainder;  // number of threads getting an
                                            // extra thread
        uint64_t threads_in_larger_groups_total =
            num_elements_getting_extra_thread *
            (threads_per_element_base +
             1);  // total number of threads in larger groups
        uint64_t current_assigned_element_id;
        uint64_t threads_working_on_my_element;
        uint64_t rank_in_element_thread_group;
        if (threadIdx.x < threads_in_larger_groups_total) {
            /* this thread is in a larger group */
            threads_working_on_my_element =
                threads_per_element_base +
                1;  // number of threads working on my element would be one more
                    // than the base
            current_assigned_element_id =
                threadIdx.x /
                threads_working_on_my_element;  // which element this thread is
                                                // assigned to
            rank_in_element_thread_group =
                threadIdx.x %
                threads_working_on_my_element;  // the rank of this thread
                                                // within the element's threads
        } else {
            /* this thread falls in later groups that have 'base' threads */
            threads_working_on_my_element =
                threads_per_element_base;  // number of threads working on my
                                           // element would be the base
            uint64_t thraeds_already_assigned_to_larger_groups =
                threads_in_larger_groups_total;  // number of threads already
                                                 // assigned to larger groups
            uint64_t threadIdx_relative_to_smaller_groups =
                threadIdx.x -
                thraeds_already_assigned_to_larger_groups;  // relative thread
                                                            // index in smaller
                                                            // groups
            current_assigned_element_id =
                threadIdx_relative_to_smaller_groups /
                    threads_working_on_my_element +
                num_elements_getting_extra_thread;  // which element this thread
                                                    // is assigned to
            rank_in_element_thread_group =
                threadIdx_relative_to_smaller_groups %
                threads_working_on_my_element;  // the rank of this thread
                                                // within the element's threads
        }
        start_index =
            current_assigned_element_id;  // start index for this thread
        end_index = min(current_assigned_element_id + 1,
                        data->num_atoms);  // end index for this thread
        uint64_t bias_per_thread =
            total_cell_bias /
            threads_working_on_my_element;  // number of biases per thread
        start_bias_index = rank_in_element_thread_group *
                           bias_per_thread;  // start bias index for this thread
        end_bias_index =
            min(start_bias_index + bias_per_thread,
                total_cell_bias);  // end bias index for this thread
    }
    for (uint64_t atom_2_index = start_index; atom_2_index < end_index;
         ++atom_2_index) {
        real_t force_neighbor_a[3] = {0.0f, 0.0f,
                                      0.0f};  // force cache for the neighbor
                                              // atom during the following loop
        real_t force_compensate_neighbor[3] = {
            0.0f, 0.0f, 0.0f};  // force compensation for the neighbor atom to
                                // improve numerical stability
        atom_t atom_2_original = data->atoms[atom_2_index];  // surrounding atom
        real_t covalent_radii_2 =
            data->rcov[data->atom_types[atom_2_index]];  // covalent radii of
                                                         // the surrounding atom
        for (uint64_t bias_index = start_bias_index;
             bias_index < end_bias_index; ++bias_index) {
            /* each thread is responsible for one atom pair, so the number of
             * threads should be equal to num_atoms * total_cell_bias */
            int64_t x_bias = (bias_index % mcb0) - (mcb0 / 2);  // x bias
            int64_t y_bias =
                ((bias_index / mcb0) % mcb1) - (mcb1 / 2);  // y bias
            int64_t z_bias =
                (bias_index / (mcb0 * mcb1) % mcb2) - (mcb2 / 2);  // z bias
            assert_(atom_2_index <
                    data->num_atoms);  // make sure the index is in bounds

            atom_t atom_2 = atom_2_original;
            /* translate atom_2 due to periodic boundaries */
            atom_2.x += x_bias * cell[0][0] + y_bias * cell[1][0] +
                        z_bias * cell[2][0];  // translate in x direction
            atom_2.y += x_bias * cell[0][1] + y_bias * cell[1][1] +
                        z_bias * cell[2][1];  // translate in y direction
            atom_2.z += x_bias * cell[0][2] + y_bias * cell[1][2] +
                        z_bias * cell[2][2];  // translate in z direction
            /* calculate the distance between the two atoms */
            real_t distance = sqrtf(powf(atom_1.x - atom_2.x, 2) +
                                    powf(atom_1.y - atom_2.y, 2) +
                                    powf(atom_1.z - atom_2.z, 2));
            /* if the distance is within cutoff range, update neighbor_flags */
            if (distance <= CN_cutoff && distance > 0.0f) {
                /* eq 15 in Grimme et al. 2010
                $CN^A = \sum_{B \neq A}^{N}
                \sqrt{1}{1+exp(-k_1(k_2(R_{A,cov}+R_{B,cov})/r_{AB}-1))}$ */
                real_t exp = expf(
                    -K1 * ((covalent_radii_1 + covalent_radii_2) / distance -
                           1.0f));  // $\exp(-k_1*(\frac{R_A+R_b}{r_{ab}}-1))$
                real_t tanh_value =
                    tanhf(CN_cutoff - distance);  // $\tanh(CN_cutoff - r_{ab})$
                real_t smooth_cutoff =
                    powf(tanh_value,
                         3);  // $\tanh^3(CN_cutoff- r_{ab}))$, this is a smooth
                              // cutoff function added in LASP code.
                real_t d_smooth_cutoff_dr =
                    3.0f * powf(tanh_value, 2) * (1.0f - powf(tanh_value, 2)) *
                    (-1.0f);  // derivative of the smooth cutoff function with
                              // respect to distance
                real_t dCN_datom = powf(1.0f + exp, -2.0f) * (-K1) * exp *
                                       (covalent_radii_1 + covalent_radii_2) *
                                       powf(distance, -3.0f) * smooth_cutoff +
                                   d_smooth_cutoff_dr * 1.0f / (1.0f + exp) /
                                       distance;  // dCN_ij/dr_ij * 1/r_ij
                real_t dE_drik =
                    dE_dCN * dCN_datom;  // dE/drik = dE/dCN * dCN/drik
                /* accumulate force for the central atom */
                // force_central[0] += dE_drik * (atom_1.x - atom_k.x);
                // force_central[1] += dE_drik * (atom_1.y - atom_k.y);
                // force_central[2] += dE_drik * (atom_1.z - atom_k.z);
                /* use Kahan summation to improve numerical stability */
                {
                    real_t y0 =
                        dE_drik * (atom_1.x - atom_2.x) -
                        force_compensate_central
                            [0];  // calculate the difference for x component
                    real_t t0 = force_central[0] +
                                y0;  // add the difference to the local force
                    force_compensate_central[0] =
                        (t0 - force_central[0]) -
                        y0;  // calculate the compensation for the next
                             // iteration
                    force_central[0] =
                        t0;  // update the local force for x component

                    real_t y1 =
                        dE_drik * (atom_1.y - atom_2.y) -
                        force_compensate_central
                            [1];  // calculate the difference for y component
                    real_t t1 = force_central[1] +
                                y1;  // add the difference to the local force
                    force_compensate_central[1] =
                        (t1 - force_central[1]) -
                        y1;  // calculate the compensation for the next
                             // iteration
                    force_central[1] =
                        t1;  // update the local force for y component

                    real_t y2 =
                        dE_drik * (atom_1.z - atom_2.z) -
                        force_compensate_central
                            [2];  // calculate the difference for z component
                    real_t t2 = force_central[2] +
                                y2;  // add the difference to the local force
                    force_compensate_central[2] =
                        (t2 - force_central[2]) -
                        y2;  // calculate the compensation for the next
                             // iteration
                    force_central[2] =
                        t2;  // update the local force for z component
                }
                /* accumulate force for the neighbor atom */
                {
                    real_t y0 =
                        -dE_drik * (atom_1.x - atom_2.x) -
                        force_compensate_neighbor
                            [0];  // calculate the difference for x component
                    real_t t0 = force_neighbor_a[0] +
                                y0;  // add the difference to the local force
                    force_compensate_neighbor[0] =
                        (t0 - force_neighbor_a[0]) -
                        y0;  // calculate the compensation for the next
                             // iteration
                    force_neighbor_a[0] =
                        t0;  // update the local force for x component

                    real_t y1 =
                        -dE_drik * (atom_1.y - atom_2.y) -
                        force_compensate_neighbor
                            [1];  // calculate the difference for y component
                    real_t t1 = force_neighbor_a[1] +
                                y1;  // add the difference to the local force
                    force_compensate_neighbor[1] =
                        (t1 - force_neighbor_a[1]) -
                        y1;  // calculate the compensation for the next
                             // iteration
                    force_neighbor_a[1] =
                        t1;  // update the local force for y component

                    real_t y2 =
                        -dE_drik * (atom_1.z - atom_2.z) -
                        force_compensate_neighbor
                            [2];  // calculate the difference for z component
                    real_t t2 = force_neighbor_a[2] +
                                y2;  // add the difference to the local force
                    force_compensate_neighbor[2] =
                        (t2 - force_neighbor_a[2]) -
                        y2;  // calculate the compensation for the next
                             // iteration
                    force_neighbor_a[2] =
                        t2;  // update the local force for z component
                }
                /* accumulate stress */
                // stress[0 * 3 + 0] += -1.0f * (atom_1.x - atom_k.x) * dE_drik
                // * (atom_1.x - atom_k.x); // stress_xx stress[0 * 3 + 1] +=
                // -1.0f * (atom_1.x - atom_k.x) * dE_drik * (atom_1.y -
                // atom_k.y); // stress_xy stress[0 * 3 + 2] += -1.0f *
                // (atom_1.x - atom_k.x) * dE_drik * (atom_1.z - atom_k.z); //
                // stress_xz stress[1 * 3 + 0] += -1.0f * (atom_1.y - atom_k.y)
                // * dE_drik * (atom_1.x - atom_k.x); // stress_yx stress[1 * 3
                // + 1] += -1.0f * (atom_1.y - atom_k.y) * dE_drik * (atom_1.y -
                // atom_k.y); // stress_yy stress[1 * 3 + 2] += -1.0f *
                // (atom_1.y - atom_k.y) * dE_drik * (atom_1.z - atom_k.z); //
                // stress_yz stress[2 * 3 + 0] += -1.0f * (atom_1.z - atom_k.z)
                // * dE_drik * (atom_1.x - atom_k.x); // stress_zx stress[2 * 3
                // + 1] += -1.0f * (atom_1.z - atom_k.z) * dE_drik * (atom_1.y -
                // atom_k.y); // stress_zy stress[2 * 3 + 2] += -1.0f *
                // (atom_1.z - atom_k.z) * dE_drik * (atom_1.z - atom_k.z); //
                // stress_zz
                /* use Kahan summation to improve numerical stability */
                {
                    real_t y0 =
                        -1.0f * (atom_1.x - atom_2.x) * dE_drik *
                            (atom_1.x - atom_2.x) -
                        stress_compensate[0];  // calculate the difference for
                                               // stress_xx
                    real_t t0 = stress[0 * 3 + 0] +
                                y0;  // add the difference to the local stress
                    stress_compensate[0] = (t0 - stress[0 * 3 + 0]) -
                                           y0;  // calculate the compensation
                                                // for the next iteration
                    stress[0 * 3 + 0] =
                        t0;  // update the local stress for stress_xx

                    real_t y1 =
                        -1.0f * (atom_1.x - atom_2.x) * dE_drik *
                            (atom_1.y - atom_2.y) -
                        stress_compensate[1];  // calculate the difference for
                                               // stress_xy
                    real_t t1 = stress[0 * 3 + 1] +
                                y1;  // add the difference to the local stress
                    stress_compensate[1] = (t1 - stress[0 * 3 + 1]) -
                                           y1;  // calculate the compensation
                                                // for the next iteration
                    stress[0 * 3 + 1] =
                        t1;  // update the local stress for stress_xy

                    real_t y2 =
                        -1.0f * (atom_1.x - atom_2.x) * dE_drik *
                            (atom_1.z - atom_2.z) -
                        stress_compensate[2];  // calculate the difference for
                                               // stress_xz
                    real_t t2 = stress[0 * 3 + 2] +
                                y2;  // add the difference to the local stress
                    stress_compensate[2] = (t2 - stress[0 * 3 + 2]) -
                                           y2;  // calculate the compensation
                                                // for the next iteration
                    stress[0 * 3 + 2] =
                        t2;  // update the local stress for stress_xz

                    real_t y3 =
                        -1.0f * (atom_1.y - atom_2.y) * dE_drik *
                            (atom_1.x - atom_2.x) -
                        stress_compensate[3];  // calculate the difference for
                                               // stress_yx
                    real_t t3 = stress[1 * 3 + 0] +
                                y3;  // add the difference to the local stress
                    stress_compensate[3] = (t3 - stress[1 * 3 + 0]) -
                                           y3;  // calculate the compensation
                                                // for the next iteration
                    stress[1 * 3 + 0] =
                        t3;  // update the local stress for stress_yx

                    real_t y4 =
                        -1.0f * (atom_1.y - atom_2.y) * dE_drik *
                            (atom_1.y - atom_2.y) -
                        stress_compensate[4];  // calculate the difference for
                                               // stress_yy
                    real_t t4 = stress[1 * 3 + 1] +
                                y4;  // add the difference to the local stress
                    stress_compensate[4] = (t4 - stress[1 * 3 + 1]) -
                                           y4;  // calculate the compensation
                                                // for the next iteration
                    stress[1 * 3 + 1] =
                        t4;  // update the local stress for stress_yy

                    real_t y5 =
                        -1.0f * (atom_1.y - atom_2.y) * dE_drik *
                            (atom_1.z - atom_2.z) -
                        stress_compensate[5];  // calculate the difference for
                                               // stress_yz
                    real_t t5 = stress[1 * 3 + 2] +
                                y5;  // add the difference to the local stress
                    stress_compensate[5] = (t5 - stress[1 * 3 + 2]) -
                                           y5;  // calculate the compensation
                                                // for the next iteration
                    stress[1 * 3 + 2] =
                        t5;  // update the local stress for stress_yz

                    real_t y6 =
                        -1.0f * (atom_1.z - atom_2.z) * dE_drik *
                            (atom_1.x - atom_2.x) -
                        stress_compensate[6];  // calculate the difference for
                                               // stress_zx
                    real_t t6 = stress[2 * 3 + 0] +
                                y6;  // add the difference to the local stress
                    stress_compensate[6] = (t6 - stress[2 * 3 + 0]) -
                                           y6;  // calculate the compensation
                                                // for the next iteration
                    stress[2 * 3 + 0] =
                        t6;  // update the local stress for stress_zx

                    real_t y7 =
                        -1.0f * (atom_1.z - atom_2.z) * dE_drik *
                            (atom_1.y - atom_2.y) -
                        stress_compensate[7];  // calculate the difference for
                                               // stress_zy
                    real_t t7 = stress[2 * 3 + 1] +
                                y7;  // add the difference to the local stress
                    stress_compensate[7] = (t7 - stress[2 * 3 + 1]) -
                                           y7;  // calculate the compensation
                                                // for the next iteration
                    stress[2 * 3 + 1] =
                        t7;  // update the local stress for stress_zy

                    real_t y8 =
                        -1.0f * (atom_1.z - atom_2.z) * dE_drik *
                            (atom_1.z - atom_2.z) -
                        stress_compensate[8];  // calculate the difference for
                                               // stress_zz
                    real_t t8 = stress[2 * 3 + 2] +
                                y8;  // add the difference to the local stress
                    stress_compensate[8] = (t8 - stress[2 * 3 + 2]) -
                                           y8;  // calculate the compensation
                                                // for the next iteration
                    stress[2 * 3 + 2] =
                        t8;  // update the local stress for stress_zz
                }
            }
        }
        /* accumulate force of neighboring atom to global memory */
        atomicAdd(
            &data->forces[atom_2_index * 3 + 0],
            force_neighbor_a[0]);  // accumulate the force for the neighbor atom
        atomicAdd(
            &data->forces[atom_2_index * 3 + 1],
            force_neighbor_a[1]);  // accumulate the force for the neighbor atom
        atomicAdd(
            &data->forces[atom_2_index * 3 + 2],
            force_neighbor_a[2]);  // accumulate the force for the neighbor atom
    }
    /* accumulate force of central atom and stress */
    real_t force_central_sum[3] = {0.0f, 0.0f,
                                   0.0f};  // force cache for the central atom
    real_t stress_sum[9] = {
        0.0f};  // stress cache for the central atom across the block
    blockReduceSumThreeBodyKernel(force_central, stress, force_central_sum,
                                  stress_sum);

    /* store the results back to global memory */
    if (threadIdx.x == 0) {
        for (uint16_t i = 0; i < 3; ++i) {
            atomicAdd(&data->forces[atom_1_index * 3 + i],
                      force_central_sum[i]);  // accumulate the force for the
                                              // central atom
        }
        for (uint16_t i = 0; i < 9; ++i) {
            atomicAdd(
                &data->stress[i],
                stress_sum[i] /
                    cell_volume);  // accumulate the stress for the central atom
        }
    }
}