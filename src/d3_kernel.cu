#include <assert.h>

#include "d3_kernel.cuh"
#include "d3_types.h"

#define ACCUMULATE_LEVELS 8
#define ACCUMULATE_STRIDE 8
template <uint64_t N>
struct accumulator_t {
    // Kahan summation state for the base level
    real_t base_sum[N]; // current sum at the base level
    real_t compensation[N]; // compensation for lost low-order bits

    // accumulation hierarchy levels
    real_t levels[ACCUMULATE_LEVELS][N]; // higher-level accumulators
    uint64_t count; // number of additions made

    /**
     * @brief initializes an accumulator
     * call this once at the start.
     */
    __device__ inline void init() {
        for (uint8_t i = 0; i < N; ++i) {
            base_sum[i] = 0.0f;
            compensation[i] = 0.0f;
            for (uint32_t level = 0; level < ACCUMULATE_LEVELS; ++level) {
                levels[level][i] = 0.0f;
            }
        }
        count = 0;
    }

    /**
     * @brief add a value to the accumulator
     * @param value the value(s) to add
     */
    __device__ inline void add(const real_t value[N]) {
        // Kahan summation for the base level
        for (uint8_t i = 0; i < N; ++i) {
            real_t y = value[i] - compensation[i];
            real_t t = base_sum[i] + y;
            compensation[i] = (t - base_sum[i]) - y;
            base_sum[i] = t;
        }
        count += 1;

        // Hierarchical accumulation
        if (count % ACCUMULATE_STRIDE == 0) {
            for (uint8_t i = 0; i < N; ++i) {
                levels[0][i] += base_sum[i];
                base_sum[i] = 0.0f;
                compensation[i] = 0.0f;
            }
            // propagate to higher levels if needed
            uint32_t current_level = 0;
            uint32_t count_at_level = count / ACCUMULATE_STRIDE;
            while (count_at_level % ACCUMULATE_STRIDE == 0 && count_at_level != 0 &&
                   current_level + 1 < ACCUMULATE_LEVELS) {
                for (uint8_t i = 0; i < N; ++i) {
                    levels[current_level + 1][i] += levels[current_level][i];
                    levels[current_level][i] = 0.0f;
                }
                current_level += 1;
                count_at_level /= ACCUMULATE_STRIDE;
            }
        }
    }

    /**
     * @brief get the final accumulated sum
     * @param result the output array to store the result
     */
    __device__ inline void get_sum(real_t result[N]) {
        for (uint8_t i = 0; i < N; ++i) {
            result[i] = base_sum[i];
        }
        for (uint32_t i = 0; i < ACCUMULATE_LEVELS; ++i) {
            for (uint8_t j = 0; j < N; ++j) {
                result[j] += levels[i][j];
            }
        }
    }
};

/**
 * @brief calculate the volume of the simulation cell
 * @param cell 3x3 matrix representing the cell vectors
 * @return the volume of the cell
 */
__device__ inline real_t calculate_cell_volume(const real_t cell[3][3]) {
    // Calculate the volume of the cell using the determinant of the matrix
    // formed by the cell vectors
    return cell[0][0] * (cell[1][1] * cell[2][2] - cell[1][2] * cell[2][1]) -
           cell[0][1] * (cell[1][0] * cell[2][2] - cell[1][2] * cell[2][0]) +
           cell[0][2] * (cell[1][0] * cell[2][1] - cell[1][1] * cell[2][0]);
}

/**
 * @brief warp-level reduction using shuffle down
 * @param val the value to be reduced
 * @return the reduced sum across the warp
 */
__device__ inline real_t warp_reduce_sum(real_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/** @brief block-wise accumulation (reduction) for coordination_number_kernel.
 * @param CN the coordination number to be reduced
 * @param dCN_dr the derivative of coordination number to be reduced
 * @param CN_sum pointer to store the reduced coordination number
 * @param dCN_dr_sum array to store the reduced derivative of coordination number
 */
__device__ inline void block_reduce_CN(real_t CN, real_t dCN_dr[3], real_t *CN_sum, real_t dCN_dr_sum[3]) {
    // shared memory to hold partial sums from each warp
    static __shared__ real_t shared_CN[32];
    static __shared__ real_t shared_dCN_dr[32 * 3];

    // determine lane and warp ID
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // reduce within the warp
    real_t CN_warp = warp_reduce_sum(CN);
    real_t dCN_dr_warp[3];
    for (int i = 0; i < 3; ++i) {
        dCN_dr_warp[i] = warp_reduce_sum(dCN_dr[i]);
    }

    // store to shared memory
    if (lane == 0) {
        shared_CN[warp_id] = CN_warp;
        for (int i = 0; i < 3; ++i) {
            shared_dCN_dr[warp_id * 3 + i] = dCN_dr_warp[i];
        }
    }
    __syncthreads();

    // final reduction by the first warp
    if (warp_id == 0) {
        CN_warp = (threadIdx.x < blockDim.x / warpSize) ? shared_CN[lane]
                                                        : 0.0; // only the first warp writes to shared memory
        for (int i = 0; i < 3; ++i) {
            dCN_dr_warp[i] = (threadIdx.x < blockDim.x / warpSize) ? shared_dCN_dr[lane * 3 + i]
                                                                   : 0.0; // only the first warp writes to shared memory
        }
        *CN_sum = warp_reduce_sum(CN_warp);
        for (int i = 0; i < 3; ++i) {
            dCN_dr_sum[i] = warp_reduce_sum(dCN_dr_warp[i]);
        }
    }
}

/**
 * @brief block-wise accumulation (reduction) for two_body_kernel.
 * @param dE_dCN the derivative of energy with respect to coordination number to be reduced
 * @param energy the energy to be reduced
 * @param force the force array to be reduced
 * @param stress the stress array to be reduced
 * @param dE_dCN_sum pointer to store the reduced derivative of energy with respect to coordination number
 * @param energy_sum pointer to store the reduced energy
 * @param force_sum array to store the reduced force
 * @param stress_sum array to store the reduced stress
 */
__device__ inline void block_reduce_twobody(real_t dE_dCN, real_t energy, real_t force[3], real_t stress[9],
                                            real_t *dE_dCN_sum, real_t *energy_sum, real_t force_sum[3],
                                            real_t stress_sum[9]) {
    // shared memory to hold partial sums from each warp
    static __shared__ real_t shared_dE_dCN[32];
    static __shared__ real_t shared_energy[32];
    static __shared__ real_t shared_force[32 * 3];
    static __shared__ real_t shared_stress[32 * 9];

    // determine lane and warp ID
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // reduce within the warp
    real_t dE_dCN_warp = warp_reduce_sum(dE_dCN);
    real_t energy_warp = warp_reduce_sum(energy);
    real_t force_warp[3];
    for (int i = 0; i < 3; ++i) {
        force_warp[i] = warp_reduce_sum(force[i]);
    }
    real_t stress_warp[9];
    for (int i = 0; i < 9; ++i) {
        stress_warp[i] = warp_reduce_sum(stress[i]);
    }

    // store to shared memory
    if (lane == 0) {
        shared_dE_dCN[warp_id] = dE_dCN_warp;
        shared_energy[warp_id] = energy_warp;
        for (int i = 0; i < 3; ++i) {
            shared_force[warp_id * 3 + i] = force_warp[i];
        }
        for (int i = 0; i < 9; ++i) {
            shared_stress[warp_id * 9 + i] = stress_warp[i];
        }
    }
    __syncthreads();

    // final reduction by the first warp
    if (warp_id == 0) {
        dE_dCN_warp = (threadIdx.x < blockDim.x / warpSize) ? shared_dE_dCN[lane] : 0.0;
        energy_warp = (threadIdx.x < blockDim.x / warpSize) ? shared_energy[lane] : 0.0;
        for (int i = 0; i < 3; ++i) {
            force_warp[i] = (threadIdx.x < blockDim.x / warpSize) ? shared_force[lane * 3 + i] : 0.0;
        }
        for (int i = 0; i < 9; ++i) {
            stress_warp[i] = (threadIdx.x < blockDim.x / warpSize) ? shared_stress[lane * 9 + i] : 0.0;
        }
        *dE_dCN_sum = warp_reduce_sum(dE_dCN_warp);
        *energy_sum = warp_reduce_sum(energy_warp);
        for (int i = 0; i < 3; ++i) {
            force_sum[i] = warp_reduce_sum(force_warp[i]);
        }
        for (int i = 0; i < 9; ++i) {
            stress_sum[i] = warp_reduce_sum(stress_warp[i]);
        }
    }
}

/**
 * @brief block-wise accumulation (reduction) for atm_kernel.
 * @param dE_dCN the derivative of energy with respect to coordination number to be reduced
 * @param energy the energy to be reduced
 * @param force the force array to be reduced
 * @param stress the stress array to be reduced
 * @param dE_dCN_sum pointer to store the reduced derivative of energy with respect to coordination number
 * @param energy_sum pointer to store the reduced energy
 * @param force_sum array to store the reduced force
 * @param stress_sum array to store the reduced stress
 */
__device__ inline void block_reduce_atm(real_t dE_dCN, real_t energy, real_t force[3], real_t stress[9],
                                        real_t *dE_dCN_sum, real_t *energy_sum, real_t force_sum[3],
                                        real_t stress_sum[9]) {
    // shared memory to hold partial sums from each warp
    static __shared__ real_t shared_dE_dCN[32];
    static __shared__ real_t shared_energy[32];
    static __shared__ real_t shared_force[32 * 3];
    static __shared__ real_t shared_stress[32 * 9];

    // determine warp ID and lane
    int warp_id = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;

    // reduce within warp
    real_t dE_dCN_warp = warp_reduce_sum(dE_dCN);
    real_t energy_warp = warp_reduce_sum(energy);
    real_t force_warp[3];
    for (int i = 0; i < 3; ++i) {
        force_warp[i] = warp_reduce_sum(force[i]);
    }
    real_t stress_warp[9];
    for (int i = 0; i < 9; ++i) {
        stress_warp[i] = warp_reduce_sum(stress[i]);
    }

    // store to shared memory
    if (lane == 0) {
        shared_dE_dCN[warp_id] = dE_dCN_warp;
        shared_energy[warp_id] = energy_warp;
        for (int i = 0; i < 3; ++i) {
            shared_force[warp_id * 3 + i] = force_warp[i];
        }
        for (int i = 0; i < 9; ++i) {
            shared_stress[warp_id * 9 + i] = stress_warp[i];
        }
    }
    __syncthreads();

    // final reduction by the first warp
    if (warp_id == 0) {
        dE_dCN_warp = (threadIdx.x < blockDim.x / warpSize) ? shared_dE_dCN[lane] : 0.0;
        energy_warp = (threadIdx.x < blockDim.x / warpSize) ? shared_energy[lane] : 0.0;
        for (int i = 0; i < 3; ++i) {
            force_warp[i] = (threadIdx.x < blockDim.x / warpSize) ? shared_force[lane * 3 + i] : 0.0;
        }
        for (int i = 0; i < 9; ++i) {
            stress_warp[i] = (threadIdx.x < blockDim.x / warpSize) ? shared_stress[lane * 9 + i] : 0.0;
        }
        *dE_dCN_sum = warp_reduce_sum(dE_dCN_warp);
        *energy_sum = warp_reduce_sum(energy_warp);
        for (int i = 0; i < 3; ++i) {
            force_sum[i] = warp_reduce_sum(force_warp[i]);
        }
        for (int i = 0; i < 9; ++i) {
            stress_sum[i] = warp_reduce_sum(stress_warp[i]);
        }
    }
}

/**
 * @brief block-wise accumulation (reduction) for three_body_kernel.
 * @param force the force array to be reduced
 * @param stress the stress array to be reduced
 * @param force_sum array to store the reduced force
 * @param stress_sum array to store the reduced stress
 */
__device__ inline void block_reduce_threebody(real_t force[3], real_t stress[9], real_t *force_sum,
                                              real_t *stress_sum) {
    // shared memory to hold partial sums from each warp
    static __shared__ real_t shared_force[32 * 3];
    static __shared__ real_t shared_stress[32 * 9];

    // determine lane and warp ID
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // reduce within warp
    real_t force_warp[3];
    for (int i = 0; i < 3; ++i) {
        force_warp[i] = warp_reduce_sum(force[i]); // reduce force components within the warp
    }
    real_t stress_warp[9];
    for (int i = 0; i < 9; ++i) {
        stress_warp[i] = warp_reduce_sum(stress[i]); // reduce stress components within the warp
    }

    // store to shared memory
    if (lane == 0) {
        for (int i = 0; i < 3; ++i) {
            shared_force[warp_id * 3 + i] = force_warp[i]; // store force components in shared memory
        }
        for (int i = 0; i < 9; ++i) {
            shared_stress[warp_id * 9 + i] = stress_warp[i]; // store stress components in shared memory
        }
    }
    __syncthreads(); // synchronize threads in the block
    if (warp_id == 0) {
        for (int i = 0; i < 3; ++i) {
            force_warp[i] = (threadIdx.x < blockDim.x / warpSize) ? shared_force[lane * 3 + i]
                                                                  : 0.0; // only the first warp writes to shared memory
        }
        for (int i = 0; i < 9; ++i) {
            stress_warp[i] = (threadIdx.x < blockDim.x / warpSize) ? shared_stress[lane * 9 + i]
                                                                   : 0.0; // only the first warp writes to shared memory
        }
        for (int i = 0; i < 3; ++i) {
            force_sum[i] = warp_reduce_sum(force_warp[i]); // final reduction across warps
        }
        for (int i = 0; i < 9; ++i) {
            stress_sum[i] = warp_reduce_sum(stress_warp[i]); // final reduction across warps
        }
    }
}

/**
 * @brief damping function and derivatives
 * @tparam damping_type Type of damping function to use
 * @param distance Distance between atoms
 * @param cutoff_radius Cutoff radius for damping
 * @param param_1 First parameter for damping. when using zero damping, it is SR_6; when using BJ damping, it is a1
 * @param param_2 Second parameter for damping. when using zero damping, it is SR_8; when using BJ damping, it is a2
 */
template <damping_type_t damping_type>
__device__ void damping(real_t distance, real_t cutoff_radius, real_t param_1, real_t param_2, real_t &damping_6,
                        real_t &damping_8, real_t &ddamping_6, real_t &ddamping_8) {
    if constexpr (damping_type == ZERO_DAMPING) {
        // calculate powers of distance/cutoff_radius
        const real_t base6 = param_1 * cutoff_radius / distance;
        const real_t base6_2 = base6 * base6;
        const real_t base6_4 = base6_2 * base6_2;
        const real_t base6_8 = base6_4 * base6_4;
        const real_t base6_14 = base6_8 * base6_4 * base6_2;
        const real_t base6_15 = base6_14 * base6;
        const real_t base8 = param_2 * cutoff_radius / distance;
        const real_t base8_2 = base8 * base8;
        const real_t base8_4 = base8_2 * base8_2;
        const real_t base8_8 = base8_4 * base8_4;
        const real_t base8_16 = base8_8 * base8_8;
        const real_t base8_17 = base8_16 * base8;

        // calculate damping
        const real_t f_dn_6 = 1.0 / (1.0 + 6.0 * base6_14); // alpha_n = 14
        const real_t f_dn_8 = 1.0 / (1.0 + 6.0 * base8_16); // alpha_n = 16
        const real_t d_f_dn_6 = 6.0 * 14.0 * f_dn_6 * f_dn_6 * base6_15 / param_1 / cutoff_radius;
        const real_t d_f_dn_8 = 6.0 * 16.0 * f_dn_8 * f_dn_8 * base8_17 / param_2 / cutoff_radius;
        // write the result back
        damping_6 = f_dn_6;
        damping_8 = f_dn_8;
        ddamping_6 = d_f_dn_6;
        ddamping_8 = d_f_dn_8;
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
        real_t d_f_dn_6 = 6 * add_entry_6 * distance_5 / ((distance_6 + add_entry_6) * (distance_6 + add_entry_6));
        real_t d_f_dn_8 = 8 * add_entry_8 * distance_7 / ((distance_8 + add_entry_8) * (distance_8 + add_entry_8));
        // write the result back
        damping_6 = f_dn_6;
        damping_8 = f_dn_8;
        ddamping_6 = d_f_dn_6;
        ddamping_8 = d_f_dn_8;
    }
    // detect bad results
    if (isnan(damping_6) || isinf(damping_6)) {
        damping_6 = 0.0;
    }
    if (isnan(ddamping_6) || isinf(ddamping_6)) {
        ddamping_6 = 0.0;
    }
    if (isnan(damping_8) || isinf(damping_8)) {
        damping_8 = 0.0;
    }
    if (isnan(ddamping_8) || isinf(ddamping_8)) {
        ddamping_8 = 0.0;
    }
}

/**
 * @brief calculate neighboring grid cells for a given home cell index
 * @note this function is used for large systems where cell list is employed (workload_distribution=CELL_LIST)
 * @param home_cell_idx index of the home cell
 * @param num_grids number of grids in each dimension
 * @param grid_start_indices starting indices of atoms in each grid cell
 * @param num_atoms total number of atoms
 * @param start_indices output array to store start indices of neighboring cells
 * @param end_indices output array to store end indices of neighboring cells
 * @param shifts output array to store periodic shift information for neighboring cells
 */
__device__ inline void calculate_neighboring_grids(uint64_t home_cell_idx, uint64_t num_grids[3],
                                                   uint64_t *grid_start_indices, uint64_t num_atoms,
                                                   uint64_t start_indices[27], uint64_t end_indices[27],
                                                   int64_t shifts[27][3]) {
    // only first 27 threads populate the neighboring cell data
    if (threadIdx.x < 27) {
        uint64_t neighbor_cell_idx; // the index of neighboring cell this thread calculates
        uint64_t total_num_grids = num_grids[0] * num_grids[1] * num_grids[2];
        // calculate 3D offset for this neighboring cell (-1, 0, or +1 in each dimension)
        int offset[3];
        offset[0] = (threadIdx.x % 3) - 1;
        offset[1] = ((threadIdx.x / 3) % 3) - 1;
        offset[2] = (threadIdx.x / 9) - 1;

        // get home cell's 3D indices
        uint64_t home[3];
        home[0] = home_cell_idx % num_grids[0];
        home[1] = (home_cell_idx / num_grids[0]) % num_grids[1];
        home[2] = home_cell_idx / (num_grids[0] * num_grids[1]);

        // calculate neighbor cell indices with periodic boundaries
        int64_t neighbor[3];
        for (int i = 0; i < 3; ++i) {
            neighbor[i] = static_cast<int64_t>(home[i]) + offset[i];
        }

        // apply periodic boundary conditions and track shifts
        for (int i = 0; i < 3; ++i) {
            shifts[threadIdx.x][i] = 0; // initialize shifts to zero
        }

        for (int i = 0; i < 3; ++i) {
            if (neighbor[i] < 0) {
                neighbor[i] += static_cast<int64_t>(num_grids[i]);
                shifts[threadIdx.x][i] = -1;
            } else if (neighbor[i] >= static_cast<int64_t>(num_grids[i])) {
                neighbor[i] -= static_cast<int64_t>(num_grids[i]);
                shifts[threadIdx.x][i] = 1;
            }
        }

        // Convert 3D indices back to 1D index
        neighbor_cell_idx = neighbor[0] + neighbor[1] * num_grids[0] + neighbor[2] * num_grids[0] * num_grids[1];
        start_indices[threadIdx.x] = grid_start_indices[neighbor_cell_idx];
        end_indices[threadIdx.x] = (neighbor_cell_idx + 1 < total_num_grids) ? grid_start_indices[neighbor_cell_idx + 1]
                                                                             : num_atoms; // handle last cell case
    }
    __syncthreads();
}

/**
 * @brief distribute workload among threads based on number of atoms and cell biases
 * @note this function is used for small systems where cell list is not employed (workload_distribution=ALL_ITERATE)
 */
__device__ inline void distribute_workload(uint64_t num_atoms, uint64_t total_cell_bias, uint64_t *start_index,
                                           uint64_t *end_index, uint64_t *start_bias_index, uint64_t *end_bias_index) {
    if (num_atoms >= blockDim.x) {
        // if the number of atoms exceed number of threads, each thread process a few atoms, going through all possible
        // bias indicies
        uint64_t workload_per_thread = (num_atoms + blockDim.x - 1) / blockDim.x; // number of atoms per thread
        *start_index = threadIdx.x * workload_per_thread;
        *end_index = min(*start_index + workload_per_thread, num_atoms); // the last thread might process fewer atoms
        *start_bias_index = 0;
        *end_bias_index = total_cell_bias; // each thread is responsible for all cell biases
    } else {
        // If the number of atoms is smaller than number of threads, multiple threads process one atom. Divide the
        // threads into groups, each group processes one atom. The first few groups will have one extra thread (larger
        // groups) to guarantee that every thread is assigned to partially equal workload.
        uint64_t base_threads_per_atom = blockDim.x / num_atoms; // base number of threads per atom
        uint64_t remainder = blockDim.x % num_atoms; // remainder threads
        uint64_t num_atoms_with_extra_thread = remainder; // number of atoms getting an extra thread
        uint64_t num_threads_in_larger_groups =
            num_atoms_with_extra_thread * (base_threads_per_atom + 1); // total number of threads in larger groups
        uint64_t atom_idx; // which atom this thread is assigned to
        uint64_t num_threads_in_group; // number of threads in this thread's group
        uint64_t rank_in_group; // the rank of this thread within the atom's threads
        if (threadIdx.x < num_threads_in_larger_groups) {
            // this thread is in a larger group
            num_threads_in_group = base_threads_per_atom + 1;
            atom_idx = threadIdx.x / num_threads_in_group;
            rank_in_group = threadIdx.x % num_threads_in_group;
        } else {
            // this thread falls in later groups that have 'base' threads
            num_threads_in_group = base_threads_per_atom;
            uint64_t threadIdx_relative_to_smaller_groups =
                threadIdx.x - num_threads_in_larger_groups; // relative thread index in smaller groups
            atom_idx = threadIdx_relative_to_smaller_groups / num_threads_in_group + num_atoms_with_extra_thread;
            rank_in_group = threadIdx_relative_to_smaller_groups % num_threads_in_group;
        }
        *start_index = atom_idx;
        *end_index = min(atom_idx + 1, num_atoms);
        uint64_t bias_per_thread = total_cell_bias / num_threads_in_group; // number of biases per thread
        *start_bias_index = rank_in_group * bias_per_thread;
        *end_bias_index = min(*start_bias_index + bias_per_thread, total_cell_bias);
    }
}

/**
 * @brief calculate the coordination number contribution of a given atom pair
 * @param delta_r vector from atom i to atom j
 * @param dist_2 squared distance between atom i and atom j
 * @param rcov_a covalent radius of atom i
 * @param rcov_b covalent radius of atom j
 * @param CN_cutoff cutoff distance for coordination number calculation
 * @param CN_accumulator accumulator to store the coordination number contribution
 * @param dCN_dr_accumulator accumulator to store the derivative of coordination number contribution
 */
__device__ inline void calculate_CN(const real_t delta_r[3], const real_t dist_2, const real_t rcov_a,
                                    const real_t rcov_b, real_t CN_cutoff, accumulator_t<1> &CN_accumulator,
                                    accumulator_t<3> &dCN_dr_accumulator) {

    const real_t dist = sqrtf(dist_2); // r_ij
    const real_t exp = expf(-K1 * ((rcov_a + rcov_b) / dist - 1.0)); // $\exp(-k_1*(\frac{R_A+R_b}{r_{ab}}-1))$
#ifdef SMOOTH_CUTOFF_CN
    const real_t tanh_value = tanhf(CN_cutoff - dist); // $\tanh(CN_cutoff - r_{ab})$
    const real_t smooth_cutoff =
        tanh_value * tanh_value *
        tanh_value; // $\tanh^3(CN_cutoff- r_{ab}))$, a smooth cutoff function added in LASP code.
    const real_t d_smooth_cutoff_dr =
        3.0 * tanh_value * tanh_value * (1.0 - tanh_value * tanh_value) * (-1.0); // d(smooth_cutoff)/dr
    const real_t dCN_datom =
        1 / (1.0 + exp) / (1.0 + exp) * (-K1) * exp * (rcov_a + rcov_b) / (dist * dist * dist) * smooth_cutoff +
        d_smooth_cutoff_dr * 1.0 / (1.0 + exp) / dist; // dCN_ij/dr_ij * 1/r_ij
    // the covalent radii table have already taken K2 coefficient into consideration
    const real_t CN = 1.0 / (1.0 + exp) * smooth_cutoff;
#else
    // the covalent radii table have already taken K2 coefficient into consideration
    const real_t dCN_datom =
        1 / (1.0 + exp) / (1.0 + exp) * (-K1) * exp * (rcov_a + rcov_b) / (dist * dist * dist); // dCN_ij/dr_ij * 1/r_ij
    const real_t CN = 1.0 / (1.0 + exp); // coordination number contribution from atom j to atom i
#endif
    CN_accumulator.add(&CN);
    real_t dCN_dr[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) {
        dCN_dr[i] = dCN_datom * delta_r[i];
    }
    dCN_dr_accumulator.add(dCN_dr);
}

/**
 * @brief this kernel is used to compute the coordination number of each atom in
 * the system.
 * @note this kernel should be launched with a 1D grid of blocks, each block
 * containing a 1D array of threads.
 */
__global__ void coordination_number_kernel(device_data_t *data) {
    const uint64_t atom1_idx = blockIdx.x; // index of the central atom
    const uint16_t atom1_type = data->atom_types[atom1_idx]; // type of the central atom
    const atom_t atom1 = data->atoms[atom1_idx]; // central atom
    const real_t cell[3][3] = {{data->cell[0][0], data->cell[0][1], data->cell[0][2]},
                               {data->cell[1][0], data->cell[1][1], data->cell[1][2]},
                               {data->cell[2][0], data->cell[2][1], data->cell[2][2]}}; // cell matrix
    const real_t CN_cutoff = data->coordination_number_cutoff; // cutoff radius of coordination number
    const real_t rcov1 = data->rcov[atom1_type]; // covalent radii of the central atom
    accumulator_t<1> CN_accumulator; // accumulator for CN
    accumulator_t<3> dCN_dr_accumulator; // accumulator for dCN/dr in x, y, z directions
    CN_accumulator.init();
    dCN_dr_accumulator.init();

    // distribute workload
    switch (data->workload_distribution_type) {
    case CELL_LIST:
        __shared__ uint64_t start_indices[27]; // start indices of atoms in neighboring cells
        __shared__ uint64_t end_indices[27]; // end indices of atoms in neighboring cells
        __shared__ int64_t cells_shifts[27][3]; // shifts corresponding to the neighboring cells
        // calculate neighboring grid cells
        calculate_neighboring_grids(atom1.home_grid_cell, data->num_grid_cells, data->grid_start_indices,
                                    data->num_atoms, start_indices, end_indices, cells_shifts);

        // iterate over neighboring cells
        for (uint8_t i = 0; i < 27; ++i) {
            uint64_t start_idx = start_indices[i]; // start index of atoms in the neighboring cell
            uint64_t end_idx = end_indices[i]; // end index of atoms in the neighboring cell
            int64_t x_shift = cells_shifts[i][0]; // shift in x direction
            int64_t y_shift = cells_shifts[i][1]; // shift in y direction
            int64_t z_shift = cells_shifts[i][2]; // shift in z direction
            for (uint64_t atom2_idx = start_idx + threadIdx.x; atom2_idx < end_idx; atom2_idx += blockDim.x) {
                const atom_t atom2_original = data->atoms[atom2_idx]; // surrounding atom without supercell translation
                const real_t rcov2 = data->rcov[data->atom_types[atom2_idx]]; // covalent radii of the surrounding atom

                atom_t atom2 = atom2_original; // surrounding atom after supercell translation
                // translate atom_2 due to periodic boundaries
                atom2.x += x_shift * cell[0][0] + y_shift * cell[1][0] + z_shift * cell[2][0];
                atom2.y += x_shift * cell[0][1] + y_shift * cell[1][1] + z_shift * cell[2][1];
                atom2.z += x_shift * cell[0][2] + y_shift * cell[1][2] + z_shift * cell[2][2];

                real_t delta_r[3] = {atom1.x - atom2.x, atom1.y - atom2.y,
                                     atom1.z - atom2.z}; // displacement vector from atom 2 to atom 1
                real_t dist_2 = delta_r[0] * delta_r[0] + delta_r[1] * delta_r[1] + delta_r[2] * delta_r[2]; // r_ij^2
                if (dist_2 <= CN_cutoff * CN_cutoff && dist_2 > 0.0) {
                    calculate_CN(delta_r, dist_2, rcov1, rcov2, CN_cutoff, CN_accumulator, dCN_dr_accumulator);
                }
            }
        }

        break;
    case ALL_ITERATE:
        const uint64_t mcb0 = data->max_cell_bias[0]; // maximum cell bias in x direction
        const uint64_t mcb1 = data->max_cell_bias[1]; // maximum cell bias in y direction
        const uint64_t mcb2 = data->max_cell_bias[2]; // maximum cell bias in z direction
        const uint64_t tot_cell_bias = mcb0 * mcb1 * mcb2; // total number of cell biases
        // distribute workload to threads
        uint64_t start_idx; // start atom index for this thread
        uint64_t end_idx; // end atom index for this thread
        uint64_t start_bias_idx; // start bias index for this thread
        uint64_t end_bias_idx; // end bias index for this thread
        distribute_workload(data->num_atoms, tot_cell_bias, &start_idx, &end_idx, &start_bias_idx, &end_bias_idx);

        for (uint64_t atom2_idx = start_idx; atom2_idx < end_idx; ++atom2_idx) {
            const atom_t atom2_original = data->atoms[atom2_idx]; // surrounding atom without supercell translation
            const real_t rcov2 = data->rcov[data->atom_types[atom2_idx]]; // covalent radii of the surrounding atom
            for (uint64_t bias_idx = start_bias_idx; bias_idx < end_bias_idx; ++bias_idx) {
                // iterate over the bias indices
                int64_t x_bias = (bias_idx % mcb0) - (mcb0 / 2);
                int64_t y_bias = ((bias_idx / mcb0) % mcb1) - (mcb1 / 2);
                int64_t z_bias = (bias_idx / (mcb0 * mcb1) % mcb2) - (mcb2 / 2);
                assert_(atom2_idx < data->num_atoms); // make sure the index is in bounds

                atom_t atom_2 = atom2_original; // surrounding atom after supercell translation
                atom_2.x += x_bias * cell[0][0] + y_bias * cell[1][0] + z_bias * cell[2][0];
                atom_2.y += x_bias * cell[0][1] + y_bias * cell[1][1] + z_bias * cell[2][1];
                atom_2.z += x_bias * cell[0][2] + y_bias * cell[1][2] + z_bias * cell[2][2];
                real_t delta_r[3] = {atom1.x - atom_2.x, atom1.y - atom_2.y,
                                     atom1.z - atom_2.z}; // displacement vector from atom 2 to atom 1
                real_t dist_2 = delta_r[0] * delta_r[0] + delta_r[1] * delta_r[1] + delta_r[2] * delta_r[2]; // r_ij^2
                if (dist_2 <= CN_cutoff * CN_cutoff && dist_2 > 0.0) {
                    calculate_CN(delta_r, dist_2, rcov1, rcov2, CN_cutoff, CN_accumulator, dCN_dr_accumulator);
                }
            }
        }
        break;
    }

    // use blockwise reduction to accumulate coordination number and dCN/dr from all threads
    real_t CN_sum, dCN_dr_sum[3], local_CN_sum, local_dCN_dr[3];
    CN_accumulator.get_sum(&local_CN_sum);
    dCN_dr_accumulator.get_sum(local_dCN_dr);
    block_reduce_CN(local_CN_sum, local_dCN_dr, &CN_sum, dCN_dr_sum);
    // write back the results to global memory
    if (threadIdx.x == 0) {
        data->coordination_numbers[atom1_idx] = CN_sum;
        for (uint16_t i = 0; i < 3; ++i) {
            data->dCN_dr[atom1_idx * 3 + i] = dCN_dr_sum[i];
        }
    }
    return;
}
/**
 * @brief this kernel is used to print the coordination numbers of all atoms
 * in the system. It's for debug use.
 * @note this kernel should be launched with a single block and a single
 * thread.
 */
__global__ void print_coordination_number_kernel(device_data_t *data) {
    if (threadIdx.x == 0) {
        printf("Coordination numbers:\n");
        for (uint64_t i = 0; i < data->num_atoms; ++i) {
            printf("Atom %llu, element: %d: %f\n", i, data->atoms[i].element, data->coordination_numbers[i]);
        }
    }
}

/**
 * @brief calculate C6ab and its derivatives with respect to CN of both atoms
 * @param data device data structure containing necessary parameters
 * @param atom1_type type of atom 1
 * @param atom2_type type of atom 2
 * @param CN1 coordination number of atom 1
 * @param CN2 coordination number of atom 2
 * @param C6_result output variable to store the calculated C6ab value
 * @param dC6_dCN1_result output variable to store the derivative of C6
 */
__device__ inline void calculate_c6ab_dual(device_data_t *data, uint64_t atom1_type, uint64_t atom2_type, real_t CN1,
                                           real_t CN2, real_t &C6_result, real_t &dC6_dCN1_result,
                                           real_t &dC6_dCN2_result) {
    /** calculate the coordination number based on
     * the equation comes from Grimme et al. 2010, eq 16.
     * formula: $C_6^{ij} = Z/W$
     * where $Z = \sum_{a,b}C_{6,ref}^{i,j}L_{a,b}$
     * $W = \sum_{a,b}L_{a,b}$
     * $L_{a,b} = \exp(-k3((CN^A-CN^A_{ref,a})^2 + (CN^B-CN^B_{ref,b})^2))$
     */
    real_t max_exp_arg = -FLT_MAX; // maximum exponent argument for L_ij
    for (uint8_t i = 0; i < NUM_REF_C6; ++i) {
        for (uint8_t j = 0; j < NUM_REF_C6; ++j) {
            // find the C6ref
            uint32_t index = atom1_type * data->c6_stride_1 + atom2_type * data->c6_stride_2 + i * data->c6_stride_3 +
                             j * data->c6_stride_4;
            // these entries could be -1.0f if they are not valid, but at least one should be valid
            real_t CN_ref1 = data->c6_ab_ref[index + 1];
            real_t CN_ref2 = data->c6_ab_ref[index + 2];
            if (CN_ref1 > -1.0f && CN_ref2 > -1.0f) {
                // if both coordination numbers are valid, we can use them
                const real_t delta_CN1 = CN1 - CN_ref1;
                const real_t delta_CN2 = CN2 - CN_ref2;
                const real_t exp_arg = -K3 * (delta_CN1 * delta_CN1 + delta_CN2 * delta_CN2);
                max_exp_arg = (max_exp_arg > exp_arg) ? max_exp_arg : exp_arg; // update the maximum exponent argument
            }
        }
    }
    // calculate C6 value
    real_t Z = 0.0;
    real_t W = 0.0;
    real_t Cref_Lij = 0.0; // C6ab_ref * L_ij
    real_t cref_dLij1 = 0.0; // C6ab_ref * dL_ij/dCN_1
    real_t cref_dLij2 = 0.0; // C6ab_ref * dL_ij/dCN_2
    real_t dLij_dCN1 = 0.0; // dL_ij/dCN_1
    real_t dLij_dCN2 = 0.0; // dL_ij/dCN_2
    for (uint8_t i = 0; i < NUM_REF_C6; ++i) {
        for (uint8_t j = 0; j < NUM_REF_C6; ++j) {
            // find the C6ref
            uint32_t index = atom1_type * data->c6_stride_1 + atom2_type * data->c6_stride_2 + i * data->c6_stride_3 +
                             j * data->c6_stride_4;
            real_t c6_ref = data->c6_ab_ref[index + 0];
            // these entries could be -1.0f if they are not valid, but at least one should be valid
            real_t CN_ref1 = data->c6_ab_ref[index + 1];
            real_t CN_ref2 = data->c6_ab_ref[index + 2];
            if (CN_ref1 > -1.0 && CN_ref2 > -1.0) {
                // if both coordination numbers are valid, we can use them
                const real_t delta_CN_1 = CN1 - CN_ref1;
                const real_t delta_CN_2 = CN2 - CN_ref2;
                const real_t exp_arg = -K3 * (delta_CN_1 * delta_CN_1 + delta_CN_2 * delta_CN_2);
                const real_t L_ij = expf(exp_arg - max_exp_arg); // normalized the L_ij value
                real_t dLij1_contribution =
                    -2.0 * K3 * (CN1 - CN_ref1) * L_ij; // part of dL_ij/dCN_1 contributed by the current valid term
                real_t dLij2_contribution =
                    -2.0 * K3 * (CN2 - CN_ref2) * L_ij; // part of dL_ij/dCN_2 contributed by the current valid term

                Z += c6_ref * L_ij;
                W += L_ij;
                Cref_Lij += c6_ref * L_ij;
                cref_dLij1 += c6_ref * dLij1_contribution;
                cref_dLij2 += c6_ref * dLij2_contribution;
                dLij_dCN1 += dLij1_contribution;
                dLij_dCN2 += dLij2_contribution;
            }
        }
    }

    // avoid division by zero
    real_t dC6_dCN1 = (W * W > 0.0) ? (cref_dLij1 * W - Cref_Lij * dLij_dCN1) / (W * W) : 0.0; // dC6ab/dCN_1
    if (isnan(dC6_dCN1) || isinf(dC6_dCN1)) {
        // NaN or inf encountered, bad result
        printf("Error: dC6ab/dCN_1 is NaN or Inf\n");
        printf("Z: %f, W: %f, c_ref_L_ij: %f, c_ref_dL_ij_1: %f, dL_ij_1: %f\n", Z, W, Cref_Lij, cref_dLij1, dLij_dCN1);
        dC6_dCN1 = 0.0f; // reset the value
    }
    real_t dC6_dCN2 = (W * W > 0.0) ? (cref_dLij2 * W - Cref_Lij * dLij_dCN2) / (W * W) : 0.0; // dC6ab/dCN_2
    if (isnan(dC6_dCN2) || isinf(dC6_dCN2)) {
        // NaN or inf encountered, bad result
        printf("Error: dC6ab/dCN_2 is NaN or Inf\n");
        printf("Z: %f, W: %f, c_ref_L_ij: %f, c_ref_dL_ij_2: %f, dL_ij_2: %f\n", Z, W, Cref_Lij, cref_dLij2, dLij_dCN2);
        dC6_dCN2 = 0.0; // reset to 0.0f if it's NaN or Inf
    }
    real_t c6_ab = (W > 0.0) ? Z / W : 0.0; // C6_ab value between atom 1 and 2
    if (isnan(c6_ab) || isinf(c6_ab)) {
        printf("Error: C6_ab is NaN or Inf\n");
        printf("Z: %f, W: %f, c_ref_L_ij: %f, c_ref_dL_ij_1: %f, dL_ij_1: %f\n", Z, W, Cref_Lij, cref_dLij1, dLij_dCN1);
        c6_ab = 0.0; // reset the value
    }
    C6_result = c6_ab;
    dC6_dCN2_result = dC6_dCN2;
    dC6_dCN1_result = dC6_dCN1;
}

__device__ inline void calculate_c6ab(device_data_t *data, uint64_t atom1_type, uint64_t atom_2_type, real_t CN1,
                                      real_t CN2, real_t &c6_ab_result, real_t &dC6_dCN1_result) {
    /** calculate the coordination number based on
     * the equation comes from Grimme et al. 2010, eq 16.
     * formula: $C_6^{ij} = Z/W$
     * where $Z = \sum_{a,b}C_{6,ref}^{i,j}L_{a,b}$
     * $W = \sum_{a,b}L_{a,b}$
     * $L_{a,b} = \exp(-k3((CN^A-CN^A_{ref,a})^2 + (CN^B-CN^B_{ref,b})^2))$
     */
    real_t max_exp_arg = -FLT_MAX; // maximum exponent argument for L_ij
    for (uint8_t i = 0; i < NUM_REF_C6; ++i) {
        for (uint8_t j = 0; j < NUM_REF_C6; ++j) {
            // find the C6ref
            uint32_t index = atom1_type * data->c6_stride_1 + atom_2_type * data->c6_stride_2 + i * data->c6_stride_3 +
                             j * data->c6_stride_4;
            // these entries could be -1.0f if they are not valid, but at least one should be valid
            real_t CN_ref1 = data->c6_ab_ref[index + 1];
            real_t CN_ref2 = data->c6_ab_ref[index + 2];
            if (CN_ref1 > -1.0 && CN_ref2 > -1.0) {
                // if both coordination numbers are valid, we can use them
                const real_t delta_CN_1 = CN1 - CN_ref1;
                const real_t delta_CN_2 = CN2 - CN_ref2;
                const real_t exp_arg = -K3 * (delta_CN_1 * delta_CN_1 + delta_CN_2 * delta_CN_2);
                max_exp_arg = (max_exp_arg > exp_arg) ? max_exp_arg : exp_arg; // update the maximum exponent argument
            }
        }
    }
    // calculate C6 value
    real_t Z = 0.0;
    real_t W = 0.0;
    real_t c_ref_L_ij = 0.0; // C6ab_ref * L_ij
    real_t c_ref_dL_ij_1 = 0.0; // C6ab_ref * dL_ij/dCN_1
    real_t dL_ij_1 = 0.0; // dL_ij/dCN_1
    for (uint8_t i = 0; i < NUM_REF_C6; ++i) {
        for (uint8_t j = 0; j < NUM_REF_C6; ++j) {
            // find the C6ref
            uint32_t index = atom1_type * data->c6_stride_1 + atom_2_type * data->c6_stride_2 + i * data->c6_stride_3 +
                             j * data->c6_stride_4;
            real_t c6_ref = data->c6_ab_ref[index + 0];
            // these entries could be -1.0f if they are not valid, but at least one should be valid
            real_t CN_ref1 = data->c6_ab_ref[index + 1];
            real_t CN_ref2 = data->c6_ab_ref[index + 2];
            if (CN_ref1 > -1.0 && CN_ref2 > -1.0) {
                // if both coordination numbers are valid, we can use them
                const real_t delta_CN_1 = CN1 - CN_ref1;
                const real_t delta_CN_2 = CN2 - CN_ref2;
                const real_t exp_arg = -K3 * (delta_CN_1 * delta_CN_1 + delta_CN_2 * delta_CN_2);
                const real_t L_ij = expf(exp_arg - max_exp_arg); // normalized the L_ij value
                real_t dL_ij_1_part =
                    -2.0 * K3 * (CN1 - CN_ref1) * L_ij; // part of dL_ij/dCN_1 contributed by the current valid term
                Z += c6_ref * L_ij;
                W += L_ij;
                c_ref_L_ij += c6_ref * L_ij;
                c_ref_dL_ij_1 += c6_ref * dL_ij_1_part;
                dL_ij_1 += dL_ij_1_part;
            }
        }
    }

    // avoid division by zero
    real_t dC6ab_dCNa = (W * W > 0.0) ? (c_ref_dL_ij_1 * W - c_ref_L_ij * dL_ij_1) / (W * W) : 0.0; // dC6ab/dCN_1
    if (isnan(dC6ab_dCNa) || isinf(dC6ab_dCNa)) {
        // NaN or inf encountered, bad result
        printf("Error: dC6ab/dCN_1 is NaN or Inf\n");
        printf("Z: %f, W: %f, c_ref_L_ij: %f, c_ref_dL_ij_1: %f, dL_ij_1: %f\n", Z, W, c_ref_L_ij, c_ref_dL_ij_1,
               dL_ij_1);
        dC6ab_dCNa = 0.0; // reset the value
    }
    dC6_dCN1_result = dC6ab_dCNa;

    // avoid division by zero
    real_t c6_ab = (W > 0.0) ? Z / W : 0.0; // C6_ab value between atom 1 and 2
    if (isnan(c6_ab) || isinf(c6_ab)) {
        printf("Error: C6_ab is NaN or Inf\n");
        printf("Z: %f, W: %f, c_ref_L_ij: %f, c_ref_dL_ij_1: %f, dL_ij_1: %f\n", Z, W, c_ref_L_ij, c_ref_dL_ij_1,
               dL_ij_1);
        c6_ab = 0.0; // reset the value
    }
    c6_ab_result = c6_ab;
}

/**
 * @brief calculate the two-body dispersion interaction between a given atom pair
 * @param delta_r vector from atom i to atom j
 * @param cell_volume volume of the simulation cell
 * @param c6_ab C6 coefficient between atom i and atom j
 * @param c8_ab C8 coefficient between atom i and atom j
 * @param dC6ab_dCNa derivative of C6 coefficient with respect to CN of atom i
 * @param dC8ab_dCNa derivative of C8 coefficient with respect to CN of atom i
 * @param r0_cutoff cutoff distance for damping function
 * @param damping_type type of damping function
 * @param damping_param_1 first parameter for damping function
 * @param damping_param_2 second parameter for damping function
 * @param s6 scaling factor for C6 term
 * @param s8 scaling factor for C8 term
 * @param energy_accumulator accumulator to store the dispersion energy
 * @param dE_dCN_accumulator accumulator to store the derivative of energy with respect to CN
 * @param force_accumulator accumulator to store the forces in x, y, z directions
 * @param stress_accumulator accumulator to store the stress tensor components
 */
__device__ inline void calculate_twobody(real_t delta_r[3], real_t dist_2, real_t cell_volume, real_t C6, real_t C8,
                                         real_t dC6_dCN1, real_t dC8_dCNa, real_t r0_cutoff,
                                         damping_type_t damping_type, real_t damping_param_1, real_t damping_param_2,
                                         real_t s6, real_t s8, accumulator_t<1> &energy_accumulator,
                                         accumulator_t<1> &dE_dCN_accumulator, accumulator_t<3> &force_accumulator,
                                         accumulator_t<9> &stress_accumulator) {
    const real_t dist = sqrtf(dist_2); // distance between atom 1 and atom 2
    // calculate distance powers
    real_t dist_3 = dist_2 * dist; // distance^3
    real_t dist_4 = dist_2 * dist_2; // distance^4
    real_t dist_6 = dist_3 * dist_3; // distance^6
    real_t dist_7 = dist_6 * dist; // distance^7
    real_t dist_8 = dist_4 * dist_4; // distance^8
    real_t dist_9 = dist_8 * dist; // distance^9
    real_t dist_10 = dist_6 * dist_4; // distance^10

    // prevent division by zero
    if (dist_6 == 0.0) {
        dist_6 = FLT_MIN;
    }
    if (dist_7 == 0.0) {
        dist_7 = FLT_MIN;
    }
    if (dist_8 == 0.0) {
        dist_8 = FLT_MIN;
    }
    if (dist_9 == 0.0) {
        dist_9 = FLT_MIN;
    }
    if (dist_10 == 0.0) {
        dist_10 = FLT_MIN;
    }

    // calculate the damping function
    real_t f_dn_6, f_dn_8, d_f_dn_6, d_f_dn_8;
    switch (damping_type) {
    case ZERO_DAMPING:
        damping<ZERO_DAMPING>(dist, r0_cutoff, damping_param_1, damping_param_2, f_dn_6, f_dn_8, d_f_dn_6, d_f_dn_8);
        break;
    case BJ_DAMPING:
        damping<BJ_DAMPING>(dist, r0_cutoff, damping_param_1, damping_param_2, f_dn_6, f_dn_8, d_f_dn_6, d_f_dn_8);
        break;
    }

    // calculate the dispersion energy
    const real_t E_6 = s6 * (C6 / dist_6) * f_dn_6; // dispersion energy with n=6
    const real_t dE6_dCN1 = s6 * f_dn_6 * dC6_dCN1 / dist_6; // dE_6/dCN1
    const real_t E_8 = s8 * (C8 / dist_8) * f_dn_8; // dispersion energy with n=8
    const real_t dE8_dCN1 = s8 * f_dn_8 * dC8_dCNa / dist_8; // dE_8/dCN1
    const real_t disp_energy = (E_6 + E_8) / 2.0; // divide by 2 because each atom pair is counted twice
    const real_t dE_dCN1 = (dE6_dCN1 + dE8_dCN1); // derivative of energy with respect to CN1

    real_t force_scalar = 0.0; // dE/dr * 1/r
    /** the first entry of two-body force:
     * $F_a = S_n C_n^{ab} f_{d,n}(r_{ab}) \frac{\partial}{\partial r_a} r_{ab}^{-n}$
     * $F_a = S_n C_n^{ab} f_{d,n}(r_{ab}) * (-n)r_{ab}^{-n-2} * \uparrow{r_{ab}}$
     */
    force_scalar += s6 * C6 * f_dn_6 * (-6.0) / dist_8; // dE_6/dr * 1/r
    force_scalar += s8 * C8 * f_dn_8 * (-8.0) / dist_10; // dE_8/dr * 1/r

    /** the second entry of two-body force:
     * $F_a = S_n C_n^{ab} r_{ab}^{-n} \frac{\partial}{\partial r_a} f_{d,n}(r_{ab})$
     * $F_a = S_n C_n^{ab} r_{ab}^{-n} -f_{d,n}^2 *
     * (6*(-\alpha_n)*(r_{ab}/{S_{r,n}R_0^{AB}})^{-\alpha_n - 1} *
     * 1/(S_{r,n}R_0^{AB})) / r_ab \vec{r_{ab}}$
     */
    force_scalar += s6 * C6 * d_f_dn_6 / dist_7; // dE_6/dr * 1/r
    force_scalar += s8 * C8 * d_f_dn_8 / dist_9; // dE_8/dr * 1/r

    // accumulate the energy, force, stress and dE/dCN using hierarchical Kahan summation
    energy_accumulator.add(&disp_energy);
    dE_dCN_accumulator.add(&dE_dCN1);
    real_t force_contribution[3] = {0.0, 0.0, 0.0};
    for (uint8_t i = 0; i < 3; ++i) {
        force_contribution[i] = force_scalar * delta_r[i];
    }
    real_t stress_contribution[9];
    for (uint8_t i = 0; i < 3; ++i) {
        for (uint8_t j = 0; j < 3; ++j) {
            stress_contribution[i * 3 + j] = -1.0 * delta_r[i] * force_scalar * delta_r[j] / 2.0 / cell_volume;
        }
    }
    force_accumulator.add(force_contribution);
    stress_accumulator.add(stress_contribution);
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
__global__ void two_body_kernel(device_data_t *data) {
    // load parameters from device data
    const real_t s6 = data->functional_params.s6;
    const real_t s8_zero = data->functional_params.s8_zero;
    const real_t s8_bj = data->functional_params.s8_bj;
    const real_t sr_6 = data->functional_params.sr6;
    const real_t sr_8 = data->functional_params.sr8;
    const real_t a1 = data->functional_params.a1;
    const real_t a2 = data->functional_params.a2;
    const real_t cutoff = data->cutoff;
    const uint64_t num_elements = data->num_elements;

    // load the central atom data
    const uint64_t atom1_idx = blockIdx.x; // index of the central atom
    const uint64_t atom1_type = data->atom_types[atom1_idx]; // type of the central atom
    const atom_t atom1 = data->atoms[atom1_idx]; // central atom
    const real_t CN1 = data->coordination_numbers[atom1_idx]; // coordination number of the central atom
    const real_t cell_volume = calculate_cell_volume(data->cell); // volume of the cell
    const real_t cell[3][3] = {{data->cell[0][0], data->cell[0][1], data->cell[0][2]},
                               {data->cell[1][0], data->cell[1][1], data->cell[1][2]},
                               {data->cell[2][0], data->cell[2][1], data->cell[2][2]}}; // cell matrix

    accumulator_t<1> energy_accumulator, dE_dCN_accumulator;
    accumulator_t<3> force_accumulator;
    accumulator_t<9> stress_accumulator;
    energy_accumulator.init();
    dE_dCN_accumulator.init();
    force_accumulator.init();
    stress_accumulator.init();

    if (data->workload_distribution_type == CELL_LIST) {
        // each thread process a few atoms in neighboring cells and all bias indices
        __shared__ uint64_t start_indices[27]; // start indices of atoms in neighboring cells
        __shared__ uint64_t end_indices[27]; // end indices of atoms in neighboring cells
        __shared__ int64_t neighbor_cells_shifts[27][3]; // shifts corresponding to the neighboring cells
        calculate_neighboring_grids(atom1.home_grid_cell, data->num_grid_cells, data->grid_start_indices,
                                    data->num_atoms, start_indices, end_indices, neighbor_cells_shifts);

        // iterate over neighboring cells
        for (uint8_t i = 0; i < 27; ++i) {
            uint64_t start_idx = start_indices[i];
            uint64_t end_idx = end_indices[i];
            int64_t x_shift = neighbor_cells_shifts[i][0];
            int64_t y_shift = neighbor_cells_shifts[i][1];
            int64_t z_shift = neighbor_cells_shifts[i][2];
            for (uint64_t atom2_idx = start_idx + threadIdx.x; atom2_idx < end_idx; atom2_idx += blockDim.x) {
                const atom_t atom2_original = data->atoms[atom2_idx]; // surrounding atom
                const uint64_t atom2_type = data->atom_types[atom2_idx]; // type of the surrounding atom
                const real_t CN2 = data->coordination_numbers[atom2_idx]; // coordination number of the surrounding atom

                real_t C6, dC6_dCN1;
                calculate_c6ab(data, atom1_type, atom2_type, CN1, CN2, C6, dC6_dCN1);

                // calculate c8_ab by $C_8^{AB} = 3C_6^{AB}\sqrt{Q^AQ^B}$
                // the values in data->r2r4 is already squared
                const real_t r2r4_1 = data->r2r4[atom1_type];
                const real_t r2r4_2 = data->r2r4[atom2_type];
                const real_t c8_ab = 3.0 * C6 * r2r4_1 * r2r4_2; // C8ab value
                const real_t dC8_dCN1 = 3.0 * dC6_dCN1 * r2r4_1 * r2r4_2; // dC8ab/dCN_1
                // acquire the R0 cutoff radius between the two atoms
                const real_t r0_cutoff = data->r0ab[atom1_type * num_elements + atom2_type];

                atom_t atom2 = atom2_original; // the actual atom2 participated in the calculation
                // translate atom_2 due to periodic boundaries
                atom2.x +=
                    x_shift * cell[0][0] + y_shift * cell[1][0] + z_shift * cell[2][0]; // translate in x direction
                atom2.y +=
                    x_shift * cell[0][1] + y_shift * cell[1][1] + z_shift * cell[2][1]; // translate in y direction
                atom2.z +=
                    x_shift * cell[0][2] + y_shift * cell[1][2] + z_shift * cell[2][2]; // translate in z direction
                real_t delta_r[3] = {
                    atom1.x - atom2.x,
                    atom1.y - atom2.y,
                    atom1.z - atom2.z,
                };
                real_t distance_square = delta_r[0] * delta_r[0] + delta_r[1] * delta_r[1] + delta_r[2] * delta_r[2];
                if (distance_square < cutoff * cutoff && distance_square > 0.0) {
                    // within cutoff and not the same atom
                    switch (data->damping_type) {
                    case ZERO_DAMPING:
                        calculate_twobody(delta_r, distance_square, cell_volume, C6, c8_ab, dC6_dCN1, dC8_dCN1,
                                          r0_cutoff, ZERO_DAMPING, sr_6, sr_8, s6, s8_zero, energy_accumulator,
                                          dE_dCN_accumulator, force_accumulator, stress_accumulator);
                        break;
                    case BJ_DAMPING:
                        calculate_twobody(delta_r, distance_square, cell_volume, C6, c8_ab, dC6_dCN1, dC8_dCN1,
                                          sqrtf(3.0 * r2r4_1 * r2r4_2), BJ_DAMPING, a1, a2, s6, s8_bj,
                                          energy_accumulator, dE_dCN_accumulator, force_accumulator,
                                          stress_accumulator);
                        break;
                    }
                }
            }
        }
    } else {
        // prefetch and calculate necessary variables during calculation
        const uint64_t mcb0 = data->max_cell_bias[0]; // maximum cell bias in x direction
        const uint64_t mcb1 = data->max_cell_bias[1]; // maximum cell bias in y direction
        const uint64_t mcb2 = data->max_cell_bias[2]; // maximum cell bias in z direction
        const uint64_t total_cell_bias = mcb0 * mcb1 * mcb2; // total number of cell bias

        // distribute workload to threads
        uint64_t start_idx; // start index for this thread
        uint64_t end_idx; // end index for this thread
        uint64_t start_bias_idx; // start bias index for this thread
        uint64_t end_bias_idx; // end bias index for this thread
        distribute_workload(data->num_atoms, total_cell_bias, &start_idx, &end_idx, &start_bias_idx, &end_bias_idx);
        // iterate over surrounding atoms
        for (uint64_t atom2_idx = start_idx; atom2_idx < end_idx; ++atom2_idx) {
            const atom_t atom2_original = data->atoms[atom2_idx]; // surrounding atom
            const uint64_t atom2_type = data->atom_types[atom2_idx]; // type of the surrounding atom
            const real_t CN2 = data->coordination_numbers[atom2_idx]; // coordination number of the surrounding atom

            real_t C6, dC6_dCN1;
            calculate_c6ab(data, atom1_type, atom2_type, CN1, CN2, C6, dC6_dCN1);

            // calculate c8_ab by $C_8^{AB} = 3C_6^{AB}\sqrt{Q^AQ^B}$
            // the values in data->r2r4 is already squared
            const real_t r2r4_1 = data->r2r4[atom1_type];
            const real_t r2r4_2 = data->r2r4[atom2_type];
            const real_t C8 = 3.0 * C6 * r2r4_1 * r2r4_2; // C8ab value
            const real_t dC8_dCN1 = 3.0 * dC6_dCN1 * r2r4_1 * r2r4_2; // dC8ab/dCN_1
            // acquire the R0 cutoff radius between the two atoms
            const real_t r0_cutoff = data->r0ab[atom1_type * data->num_elements + atom2_type];
            // loop over supercells
            for (uint64_t bias_idx = start_bias_idx; bias_idx < end_bias_idx; ++bias_idx) {
                const int64_t x_bias = (bias_idx % mcb0) - (mcb0 / 2); // x bias
                const int64_t y_bias = ((bias_idx / mcb0) % mcb1) - (mcb1 / 2); // y bias
                const int64_t z_bias = (bias_idx / (mcb0 * mcb1) % mcb2) - (mcb2 / 2); // z bias
                assert_(atom2_idx < data->num_atoms); // make sure the index is in bounds

                atom_t atom2 = atom2_original; // the actual atom2 participated in the calculation
                // translate atom_2 due to periodic boundaries
                atom2.x += x_bias * cell[0][0] + y_bias * cell[1][0] + z_bias * cell[2][0]; // translate in x direction
                atom2.y += x_bias * cell[0][1] + y_bias * cell[1][1] + z_bias * cell[2][1]; // translate in y direction
                atom2.z += x_bias * cell[0][2] + y_bias * cell[1][2] + z_bias * cell[2][2]; // translate in z direction
                real_t delta_r[3] = {
                    atom1.x - atom2.x,
                    atom1.y - atom2.y,
                    atom1.z - atom2.z,
                };
                real_t distance_square = delta_r[0] * delta_r[0] + delta_r[1] * delta_r[1] + delta_r[2] * delta_r[2];
                if (distance_square < cutoff * cutoff && distance_square > 0.0) {
                    switch (data->damping_type) {
                    case ZERO_DAMPING:
                        calculate_twobody(delta_r, distance_square, cell_volume, C6, C8, dC6_dCN1, dC8_dCN1, r0_cutoff,
                                          ZERO_DAMPING, sr_6, sr_8, s6, s8_zero, energy_accumulator, dE_dCN_accumulator,
                                          force_accumulator, stress_accumulator);
                        break;
                    case BJ_DAMPING:
                        calculate_twobody(delta_r, distance_square, cell_volume, C6, C8, dC6_dCN1, dC8_dCN1,
                                          sqrtf(3.0 * r2r4_1 * r2r4_2), BJ_DAMPING, a1, a2, s6, s8_bj, energy_accumulator,
                                          dE_dCN_accumulator, force_accumulator, stress_accumulator);
                        break;
                    }
                }
            }
        }
    }

    real_t local_dE_dCN;
    real_t local_energ;
    real_t local_stress[9];
    real_t local_force[3];
    dE_dCN_accumulator.get_sum(&local_dE_dCN);
    energy_accumulator.get_sum(&local_energ);
    force_accumulator.get_sum(local_force);
    stress_accumulator.get_sum(local_stress);

    real_t dE_dCN_sum = 0; // sum of dE/dCN across the block
    real_t energy_sum = 0; // sum of energy across the block
    real_t force_central_sum[3] = {0.0f}; // sum of force of central atom across the block
    real_t stress_sum[9] = {0.0f}; // sum of stress of central atom across the block
    // accumulate the results across the block
    block_reduce_twobody(local_dE_dCN, local_energ, local_force, local_stress, &dE_dCN_sum, &energy_sum,
                         force_central_sum, stress_sum);

    if (threadIdx.x == 0) {
        /** Only the first thread in the block is responsible for writing back the accumulated result.
         * We directly write to global memory for dE/dCN, energy, and force without atomic operations.
         * This is safe because each block processes a single atom, therefore no data race :).
         * However, stress accumulation requires atomic operations to avoid data races.
         * Note that the atoms are rearranged, so when writing to result arrays(energy and force),
         * we use orginal index instead of atom_1_index, which is the rearranged index.
         * However, for intermediates like dE/dCN, we still use atom_1_index for convenience
         */
        uint64_t original_atom1_idx = atom1.original_index;
        // dE/dCN
        data->dE_dCN[atom1_idx] += dE_dCN_sum;
        // energy
        data->energy[original_atom1_idx] += energy_sum;
        // force and stress
        for (uint8_t i = 0; i < 3; ++i) {
            // write back the force without atomic operation, safe because no other thread writes to this memory
            data->forces[original_atom1_idx * 3 + i] += force_central_sum[i];
            for (uint8_t j = 0; j < 3; ++j) {
                atomicAdd(&data->stress[i * 3 + j], stress_sum[i * 3 + j]); // atomic operation here to avoid data races
            }
        }
    }
}

__global__ void atm_kernel(device_data_t *data) {
    // load central atom data
    const uint64_t atom_1_index = blockIdx.x; // index of the central atom
    const uint16_t atom_1_type = data->atom_types[atom_1_index]; // type of the central atom
    const atom_t atom_1 = data->atoms[atom_1_index]; // central atom
    const real_t CN_1 = data->coordination_numbers[atom_1_index]; // coordination number of the central atom

    const real_t cell[3][3] = {{data->cell[0][0], data->cell[0][1], data->cell[0][2]},
                               {data->cell[1][0], data->cell[1][1], data->cell[1][2]},
                               {data->cell[2][0], data->cell[2][1], data->cell[2][2]}}; // cell matrix
    const real_t cell_volume = calculate_cell_volume(cell); // volume of the cell
    const real_t cutoff = data->atm_cutoff;

    accumulator_t<1> energy_accumulator; // energy accumulator for central atom
    accumulator_t<1> dE_dCN_accumulator_central; // dE/dCN accumulator for central atom
    accumulator_t<3> force_accumulator_central; // force accumulator for central atom
    accumulator_t<9> stress_accumulator; // stress accumulator for central atom
    energy_accumulator.init();
    dE_dCN_accumulator_central.init();
    force_accumulator_central.init();
    stress_accumulator.init();

    const uint64_t mcb0 = data->max_cell_bias[0]; // maximum cell bias in x direction
    const uint64_t mcb1 = data->max_cell_bias[1]; // maximum cell bias in y direction
    const uint64_t mcb2 = data->max_cell_bias[2]; // maximum cell bias in z direction
    const uint64_t total_cell_bias = mcb0 * mcb1 * mcb2; // total number of cell bias

    // // distribute workload to threads
    // uint64_t start_index; // start index for this thread
    // uint64_t end_index; // end index for this thread
    // uint64_t start_bias_index; // start bias index for this thread
    // uint64_t end_bias_index; // end bias index for this thread
    // distribute_workload(atom_1_index + 1, total_cell_bias, &start_index, &end_index, &start_bias_index,
    //                     &end_bias_index);

    // iterate over neighbor atom 1
    for (uint64_t atom_2_index = threadIdx.x; atom_2_index <= atom_1_index; atom_2_index += blockDim.x) {
        const atom_t atom_2_original = data->atoms[atom_2_index]; // neighbor atom 1
        const uint16_t atom_2_type = data->atom_types[atom_2_index]; // type of neighbor atom 1
        const real_t CN_2 = data->coordination_numbers[atom_2_index]; // coordination number of neighbor atom 1
        real_t r0ab = data->r0ab[atom_1_type * data->num_elements + atom_2_type]; // R0 cutoff between atom 1 and 2

        accumulator_t<1> energy_accumulator_neighbor1; // energy accumulator for neighbor atom 1
        accumulator_t<1> dE_dCN_accumulator_neighbor1; // dE/dCN accumulator for neighbor atom 1
        accumulator_t<3> force_accumulator_neighbor1; // force accumulator for neighbor atom 1
        energy_accumulator_neighbor1.init();
        dE_dCN_accumulator_neighbor1.init();
        force_accumulator_neighbor1.init();

        // calculate c6_ab and derivative
        real_t c6_ab, dc6ab_dCNa, dc6ab_dCNb;
        calculate_c6ab_dual(data, atom_1_type, atom_2_type, CN_1, CN_2, c6_ab, dc6ab_dCNa, dc6ab_dCNb);
        // loop over periodic imagers of atom 2
        for (uint64_t bias_index_2 = 0; bias_index_2 < total_cell_bias; ++bias_index_2) {
            const int64_t x_bias_2 = (bias_index_2 % mcb0) - (mcb0 / 2); // x bias
            const int64_t y_bias_2 = ((bias_index_2 / mcb0) % mcb1) - (mcb1 / 2); // y bias
            const int64_t z_bias_2 = (bias_index_2 / (mcb0 * mcb1) % mcb2) - (mcb2 / 2); // z bias

            atom_t atom_2 = atom_2_original; // the actual atom2 participated in the calculation
            // translate atom_2 due to periodic boundaries
            atom_2.x +=
                x_bias_2 * cell[0][0] + y_bias_2 * cell[1][0] + z_bias_2 * cell[2][0]; // translate in x direction
            atom_2.y +=
                x_bias_2 * cell[0][1] + y_bias_2 * cell[1][1] + z_bias_2 * cell[2][1]; // translate in y direction
            atom_2.z +=
                x_bias_2 * cell[0][2] + y_bias_2 * cell[1][2] + z_bias_2 * cell[2][2]; // translate in z direction

            real_t rab[3] = {atom_1.x - atom_2.x, atom_1.y - atom_2.y, atom_1.z - atom_2.z};
            real_t dist_ab_2 = rab[0] * rab[0] + rab[1] * rab[1] + rab[2] * rab[2];
            if (dist_ab_2 > cutoff * cutoff || dist_ab_2 < 1e-12) {
                continue; // skip if beyond cutoff
            }

            // loop over atom 3
            for (uint64_t atom_3_index = 0; atom_3_index <= atom_2_index; ++atom_3_index) {
                const atom_t atom_3_original = data->atoms[atom_3_index]; // third atom
                const uint64_t atom_3_type = data->atom_types[atom_3_index]; // type of the third atom
                const real_t CN_3 = data->coordination_numbers[atom_3_index]; // coordination number of the third atom
                real_t r0ac =
                    data->r0ab[atom_1_type * data->num_elements + atom_3_type]; // R0 cutoff between atom 1 and 3
                real_t r0bc =
                    data->r0ab[atom_2_type * data->num_elements + atom_3_type]; // R0 cutoff between atom 2 and 3

                real_t scaling_factor = 1.0;
                if (atom_3_index == atom_2_index && atom_2_index == atom_1_index) {
                    scaling_factor = 1.0 / 6.0;
                } else if (atom_3_index == atom_2_index || atom_3_index == atom_1_index ||
                           atom_2_index == atom_1_index) {
                    scaling_factor = 1.0 / 2.0;
                } else {
                    scaling_factor = 1.0;
                }

                accumulator_t<1> energy_accumulator_neighbor2; // energy accumulator for neighbor atom 2
                accumulator_t<1> dE_dCN_accumulator_neighbor2; // dE/dCN accumulator for neighbor atom 2
                accumulator_t<3> force_accumulator_neighbor2; // force accumulator for neighbor atom 2
                energy_accumulator_neighbor2.init();
                dE_dCN_accumulator_neighbor2.init();
                force_accumulator_neighbor2.init();

                // calculate c6_ac and c6_bc and derivatives
                real_t c6_ac, dC6ac_dCNa, dC6ac_dCNc;
                calculate_c6ab_dual(data, atom_1_type, atom_3_type, CN_1, CN_3, c6_ac, dC6ac_dCNa, dC6ac_dCNc);
                real_t c6_bc, dC6bc_dCNb, dC6bc_dCNc;
                calculate_c6ab_dual(data, atom_2_type, atom_3_type, CN_2, CN_3, c6_bc, dC6bc_dCNb, dC6bc_dCNc);
                // loop over supercells for atom 3
                for (uint64_t bias_index_3 = 0; bias_index_3 < total_cell_bias; ++bias_index_3) {
                    const int64_t x_bias_3 = (bias_index_3 % mcb0) - (mcb0 / 2); // x bias
                    const int64_t y_bias_3 = ((bias_index_3 / mcb0) % mcb1) - (mcb1 / 2); // y bias
                    const int64_t z_bias_3 = (bias_index_3 / (mcb0 * mcb1) % mcb2) - (mcb2 / 2); // z bias

                    atom_t atom_3 = atom_3_original; // the actual atom3 participated in the calculation
                    // translate atom_3 due to periodic boundaries
                    atom_3.x += x_bias_3 * cell[0][0] + y_bias_3 * cell[1][0] +
                                z_bias_3 * cell[2][0]; // translate in x direction
                    atom_3.y += x_bias_3 * cell[0][1] + y_bias_3 * cell[1][1] +
                                z_bias_3 * cell[2][1]; // translate in y direction
                    atom_3.z += x_bias_3 * cell[0][2] + y_bias_3 * cell[1][2] +
                                z_bias_3 * cell[2][2]; // translate in z direction

                    real_t rac[3] = {atom_1.x - atom_3.x, atom_1.y - atom_3.y, atom_1.z - atom_3.z};
                    real_t dist_ac_2 = rac[0] * rac[0] + rac[1] * rac[1] + rac[2] * rac[2];
                    real_t rbc[3] = {atom_2.x - atom_3.x, atom_2.y - atom_3.y, atom_2.z - atom_3.z};
                    real_t dist_bc_2 = rbc[0] * rbc[0] + rbc[1] * rbc[1] + rbc[2] * rbc[2];

                    if (dist_ac_2 > cutoff * cutoff || dist_bc_2 > cutoff * cutoff || dist_ac_2 < 1e-12 ||
                        dist_bc_2 < 1e-12) {
                        continue; // skip if beyond cutoff
                    }

                    // calculate ATM interaction
                    real_t distance_ab_2 = rab[0] * rab[0] + rab[1] * rab[1] + rab[2] * rab[2];
                    real_t distance_ab = sqrtf(distance_ab_2);
                    real_t distance_ab_3 = distance_ab_2 * distance_ab;
                    real_t distance_ab_4 = distance_ab_2 * distance_ab_2;
                    real_t distance_ac_2 = rac[0] * rac[0] + rac[1] * rac[1] + rac[2] * rac[2];
                    real_t distance_ac = sqrtf(distance_ac_2);
                    real_t distance_ac_3 = distance_ac_2 * distance_ac;
                    real_t distance_ac_4 = distance_ac_2 * distance_ac_2;
                    real_t distance_bc_2 = rbc[0] * rbc[0] + rbc[1] * rbc[1] + rbc[2] * rbc[2];
                    real_t distance_bc = sqrtf(distance_bc_2);
                    real_t distance_bc_3 = distance_bc_2 * distance_bc;
                    real_t distance_bc_4 = distance_bc_2 * distance_bc_2;
                    real_t r_square = distance_ab_2 * distance_ac_2 * distance_bc_2;
                    real_t r = sqrtf(r_square);
                    // calculate $\cos\theta_{abc}\cos\theta_{acb}\cos\theta_{bca}$
                    real_t tmp_bac = (distance_ab_2 + distance_ac_2 - distance_bc_2); // part of cosine for angle BAC
                    real_t tmp_cab = (distance_ab_2 - distance_ac_2 + distance_bc_2); // part of cosine for angle CAB
                    real_t tmp_abc = (-distance_ab_2 + distance_ac_2 + distance_bc_2); // part of cosine for angle ABC
                    real_t cosine = tmp_bac * tmp_cab * tmp_abc / (8.0 * r_square);
                    real_t angle_term = (1.0 + 3.0 * cosine) / (r_square * r);

                    real_t dangle_ab =
                        3.0 / 8.0 *
                        (distance_ab_3 * distance_ab_3 + distance_ab_4 * (distance_ac_2 + distance_bc_2) +
                         distance_ab_2 * (3.0 * distance_bc_2 * distance_bc_2 + 2.0 * distance_bc_2 * distance_ac_2 +
                                          3.0 * distance_ac_2 * distance_ac_2) -
                         5.0 * (distance_bc_2 - distance_ac_2) * (distance_bc_2 - distance_ac_2) *
                             (distance_bc_2 + distance_ac_2)) /
                        (r_square * r_square * r); // d(angle_term)/d(distance_ab)

                    real_t dangle_ac =
                        3.0 / 8.0 *
                        (distance_ac_3 * distance_ac_3 + distance_ac_4 * (distance_bc_2 + distance_ab_2) +
                         distance_ac_2 * (3.0 * distance_bc_2 * distance_bc_2 + 2.0 * distance_bc_2 * distance_ab_2 +
                                          3.0 * distance_ab_2 * distance_ab_2) -
                         5.0 * (distance_bc_2 - distance_ab_2) * (distance_bc_2 - distance_ab_2) *
                             (distance_bc_2 + distance_ab_2)) /
                        (r_square * r_square * r); // d(angle_term)/d(distance_ac)

                    real_t dangle_bc =
                        3.0 / 8.0 *
                        (distance_bc_3 * distance_bc_3 + distance_bc_4 * (distance_ac_2 + distance_ab_2) +
                         distance_bc_2 * (3.0 * distance_ac_2 * distance_ac_2 + 2.0 * distance_ac_2 * distance_ab_2 +
                                          3.0 * distance_ab_2 * distance_ab_2) -
                         5.0 * (distance_ac_2 - distance_ab_2) * (distance_ac_2 - distance_ab_2) *
                             (distance_ac_2 + distance_ab_2)) /
                        (r_square * r_square * r); // d(angle_term)/d(distance_bc)

                    real_t C9 = -sqrtf(c6_ab * c6_ac * c6_bc); // C9 value
                    real_t dC9_dc6ab = -0.5 * C9 / c6_ab; // dC9/dC6ab
                    real_t dC9_dc6ac = -0.5 * C9 / c6_ac; // dC9/dC6ac
                    real_t dC9_dc6bc = -0.5 * C9 / c6_bc; // dC9/dC6bc
                    real_t dC9_dCNa = dC9_dc6ab * dc6ab_dCNa + dC9_dc6ac * dC6ac_dCNa; // dC9/dCNa
                    real_t dC9_dCNb = dC9_dc6ab * dc6ab_dCNb + dC9_dc6bc * dC6bc_dCNb; // dC9/dCNb
                    real_t dC9_dCNc = dC9_dc6ac * dC6ac_dCNc + dC9_dc6bc * dC6bc_dCNc; // dC9/dCNc

                    real_t distance_avg = powf(distance_ab_2 * distance_ac_2 * distance_bc_2, 1.0f / 6.0f);
                    real_t r0_cutoff = powf(r0ab * r0ac * r0bc, 1.0f / 3.0f);

                    real_t damping_8 = 1.0 / (1.0 + 6.0 * powf(4.0 / 3.0 * r0_cutoff / distance_avg, 16.0));
                    real_t d_damping_8 =
                        2.0 * 16.0 * powf(4.0 / 3.0 * r0_cutoff / distance_avg, 16.0) * damping_8 * damping_8;

                    // calculate energy
                    real_t energy = C9 * angle_term * damping_8 * scaling_factor /
                                    3.0; // contribution to energy for each atom in the triplet
                    // accumulate energy
                    energy_accumulator.add(&energy);
                    energy_accumulator_neighbor1.add(&energy);
                    energy_accumulator_neighbor2.add(&energy);

                    // calculate dE_dCN
                    real_t dE_dCNa = -dC9_dCNa * angle_term * damping_8 * scaling_factor;
                    real_t dE_dCNb = -dC9_dCNb * angle_term * damping_8 * scaling_factor;
                    real_t dE_dCNc = -dC9_dCNc * angle_term * damping_8 * scaling_factor;
                    dE_dCN_accumulator_central.add(&dE_dCNa);
                    dE_dCN_accumulator_neighbor1.add(&dE_dCNb);
                    dE_dCN_accumulator_neighbor2.add(&dE_dCNc);

                    real_t force_a[3] = {0.0f, 0.0f, 0.0f}; // force on atom a
                    real_t force_b[3] = {0.0f, 0.0f, 0.0f}; // force on atom b
                    real_t force_c[3] = {0.0f, 0.0f, 0.0f}; // force on atom c
                    real_t stress[9] = {0.0}; // stress contribution
                    /**
                     * Fortran calculation code for stress:
                     * dS(:, :) = spread(dGij, 1, 3) * spread(vij, 2, 3)&
                     * & + spread(dGik, 1, 3) * spread(vik, 2, 3)&
                     * & + spread(dGjk, 1, 3) * spread(vjk, 2, 3)
                     * here, dS is stress[9] in this code,
                     * dGij is force_ab, dGik is force_ac, dGjk is force_bc
                     * vij is rab, vik is rac, vjk is rbc
                     */
                    for (uint8_t i = 0; i < 3; ++i) {
                        real_t force_ab_scalar =
                            -C9 * (dangle_ab * damping_8 - angle_term * d_damping_8) / distance_ab_2;
                        real_t force_ac_scalar =
                            -C9 * (dangle_ac * damping_8 - angle_term * d_damping_8) / distance_ac_2;
                        real_t force_bc_scalar =
                            -C9 * (dangle_bc * damping_8 - angle_term * d_damping_8) / distance_bc_2;
                        for (uint8_t j = 0; j < 3; ++j) {
                            stress[i * 3 + j] +=
                                (-1.0f * force_ab_scalar * rab[i] * rab[j] + -1.0f * force_ac_scalar * rac[i] * rac[j] +
                                 -1.0f * force_bc_scalar * rbc[i] * rbc[j]) /
                                cell_volume * scaling_factor;
                        }
                        real_t force_ab = force_ab_scalar * rab[i];
                        real_t force_ac = force_ac_scalar * rac[i];
                        real_t force_bc = force_bc_scalar * rbc[i];
                        force_a[i] = (force_ab + force_ac);
                        force_b[i] = (-force_ab + force_bc);
                        force_c[i] = (-force_ac - force_bc);
                    }
                    // accumulate forces and stress
                    force_accumulator_central.add(force_a);
                    force_accumulator_neighbor1.add(force_b);
                    force_accumulator_neighbor2.add(force_c);
                    stress_accumulator.add(stress);
                }
                // accumulate neighbor 2 results
                real_t energy_neighbor2;
                real_t dE_dCN_neighbor2;
                real_t force_neighbor2[3];
                energy_accumulator_neighbor2.get_sum(&energy_neighbor2);
                dE_dCN_accumulator_neighbor2.get_sum(&dE_dCN_neighbor2);
                force_accumulator_neighbor2.get_sum(force_neighbor2);
                // atomic additions to global memory
                uint64_t atom_3_original_index = atom_3_original.original_index;
                atomicAdd(&data->energy[atom_3_original_index], energy_neighbor2);
                atomicAdd(&data->dE_dCN[atom_3_index], dE_dCN_neighbor2);
                for (uint8_t i = 0; i < 3; ++i) {
                    atomicAdd(&data->forces[atom_3_original_index * 3 + i], force_neighbor2[i]);
                }
            }
        }
        // accumulate neighbor 1 results
        real_t energy_neighbor1;
        real_t dE_dCN_neighbor1;
        real_t force_neighbor1[3];
        energy_accumulator_neighbor1.get_sum(&energy_neighbor1);
        dE_dCN_accumulator_neighbor1.get_sum(&dE_dCN_neighbor1);
        force_accumulator_neighbor1.get_sum(force_neighbor1);
        // atomic additions to global memory
        uint64_t atom_2_original_index = atom_2_original.original_index;
        atomicAdd(&data->energy[atom_2_original_index], energy_neighbor1);
        atomicAdd(&data->dE_dCN[atom_2_index], dE_dCN_neighbor1);
        for (uint8_t i = 0; i < 3; ++i) {
            atomicAdd(&data->forces[atom_2_original_index * 3 + i], force_neighbor1[i]);
        }
    }
    // accumulate central atom results
    real_t energy_central_local;
    real_t dE_dCN_central_local;
    real_t force_central_local[3];
    real_t stress_local[9];
    energy_accumulator.get_sum(&energy_central_local);
    dE_dCN_accumulator_central.get_sum(&dE_dCN_central_local);
    force_accumulator_central.get_sum(force_central_local);
    stress_accumulator.get_sum(stress_local);
    // blockwise reduction
    real_t energy_central_sum;
    real_t dE_dCN_central_sum;
    real_t force_central_sum[3];
    real_t stress_sum[9];
    block_reduce_twobody(dE_dCN_central_local, energy_central_local, force_central_local, stress_local,
                         &dE_dCN_central_sum, &energy_central_sum, force_central_sum, stress_sum);
    // atomic additions to global memory
    if (threadIdx.x == 0) {
        uint64_t atom_1_original_index = atom_1.original_index;
        atomicAdd(&data->energy[atom_1_original_index], energy_central_sum);
        atomicAdd(&data->dE_dCN[atom_1_index], dE_dCN_central_sum);
        for (uint8_t i = 0; i < 3; ++i) {
            atomicAdd(&data->forces[atom_1_original_index * 3 + i], force_central_sum[i]);
        }
        // atomic addition to global stress
        for (uint8_t i = 0; i < 9; ++i) {
            atomicAdd(&data->stress[i], stress_sum[i]);
        }
    }
    return;
}

__device__ inline void calculate_three_body_interaction(real_t delta_r[3], real_t dist_2, real_t rcov1, real_t rcov2,
                                                        real_t CN_cutoff, real_t dE_dCN1, real_t dE_dCN2,
                                                        accumulator_t<3> &force_accumulator,
                                                        accumulator_t<9> &stress_accumulator) {
    const real_t dist = sqrtf(dist_2); // distance between atom 1 and atom 2
    const real_t dist_3 = dist_2 * dist; // distance^3

    /**
     * eq 15 in Grimme et al. 2010
     * $CN^A = \sum_{B \neq A}^{N}
     * \sqrt{1}{1+exp(-k_1(k_2(R_{A,cov}+R_{B,cov})/r_{AB}-1))}$
     */
    real_t exp = expf(-K1 * ((rcov1 + rcov2) / dist - 1.0)); // $\exp(-k_1*(\frac{R_A+R_b}{r_{ab}}-1))$
#ifdef SMOOTH_CUTOFF_CN
    real_t tanh_value = tanhf(CN_cutoff - dist); // $\tanh(CN_cutoff - r_{ab})$
    real_t smooth_cutoff =
        powf(tanh_value, 3); // $\tanh^3(CN_cutoff- r_{ab}))$, this is a smooth cutoff function added in LASP code.
    real_t d_smooth_cutoff_dr = 3.0 * powf(tanh_value, 2) * (1.0 - powf(tanh_value, 2)) * (-1.0); // d(smooth_cutoff)/dr
    // the covalent radii table have already taken K2 coefficient into consideration
    real_t dCN_datom = powf(1.0 + exp, -2.0) * (-K1) * exp * (rcov1 + rcov2) / dist_3 * smooth_cutoff +
                       d_smooth_cutoff_dr * 1.0 / (1.0 + exp) / dist; // dCN_ij/dr_ij * 1/r_ij
#else
    // the covalent radii table have already taken K2 coefficient into consideration
    real_t dCN_datom = powf(1.0 + exp, -2.0) * (-K1) * exp * (rcov1 + rcov2) / dist_3; // dCN_ij/dr_ij * 1/r_ij
#endif
    // dE/drik = dE/dCN * dCN/drik
    real_t dE_drik = (dE_dCN1 + dE_dCN2) * dCN_datom;
    if (isnan(dE_drik) || isinf(dE_drik)) {
        // NaN or inf encountered, bad result
        printf("Error: dE_dCN: %f, dCN_datom: %f\n", dE_dCN1 + dE_dCN2, dCN_datom);
        dE_drik = 0.0; // reset to 0.0
    }
    // accumulate force for the central atom and neighboring atom
    // force_central += dE/drik * delta_r
    // use Kahan summation to improve numerical stability
    real_t force_contribution[3];
    real_t stress_contribution[9];
    for (uint8_t i = 0; i < 3; ++i) {
        force_contribution[i] = (dE_drik * delta_r[i]);
        for (uint8_t j = 0; j < 3; ++j) {
            stress_contribution[i * 3 + j] = (-1.0 * delta_r[i] * dE_drik * delta_r[j]) / 2.0;
        }
    }
    force_accumulator.add(force_contribution);
    stress_accumulator.add(stress_contribution);
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
__global__ void three_body_kernel(device_data_t *data) {
    uint64_t atom1_idx = blockIdx.x; // index of central atom
    atom_t atom1 = data->atoms[atom1_idx]; // central atom
    uint64_t atom1_type = data->atom_types[atom1_idx]; // type of the central atom
    real_t rcov1 = data->rcov[atom1_type]; // covalent radius of the central atom
    real_t dE_dCN1 = data->dE_dCN[atom1_idx]; // dE/dCN of the central atom

    accumulator_t<3> force_accumulator;
    accumulator_t<9> stress_accumulator;
    force_accumulator.init();
    stress_accumulator.init();

    const uint64_t mcb0 = data->max_cell_bias[0]; // maximum cell bias in x direction
    const uint64_t mcb1 = data->max_cell_bias[1]; // maximum cell bias in y direction
    const uint64_t mcb2 = data->max_cell_bias[2]; // maximum cell bias in z direction
    uint64_t total_cell_bias = mcb0 * mcb1 * mcb2; // total number of cell bias
    const real_t cell[3][3] = {{data->cell[0][0], data->cell[0][1], data->cell[0][2]},
                               {data->cell[1][0], data->cell[1][1], data->cell[1][2]},
                               {data->cell[2][0], data->cell[2][1], data->cell[2][2]}}; // local cell matrix
    const real_t cell_volume = calculate_cell_volume(cell); // cell volume
    const real_t CN_cutoff = data->coordination_number_cutoff; // cutoff radius of coordination number

    switch (data->workload_distribution_type) {
    case CELL_LIST:
        // each thread process a few atoms in neighboring cells and all bias indices
        __shared__ uint64_t start_indices[27]; // start indices of atoms in neighboring cells
        __shared__ uint64_t end_indices[27]; // end indices of atoms in neighboring cells
        __shared__ int64_t neighbor_cells_shifts[27][3]; // shifts corresponding to the neighboring cells
        calculate_neighboring_grids(atom1.home_grid_cell, data->num_grid_cells, data->grid_start_indices,
                                    data->num_atoms, start_indices, end_indices, neighbor_cells_shifts);

        for (uint8_t i = 0; i < 27; ++i) {
            uint64_t start_idx = start_indices[i];
            uint64_t end_idx = end_indices[i];
            int64_t x_shift = neighbor_cells_shifts[i][0];
            int64_t y_shift = neighbor_cells_shifts[i][1];
            int64_t z_shift = neighbor_cells_shifts[i][2];
            for (uint64_t atom2_idx = start_idx + threadIdx.x; atom2_idx < end_idx; atom2_idx += blockDim.x) {
                real_t dE_dCN2 = data->dE_dCN[atom2_idx]; // dE/dCN of the surrounding atom
                atom_t atom2_original = data->atoms[atom2_idx]; // surrounding atom without supercell translation
                real_t rcov2 = data->rcov[data->atom_types[atom2_idx]]; // covalent radii of the surrounding atom

                atom_t atom2 = atom2_original;
                // translate atom_2 due to periodic boundaries
                atom2.x += x_shift * cell[0][0] + y_shift * cell[1][0] + z_shift * cell[2][0];
                atom2.y += x_shift * cell[0][1] + y_shift * cell[1][1] + z_shift * cell[2][1];
                atom2.z += x_shift * cell[0][2] + y_shift * cell[1][2] + z_shift * cell[2][2];

                // calculate distance square
                real_t delta_r[3] = {atom1.x - atom2.x, atom1.y - atom2.y, atom1.z - atom2.z};
                real_t dist_2 = delta_r[0] * delta_r[0] + delta_r[1] * delta_r[1] + delta_r[2] * delta_r[2];

                // calcualte three-body interaction if within cutoff
                if (dist_2 <= CN_cutoff * CN_cutoff && dist_2 > 0.0) {
                    calculate_three_body_interaction(delta_r, dist_2, rcov1, rcov2, CN_cutoff, dE_dCN1, dE_dCN2,
                                                     force_accumulator, stress_accumulator);
                }
            }
        }
        break;
    case ALL_ITERATE:
        uint64_t start_idx;
        uint64_t end_idx;
        uint64_t start_bias_idx;
        uint64_t end_bias_idx;
        distribute_workload(data->num_atoms, total_cell_bias, &start_idx, &end_idx, &start_bias_idx, &end_bias_idx);

        // iterate over surrounding atoms
        for (uint64_t atom2_idx = start_idx; atom2_idx < end_idx; ++atom2_idx) {
            real_t dE_dCN2 = data->dE_dCN[atom2_idx]; // dE/dCN of the surrounding atom
            atom_t atom2_original = data->atoms[atom2_idx]; // original surrounding atom before translation
            real_t rcov2 = data->rcov[data->atom_types[atom2_idx]]; // covalent radii of the surrounding atom

            // iterate over cell biases
            for (uint64_t bias_idx = start_bias_idx; bias_idx < end_bias_idx; ++bias_idx) {
                int64_t x_bias = (bias_idx % mcb0) - (mcb0 / 2); // x bias
                int64_t y_bias = ((bias_idx / mcb0) % mcb1) - (mcb1 / 2); // y bias
                int64_t z_bias = (bias_idx / (mcb0 * mcb1) % mcb2) - (mcb2 / 2); // z bias
                assert_(atom2_idx < data->num_atoms); // make sure the index is in bounds

                atom_t atom2 = atom2_original;
                // translate atom_2 due to periodic boundaries
                atom2.x += x_bias * cell[0][0] + y_bias * cell[1][0] + z_bias * cell[2][0]; // translate in x direction
                atom2.y += x_bias * cell[0][1] + y_bias * cell[1][1] + z_bias * cell[2][1]; // translate in y direction
                atom2.z += x_bias * cell[0][2] + y_bias * cell[1][2] + z_bias * cell[2][2]; // translate in z direction

                // calculate distance square
                real_t delta_r[3] = {atom1.x - atom2.x, atom1.y - atom2.y, atom1.z - atom2.z};
                real_t dist_2 = delta_r[0] * delta_r[0] + delta_r[1] * delta_r[1] + delta_r[2] * delta_r[2];

                // calculate three-body interaction if within cutoff
                if (dist_2 <= CN_cutoff * CN_cutoff && dist_2 > 0.0) {
                    calculate_three_body_interaction(delta_r, dist_2, rcov1, rcov2, CN_cutoff, dE_dCN1, dE_dCN2,
                                                     force_accumulator, stress_accumulator);
                }
            }
        }
        break;
    }

    // accumulate force of central atom and stress
    real_t local_force_sum[3];
    real_t local_stress_sum[9];
    force_accumulator.get_sum(local_force_sum);
    stress_accumulator.get_sum(local_stress_sum);

    real_t force_central_sum[3] = {0.0f}; // force sum for the central atom across the block
    real_t stress_sum[9] = {0.0f}; // stress sum across the block
    block_reduce_threebody(local_force_sum, local_stress_sum, force_central_sum, stress_sum);

    // the first thread accumulates the results back to global memory
    /**
     * note that the atoms are rearranged, so when writing to result arrays(energy and force),
     * we use orginal index instead of atom_1_index, which is the rearranged index.
     * However, for intermediates like dE/dCN, we still use atom_1_index for convenience
     */
    if (threadIdx.x == 0) {
        uint64_t original_atom_1_index = atom1.original_index;
        for (uint8_t i = 0; i < 3; ++i) {
            // accumulate force
            data->forces[original_atom_1_index * 3 + i] += force_central_sum[i];
            for (uint8_t j = 0; j < 3; ++j) {
                // accumulate stress
                atomicAdd(&data->stress[i * 3 + j], stress_sum[i * 3 + j] / cell_volume);
            }
        }
    }
}