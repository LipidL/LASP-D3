#include <assert.h>

#include "d3_kernel.cuh"
#include "d3_types.h"

#define ACCUMULATE_LEVELS 1
#define ACCUMULATE_STRIDE 8
template <uint64_t N>
struct HierarchicalKahanAccumulator
{
    // Kahan summation state for the base level
    real_t base_sum[N];     // current sum at the base level
    real_t compensation[N]; // compensation for lost low-order bits

    // accumulation hierarchy levels
    real_t levels[ACCUMULATE_LEVELS][N]; // higher-level accumulators
    uint64_t count;                      // number of additions made

    /**
     * @brief initializes an accumulator
     * Call this once at the start.
     */
    __inline__ __device__ void init()
    {
        for (uint8_t i = 0; i < N; ++i)
        {
            base_sum[i] = 0.0f;
            compensation[i] = 0.0f;
            for (uint32_t level = 0; level < ACCUMULATE_LEVELS; ++level)
            {
                levels[level][i] = 0.0f;
            }
        }
        count = 0;
    }

    /**
     * @brief add a value to the accumulator
     */
    __inline__ __device__ void add(const real_t value[N])
    {
        // Kahan summation for the base level
        for (uint8_t i = 0; i < N; ++i)
        {
            real_t y = value[i] - compensation[i];
            real_t t = base_sum[i] + y;
            compensation[i] = (t - base_sum[i]) - y;
            base_sum[i] = t;
        }
        count += 1;

        // Hierarchical accumulation
        if (count % ACCUMULATE_STRIDE == 0)
        {
            for (uint8_t i = 0; i < N; ++i)
            {
                levels[0][i] += base_sum[i];
                base_sum[i] = 0.0f;
                compensation[i] = 0.0f;
            }
            // propagate to higher levels if needed
            uint32_t current_level = 0;
            uint32_t count_at_level = count / ACCUMULATE_STRIDE;
            while (count_at_level % ACCUMULATE_STRIDE == 0 && count_at_level != 0 && current_level + 1 < ACCUMULATE_LEVELS)
            {
                for (uint8_t i = 0; i < N; ++i)
                {
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
     */
    __inline__ __device__ void get_sum(real_t result[N])
    {
        for (uint8_t i = 0; i < N; ++i)
        {
            result[i] = base_sum[i];
        }
        for (uint32_t i = 0; i < ACCUMULATE_LEVELS; ++i)
        {
            for (uint8_t j = 0; j < N; ++j)
            {
                result[j] += levels[i][j];
            }
        }
    }
};

__inline__ __device__ real_t calculate_cell_volume(const real_t cell[3][3])
{
    // Calculate the volume of the cell using the determinant of the matrix
    // formed by the cell vectors
    return cell[0][0] * (cell[1][1] * cell[2][2] - cell[1][2] * cell[2][1]) -
           cell[0][1] * (cell[1][0] * cell[2][2] - cell[1][2] * cell[2][0]) +
           cell[0][2] * (cell[1][0] * cell[2][1] - cell[1][1] * cell[2][0]);
}

__inline__ __device__ real_t warpReduceSum(real_t val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/** @brief blockReduceSum for coordination_number_kernel.
 * The entries taht need to be added are: CN and dCN_dr (3 components).
 */
__inline__ __device__ void blockReduceCNKernel(
    real_t coordination_number, real_t dCN_dr[3],
    real_t *coordination_number_sum, real_t dCN_dr_sum[3])
{
    static __shared__ real_t shared_coordination_number[32];
    static __shared__ real_t shared_dCN_dr[32 * 3]; // 3 components for dCN_dr

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    coordination_number =
        warpReduceSum(coordination_number); // reduce CN within the warp
    for (int i = 0; i < 3; ++i)
    {
        dCN_dr[i] =
            warpReduceSum(dCN_dr[i]); // reduce dCN_dr components within the warp
    }

    if (lane == 0)
    {
        shared_coordination_number[warp_id] =
            coordination_number; // store CN in shared memory
        for (int i = 0; i < 3; ++i)
        {
            shared_dCN_dr[warp_id * 3 + i] =
                dCN_dr[i]; // store dCN_dr components in shared memory
        }
    }
    __syncthreads(); // synchronize threads in the block
    if (warp_id == 0)
    {
        coordination_number = (threadIdx.x < blockDim.x / warpSize)
                                  ? shared_coordination_number[lane]
                                  : 0.0f; // only the first warp writes to shared memory
        for (int i = 0; i < 3; ++i)
        {
            dCN_dr[i] = (threadIdx.x < blockDim.x / warpSize)
                            ? shared_dCN_dr[lane * 3 + i]
                            : 0.0f; // only the first warp writes to shared memory
        }
        *coordination_number_sum = warpReduceSum(coordination_number);
        for (int i = 0; i < 3; ++i)
        {
            dCN_dr_sum[i] = warpReduceSum(dCN_dr[i]);
        }
    }
}

/**
 * @brief blockReduceSum for two_body_kernel. the entries that need to be added
 * are: dE_dCN, energy, force, stress
 */
__inline__ __device__ void blockReduceTwoBodyKernel(
    real_t dE_dCN, real_t energy, real_t force[3], real_t stress[9],
    real_t *dE_dCN_sum, real_t *energy_sum, real_t force_central_sum[3],
    real_t stress_central_sum[9])
{
    static __shared__ real_t shared_dE_dCN[32];
    static __shared__ real_t shared_energy[32];
    static __shared__ real_t shared_force[32 * 3];  // 3 components for force
    static __shared__ real_t shared_stress[32 * 9]; // 9 components for stress

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    dE_dCN = warpReduceSum(dE_dCN); // reduce dE_dCN within the warp
    energy = warpReduceSum(energy); // reduce energy within the warp
    for (int i = 0; i < 3; ++i)
    {
        force[i] =
            warpReduceSum(force[i]); // reduce force components within the warp
    }
    for (int i = 0; i < 9; ++i)
    {
        stress[i] = warpReduceSum(
            stress[i]); // reduce stress components within the warp
    }

    if (lane == 0)
    {
        shared_dE_dCN[warp_id] = dE_dCN; // store dE_dCN in shared memory
        shared_energy[warp_id] = energy; // store energy in shared memory
        for (int i = 0; i < 3; ++i)
        {
            shared_force[warp_id * 3 + i] =
                force[i]; // store force components in shared memory
        }
        for (int i = 0; i < 9; ++i)
        {
            shared_stress[warp_id * 9 + i] =
                stress[i]; // store stress components in shared memory
        }
    }
    __syncthreads(); // synchronize threads in the block
    if (warp_id == 0)
    {
        dE_dCN = (threadIdx.x < blockDim.x / warpSize)
                     ? shared_dE_dCN[lane]
                     : 0.0f; // only the first warp writes to shared memory
        energy = (threadIdx.x < blockDim.x / warpSize)
                     ? shared_energy[lane]
                     : 0.0f; // only the first warp writes to shared memory
        for (int i = 0; i < 3; ++i)
        {
            force[i] =
                (threadIdx.x < blockDim.x / warpSize)
                    ? shared_force[lane * 3 + i]
                    : 0.0f; // only the first warp writes to shared memory
        }
        for (int i = 0; i < 9; ++i)
        {
            stress[i] =
                (threadIdx.x < blockDim.x / warpSize)
                    ? shared_stress[lane * 9 + i]
                    : 0.0f; // only the first warp writes to shared memory
        }
        *dE_dCN_sum = warpReduceSum(dE_dCN);
        *energy_sum = warpReduceSum(energy);
        for (int i = 0; i < 3; ++i)
        {
            force_central_sum[i] = warpReduceSum(force[i]);
        }
        for (int i = 0; i < 9; ++i)
        {
            stress_central_sum[i] = warpReduceSum(stress[i]);
        }
    }
}

/**
 * @brief blockReduceSum for three_body_kernel. the entries that need to be
 * added are: force, stress
 */
__inline__ __device__ void blockReduceSumThreeBodyKernel(
    real_t force[3], real_t stress[9], real_t *force_central_sum,
    real_t *stress_central_sum)
{
    static __shared__ real_t shared_force[32 * 3];  // 3 components for force
    static __shared__ real_t shared_stress[32 * 9]; // 9 components for stress
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    for (int i = 0; i < 3; ++i)
    {
        force[i] =
            warpReduceSum(force[i]); // reduce force components within the warp
    }
    for (int i = 0; i < 9; ++i)
    {
        stress[i] = warpReduceSum(
            stress[i]); // reduce stress components within the warp
    }
    if (lane == 0)
    {
        for (int i = 0; i < 3; ++i)
        {
            shared_force[warp_id * 3 + i] =
                force[i]; // store force components in shared memory
        }
        for (int i = 0; i < 9; ++i)
        {
            shared_stress[warp_id * 9 + i] =
                stress[i]; // store stress components in shared memory
        }
    }
    __syncthreads(); // synchronize threads in the block
    if (warp_id == 0)
    {
        for (int i = 0; i < 3; ++i)
        {
            force[i] =
                (threadIdx.x < blockDim.x / warpSize)
                    ? shared_force[lane * 3 + i]
                    : 0.0f; // only the first warp writes to shared memory
        }
        for (int i = 0; i < 9; ++i)
        {
            stress[i] =
                (threadIdx.x < blockDim.x / warpSize)
                    ? shared_stress[lane * 9 + i]
                    : 0.0f; // only the first warp writes to shared memory
        }
        for (int i = 0; i < 3; ++i)
        {
            force_central_sum[i] =
                warpReduceSum(force[i]); // final reduction across warps
        }
        for (int i = 0; i < 9; ++i)
        {
            stress_central_sum[i] =
                warpReduceSum(stress[i]); // final reduction across warps
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
                        real_t *damping_6, real_t *damping_8,
                        real_t *d_damping_6, real_t *d_damping_8)
{
    if constexpr (damping_type == ZERO_DAMPING)
    {
        // calculate damping
        const real_t f_dn_6 = 1 / (1 + 6.0f * powf(param_1 * cutoff_radius / distance, 14.0f)); // alpha_n = 14
        const real_t f_dn_8 = 1 / (1 + 6.0f * powf(param_2 * cutoff_radius / distance, 16.0f)); // alpha_n = 16
        const real_t d_f_dn_6 = 6.0f * 14.0f * f_dn_6 * f_dn_6 * powf(param_1 * cutoff_radius / distance, 15.0f) / param_1 / cutoff_radius;
        const real_t d_f_dn_8 = 6.0f * 16.0f * f_dn_8 * f_dn_8 * powf(param_2 * cutoff_radius / distance, 17.0f) / param_2 / cutoff_radius;
        // write the result back
        *damping_6 = f_dn_6;
        *damping_8 = f_dn_8;
        *d_damping_6 = d_f_dn_6;
        *d_damping_8 = d_f_dn_8;
    }
    else if constexpr (damping_type == BJ_DAMPING)
    {
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
        *damping_6 = f_dn_6;
        *damping_8 = f_dn_8;
        *d_damping_6 = d_f_dn_6;
        *d_damping_8 = d_f_dn_8;
    }
}

__device__ inline void calculate_neighboring_grids(
    uint64_t home_cell_index, uint64_t num_grids[3],
    uint64_t *grid_start_indices, uint64_t num_atoms,
    uint64_t start_indices[27], uint64_t end_indices[27],
    int64_t shifts[27][3]
) {
    // Only first 27 threads populate the neighboring cell data
    if (threadIdx.x < 27) {
        uint64_t neighbor_cell_index; // the index of neighboring cell this thread calculates
        uint64_t total_num_grids = num_grids[0] * num_grids[1] * num_grids[2];
        // Calculate 3D offset for this neighboring cell (-1, 0, or +1 in each dimension)
        int offset_x = (threadIdx.x % 3) - 1;
        int offset_y = ((threadIdx.x / 3) % 3) - 1;
        int offset_z = (threadIdx.x / 9) - 1;
        
        // Get home cell's 3D indices
        uint64_t home_x = home_cell_index % num_grids[0];
        uint64_t home_y = (home_cell_index / num_grids[0]) % num_grids[1];
        uint64_t home_z = home_cell_index / (num_grids[0] * num_grids[1]);
        
        // Calculate neighbor cell indices with periodic boundaries
        int64_t neighbor_x = (int64_t)home_x + offset_x;
        int64_t neighbor_y = (int64_t)home_y + offset_y;
        int64_t neighbor_z = (int64_t)home_z + offset_z;
        
        // Apply periodic boundary conditions and track shifts
        shifts[threadIdx.x][0] = 0;
        shifts[threadIdx.x][1] = 0;
        shifts[threadIdx.x][2] = 0;
        
        if (neighbor_x < 0) {
            neighbor_x += num_grids[0];
            shifts[threadIdx.x][0] = -1;
        } else if (neighbor_x >= (int64_t)num_grids[0]) {
            neighbor_x -= num_grids[0];
            shifts[threadIdx.x][0] = 1;
        }
        
        if (neighbor_y < 0) {
            neighbor_y += num_grids[1];
            shifts[threadIdx.x][1] = -1;
        } else if (neighbor_y >= (int64_t)num_grids[1]) {
            neighbor_y -= num_grids[1];
            shifts[threadIdx.x][1] = 1;
        }
        if (neighbor_z < 0) {
            neighbor_z += num_grids[2];
            shifts[threadIdx.x][2] = -1;
        } else if (neighbor_z >= (int64_t)num_grids[2]) {
            neighbor_z -= num_grids[2];
            shifts[threadIdx.x][2] = 1;
        }
        
        // Convert 3D indices back to 1D index
        neighbor_cell_index = neighbor_x + neighbor_y * num_grids[0] + neighbor_z * num_grids[0] * num_grids[1];
        start_indices[threadIdx.x] = grid_start_indices[neighbor_cell_index];
        end_indices[threadIdx.x] = (neighbor_cell_index + 1 < total_num_grids) ? grid_start_indices[neighbor_cell_index + 1] : num_atoms; // handle last cell case
    }
    __syncthreads();
}

__device__ void distribute_workload(uint64_t num_atoms, uint64_t total_cell_bias, uint64_t *start_index, uint64_t *end_index, uint64_t *start_bias_index, uint64_t *end_bias_index)
{
    // distribute workload to threads
    if (num_atoms >= blockDim.x)
    {
        /* if the number of atoms exceed number of threads, each thread process
         * a few atoms, going through all possible bias indicies */
        uint64_t workload_per_thread = (num_atoms + blockDim.x - 1) / blockDim.x; // number of atoms per thread
        *start_index = threadIdx.x * workload_per_thread;
        *end_index = min(*start_index + workload_per_thread, num_atoms); // the last thread might process fewer atoms
        *start_bias_index = 0;
        *end_bias_index = total_cell_bias; // each thread is responsible for all cell biases
    }
    else
    {
        /* If the number of atoms is smaller than number of threads, multiple
         * threads process one atom.
         * Divide the threads into groups, each group processes one atom.
         * The first few groups will have one extra thread to guarantee that
         * every thread is assigned to partially equal workload.
         */
        uint64_t threads_per_atom_base = blockDim.x / num_atoms;                                                // base number of threads per atom
        uint64_t threads_per_atom_remainder = blockDim.x % num_atoms;                                           // remainder threads
        uint64_t num_atoms_getting_extra_thread = threads_per_atom_remainder;                                   // number of atoms getting an extra thread
        uint64_t threads_in_larger_groups_total = num_atoms_getting_extra_thread * (threads_per_atom_base + 1); // total number of threads in larger groups
        uint64_t current_assigned_atom_id;                                                                      // which atom this thread is assigned to
        uint64_t threads_working_on_my_atom;                                                                    // number of threads working on my atom
        uint64_t rank_in_element_thread_group;                                                                  // the rank of this thread within the atom's threads
        if (threadIdx.x < threads_in_larger_groups_total)
        {
            // this thread is in a larger group
            threads_working_on_my_atom = threads_per_atom_base + 1;
            current_assigned_atom_id = threadIdx.x / threads_working_on_my_atom;
            rank_in_element_thread_group = threadIdx.x % threads_working_on_my_atom;
        }
        else
        {
            // this thread falls in later groups that have 'base' threads
            threads_working_on_my_atom = threads_per_atom_base;
            uint64_t threads_already_assigned_to_larger_groups = threads_in_larger_groups_total;                     // number of threads already assigned to larger groups
            uint64_t threadIdx_relative_to_smaller_groups = threadIdx.x - threads_already_assigned_to_larger_groups; // relative thread index in smaller groups
            current_assigned_atom_id =
                threadIdx_relative_to_smaller_groups / threads_working_on_my_atom + num_atoms_getting_extra_thread;
            rank_in_element_thread_group = threadIdx_relative_to_smaller_groups % threads_working_on_my_atom;
        }
        *start_index = current_assigned_atom_id;
        *end_index = min(current_assigned_atom_id + 1, num_atoms);
        uint64_t bias_per_thread = total_cell_bias / threads_working_on_my_atom; // number of biases per thread
        *start_bias_index = rank_in_element_thread_group * bias_per_thread;
        *end_bias_index = min(*start_bias_index + bias_per_thread, total_cell_bias);
    }
}

__device__ inline void calculate_CN(
    atom_t atom_1, atom_t atom_2,
    real_t covalent_radii_1, real_t covalent_radii_2,
    real_t CN_cutoff,
    HierarchicalKahanAccumulator<1> &CN_accumulator,
    HierarchicalKahanAccumulator<3> &dCN_dr_accumulator
) {
    real_t delta_r[3] = {
        atom_1.x - atom_2.x,  // delta x
        atom_1.y - atom_2.y,  // delta y
        atom_1.z - atom_2.z   // delta z
    }; // delta_r between atom 1 and atom 2
    // calculate the distance between the two atoms
    real_t distance = sqrtf(powf(atom_1.x - atom_2.x, 2) + powf(atom_1.y - atom_2.y, 2) + powf(atom_1.z - atom_2.z, 2));
    // if the distance is within cutoff range, calculate the coordination number
    if (distance <= CN_cutoff && distance > 0.0f) {
        real_t exp = expf(-K1 * ((covalent_radii_1 + covalent_radii_2) / distance - 1.0f));  // $\exp(-k_1*(\frac{R_A+R_b}{r_{ab}}-1))$
        // real_t tanh_value = tanhf(CN_cutoff - distance);  // $\tanh(CN_cutoff - r_{ab})$
        // real_t smooth_cutoff = powf(tanh_value, 3);  // $\tanh^3(CN_cutoff- r_{ab}))$, this is a smooth cutoff function added in LASP code.
        // real_t d_smooth_cutoff_dr = 3.0f * powf(tanh_value, 2) * (1.0f - powf(tanh_value, 2)) * (-1.0f);  // d(smooth_cutoff)/dr
        real_t dCN_datom = powf(1.0f + exp, -2.0f) * (-K1) * exp * (covalent_radii_1 + covalent_radii_2) * powf(distance, -3.0f); // * smooth_cutoff + d_smooth_cutoff_dr * 1.0f / (1.0f + exp) / distance;  // dCN_ij/dr_ij * 1/r_ij

        // the covalent radii table have already taken K2 coefficient into consideration
        real_t coordination_number = 1.0f / (1.0f + exp); // * smooth_cutoff;
        // accumulate coordination number and dCN/dr using Kahan summation with batching
        CN_accumulator.add(&coordination_number);
        real_t dCN_dr_contribution[3] = {0.0f, 0.0f, 0.0f};
        for (uint16_t i = 0; i < 3; ++i) {
            dCN_dr_contribution[i] = dCN_datom * delta_r[i];
        }
        dCN_dr_accumulator.add(dCN_dr_contribution);
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
__global__ void coordination_number_kernel(device_data_t *data)
{
    const uint64_t atom_1_index = blockIdx.x;                                                                  // each block is responsible for one central atom
    const uint64_t atom_1_type = data->atom_types[atom_1_index];                                               // type of the central atom
    const atom_t atom_1 = data->atoms[atom_1_index];                                                           // central atom
    const real_t cell[3][3] = {
        {data->cell[0][0], data->cell[0][1], data->cell[0][2]},
        {data->cell[1][0], data->cell[1][1], data->cell[1][2]},
        {data->cell[2][0], data->cell[2][1], data->cell[2][2]}}; // cell matrix
    const real_t CN_cutoff = data->coordination_number_cutoff;   // cutoff radius of coordination number
    real_t covalent_radii_1 = data->rcov[atom_1_type]; // covalent radii of the central atom
    HierarchicalKahanAccumulator<1> CN_accumulator;
    HierarchicalKahanAccumulator<3> dCN_dr_accumulators; // accumulators for dCN/dr in x, y, z directions
    CN_accumulator.init();
    dCN_dr_accumulators.init();

    // distribute workload
    switch (data->workload_distribution_type) {
        case CELL_LIST:
            // each thread process a few atoms in neighboring cells and all bias indices
            __shared__ uint64_t start_indices[27]; // start indices of atoms in neighboring cells
            __shared__ uint64_t end_indices[27];   // end indices of atoms in neighboring cells
            __shared__ int64_t neighbor_cells_shifts[27][3]; // shifts corresponding to the neighboring cells
            calculate_neighboring_grids(
                atom_1.home_grid_cell,
                data->num_grid_cells,
                data->grid_start_indices,
                data->num_atoms,
                start_indices,
                end_indices,
                neighbor_cells_shifts
            );

            for (uint8_t i = 0; i < 27; ++i) {
                uint64_t start_index = start_indices[i];
                uint64_t end_index = end_indices[i];
                int64_t x_shift = neighbor_cells_shifts[i][0];
                int64_t y_shift = neighbor_cells_shifts[i][1];
                int64_t z_shift = neighbor_cells_shifts[i][2];
                for (uint64_t atom_2_index = start_index + threadIdx.x; atom_2_index < end_index; atom_2_index += blockDim.x) {
                    atom_t atom_2_original = data->atoms[atom_2_index];                   // surrounding atom without supercell translation
                    real_t covalent_radii_2 = data->rcov[data->atom_types[atom_2_index]]; // covalent radii of the surrounding atom

                    atom_t atom_2 = atom_2_original;
                    // translate atom_2 due to periodic boundaries
                    atom_2.x += x_shift * cell[0][0] + y_shift * cell[1][0] + z_shift * cell[2][0];
                    atom_2.y += x_shift * cell[0][1] + y_shift * cell[1][1] + z_shift * cell[2][1];
                    atom_2.z += x_shift * cell[0][2] + y_shift * cell[1][2] + z_shift * cell[2][2];

                    calculate_CN(
                        atom_1, atom_2,
                        covalent_radii_1, covalent_radii_2,
                        CN_cutoff,
                        CN_accumulator,
                        dCN_dr_accumulators
                    );
                }
            }

            break;
        case ALL_ITERATE:
            // no special preparation needed
            const uint64_t total_cell_bias = data->max_cell_bias[0] * data->max_cell_bias[1] * data->max_cell_bias[2]; // total number of cell bias
            const uint64_t mcb0 = data->max_cell_bias[0];                                                              // maximum cell bias in x direction
            const uint64_t mcb1 = data->max_cell_bias[1];                                                              // maximum cell bias in y direction
            const uint64_t mcb2 = data->max_cell_bias[2];                                                              // maximum cell bias in z direction
            // distribute workload to threads
            uint64_t start_index;      // start atom index for this thread
            uint64_t end_index;        // end atom index for this thread
            uint64_t start_bias_index; // start bias index for this thread
            uint64_t end_bias_index;   // end bias index for this thread
            distribute_workload(data->num_atoms, total_cell_bias, &start_index, &end_index, &start_bias_index, &end_bias_index);

            for (uint64_t atom_2_index = start_index; atom_2_index < end_index; ++atom_2_index)
            {
                atom_t atom_2_original = data->atoms[atom_2_index];                   // surrounding atom without supercell translation
                real_t covalent_radii_2 = data->rcov[data->atom_types[atom_2_index]]; // covalent radii of the surrounding atom
                for (uint64_t bias_index = start_bias_index; bias_index < end_bias_index; ++bias_index)
                {
                    // iterate over the bias indices
                    int64_t x_bias = (bias_index % mcb0) - (mcb0 / 2);                 // x bias
                    int64_t y_bias = ((bias_index / mcb0) % mcb1) - (mcb1 / 2);        // y bias
                    int64_t z_bias = (bias_index / (mcb0 * mcb1) % mcb2) - (mcb2 / 2); // z bias
                    assert_(atom_2_index < data->num_atoms);                           // make sure the index is in bounds

                    atom_t atom_2 = atom_2_original;
                    // translate atom_2 due to periodic boundaries
                    atom_2.x += x_bias * cell[0][0] + y_bias * cell[1][0] + z_bias * cell[2][0];
                    atom_2.y += x_bias * cell[0][1] + y_bias * cell[1][1] + z_bias * cell[2][1];
                    atom_2.z += x_bias * cell[0][2] + y_bias * cell[1][2] + z_bias * cell[2][2];

                    calculate_CN(
                        atom_1, atom_2,
                        covalent_radii_1, covalent_radii_2,
                        CN_cutoff,
                        CN_accumulator,
                        dCN_dr_accumulators
                    );
                }
            }
            break;
    }

    // use blockwise reduction to accumulate coordination number and dCN/dr from all threads
    real_t CN_sum, dCN_dr_sum[3], local_CN_sum, local_dCN_dr[3];
    CN_accumulator.get_sum(&local_CN_sum);
    dCN_dr_accumulators.get_sum(local_dCN_dr);
    blockReduceCNKernel(local_CN_sum, local_dCN_dr, &CN_sum, dCN_dr_sum);
    // write back the results to global memory
    if (threadIdx.x == 0)
    {
        data->coordination_numbers[atom_1_index] = CN_sum;
        for (uint16_t i = 0; i < 3; ++i)
        {
            data->dCN_dr[atom_1_index * 3 + i] = dCN_dr_sum[i];
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
__global__ void print_coordination_number_kernel(device_data_t *data)
{
    if (threadIdx.x == 0)
    {
        printf("Coordination numbers:\n");
        for (uint64_t i = 0; i < data->num_atoms; ++i)
        {
            printf("Atom %llu, element: %d: %f\n", i, data->atoms[i].element,
                   data->coordination_numbers[i]);
        }
    }
}

__device__ inline void calculate_c6ab(
    device_data_t *data,
    uint64_t atom_1_type, uint64_t atom_2_type,
    real_t coordination_number_1, real_t coordination_number_2,
    real_t &c6_ab_result, real_t &dC6_dCN_1_result
) {
    /** calculate the coordination number based on 
     * the equation comes from Grimme et al. 2010, eq 16.
     * formula: $C_6^{ij} = Z/W$
     * where $Z = \sum_{a,b}C_{6,ref}^{i,j}L_{a,b}$
     * $W = \sum_{a,b}L_{a,b}$
     * $L_{a,b} = \exp(-k3((CN^A-CN^A_{ref,a})^2 + (CN^B-CN^B_{ref,b})^2))$
     */
    real_t max_exponent_arg = -FLT_MAX;  // maximum exponent argument for L_ij
    for (uint8_t i = 0; i < NUM_REF_C6; ++i) {
        for (uint8_t j = 0; j < NUM_REF_C6; ++j) {
            // find the C6ref
            uint32_t index = atom_1_type * data->c6_stride_1 +
                                atom_2_type * data->c6_stride_2 + 
                                i * data->c6_stride_3 +
                                j * data->c6_stride_4;
            // these entries could be -1.0f if they are not valid, but at least one should be valid
            real_t coordination_number_ref_1 = data->c6_ab_ref[index + 1];
            real_t coordination_number_ref_2 = data->c6_ab_ref[index + 2];
            if (coordination_number_ref_1 > -1.0f &&
                coordination_number_ref_2 > -1.0f) {
                // if both coordination numbers are valid, we can use them
                const real_t delta_CN_1 = coordination_number_1 - coordination_number_ref_1;
                const real_t delta_CN_2 = coordination_number_2 - coordination_number_ref_2;
                const real_t exponent_arg = -K3 * (delta_CN_1 * delta_CN_1 + delta_CN_2 * delta_CN_2);
                max_exponent_arg =
                    (max_exponent_arg > exponent_arg)
                        ? max_exponent_arg
                        : exponent_arg;  // update the maximum exponent argument
            }
        }
    }
    // calculate C6 value
    real_t Z = 0.0f;
    real_t W = 0.0f;
    real_t c_ref_L_ij = 0.0f; // C6ab_ref * L_ij
    real_t c_ref_dL_ij_1 = 0.0f; // C6ab_ref * dL_ij/dCN_1
    real_t dL_ij_1 = 0.0f; // dL_ij/dCN_1
    for (uint8_t i = 0; i < NUM_REF_C6; ++i) {
        for (uint8_t j = 0; j < NUM_REF_C6; ++j) {
            // find the C6ref
            uint32_t index = atom_1_type * data->c6_stride_1 +
                                atom_2_type * data->c6_stride_2 + 
                                i * data->c6_stride_3 +
                                j * data->c6_stride_4;
            real_t c6_ref = data->c6_ab_ref[index + 0];
            // these entries could be -1.0f if they are not valid, but at least one should be valid
            real_t coordination_number_ref_1 = data->c6_ab_ref[index + 1];
            real_t coordination_number_ref_2 = data->c6_ab_ref[index + 2];
            if (coordination_number_ref_1 > -1.0f &&
                coordination_number_ref_2 > -1.0f) {
                // if both coordination numbers are valid, we can use them
                const real_t delta_CN_1 = coordination_number_1 - coordination_number_ref_1;
                const real_t delta_CN_2 = coordination_number_2 - coordination_number_ref_2;
                const real_t exponent_arg = -K3 * (delta_CN_1 * delta_CN_1 + delta_CN_2 * delta_CN_2);
                const real_t L_ij = expf(exponent_arg - max_exponent_arg);  // normalized the L_ij value
                real_t dL_ij_1_part = -2.0f * K3 * (coordination_number_1 - coordination_number_ref_1) * L_ij;  // part of dL_ij/dCN_1 contributed by the current valid term
                Z += c6_ref * L_ij;
                W += L_ij;
                c_ref_L_ij += c6_ref * L_ij;
                c_ref_dL_ij_1 += c6_ref * dL_ij_1_part;
                dL_ij_1 += dL_ij_1_part;
            }
        }
    }

    // avoid division by zero
    real_t dC6ab_dCN_1 =
        (W * W > 0.0f)
            ? (c_ref_dL_ij_1 * W - c_ref_L_ij * dL_ij_1) / (W * W)
            : 0.0f; // dC6ab/dCN_1
    if (isnan(dC6ab_dCN_1) || isinf(dC6ab_dCN_1)) {
        // NaN or inf encountered, bad result
        printf("Error: dC6ab/dCN_1 is NaN or Inf\n");
        printf("Z: %f, W: %f, c_ref_L_ij: %f, c_ref_dL_ij_1: %f, dL_ij_1: %f\n", Z, W, c_ref_L_ij, c_ref_dL_ij_1, dL_ij_1);
        dC6ab_dCN_1 = 0.0f;  // reset to 0.0f if it's NaN or Inf
    }
    dC6_dCN_1_result = dC6ab_dCN_1;

    // avoid division by zero
    real_t c6_ab = (W > 0.0f) ? Z / W : 0.0f;   // C6_ab value between atom 1 and 2
    if (isnan(c6_ab) || isinf(c6_ab)) {
        printf("Error: C6_ab is NaN or Inf\n");
        printf("Z: %f, W: %f, c_ref_L_ij: %f, c_ref_dL_ij_1: %f, dL_ij_1: %f\n", Z, W, c_ref_L_ij, c_ref_dL_ij_1, dL_ij_1);
        c6_ab = 0.0f;  // reset to 0.0f if it's NaN or Inf
    }
    c6_ab_result = c6_ab;
}

__device__ inline void calculate_two_body_interaction(
    real_t cell_volume,
    atom_t atom_1, atom_t atom_2,
    real_t c6_ab, real_t c8_ab,
    real_t dC6ab_dCN_1, real_t dC8ab_dCN_1,
    real_t r0_cutoff, real_t cutoff_radius,
    DampingType damping_type,
    real_t damping_param_1, real_t damping_param_2,
    real_t s6, real_t s8,
    HierarchicalKahanAccumulator<1> &energy_accumulator,
    HierarchicalKahanAccumulator<1> &dE_dCN_accumulator,
    HierarchicalKahanAccumulator<3> &force_accumulator,
    HierarchicalKahanAccumulator<9> &stress_accumulator)
{
    real_t delta_r[3] = {
        atom_1.x - atom_2.x, // delta x
        atom_1.y - atom_2.y, // delta y
        atom_1.z - atom_2.z  // delta z
    }; // delta_r between atom 1 and atom 2
    const real_t distance_2 = delta_r[0] * delta_r[0] +
                              delta_r[1] * delta_r[1] +
                              delta_r[2] * delta_r[2];        // distance^2 between atom 1 and atom 2
    const real_t cutoff_square = cutoff_radius * cutoff_radius; // global cutoff for distance check
    if (distance_2 <= cutoff_square && distance_2 > 0.0f)
    {
        const real_t distance = sqrtf(distance_2); // distance between atom 1 and atom 2
        // calculate distance^6 and distance^8 using fast power
        const real_t distance_3 = distance_2 * distance;    // distance^3
        const real_t distance_4 = distance_2 * distance_2;  // distance^4
        const real_t distance_6 = distance_3 * distance_3;  // distance^6
        const real_t distance_8 = distance_4 * distance_4;  // distance^8
        const real_t distance_10 = distance_6 * distance_4; // distance^10
        // calculate the damping function
        real_t f_dn_6, f_dn_8, d_f_dn_6, d_f_dn_8;
        switch (damping_type)
        {
        case ZERO_DAMPING:
            damping<ZERO_DAMPING>(distance, r0_cutoff, damping_param_1, damping_param_2,
                                  &f_dn_6, &f_dn_8,
                                  &d_f_dn_6, &d_f_dn_8);
            break;
        case BJ_DAMPING:
            damping<BJ_DAMPING>(distance, r0_cutoff, damping_param_1, damping_param_2,
                                &f_dn_6, &f_dn_8, &d_f_dn_6,
                                &d_f_dn_8);
            break;
        }
        // calculate the dispersion energy according to Grimme et al. 2010, eq3
        const real_t dispersion_energy_6 = s6 * (c6_ab / distance_6) * f_dn_6;               // E_6
        const real_t d_E6_dCN = s6 * f_dn_6 * dC6ab_dCN_1 / distance_6;                      // dE_6/dCN
        const real_t dispersion_energy_8 = s8 * (c8_ab / distance_8) * f_dn_8;               // E_8
        const real_t d_E8_dCN = s8 * f_dn_8 * dC8ab_dCN_1 / distance_8;                      // dE_8/dCN
        const real_t dispersion_energy = (dispersion_energy_6 + dispersion_energy_8) / 2.0f; // divide by 2 because each atom pair is counted twice
        const real_t dE_dCN = (d_E6_dCN + d_E8_dCN);                                         // dE/dCN

        /** the first entry of two-body force:
         * $F_a = S_n C_n^{ab} f_{d,n}(r_{ab}) \frac{\partial}{\partial r_a} r_{ab}^{-n}$
         * $F_a = S_n C_n^{ab} f_{d,n}(r_{ab}) * (-n)r_{ab}^{-n-2} * \uparrow{r_{ab}}$
         */
        real_t force = 0.0f;                                  // dE/dr * 1/r
        force += s6 * c6_ab * f_dn_6 * (-6.0f) / distance_8;  // dE_6/dr * 1/r
        force += s8 * c8_ab * f_dn_8 * (-8.0f) / distance_10; // dE_8/dr * 1/r

        /** the second entry of two-body force:
         * $F_a = S_n C_n^{ab} r_{ab}^{-n} \frac{\partial}{\partial r_a} f_{d,n}(r_{ab})$
         * $F_a = S_n C_n^{ab} r_{ab}^{-n} -f_{d,n}^2 *
         * (6*(-\alpha_n)*(r_{ab}/{S_{r,n}R_0^{AB}})^{-\alpha_n - 1} *
         * 1/(S_{r,n}R_0^{AB})) / r_ab \vec{r_{ab}}$
         */
        force += s6 * c6_ab / distance_6 * d_f_dn_6 / distance; // dE_6/dr * 1/r
        force += s8 * c8_ab / distance_8 * d_f_dn_8 / distance; // dE_8/dr * 1/r

        // accumulate the energy, force, stress and dE/dCN using hierarchical Kahan summation
        energy_accumulator.add(&dispersion_energy);
        dE_dCN_accumulator.add(&dE_dCN);
        real_t force_contribution[3] = {0.0f, 0.0f, 0.0f};
        for (uint8_t i = 0; i < 3; ++i)
        {
            force_contribution[i] = force * delta_r[i];
        }
        real_t stress_contribution[9];
        for (uint8_t i = 0; i < 3; ++i)
        {
            for (uint8_t j = 0; j < 3; ++j)
            {
                stress_contribution[i * 3 + j] = -1.0f * delta_r[i] * force * delta_r[j] / 2.0f / cell_volume;
            }
        }
        force_accumulator.add(force_contribution);
        stress_accumulator.add(stress_contribution);
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
__global__ void two_body_kernel(device_data_t *data)
{
    // load parameters from device data
    const real_t s6 = data->functional_params.s6;
    const real_t s8 = data->functional_params.s8;
    const real_t sr_6 = data->functional_params.sr6;
    const real_t sr_8 = data->functional_params.sr8;
    const real_t a1 = data->functional_params.a1;
    const real_t a2 = data->functional_params.a2;
    const real_t cutoff = data->cutoff;

    // load the central atom data
    const uint64_t atom_1_index = blockIdx.x;                                      // index of the central atom
    const uint64_t atom_1_type = data->atom_types[atom_1_index];                   // type of the central atom
    const atom_t atom_1 = data->atoms[atom_1_index];                               // central atom
    const real_t coordination_number_1 = data->coordination_numbers[atom_1_index]; // coordination number of the central atom

    HierarchicalKahanAccumulator<1> energy_accumulator, dE_dCN_accumulator;
    HierarchicalKahanAccumulator<3> force_accumulators;
    HierarchicalKahanAccumulator<9> stress_accumulators;
    energy_accumulator.init();
    dE_dCN_accumulator.init();
    force_accumulators.init();
    stress_accumulators.init();

    // prefetch and calculate necessary variables during calculation
    const real_t cell_volume = calculate_cell_volume(data->cell); // volume of the cell
    const uint64_t mcb0 = data->max_cell_bias[0];                 // maximum cell bias in x direction
    const uint64_t mcb1 = data->max_cell_bias[1];                 // maximum cell bias in y direction
    const uint64_t mcb2 = data->max_cell_bias[2];                 // maximum cell bias in z direction
    const uint64_t total_cell_bias = mcb0 * mcb1 * mcb2;          // total number of cell bias
    const real_t cell[3][3] = {
        {data->cell[0][0], data->cell[0][1], data->cell[0][2]},
        {data->cell[1][0], data->cell[1][1], data->cell[1][2]},
        {data->cell[2][0], data->cell[2][1], data->cell[2][2]}}; // cell matrix

    // print some debug information
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        debug("Coordination number kernel launched with %llu atoms, cell: \n",
              data->num_atoms);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                debug("%f ", cell[i][j]);
            }
            debug("\n");
        }
        debug("max_cell_bias: %llu %llu %llu\n", mcb0, mcb1, mcb2);
    }

    // distribute workload to threads
    uint64_t start_index;      // start index for this thread
    uint64_t end_index;        // end index for this thread
    uint64_t start_bias_index; // start bias index for this thread
    uint64_t end_bias_index;   // end bias index for this thread
    distribute_workload(data->num_atoms, total_cell_bias, &start_index, &end_index, &start_bias_index, &end_bias_index);
    // iterate over surrounding atoms
    for (uint64_t atom_2_index = start_index; atom_2_index < end_index; ++atom_2_index)
    {
        const atom_t atom_2_original = data->atoms[atom_2_index];                      // surrounding atom
        const uint64_t atom_2_type = data->atom_types[atom_2_index];                   // type of the surrounding atom
        const real_t coordination_number_2 = data->coordination_numbers[atom_2_index]; // coordination number of the surrounding atom

        real_t c6_ab, dC6ab_dCN_1;
        calculate_c6ab(
            data,
            atom_1_type, atom_2_type,
            coordination_number_1, coordination_number_2,
            c6_ab, dC6ab_dCN_1
        );

        // calculate c8_ab by $C_8^{AB} = 3C_6^{AB}\sqrt{Q^AQ^B}$
        // the values in data->r2r4 is already squared
        const real_t r2r4_1 = data->r2r4[atom_1_type];
        const real_t r2r4_2 = data->r2r4[atom_2_type];
        const real_t c8_ab = 3.0f * c6_ab * r2r4_1 * r2r4_2;             // C8ab value
        const real_t dC8ab_dCN_1 = 3.0f * dC6ab_dCN_1 * r2r4_1 * r2r4_2; // dC8ab/dCN_1
        // acquire the R0 cutoff radius between the two atoms
        const real_t r0_cutoff = data->r0ab[atom_1_type * data->num_elements + atom_2_type];
        // loop over supercells
        for (uint64_t bias_index = start_bias_index; bias_index < end_bias_index; ++bias_index)
        {
            const int64_t x_bias = (bias_index % mcb0) - (mcb0 / 2);                 // x bias
            const int64_t y_bias = ((bias_index / mcb0) % mcb1) - (mcb1 / 2);        // y bias
            const int64_t z_bias = (bias_index / (mcb0 * mcb1) % mcb2) - (mcb2 / 2); // z bias
            assert_(atom_2_index < data->num_atoms);                                 // make sure the index is in bounds

            atom_t atom_2 = atom_2_original; // the actual atom2 participated in the calculation
            // translate atom_2 due to periodic boundaries
            atom_2.x += x_bias * cell[0][0] + y_bias * cell[1][0] + z_bias * cell[2][0]; // translate in x direction
            atom_2.y += x_bias * cell[0][1] + y_bias * cell[1][1] + z_bias * cell[2][1]; // translate in y direction
            atom_2.z += x_bias * cell[0][2] + y_bias * cell[1][2] + z_bias * cell[2][2]; // translate in z direction

            switch (data->damping_type)
            {
            case ZERO_DAMPING:
                calculate_two_body_interaction(
                    cell_volume,
                    atom_1, atom_2,
                    c6_ab, c8_ab,
                    dC6ab_dCN_1, dC8ab_dCN_1,
                    r0_cutoff, cutoff,
                    ZERO_DAMPING,
                    sr_6, sr_8,
                    s6, s8,
                    energy_accumulator,
                    dE_dCN_accumulator,
                    force_accumulators,
                    stress_accumulators);
                break;
            case BJ_DAMPING:
                calculate_two_body_interaction(
                    cell_volume,
                    atom_1, atom_2,
                    c6_ab, c8_ab,
                    dC6ab_dCN_1, dC8ab_dCN_1,
                    r0_cutoff, cutoff,
                    BJ_DAMPING,
                    a1, a2,
                    s6, s8,
                    energy_accumulator,
                    dE_dCN_accumulator,
                    force_accumulators,
                    stress_accumulators);
                break;
            default:
                break;
            }
        }
    }

    real_t local_dE_dCN;
    real_t local_energ;
    real_t local_stress[9];
    real_t local_force[3];
    dE_dCN_accumulator.get_sum(&local_dE_dCN);
    energy_accumulator.get_sum(&local_energ);
    force_accumulators.get_sum(local_force);
    stress_accumulators.get_sum(local_stress);

    real_t dE_dCN_sum = 0;                // sum of dE/dCN across the block
    real_t energy_sum = 0;                // sum of energy across the block
    real_t force_central_sum[3] = {0.0f}; // sum of force of central atom across the block
    real_t stress_sum[9] = {0.0f};        // sum of stress of central atom across the block
    // accumulate the results across the block
    blockReduceTwoBodyKernel(local_dE_dCN, local_energ, local_force,
                             local_stress, &dE_dCN_sum, &energy_sum,
                             force_central_sum, stress_sum);

    if (threadIdx.x == 0)
    {
        /** Only the first thread in the block is responsible for writing back the accumulated result.
         * We directly write to global memory for dE/dCN, energy, and force without atomic operations.
         * This is safe because each block processes a single atom, therefore no data race :).
         * However, stress accumulation requires atomic operations to avoid data races.
         * Note that the atoms are rearranged, so when writing to result arrays(energy and force),
         * we use orginal index instead of atom_1_index, which is the rearranged index.
         * However, for intermediates like dE/dCN, we still use atom_1_index for convenience
         */
        uint64_t original_atom_1_index = atom_1.original_index;
        // dE/dCN
        data->dE_dCN[atom_1_index] = dE_dCN_sum;
        // energy
        data->energy[original_atom_1_index] = energy_sum;
        // force and stress
        for (uint8_t i = 0; i < 3; ++i)
        {
            force_central_sum[i] += dE_dCN_sum * data->dCN_dr[atom_1_index * 3 + i]; // another force entry: $F_i = dE/dCN_i * dCN_i/dr_i$
            // write back the force without atomic operation, safe because no other thread writes to this memory
            data->forces[original_atom_1_index * 3 + i] = force_central_sum[i];
            for (uint8_t j = 0; j < 3; ++j)
            {
                atomicAdd(&data->stress[i * 3 + j], stress_sum[i * 3 + j]); // atomic operation here to avoid data races
            }
        }
    }
}

__device__ inline void calculate_three_body_interaction(
    atom_t atom_1, atom_t atom_2,
    real_t covalent_radii_1, real_t covalent_radii_2,
    real_t CN_cutoff,
    real_t dE_dCN,
    HierarchicalKahanAccumulator<3> &force_accumulators,
    HierarchicalKahanAccumulator<9> &stress_accumulators
) {
    real_t delta_r[3] = {
        atom_1.x - atom_2.x,  // delta x
        atom_1.y - atom_2.y,  // delta y
        atom_1.z - atom_2.z   // delta z
    }; // delta_r between atom 1 and atom 2
    // calculate the distance between the two atoms
    const real_t distance_square = delta_r[0] * delta_r[0] +
                                        delta_r[1] * delta_r[1] +
                                        delta_r[2] * delta_r[2];
    real_t distance = sqrtf(distance_square);
    /* if the distance is within cutoff range, update neighbor_flags */
    if (distance <= CN_cutoff && distance > 0.0f) {
        /**
         * eq 15 in Grimme et al. 2010
         * $CN^A = \sum_{B \neq A}^{N}
         * \sqrt{1}{1+exp(-k_1(k_2(R_{A,cov}+R_{B,cov})/r_{AB}-1))}$
         */
        real_t exp = expf(-K1 * ((covalent_radii_1 + covalent_radii_2) / distance - 1.0f));  // $\exp(-k_1*(\frac{R_A+R_b}{r_{ab}}-1))$
        real_t tanh_value = tanhf(CN_cutoff - distance);  // $\tanh(CN_cutoff - r_{ab})$
        real_t smooth_cutoff = powf(tanh_value, 3);  // $\tanh^3(CN_cutoff- r_{ab}))$, this is a smooth cutoff function added in LASP code.
        real_t d_smooth_cutoff_dr = 3.0f * powf(tanh_value, 2) * (1.0f - powf(tanh_value, 2)) * (-1.0f);  // d(smooth_cutoff)/dr
        real_t dCN_datom = powf(1.0f + exp, -2.0f) * (-K1) * exp * (covalent_radii_1 + covalent_radii_2) * powf(distance, -3.0f) * smooth_cutoff + d_smooth_cutoff_dr * 1.0f / (1.0f + exp) / distance;  // dCN_ij/dr_ij * 1/r_ij
        // dE/drik = dE/dCN * dCN/drik
        real_t dE_drik = dE_dCN * dCN_datom;
        // accumulate force for the central atom and neighboring atom
        // force_central += dE/drik * delta_r
        // use Kahan summation to improve numerical stability
        real_t force_contribution[3];
        real_t stress_contribution[9];
        for (uint8_t i = 0; i < 3; ++i) {
            force_contribution[i] = (dE_drik * delta_r[i]);
            for (uint8_t j = 0; j < 3; ++j) {
                stress_contribution[i * 3 + j] = (-1.0f * delta_r[i] * dE_drik * delta_r[j]);
            }
        }
        force_accumulators.add(force_contribution);
        stress_accumulators.add(stress_contribution);
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
__global__ void three_body_kernel(device_data_t *data)
{
    real_t cell_volume = calculate_cell_volume(data->cell);
    uint64_t atom_1_index = blockIdx.x;                    // index of central atom
    atom_t atom_1 = data->atoms[atom_1_index];             // central atom
    uint64_t atom_1_type = data->atom_types[atom_1_index]; // type of the central atom
    real_t covalent_radii_1 = data->rcov[atom_1_type];     // covalent radius of the central atom

    HierarchicalKahanAccumulator<3> force_accumulators;
    HierarchicalKahanAccumulator<9> stress_accumulators;
    force_accumulators.init();
    stress_accumulators.init();

    uint64_t total_cell_bias = data->max_cell_bias[0] * data->max_cell_bias[1] * data->max_cell_bias[2]; // total number of cell biases
    const uint64_t mcb0 = data->max_cell_bias[0];                                                        // maximum cell bias in x direction
    const uint64_t mcb1 = data->max_cell_bias[1];                                                        // maximum cell bias in y direction
    const uint64_t mcb2 = data->max_cell_bias[2];                                                        // maximum cell bias in z direction
    const real_t cell[3][3] = {
        {data->cell[0][0], data->cell[0][1], data->cell[0][2]},
        {data->cell[1][0], data->cell[1][1], data->cell[1][2]},
        {data->cell[2][0], data->cell[2][1], data->cell[2][2]}}; // local cell matrix
    const real_t CN_cutoff = data->coordination_number_cutoff;
    // print some debug information
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        debug("Coordination number kernel launched with %llu atoms, cell: \n", data->num_atoms);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                debug("%f ", cell[i][j]);
            }
            debug("\n");
        }
        debug("max_cell_bias: %llu %llu %llu\n", mcb0, mcb1, mcb2);
    }
    // distribute workload
    uint64_t start_index;
    uint64_t end_index;
    uint64_t start_bias_index;
    uint64_t end_bias_index;
    distribute_workload(data->num_atoms, total_cell_bias, &start_index, &end_index, &start_bias_index, &end_bias_index);

    // iterate over surrounding atoms
    for (uint64_t atom_2_index = start_index; atom_2_index < end_index; ++atom_2_index)
    {
        real_t dE_dCN = data->dE_dCN[atom_2_index];                           // dE/dCN of the surrounding atom
        atom_t atom_2_original = data->atoms[atom_2_index];                   // original surrounding atom before translation
        real_t covalent_radii_2 = data->rcov[data->atom_types[atom_2_index]]; // covalent radii of the surrounding atom

        // iterate over cell biases
        for (uint64_t bias_index = start_bias_index; bias_index < end_bias_index; ++bias_index)
        {
            int64_t x_bias = (bias_index % mcb0) - (mcb0 / 2);                 // x bias
            int64_t y_bias = ((bias_index / mcb0) % mcb1) - (mcb1 / 2);        // y bias
            int64_t z_bias = (bias_index / (mcb0 * mcb1) % mcb2) - (mcb2 / 2); // z bias
            assert_(atom_2_index < data->num_atoms);                           // make sure the index is in bounds

            atom_t atom_2 = atom_2_original;
            // translate atom_2 due to periodic boundaries
            atom_2.x += x_bias * cell[0][0] + y_bias * cell[1][0] + z_bias * cell[2][0];  // translate in x direction
            atom_2.y += x_bias * cell[0][1] + y_bias * cell[1][1] + z_bias * cell[2][1];  // translate in y direction
            atom_2.z += x_bias * cell[0][2] + y_bias * cell[1][2] + z_bias * cell[2][2];  // translate in z direction

            calculate_three_body_interaction(
                atom_1, atom_2,
                covalent_radii_1, covalent_radii_2,
                CN_cutoff,
                dE_dCN,
                force_accumulators,
                stress_accumulators
            );

        }
    }

    // accumulate force of central atom and stress
    real_t local_force_sum[3];
    real_t local_stress_sum[9];
    force_accumulators.get_sum(local_force_sum);
    stress_accumulators.get_sum(local_stress_sum);

    real_t force_central_sum[3] = {0.0f}; // force sum for the central atom across the block
    real_t stress_sum[9] = {0.0f};        // stress sum across the block
    blockReduceSumThreeBodyKernel(local_force_sum, local_stress_sum, force_central_sum, stress_sum);

    // the first thread accumulates the results back to global memory
    /**
     * note that the atoms are rearranged, so when writing to result arrays(energy and force),
     * we use orginal index instead of atom_1_index, which is the rearranged index.
     * However, for intermediates like dE/dCN, we still use atom_1_index for convenience
     */
    if (threadIdx.x == 0)
    {
        uint64_t original_atom_1_index = atom_1.original_index;
        for (uint8_t i = 0; i < 3; ++i)
        {
            // accumulate force
            data->forces[original_atom_1_index * 3 + i] += force_central_sum[i];
            for (uint8_t j = 0; j < 3; ++j)
            {
                // accumulate stress
                atomicAdd(&data->stress[i * 3 + j], stress_sum[i * 3 + j] / cell_volume);
            }
        }
    }
}