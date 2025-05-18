#include "constants.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "d3.h"

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
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(error)); \
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
#define MAX_NEIGHBORS 10000 // the maximum number of neighbors, dependent on the cutoff choice
#define MAX_LOCAL_NEIGHBORS 800 // the maximum neighbor of one thread, equal to max_supercell_size * (num_atoms / num_threads)
#define MAX_ATOMS 1000

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

// Calculate inverse of a 3x3 matrix
void matrix_inverse(const real_t mat[3][3], real_t inv[3][3]) {
    // Calculate determinant
    real_t det = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
              - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
              + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
    
              real_t inv_det = 1.0 / det;
    
    // Calculate cofactor matrix (transposed)
    inv[0][0] = (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) * inv_det;
    inv[0][1] = (mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2]) * inv_det;
    inv[0][2] = (mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1]) * inv_det;
    
    inv[1][0] = (mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2]) * inv_det;
    inv[1][1] = (mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0]) * inv_det;
    inv[1][2] = (mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2]) * inv_det;
    
    inv[2][0] = (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]) * inv_det;
    inv[2][1] = (mat[0][1] * mat[2][0] - mat[0][0] * mat[2][1]) * inv_det;
    inv[2][2] = (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]) * inv_det;
}

// Transpose a 3x3 matrix
void matrix_transpose(const real_t mat[3][3], real_t trans[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            trans[j][i] = mat[i][j];
        }
    }
}

// Calculate row-wise norm of a 3x3 matrix
void row_norms(const real_t mat[3][3], real_t norms[3]) {
    for (int i = 0; i < 3; i++) {
        norms[i] = sqrt(mat[i][0] * mat[i][0] + 
                         mat[i][1] * mat[i][1] + 
                         mat[i][2] * mat[i][2]);
    }
}

// Equivalent to torch.ceil(cutoff * inv_distances).long()
void calculate_cell_repeats(const real_t cell[3][3], real_t cutoff, uint64_t repeats[3]) {
    real_t inv[3][3];
    real_t trans[3][3];
    real_t norms[3];
    
    // Calculate inverse of cell matrix
    matrix_inverse(cell, inv);
    
    // Transpose the inverse matrix
    matrix_transpose(inv, trans);
    
    // Calculate norms of each row
    row_norms(trans, norms);
    
    // Multiply by cutoff and round up to nearest integer
    for (int i = 0; i < 3; i++) {
        repeats[i] = ((int)(cutoff * norms[i]) + 2) + 1; // the number of repeats need to be timed by 2 due to two directions, and add 1 due to the central unit (no translation at all)
    }
}

typedef struct neighbor {
    uint64_t index; // index of the neighbor atom
    atom_t atom; // atom data of the neighbor atom
    real_t distance; // distance to the neighbor atom
    real_t dCN_dr; // dCN_ab/dr_ab * 1/r for the neighbor atom
} neighbor_t;

typedef struct device_data {
    uint64_t num_atoms; // number of atoms in the system
    uint64_t num_elements; // number of unique elements in the system
    // uint64_t *unique_elements; // array of unique elements in the system, length: num_elements
    uint64_t *atom_types; // array of atom types, length: num_atoms. the entries is not the atomic number, but the index of the corresponding entry in constants.
    atom_t *atoms; // array of atom data
    real_t *c6_ab_ref; // size: num_elements*num_elements*NUM_REF_C6*NUM_REF_C6*NUM_C6AB_ENTRIES
    uint64_t c6_stride_1, c6_stride_2, c6_stride_3, c6_stride_4; // strides for c6ab array
    real_t *r0ab; // size: num_elements*num_elements
    real_t *rcov; // size: num_elements
    real_t *r2r4;// size: num_elements
    // d3_constant_t *constants; // constants for the simulation
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
    uint64_t max_num_CN_neighbor;
    real_t *energy; // energy of the system, length: 1
    real_t *forces; // forces on each atom, length: 3*num_atoms.
    real_t *stress; // stress tensor, length: 9
} device_data_t;

/**
 * @brief class to store unique elements in the system
 * @note the user must not construct a Unique_Elements object directly, but use the Unique_Elements constructor instead
 */
class Unique_Elements {
public:
    uint16_t num_elements; // number of unique elements in the system
    Unique_Elements(uint16_t *elements, uint16_t length) {
        uint16_t *all_elements = (uint16_t *)malloc(MAX_ELEMENTS * sizeof(uint16_t));
        memset(all_elements, 0, MAX_ELEMENTS * sizeof(uint16_t)); // initialize all elements to 0
        this->num_elements = 0;
        /* bucket sort */
        for (uint16_t i = 0; i < length; ++i) {
            if (elements[i] >= MAX_ELEMENTS) {
                /* check that no element number exceed MAX_LEMENTS */
                fprintf(stderr, "Error: element %d is out of range\n", elements[i]);
                free(all_elements);
                exit(EXIT_FAILURE);
            }
            if (all_elements[elements[i]] == 0) {
                this->num_elements++;
            }
            all_elements[elements[i]] = 1; // mark the element as present
        }
        this->elements_ = (uint16_t *)malloc(this->num_elements * sizeof(uint16_t));
        uint16_t index = 0;
        for (uint16_t i = 0; i < MAX_ELEMENTS; ++i) {
            if (all_elements[i] == 1) {
                this->elements_[index] = i;
                index++;
            }
        }
        assert(index == this->num_elements); // check if the number of unique elements is correct
        free(all_elements); // free the temporary array        
    } // Unique_Elements constructor
    ~Unique_Elements() {
        free(this->elements_); // free the unique elements array
    } // Unique_Elements destructor
    /* find an element by its atomic number */
    uint16_t find(uint16_t element) {
        for (uint16_t i = 0; i < this->num_elements; ++i) {
            if (this->elements_[i] == element) {
                return i; // return the index of the element
            }
        }
        fprintf(stderr, "Error: element %d not found\n", element);
        exit(EXIT_FAILURE);
    }
    /* access an element using its index */
    uint16_t operator[](uint16_t index) {
        if (index >= this->num_elements) {
            fprintf(stderr, "Error: index %d is out of range\n", index);
            exit(EXIT_FAILURE);
        }
        return this->elements_[index]; // return the element at the given index
    }

private:
    uint16_t *elements_; // array of unique elements in the system
}; // Unique_Elements

/**
 * @brief class on host to store the device data
 * @note the data is stored in device memory, so no host access is allowed
 */
class Device_Buffer {
public:
    __host__ Device_Buffer(real_t coords[][3], uint16_t *elements, real_t cell[3][3], uint64_t length, real_t cutoff, real_t CN_cutoff){
        Unique_Elements unique_elements(elements, length); // create the unique elements object
        {
            /* construct elements */
            this->host_data_.num_atoms = length; // number of atoms in the system
            this->host_data_.num_elements = unique_elements.num_elements; // number of unique elements in the system

   
            uint64_t *h_atom_types = (uint64_t *)malloc(sizeof(uint64_t) * length);
            for (uint64_t i = 0; i < length; ++i) {
                h_atom_types[i] = unique_elements.find(elements[i]);
            }
       
            uint64_t *d_atom_types;
            CHECK_CUDA(cudaMalloc((void**)&d_atom_types, sizeof(uint64_t) * length));
            CHECK_CUDA(cudaMemcpy(d_atom_types, h_atom_types, sizeof(uint64_t)*length, cudaMemcpyHostToDevice));
            free(h_atom_types);
            this->host_data_.atom_types = d_atom_types;
        }
        {
            /* construct cell */
            for (uint16_t i = 0; i < 3; ++i) {
                for (uint16_t j = 0; j < 3; ++j) {
                    this->host_data_.cell[i][j] = cell[i][j];
                }
            }
        }
        {
            /* construct atoms */
            atom_t *h_atoms = (atom_t *)malloc(length * sizeof(atom_t));
            if (h_atoms == NULL) {
                fprintf(stderr, "Error: failed to allocate memory for atoms on host");
                exit(EXIT_FAILURE);
            }
            for (uint64_t i = 0; i < length; ++i) {
                h_atoms[i].element = elements[i];
                h_atoms[i].x = coords[i][0];
                h_atoms[i].y = coords[i][1];
                h_atoms[i].z = coords[i][2];
            }
            atom_t *d_atoms;
            CHECK_CUDA(cudaMalloc((void**)&d_atoms, length * sizeof(atom_t)));
            CHECK_CUDA(cudaMemcpy(d_atoms, h_atoms, length * sizeof(atom_t), cudaMemcpyHostToDevice));
            this->host_data_.atoms = d_atoms; // set the atoms pointer in device data
            free(h_atoms); // free the host atoms array
        }

        /* construct constants */
        uint16_t num_elements = unique_elements.num_elements;
        {
            /* c6ab_ref array */
            this->host_data_.c6_stride_1 = num_elements*NUM_REF_C6*NUM_REF_C6*NUM_C6AB_ENTRIES;
            this->host_data_.c6_stride_2 = NUM_REF_C6*NUM_REF_C6*NUM_C6AB_ENTRIES;
            this->host_data_.c6_stride_3 = NUM_REF_C6*NUM_C6AB_ENTRIES;
            this->host_data_.c6_stride_4 = NUM_C6AB_ENTRIES;
            real_t *h_c6ab_ref = (real_t *)malloc(num_elements*num_elements*NUM_REF_C6*NUM_REF_C6*NUM_C6AB_ENTRIES*sizeof(real_t));
            for (uint16_t i = 0; i < num_elements; ++i) {
                for(uint16_t j = 0; j < num_elements; ++j) {
                    uint16_t element_i = unique_elements[i];
                    uint16_t element_j = unique_elements[j];
                    for(uint16_t k = 0; k < NUM_REF_C6; ++k) {
                        for (uint16_t l = 0; l < NUM_REF_C6; ++l) {
                            uint64_t index = this->host_data_.c6_stride_1 * i + this->host_data_.c6_stride_2 * j + this->host_data_.c6_stride_3 * k + this->host_data_.c6_stride_4 * l;
                            for(uint16_t m = 0; m < NUM_C6AB_ENTRIES; ++m) {
                                h_c6ab_ref[index+m] = c6ab_ref[element_i][element_j][k][l][m];
                            }
                        }
                    }
                }
            }
            real_t *d_c6ab_ref;
            CHECK_CUDA(cudaMalloc((void**)&d_c6ab_ref, num_elements*num_elements*NUM_REF_C6*NUM_REF_C6*NUM_C6AB_ENTRIES*sizeof(real_t)));
            CHECK_CUDA(cudaMemcpy(d_c6ab_ref, h_c6ab_ref, num_elements*num_elements*NUM_REF_C6*NUM_REF_C6*NUM_C6AB_ENTRIES*sizeof(real_t), cudaMemcpyHostToDevice));
            this->host_data_.c6_ab_ref = d_c6ab_ref;
            free(h_c6ab_ref);
        } // c6ab_ref array
        {
            /* r0ab array */
            real_t *h_r0ab = (real_t *)malloc(num_elements * num_elements * sizeof(real_t));
            for(uint16_t i = 0; i < num_elements; ++i) {
                for(uint16_t j = 0; j < num_elements; ++j) {
                    uint16_t element_i = unique_elements[i];
                    uint16_t element_j = unique_elements[j];
                    h_r0ab[i*num_elements+j] = r0ab[element_i][element_j];
                }
            }
            real_t *d_r0ab;
            CHECK_CUDA(cudaMalloc((void**)&d_r0ab, num_elements*num_elements*sizeof(real_t)));
            CHECK_CUDA(cudaMemcpy(d_r0ab, h_r0ab, num_elements*num_elements*sizeof(real_t), cudaMemcpyHostToDevice));
            this->host_data_.r0ab = d_r0ab;
            free(h_r0ab);
        } // r0ab array
        {
            /* rcov array */
            real_t *h_rcov = (real_t *)malloc(num_elements * sizeof(real_t));
            for(uint16_t i = 0; i < num_elements; ++i) {
                h_rcov[i] = rcov[unique_elements[i]];
            }
            real_t *d_rcov;
            CHECK_CUDA(cudaMalloc((void**)&d_rcov, num_elements*sizeof(real_t)));
            CHECK_CUDA(cudaMemcpy(d_rcov, h_rcov, num_elements*sizeof(real_t), cudaMemcpyHostToDevice));
            this->host_data_.rcov = d_rcov;
            free(h_rcov);
        } // rcov array
        {
            /* r2r4 array */
            real_t *h_r2r4 = (real_t *)malloc(num_elements * sizeof(real_t));
            for(uint16_t i = 0; i < num_elements; ++i) {
                h_r2r4[i] = r2r4[unique_elements[i]];
            }
            real_t *d_r2r4;
            CHECK_CUDA(cudaMalloc((void**)&d_r2r4, num_elements*sizeof(real_t)));
            CHECK_CUDA(cudaMemcpy(d_r2r4, h_r2r4, num_elements*sizeof(real_t), cudaMemcpyHostToDevice));
            this->host_data_.r2r4 = d_r2r4;
            free(h_r2r4);
        } // r2r4 array
        {
            /* construct supercell information */
            this->host_data_.coordination_number_cutoff = CN_cutoff;
            this->host_data_.cutoff = cutoff;
            real_t larger_cutoff = CN_cutoff > cutoff ? CN_cutoff : cutoff;
            calculate_cell_repeats(cell, larger_cutoff, this->host_data_.max_cell_bias); 
            printf("max_cell_bias: %zu %zu %zu\n", this->host_data_.max_cell_bias[0], this->host_data_.max_cell_bias[1], this->host_data_.max_cell_bias[2]);
        } // cupercell information
        {
            /* construct other fields */
            neighbor_t *neighbors;
            cudaMalloc((void**)&neighbors, length * MAX_NEIGHBORS * sizeof(neighbor_t));
            cudaMemset(neighbors, 0, length * MAX_NEIGHBORS * sizeof(neighbor_t));
            this->host_data_.neighbors = neighbors;
            neighbor_t *CN_neighbors;
            cudaMalloc((void**)&CN_neighbors, length * MAX_NEIGHBORS * sizeof(neighbor_t));
            cudaMemset(CN_neighbors, 0, length * MAX_NEIGHBORS * sizeof(neighbor_t));
            this->host_data_.CN_neighbors = CN_neighbors;
            real_t *coordination_numbers;
            cudaMalloc((void**)&coordination_numbers, length * sizeof(real_t));
            cudaMemset(coordination_numbers, 0, length * sizeof(real_t));
            this->host_data_.coordination_numbers = coordination_numbers;
            uint64_t *num_neighbors;
            cudaMalloc((void**)&num_neighbors, length * sizeof(uint64_t));
            cudaMemset(num_neighbors, 0, length * sizeof(uint64_t));
            this->host_data_.num_neighbors = num_neighbors;
            uint64_t *num_CN_neighbors;
            cudaMalloc((void**)&num_CN_neighbors, length * sizeof(uint64_t));
            cudaMemset(num_CN_neighbors, 0, length * sizeof(uint64_t));
            this->host_data_.num_CN_neighbors = num_CN_neighbors;
            this->host_data_.max_num_CN_neighbor = 0; // initialize the maximum number of CN neighbors to 0
            real_t *energy;
            cudaMalloc((void**)&energy, sizeof(real_t));
            cudaMemset(energy, 0, sizeof(real_t));
            this->host_data_.energy = energy;
            real_t *forces;
            cudaMalloc((void**)&forces, length * 3 * sizeof(real_t));
            cudaMemset(forces, 0, length * 3 * sizeof(real_t));
            this->host_data_.forces = forces;
            real_t *stress;
            cudaMalloc((void**)&stress, 9 * sizeof(real_t));
            cudaMemset(stress, 0, 9 * sizeof(real_t));
            this->host_data_.stress = stress;
        }
        /* copy the data to device */
        device_data_t *d_data;
        CHECK_CUDA(cudaMalloc((void**)&d_data, sizeof(device_data_t)));
        CHECK_CUDA(cudaMemcpy(d_data, &this->host_data_, sizeof(device_data_t), cudaMemcpyHostToDevice));
        this->device_data_ = d_data; // set the data pointer in the class
    } // Device_Buffer constructor
    __host__ ~Device_Buffer() {
        if (this->device_data_ != NULL) {
            CHECK_CUDA(cudaFree(this->host_data_.atom_types)); // free the atom types array
            CHECK_CUDA(cudaFree(this->host_data_.atoms)); // free the atoms array
            CHECK_CUDA(cudaFree(this->host_data_.c6_ab_ref)); // free the c6ab_ref array
            CHECK_CUDA(cudaFree(this->host_data_.r0ab)); // free the r0ab array
            CHECK_CUDA(cudaFree(this->host_data_.rcov)); // free the rcov array
            CHECK_CUDA(cudaFree(this->host_data_.r2r4)); // free the r2r4 array
            CHECK_CUDA(cudaFree(this->host_data_.neighbors)); // free the neighbors array
            CHECK_CUDA(cudaFree(this->host_data_.CN_neighbors)); // free the CN neighbors array
            CHECK_CUDA(cudaFree(this->host_data_.coordination_numbers)); // free the coordination numbers array
            CHECK_CUDA(cudaFree(this->host_data_.num_neighbors)); // free the number of neighbors array
            CHECK_CUDA(cudaFree(this->host_data_.num_CN_neighbors)); // free the number of CN neighbors array
            CHECK_CUDA(cudaFree(this->host_data_.energy)); // free the energy array
            CHECK_CUDA(cudaFree(this->host_data_.forces)); // free the forces array
            CHECK_CUDA(cudaFree(this->host_data_.stress)); // free the stress array
        }
    } // Device_Buffer destructor

    /* disable copying */
    Device_Buffer(const Device_Buffer&) = delete; // disable copy constructor
    Device_Buffer& operator=(const Device_Buffer&) = delete; // disable copy assignment operator

    /* enable moving */
    __host__ Device_Buffer(Device_Buffer&& other) noexcept : device_data_(other.device_data_) {
        other.device_data_ = nullptr; // transfer ownership of the data pointer
    } // move constructor
    __host__ Device_Buffer& operator=(Device_Buffer&& other) noexcept {
        if (this != &other) {
            if (this->device_data_ != nullptr){
                CHECK_CUDA(cudaFree(this->device_data_)); // free the current data
            }
            this->device_data_ = other.device_data_; // transfer ownership of the data pointer
            other.device_data_ = nullptr; // set the other data pointer to null
        }
        return *this;
    } // move assignment operator

    __host__ device_data_t* get_device_data() {
        return this->device_data_; // return the device data pointer
    } // get device data pointer
    __host__ device_data_t get_host_data() {
        return this->host_data_; // return the host data
    } // get host data
private:
    device_data_t *device_data_; // pointer to the device data
    device_data_t host_data_;
}; // Device_Buffer

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
    uint64_t num_threads = blockDim.x; // total number of threads in the block
    uint64_t thread_index = threadIdx.x; // the linear index of thread in current block
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
    /* loop over three dimentions to find neighbor of atom 1 */
    neighbor_t neighbors[MAX_LOCAL_NEIGHBORS]; // array of neighbors for the central atom
    neighbor_t CN_neighbors[MAX_LOCAL_NEIGHBORS]; // array of neighbors for the CN calculation
    uint64_t neighbors_index = 0; // index of the neighbor in the neighbors array
    uint64_t CN_neighbors_index = 0; // index of the neighbor in the CN neighbors array
    for(uint64_t atom_2_index = thread_index; atom_2_index < data->num_atoms; atom_2_index += num_threads) {
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
                if (CN_neighbors_index < MAX_LOCAL_NEIGHBORS) {
                    CN_neighbor_flags[thread_index] += 1; // mark the atom as a neighbor
                    CN_neighbors[CN_neighbors_index].index = atom_2_index; // set the index of the neighbor atom
                    CN_neighbors[CN_neighbors_index].distance = distance; // set the distance to the neighbor atom
                    CN_neighbors[CN_neighbors_index].atom = atom_2; // set the atom data of the neighbor atom
                    CN_neighbors_index++; // increment the index of the neighbor
                } else {
                    printf("Warning: too many CN neighbors for atom %llu and atom %llu\n", atom_1_index, atom_2_index); // too many neighbors, skip this one
                }
            }
            if (distance <= data->cutoff && distance > 0.0f) {
                if (neighbors_index < MAX_LOCAL_NEIGHBORS) {
                    neighbor_flags[thread_index] += 1; // mark the atom as a neighbor for CN calculation
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
    __syncthreads();
    /* now we need to convert entries in neighbor_flags to indicies in neighbors. */
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
    real_t local_coordination_number = 0.0f; // local coordination number for the central atom
    for(uint64_t i = 0; i < CN_neighbors_index; ++i) {
        real_t distance = CN_neighbors[i].distance; // distance to the neighbor atom
        assert(distance <= data->coordination_number_cutoff); // make sure the distance is in bounds
        /* if the distance is within cutoff range, update neighbors */
        uint64_t neighbor_index = CN_neighbor_flags[thread_index]; // index of the neighbor in the neighbors array
        uint64_t atom_2_index = CN_neighbors[i].index; // index of the second atom in the pair
        uint64_t atom_2_type = data->atom_types[atom_2_index]; // type of the surrounding atom
        neighbor_t *data_neighbors = &data->CN_neighbors[atom_1_index * MAX_NEIGHBORS]; // pointer to the neighbors array for the central atom
        assert(neighbor_index + i < MAX_NEIGHBORS); // make sure the index is in bounds
        assert(data_neighbors[neighbor_index+i].index == 0); // make sure the index is not already set
        assert(data_neighbors[neighbor_index+i].distance == 0); // make sure the distance is not already set
        data_neighbors[neighbor_index+i].index = CN_neighbors[i].index;
        data_neighbors[neighbor_index+i].distance = distance;
        data_neighbors[neighbor_index+i].atom = CN_neighbors[i].atom;
        /* compute the coordination number and add to the CN of atom 1 and atom 2 */
        real_t covalent_radii_1 = data->rcov[atom_1_type];
        real_t covalent_radii_2 = data->rcov[atom_2_type];
        /* eq 15 in Grimme et al. 2010
        $CN^A = \sum_{B \neq A}^{N} \sqrt{1}{1+exp(-k_1(k_2(R_{A,cov}+R_{B,cov})/r_{AB}-1))}$ */
        real_t exp = expf(-K1*((covalent_radii_1 + covalent_radii_2)/distance - 1.0f)); // $\exp(-k_1*(\frac{R_A+R_b}{r_{ab}}-1))$
        real_t tanh_value = tanhf(data->coordination_number_cutoff - distance); // $\tanh(CN_cutoff - r_{ab})$
        real_t smooth_cutoff = 1; // powf(tanh_value, 3); // $\tanh^3(CN_cutoff- r_{ab})})$, this is a smooth cutoff function added in LASP code.
        real_t d_smooth_cutoff_dr = 0; // 3.0f * powf(tanh_value, 2) * (1.0f - powf(tanh_value,2)) * (-1.0f); // derivative of the smooth cutoff function with respect to distance       
        real_t coordination_number = 1.0f/(1.0f+exp) * smooth_cutoff; // the covalent radii in input table have already taken K2 coefficient into onsideration
        real_t dCN_datom = powf(1.0f+exp,-2.0f)*(-K1)*exp*(covalent_radii_1 + covalent_radii_2)*powf(distance, -3.0f) * smooth_cutoff + d_smooth_cutoff_dr * 1.0f/(1.0f+exp) / distance; // dCN_ij/dr_ij * 1/r_ij
        data_neighbors[neighbor_index+i].dCN_dr = dCN_datom; // set the dCN/dr for the neighbor atom
        // increment the data.coordination_number array for both atoms
        // atomicAdd(&data->coordination_numbers[atom_1_index], coordination_number); // add the coordination number to the central atom
        local_coordination_number += coordination_number; // add the coordination number to the local coordination number
    }
    /* now the coordination number of the central atom is stored in local_coordination_number, add it back to global memory. */
    atomicAdd(&data->coordination_numbers[atom_1_index], local_coordination_number); // accumulate the coordination number for the central atom
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
    return; // return from the kernel
}

/**
 * @brief this kernel is used to compute the two-body interactions between atoms in the system.
 * @brief i.e the energy and two-atom part of force
 * @note this kernel should be launched with a 1D grid of blocks, each block containining a 1D array of threads.
 * @note the number of blocks should be equal to the number of atoms in the system.
 * @note the number of threads in each block can be any value, but it's better to set a value smaller than the number of neighbors.
 */
__global__ void two_body_kernel(device_data_t *data){
    extern __shared__ real_t force_cache[]; // shared memory for forces, size: 3 * data->num_atoms
    uint64_t atom_1_index = blockIdx.x; // each block is responsible for one central atom
    atom_t atom_1 = data->atoms[atom_1_index]; // central atom
    /* local variables to reduce calls to `atomicAdd` */
    real_t local_energy = 0.0f; // local energy for the central atom
    real_t local_force_central[3] = {0.0f, 0.0f, 0.0f}; // local force for the central atom
    real_t local_stress[9] = {0.0f}; // local stress matrix
    assert(data->num_neighbors[atom_1_index] <= MAX_NEIGHBORS); // make sure the number of neighbors is in bounds
    if (atom_1_index != data->num_atoms - 1) {
        assert(data->neighbors[atom_1_index * MAX_NEIGHBORS + data->num_neighbors[atom_1_index]].index == 0); // make sure the last entry is empty
    }
    for (uint64_t neighbor_index = threadIdx.x; neighbor_index < data->num_neighbors[atom_1_index]; neighbor_index += blockDim.x) {
        /* each thread is responsible for one atom pair, so the number of threads should be equal to num_atoms * total_cell_bias */
        if (neighbor_index >= data->num_neighbors[atom_1_index]) {
            break; // exit the loop if the index is out of bounds
        }
        uint64_t atom_2_index = data->neighbors[atom_1_index * MAX_NEIGHBORS + neighbor_index].index; // index of the second atom in the pair
        atom_t atom_2 = data->neighbors[atom_1_index * MAX_NEIGHBORS + neighbor_index].atom; // surrounding atom
        real_t distance = data->neighbors[atom_1_index * MAX_NEIGHBORS + neighbor_index].distance; // distance to the neighbor atom
        real_t coordination_number_1 = data->coordination_numbers[atom_1_index];
        real_t coordination_number_2 = data->coordination_numbers[atom_2_index];
        uint64_t atom_1_type = data->atom_types[atom_1_index];
        uint64_t atom_2_type = data->atom_types[atom_2_index];
        /* calculate cell volume */
        real_t cell_volume = data->cell[0][0] * data->cell[1][1] * data->cell[2][2] - data->cell[0][1] * data->cell[1][0] * data->cell[2][2] - data->cell[0][2] * data->cell[1][1] * data->cell[2][0] + data->cell[0][1] * data->cell[1][2] * data->cell[2][0] + data->cell[0][2] * data->cell[1][0] * data->cell[2][1];
        /* calculate the coordination number based on dispersion coefficient
            formula: $C_6^{ij} = Z/W$ 
            where $Z = \sum_{a,b}C_{6,ref}^{i,j}L_{a,b}$
            $W = \sum_{a,b}L_{a,b}$
            $L_{a,b} = \exp(-k3((CN^A-CN^A_{ref,a})^2 + (CN^B-CN^B_{ref,b})^2))$ */
        real_t Z = 0.0f;
        real_t W = 0.0f;
        real_t c_ref_L_ij = 0.0f;
        real_t c_ref_dL_ij_1 = 0.0f;
        real_t dL_ij_1 = 0.0f;
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
                /* because they could be invalid, L_ij cannot be used directly */
                real_t L_ij_candidate = expf(-K3 * (powf(coordination_number_1 - coordination_number_ref_1, 2) + powf(coordination_number_2 - coordination_number_ref_2, 2)));
                real_t dL_ij_candidate_1 = -2.0f * K3 * (coordination_number_1 - coordination_number_ref_1) * L_ij_candidate; // dL_ij/dCN_1
                /* since we need the value $\frac{\sum_{i,j}C_{6,ref}^{A,B}L_{i,j}}{\sum_{i,j}L_{i,j}}$
                    we can set invalid L_ij to 0.0f and perform the summation in the same loop
                    invalid entry: have -1.0f in c6_ref, coordination_number_ref_1 and coordination_number_ref_2
                    we check coordination_number_ref_1 here. */
                if (coordination_number_ref_1 - (-1.0f) > 1e-5f) {
                    Z += c6_ref * L_ij_candidate; // accumulate the value of Z
                    W += L_ij_candidate; // accumulate the value of W
                    /* accumulate the value of c_ref_L_ij, dL_ij_1 and dL_ij_2 */
                    c_ref_L_ij += c6_ref * L_ij_candidate;
                    c_ref_dL_ij_1 += c6_ref * dL_ij_candidate_1;
                    dL_ij_1 += dL_ij_candidate_1;
                }
            }
        }
        real_t L_ij = W;
        real_t dC6ab_dCN_1 = (L_ij > 0.0f) ? (c_ref_dL_ij_1*L_ij - c_ref_L_ij * dL_ij_1) / powf(L_ij,2.0f) : 0.0f; // avoid division by zero
        real_t c6_ab = (W > 0.0f) ? Z / W : 0.0f; // avoid division by zero
        /* calculate c8_ab by $C_8^{AB} = 3C_6^{AB}\sqrt{Q^AQ^B}$*/
        real_t r2r4_1 = data->r2r4[atom_1_type];
        real_t r2r4_2 = data->r2r4[atom_2_type];
        real_t c8_ab = 3.0f * c6_ab * r2r4_1 * r2r4_2; // the value in r2r4 is already squared
        real_t dC8ab_dCN_1 = 3.0f * dC6ab_dCN_1 * r2r4_1 * r2r4_2; // dC8ab/dCN_1
        /* acquire the cutoff radius between the two atoms */
        real_t cutoff_radius = data->r0ab[atom_1_type*data->num_elements + atom_2_type];
        /* calculate the dampling function as Grimme et al. 2010, eq4 */
        real_t f_dn_6 = 1/(1+6.0f*powf(distance/(SR_6*cutoff_radius), -ALPHA_N(6.0f)));
        real_t f_dn_8 = 1/(1+6.0f*powf(distance/(SR_8*cutoff_radius), -ALPHA_N(8.0f)));
        /* calculate the dispersion enegry as Grimme et al. 2010, eq3 */
        real_t dispersion_energy_6 = S6*(c6_ab/powf(distance, 6.0f))*f_dn_6;
        real_t dispersion_energy_8 = S8*(c8_ab/powf(distance, 8.0f))*f_dn_8;
        // printf("dispersion energy between atoms (%llu,%llu): %f\n",atom_1_index,atom_2_index,dispersion_energy_6 + dispersion_energy_8);
        real_t dispersion_energy = dispersion_energy_6 + dispersion_energy_8;
        /* add the energy back to results */
        local_energy += dispersion_energy / 2.0f; // add the energy to the local energy
       
        /* the first entry of two-body force
         $F_a = S_n C_n^{ab} f_{d,n}(r_{ab}) \frac{\partial}{\partial r_a} r_{ab}^{-n}$
         $F_a = S_n C_n^{ab} f_{d,n}(r_{ab}) * (-n)r_{ab}^{-n-2} * \uparrow{r_{ab}}$ */
        real_t force = 0.0f; // dE/dr * 1/r
        force += S6 * c6_ab * f_dn_6 * (-6.0f) * powf(distance, -8.0f); // dE_6/dr * 1/r
        force += S8 * c8_ab * f_dn_8 * (-8.0f) * powf(distance, -10.0f); // dE_8/dr * 1/r
        /* the second entry of two-body force 
         $F_a = S_n C_n^{ab} r_{ab}^{-n} \frac{\partial}{\partial r_a} f_{d,n}(r_{ab})$
         $F_a = S_n C_n^{ab} r_{ab}^{-n} -f_{d,n}^2 * (6*(-\alpha_n)*(r_{ab}/{S_{r,n}R_0^{AB}})^(-\alpha_n - 1) * 1/(S_{r,n}R_0^{AB}})) / r_ab \uparrow{r_{ab}}$*/
        force += S6 * c6_ab * powf(distance, -6.0f) * (-f_dn_6 * f_dn_6) * (6.0f * (-ALPHA_N(6.0f))* powf(distance/(SR_6*cutoff_radius), -ALPHA_N(6.0f) - 1.0f) / (SR_6*cutoff_radius)) / distance; // dE_6/dr * 1/r
        force += S8 * c8_ab * powf(distance, -8.0f) * (-f_dn_8 * f_dn_8) * (6.0f * (-ALPHA_N(8.0f))* powf(distance/(SR_8*cutoff_radius), -ALPHA_N(8.0f) - 1.0f) / (SR_8*cutoff_radius)) / distance; // dE_8/dr * 1/r
        /* accumulate force for the central atom */
        local_force_central[0] += force * (atom_1.x - atom_2.x); // x component of the force
        local_force_central[1] += force * (atom_1.y - atom_2.y); // y component of the force
        local_force_central[2] += force * (atom_1.z - atom_2.z); // z component of the force

        /* accumulate stress to local matrix instead of directly using atomicAdd, divide by 2 becuase the same entry will be calculated twice (when atom_2 is the central atom) */
        local_stress[0*3+0] += -1.0f * (atom_1.x - atom_2.x) * force * (atom_1.x - atom_2.x)/2.0f / cell_volume; // stress_xx
        local_stress[0*3+1] += -1.0f * (atom_1.x - atom_2.x) * force * (atom_1.y - atom_2.y)/2.0f / cell_volume; // stress_xy
        local_stress[0*3+2] += -1.0f * (atom_1.x - atom_2.x) * force * (atom_1.z - atom_2.z)/2.0f / cell_volume; // stress_xz
        local_stress[1*3+0] += -1.0f * (atom_1.y - atom_2.y) * force * (atom_1.x - atom_2.x)/2.0f / cell_volume; // stress_yx
        local_stress[1*3+1] += -1.0f * (atom_1.y - atom_2.y) * force * (atom_1.y - atom_2.y)/2.0f / cell_volume; // stress_yy
        local_stress[1*3+2] += -1.0f * (atom_1.y - atom_2.y) * force * (atom_1.z - atom_2.z)/2.0f / cell_volume; // stress_yz
        local_stress[2*3+0] += -1.0f * (atom_1.z - atom_2.z) * force * (atom_1.x - atom_2.x)/2.0f / cell_volume; // stress_zx
        local_stress[2*3+1] += -1.0f * (atom_1.z - atom_2.z) * force * (atom_1.y - atom_2.y)/2.0f / cell_volume; // stress_zy
        local_stress[2*3+2] += -1.0f * (atom_1.z - atom_2.z) * force * (atom_1.z - atom_2.z)/2.0f / cell_volume; // stress_zz

        /* increment contribution of dC6ab/drai where i is neighbor of a to force of a and i */
        for (uint64_t neighbor_a = 0; neighbor_a < data->num_CN_neighbors[atom_1_index]; ++neighbor_a) {
            uint64_t neighbor_a_index = data->CN_neighbors[atom_1_index * MAX_NEIGHBORS + neighbor_a].index; // index of the neighbor atom
            atom_t neighbor_a_atom = data->CN_neighbors[atom_1_index * MAX_NEIGHBORS + neighbor_a].atom; // atom data of the neighbor atom
            real_t dC6ab_drai = dC6ab_dCN_1 * data->CN_neighbors[atom_1_index * MAX_NEIGHBORS + neighbor_a].dCN_dr; // dC6ab/dr_i * 1/r_i
            real_t dC8ab_drai = dC8ab_dCN_1 * data->CN_neighbors[atom_1_index * MAX_NEIGHBORS + neighbor_a].dCN_dr; // dC8ab/dr_i * 1/r_i
            real_t dE_drai = 0.0f;
            dE_drai += S6 * f_dn_6 * powf(distance, -6.0f) * dC6ab_drai; // dE_6/dr * 1/r
            dE_drai += S8 * f_dn_8 * powf(distance, -8.0f) * dC8ab_drai; // dE_8/dr * 1/r
            /* accumulate force */
            atomicAdd(&force_cache[neighbor_a_index*3+0], dE_drai * (neighbor_a_atom.x - atom_1.x));
            atomicAdd(&force_cache[neighbor_a_index*3+1], dE_drai * (neighbor_a_atom.y - atom_1.y));
            atomicAdd(&force_cache[neighbor_a_index*3+2], dE_drai * (neighbor_a_atom.z - atom_1.z));
            local_force_central[0] += -dE_drai * (neighbor_a_atom.x - atom_1.x);
            local_force_central[1] += -dE_drai * (neighbor_a_atom.y - atom_1.y);
            local_force_central[2] += -dE_drai * (neighbor_a_atom.z - atom_1.z);
            /* accumulate stress */
            local_stress[0*3+0] += -1.0f * (atom_1.x - neighbor_a_atom.x) * dE_drai * (atom_1.x - neighbor_a_atom.x) / cell_volume; // stress_xx
            local_stress[0*3+1] += -1.0f * (atom_1.x - neighbor_a_atom.x) * dE_drai * (atom_1.y - neighbor_a_atom.y) / cell_volume; // stress_xy
            local_stress[0*3+2] += -1.0f * (atom_1.x - neighbor_a_atom.x) * dE_drai * (atom_1.z - neighbor_a_atom.z) / cell_volume; // stress_xz
            local_stress[1*3+0] += -1.0f * (atom_1.y - neighbor_a_atom.y) * dE_drai * (atom_1.x - neighbor_a_atom.x) / cell_volume; // stress_yx
            local_stress[1*3+1] += -1.0f * (atom_1.y - neighbor_a_atom.y) * dE_drai * (atom_1.y - neighbor_a_atom.y) / cell_volume; // stress_yy
            local_stress[1*3+2] += -1.0f * (atom_1.y - neighbor_a_atom.y) * dE_drai * (atom_1.z - neighbor_a_atom.z) / cell_volume; // stress_yz
            local_stress[2*3+0] += -1.0f * (atom_1.z - neighbor_a_atom.z) * dE_drai * (atom_1.x - neighbor_a_atom.x) / cell_volume; // stress_zx
            local_stress[2*3+1] += -1.0f * (atom_1.z - neighbor_a_atom.z) * dE_drai * (atom_1.y - neighbor_a_atom.y) / cell_volume; // stress_zy
            local_stress[2*3+2] += -1.0f * (atom_1.z - neighbor_a_atom.z) * dE_drai * (atom_1.z - neighbor_a_atom.z) / cell_volume; // stress_zz
        }
    }
    /* accumulate energy */
    atomicAdd(data->energy, local_energy); // accumulate the energy for the central atom

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

    /* accumulate force cache to global memory */
    __syncthreads(); // make sure all threads see the updated force cache
    for (uint64_t i = threadIdx.x; i < data->num_atoms * 3; i += blockDim.x) {
        atomicAdd(&data->forces[i], force_cache[i]); // accumulate the forces for the central atom
    }
}

/**
 * @brief the function is used to compute the dispersion energy of the system using the D3 potential.
 * 
 * @param atoms the array of atoms in the system. The array is of size num_atoms * 4, where the first entry is atomic number and the last 3 entries are the coordinates of the atom.
 * @param length the number of atoms in the system.
 * @note the function is not thread safe, and should be called from a single thread.
 */
__host__ void compute_dispersion_energy(
    real_t coords[][3], 
    uint16_t *elements,
    uint64_t length, 
    real_t cell[3][3],
    real_t cutoff_radius,
    real_t coordination_number_cutoff,
    real_t *energy,
    real_t *force,
    real_t *stress
    ) {
    // allocate memory for device_data_t
    debug("starting compute_dispersion_energy...\n");
    Device_Buffer buffer(coords, elements, cell, length, cutoff_radius, coordination_number_cutoff); // create a buffer to hold the data
    // launch the kernel
    printf("launching coordination_number_kernel, size: %zu, %zu\n", length, length);
    coordination_number_kernel<<<length, length>>>(buffer.get_device_data()); // launch the kernel to compute the coordination numbers
    CHECK_CUDA(cudaDeviceSynchronize()); // synchronize the device to ensure all threads are finished
    printf("launching two_body_kernel, size: %zu, %zu\n", length, (uint64_t)512);
    two_body_kernel<<<length, 512, length * 3 * sizeof(real_t)>>>(buffer.get_device_data());
    CHECK_CUDA(cudaDeviceSynchronize()); // synchronize the device to ensure all threads are finished

    cudaMemcpy(force, buffer.get_host_data().forces, length * 3 * sizeof(real_t), cudaMemcpyDeviceToHost); // copy the forces back to host memory
    cudaMemcpy(energy, buffer.get_host_data().energy, sizeof(real_t), cudaMemcpyDeviceToHost); // copy the energy back to host memory
    cudaMemcpy(stress, buffer.get_host_data().stress, 9 * sizeof(real_t), cudaMemcpyDeviceToHost); // copy the stress back to host memory
    real_t angstron_to_bohr = 1/0.52917726f; // angstron to bohr conversion factor
    real_t hartree_to_eV = 27.211396641308f; // hartree to eV conversion factor
    *energy *= hartree_to_eV; // convert energy to eV
    for (uint64_t i = 0; i < length; ++i) {
        /* convert force from hartree/bohr to eV/angstron */
        force[i * 3 + 0] *= hartree_to_eV * angstron_to_bohr;
        force[i * 3 + 1] *= hartree_to_eV * angstron_to_bohr;
        force[i * 3 + 2] *= hartree_to_eV * angstron_to_bohr;
    }
    for (uint64_t i = 0; i < 3; ++i) {
        for (uint64_t j = 0; j < 3; ++j) {
            stress[i * 3 + j] *= hartree_to_eV * powf(angstron_to_bohr,3); // convert stress to from hartree/bohr^3 to eV/angstron^3
        }
    }
}

#ifndef BUILD_LIBRARY
int main()
{
    // example usage of the compute_dispersion_energy function
    real_t atoms[10][3] = {
        {5.1372f, 5.5512f, 10.1047f},
        {4.5169f, 6.1365f, 11.3604f},
        {6.1937f, 4.4752f, 10.2703f},
        {4.7872f, 5.9358f, 8.9937f},
        {6.7474f, 4.3475f, 9.3339f},
        {5.6975f, 3.5214f, 10.5181f},
        {6.8870f, 4.7006f, 11.0939f},
        {4.8579f, 5.6442f, 12.2774f},
        {3.4204f, 6.0677f, 11.2935f},
        {4.7678f, 7.2075f, 11.4098f}
    };
    uint16_t elements[10] = {6, 6, 6, 8, 1, 1, 1, 1, 1, 1}; // atomic numbers of the atoms
    real_t angstron_to_bohr = 1/0.52917726f; // angstron to bohr conversion factor
    for(uint64_t i = 0; i < 10; ++i) {
        atoms[i][0] *= angstron_to_bohr; // convert to bohr
        atoms[i][1] *= angstron_to_bohr; // convert to bohr
        atoms[i][2] *= angstron_to_bohr; // convert to bohr
    }
    // fill the atoms array with Po element
    debug("Computing dispersion energy for %zu atoms...\n", sizeof(atoms)/sizeof(atoms[0]));
    // initialize parameters
    init_params();
    debug("Computing dispersion energy...\n");
    real_t cell[3][3] = {
        {200.0f, 0.0f, 0.0f},
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
    real_t energy;
    real_t *force = (real_t *)malloc(sizeof(real_t) * 10 * 3); // allocate memory for force
    real_t *stress = (real_t *)malloc(sizeof(real_t) * 9); // allocate memory for stress
    compute_dispersion_energy(atoms, elements, 10, cell, cutoff_radius, CN_cutoff_radius,&energy, force, stress);
    printf("energy: %f eV\n", energy);
    real_t force_sum[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 10; ++i) {
        real_t force_x = force[0 + i * 3];
        real_t force_y = force[1 + i * 3];
        real_t force_z = force[2 + i * 3];
        force_sum[0] += force_x;
        force_sum[1] += force_y;
        force_sum[2] += force_z;
        printf("force[%d]: %.13f %.13f %.13f\n", i, force_x, force_y, force_z);
    }
    printf("force sum: %.13f %.13f %.13f\n", force_sum[0], force_sum[1], force_sum[2]);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            printf("stress[%d][%d]: %.13f\n", i, j, stress[i * 3 + j]);
        }
    }
    return 0;
}
#endif