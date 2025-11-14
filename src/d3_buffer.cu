#include <assert.h>

#include <cmath>
#include <stdexcept>

#include "d3_buffer.cuh"
#include "d3_internal.h"
#include "d3_types.h"
#include "constants_include.h"

// Calculate inverse of a 3x3 matrix
template <typename T1, typename T2>
void matrix_inverse(const T1 mat[3][3], T2 inv[3][3])
{
    // Calculate determinant
    T2 det = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) -
                 mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) +
                 mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);

    T2 inv_det = 1.0 / det;
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
template <typename T1, typename T2>
void matrix_transpose(const T1 mat[3][3], T2 trans[3][3])
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            trans[j][i] = mat[i][j];
        }
    }
}

// Calculate row-wise norm of a 3x3 matrix
template <typename T1, typename T2>
void row_norms(const T1 mat[3][3], T2 norms[3])
{
    for (int i = 0; i < 3; i++)
    {
        norms[i] = sqrt(mat[i][0] * mat[i][0] + mat[i][1] * mat[i][1] +
                        mat[i][2] * mat[i][2]);
    }
}

// Equivalent to torch.ceil(cutoff * inv_distances).long()
void calculate_cell_repeats(
    real_t cell[3][3],
    real_t cutoff,
    uint64_t max_cell_bias[3])
{
    double inv[3][3];
    double trans[3][3];
    double norms[3];

    // Calculate inverse of cell matrix
    matrix_inverse<real_t, double>(cell, inv);

    // Transpose the inverse matrix
    matrix_transpose<double, double>(inv, trans);

    // Calculate norms of each row
    row_norms<double, double>(trans, norms);
    // Multiply by cutoff and round up to nearest integer
    for (int i = 0; i < 3; i++)
    {
        /**
         * The number of repeats need to be timed by 2 due to two directions,
         * and add 1 due to the central unit (no translation at all)
         */
        max_cell_bias[i] = ((int)(cutoff * norms[i]) + 1) * 2 + 1;
    }
}

Unique_Elements::Unique_Elements(uint16_t *elements, uint16_t length)
{
    uint16_t *all_elements = (uint16_t *)malloc(MAX_ELEMENTS * sizeof(uint16_t));
    if (all_elements == NULL)
    {
        throw std::runtime_error("Error: failed to allocate memory for unique elements");
    }
    memset(all_elements, 0, MAX_ELEMENTS * sizeof(uint16_t)); // initialize all elements to 0
    this->num_elements = 0;
    // bucket sort
    for (uint16_t i = 0; i < length; ++i)
    {
        // check that no element number exceed MAX_ELEMENTS
        if (elements[i] >= MAX_ELEMENTS)
        {
            free(all_elements);
            throw std::runtime_error("Error: element exceeds maximum allowed value");
        }
        if (all_elements[elements[i]] == 0)
        {
            this->num_elements++;
        }
        all_elements[elements[i]] = 1; // mark the element as present
    }
    // allocate memory for unique elements array
    this->elements_ = (uint16_t *)malloc(this->num_elements * sizeof(uint16_t));
    if (this->elements_ == NULL)
    {
        free(all_elements);
        throw std::runtime_error("Error: failed to allocate memory for unique elements");
    }
    // fill the unique elements array
    uint16_t index = 0;
    for (uint16_t i = 0; i < MAX_ELEMENTS; ++i)
    {
        if (all_elements[i] == 1)
        {
            this->elements_[index] = i;
            index++;
        }
    }
    // sanity check
    if (index != this->num_elements)
    {
        free(this->elements_);
        free(all_elements);
        throw std::runtime_error("Error: failed to construct unique elements");
    }
    free(all_elements); // free the temporary array
} // Unique_Elements constructor

Unique_Elements::~Unique_Elements()
{
    free(this->elements_); // free the unique elements array
} // Unique_Elements destructor

uint16_t Unique_Elements::find(uint16_t element)
{
    // this could be faster with a hash map or binary search, but linear search is enough
    for (uint16_t i = 0; i < this->num_elements; ++i)
    {
        if (this->elements_[i] == element)
        {
            return i; // return the index of the element
        }
    }
    throw std::runtime_error("Error: element not found in unique elements");
}

uint16_t Unique_Elements::operator[](uint16_t index)
{
    // check that the index is within range
    if (index >= this->num_elements)
    {
        throw std::runtime_error("Error: index is out of range");
    }
    return this->elements_[index]; // return the element at the given index
} // operator to access the element at the given index

// implementation for Device_Buffer class
__host__ Device_Buffer::Device_Buffer(
    real_t coords[][3],
    uint16_t *elements,
    uint64_t length_elements,
    real_t cell[3][3],
    uint64_t length,
    real_t cutoff,
    real_t CN_cutoff,
    DampingType damping_type,
    FunctionalType functional_type)
{
    memset(&this->host_data_, 0, sizeof(device_data_t));        // initialize the host data to 0
    this->device_data_ = nullptr;                               // initialize the device data pointer to null
    Unique_Elements unique_elements(elements, length_elements); // create the unique elements object
    {
        // construct elements
        this->host_data_.num_atoms = length;                          // number of atoms in the system
        this->host_data_.num_elements = unique_elements.num_elements; // number of unique elements in the system

        uint64_t *h_atom_types = (uint64_t *)malloc(sizeof(uint64_t) * length);
        if (h_atom_types == NULL)
        {
            throw std::runtime_error("Error: failed to allocate host memory for atom types");
        }
        for (uint64_t i = 0; i < length; ++i)
        {
            h_atom_types[i] = unique_elements.find(elements[i]);
        }

        uint64_t *d_atom_types;
        CHECK_CUDA(cudaMalloc((void **)&d_atom_types, sizeof(uint64_t) * length));
        CHECK_CUDA(cudaMemcpy(d_atom_types, h_atom_types, sizeof(uint64_t) * length, cudaMemcpyHostToDevice));
        free(h_atom_types);
        this->host_data_.atom_types = d_atom_types;
    }
    {
        // construct cell
        for (uint16_t i = 0; i < 3; ++i)
        {
            for (uint16_t j = 0; j < 3; ++j)
            {
                this->host_data_.cell[i][j] = cell[i][j];
            }
        }

        // set cutoff parameters
        this->host_data_.coordination_number_cutoff = CN_cutoff;
        this->host_data_.cutoff = cutoff;

        // construct number of grid cells in each direction
        double inversed_cell_matrix[3][3];
        // we hypothesize that the CN cutoff and dispersion cutoff is close,
        // so only using the larger one to determine the grid size doesn't affect performace too much.
        double larger_cutoff = CN_cutoff > cutoff ? CN_cutoff : cutoff; // the larger cutoff value among CN cutoff and disp cutoff
        matrix_inverse<real_t, double>(this->host_data_.cell, inversed_cell_matrix);
        for (uint16_t i = 0; i < 3; ++i)
        {
            // calculate the norm of reciprocal lattice vector, note that in host_data_.cell, cell vectors are stored in rows
            double vec_norm = std::sqrt(
                inversed_cell_matrix[i][0] * inversed_cell_matrix[i][0] +
                inversed_cell_matrix[i][1] * inversed_cell_matrix[i][1] +
                inversed_cell_matrix[i][2] * inversed_cell_matrix[i][2]);
            double perpendicular_height = 1 / vec_norm;
            this->host_data_.num_grid_cells[i] = (uint64_t)std::ceil(perpendicular_height / larger_cutoff);
        }
        // construct supercell information
        calculate_cell_repeats(cell, larger_cutoff, this->host_data_.max_cell_bias);
        debug("max_cell_bias: %zu %zu %zu\n",
              this->host_data_.max_cell_bias[0],
              this->host_data_.max_cell_bias[1],
              this->host_data_.max_cell_bias[2]);
    } // supercell information
    {
        // construct atoms
        // the rearrangement of atoms is performed before calculation.
        // here we just allocate memory and copy the data to device.
        atom_t *h_atoms = (atom_t *)malloc(sizeof(atom_t) * length);
        for (uint64_t i = 0; i < length; ++i)
        {
            h_atoms[i].element = elements[i];
            h_atoms[i].original_index = i;
            h_atoms[i].home_grid_cell = 0; // to be filled in later
            h_atoms[i].x = coords[i][0];
            h_atoms[i].y = coords[i][1];
            h_atoms[i].z = coords[i][2];
        }
        atom_t *d_atoms;
        CHECK_CUDA(cudaMalloc((void **)&d_atoms, sizeof(atom_t) * length));
        CHECK_CUDA(cudaMemcpy(d_atoms, h_atoms, sizeof(atom_t) * length, cudaMemcpyHostToDevice));
        this->host_data_.atoms = d_atoms;
        // cleanup
        free(h_atoms);
    }

    // construct constants
    uint16_t num_elements = unique_elements.num_elements;
    {
        // c6ab_ref array
        this->host_data_.c6_stride_1 = num_elements * NUM_REF_C6 * NUM_REF_C6 * NUM_C6AB_ENTRIES;
        this->host_data_.c6_stride_2 = NUM_REF_C6 * NUM_REF_C6 * NUM_C6AB_ENTRIES;
        this->host_data_.c6_stride_3 = NUM_REF_C6 * NUM_C6AB_ENTRIES;
        this->host_data_.c6_stride_4 = NUM_C6AB_ENTRIES;
        real_t *h_c6ab_ref = (real_t *)malloc(num_elements * num_elements * NUM_REF_C6 * NUM_REF_C6 * NUM_C6AB_ENTRIES * sizeof(real_t));
        if (h_c6ab_ref == NULL)
        {
            throw std::runtime_error("Error: failed to allocate host memory for c6ab_ref");
        }
        for (uint16_t i = 0; i < num_elements; ++i)
        {
            for (uint16_t j = 0; j < num_elements; ++j)
            {
                uint16_t element_i = unique_elements[i];
                uint16_t element_j = unique_elements[j];
                for (uint16_t k = 0; k < NUM_REF_C6; ++k)
                {
                    for (uint16_t l = 0; l < NUM_REF_C6; ++l)
                    {
                        uint64_t index = this->host_data_.c6_stride_1 * i +
                                         this->host_data_.c6_stride_2 * j +
                                         this->host_data_.c6_stride_3 * k +
                                         this->host_data_.c6_stride_4 * l;
                        for (uint16_t m = 0; m < NUM_C6AB_ENTRIES; ++m)
                        {
                            h_c6ab_ref[index + m] =
                                c6ab_ref[element_i][element_j][k][l][m];
                        }
                    }
                }
            }
        }
        real_t *d_c6ab_ref;
        CHECK_CUDA(cudaMalloc((void **)&d_c6ab_ref, num_elements * num_elements * NUM_REF_C6 * NUM_REF_C6 * NUM_C6AB_ENTRIES * sizeof(real_t)));
        CHECK_CUDA(cudaMemcpy(d_c6ab_ref, h_c6ab_ref, num_elements * num_elements * NUM_REF_C6 * NUM_REF_C6 * NUM_C6AB_ENTRIES * sizeof(real_t), cudaMemcpyHostToDevice));
        this->host_data_.c6_ab_ref = d_c6ab_ref;
        free(h_c6ab_ref);
    } // c6ab_ref array
    {
        // r0ab array
        real_t *h_r0ab = (real_t *)malloc(num_elements * num_elements * sizeof(real_t));
        if (h_r0ab == NULL)
        {
            throw std::runtime_error("Error: failed to allocate host memory for r0ab");
        }
        for (uint16_t i = 0; i < num_elements; ++i)
        {
            for (uint16_t j = 0; j < num_elements; ++j)
            {
                uint16_t element_i = unique_elements[i];
                uint16_t element_j = unique_elements[j];
                h_r0ab[i * num_elements + j] = r0ab[element_i][element_j];
            }
        }
        real_t *d_r0ab;
        CHECK_CUDA(cudaMalloc((void **)&d_r0ab, num_elements * num_elements * sizeof(real_t)));
        CHECK_CUDA(cudaMemcpy(d_r0ab, h_r0ab, num_elements * num_elements * sizeof(real_t), cudaMemcpyHostToDevice));
        this->host_data_.r0ab = d_r0ab;
        free(h_r0ab);
    } // r0ab array
    {
        // rcov array
        real_t *h_rcov = (real_t *)malloc(num_elements * sizeof(real_t));
        if (h_rcov == NULL)
        {
            throw std::runtime_error("Error: failed to allocate host memory for rcov");
        }
        for (uint16_t i = 0; i < num_elements; ++i)
        {
            h_rcov[i] = rcov[unique_elements[i]];
        }
        real_t *d_rcov;
        CHECK_CUDA(cudaMalloc((void **)&d_rcov, num_elements * sizeof(real_t)));
        CHECK_CUDA(cudaMemcpy(d_rcov, h_rcov, num_elements * sizeof(real_t), cudaMemcpyHostToDevice));
        this->host_data_.rcov = d_rcov;
        free(h_rcov);
    } // rcov array
    {
        // r2r4 array
        real_t *h_r2r4 = (real_t *)malloc(num_elements * sizeof(real_t));
        if (h_r2r4 == NULL)
        {
            throw std::runtime_error("Error: failed to allocate host memory for r2r4");
        }
        for (uint16_t i = 0; i < num_elements; ++i)
        {
            h_r2r4[i] = r2r4[unique_elements[i]];
        }
        real_t *d_r2r4;
        CHECK_CUDA(cudaMalloc((void **)&d_r2r4, num_elements * sizeof(real_t)));
        CHECK_CUDA(cudaMemcpy(d_r2r4, h_r2r4, num_elements * sizeof(real_t), cudaMemcpyHostToDevice));
        this->host_data_.r2r4 = d_r2r4;
        free(h_r2r4);
    } // r2r4 array
    {
        // construct other fields
        this->host_data_.damping_type = damping_type;
        this->host_data_.functional_type = functional_type;
        this->host_data_.workload_distribution_type = ALL_ITERATE;               // default to all iterate
        this->host_data_.functional_params = FUNCTIONAL_PARAMS[functional_type]; // set the functional parameters
        uint64_t num_grids =
            this->host_data_.num_grid_cells[0] *
            this->host_data_.num_grid_cells[1] *
            this->host_data_.num_grid_cells[2];
        uint64_t *d_grid_start_indices;
        CHECK_CUDA(cudaMalloc((void **)&d_grid_start_indices, sizeof(uint64_t) * num_grids));
        CHECK_CUDA(cudaMemset(d_grid_start_indices, 0, sizeof(uint64_t) * num_grids));
        this->host_data_.grid_start_indices = d_grid_start_indices;
        real_t *coordination_numbers;
        CHECK_CUDA(cudaMalloc((void **)&coordination_numbers, length * sizeof(real_t)));
        CHECK_CUDA(cudaMemset(coordination_numbers, 0, length * sizeof(real_t)));
        this->host_data_.coordination_numbers = coordination_numbers;
        this->host_data_.status = COMPUTE_SUCCESS; // set the status to normal
        real_t *dCN_dr;
        CHECK_CUDA(cudaMalloc((void **)&dCN_dr, length * 3 * sizeof(real_t)));
        CHECK_CUDA(cudaMemset(dCN_dr, 0, length * 3 * sizeof(real_t)));
        this->host_data_.dCN_dr = dCN_dr;
        real_t *dE_dCN;
        CHECK_CUDA(cudaMalloc((void **)&dE_dCN, length * sizeof(real_t)));
        CHECK_CUDA(cudaMemset(dE_dCN, 0, length * sizeof(real_t)));
        this->host_data_.dE_dCN = dE_dCN;
        real_t *energy;
        CHECK_CUDA(cudaMalloc((void **)&energy, length * sizeof(real_t)));
        CHECK_CUDA(cudaMemset(energy, 0, length * sizeof(real_t)));
        this->host_data_.energy = energy;
        real_t *forces;
        CHECK_CUDA(cudaMalloc((void **)&forces, length * 3 * sizeof(real_t)));
        CHECK_CUDA(cudaMemset(forces, 0, length * 3 * sizeof(real_t)));
        this->host_data_.forces = forces;
        real_t *stress;
        CHECK_CUDA(cudaMalloc((void **)&stress, 9 * sizeof(real_t)));
        CHECK_CUDA(cudaMemset(stress, 0, 9 * sizeof(real_t)));
        this->host_data_.stress = stress;
    }
    // copy the data to device
    device_data_t *d_data;
    CHECK_CUDA(cudaMalloc((void **)&d_data, sizeof(device_data_t)));
    CHECK_CUDA(cudaMemcpy(d_data, &this->host_data_, sizeof(device_data_t), cudaMemcpyHostToDevice));
    this->device_data_ = d_data; // set the data pointer in the class
} // Device_Buffer constructor

__host__ Device_Buffer::~Device_Buffer()
{
    CHECK_CUDA(cudaFree(this->host_data_.atom_types));           // free the atom types array
    CHECK_CUDA(cudaFree(this->host_data_.atoms));                // free the atoms array
    CHECK_CUDA(cudaFree(this->host_data_.c6_ab_ref));            // free the c6ab_ref array
    CHECK_CUDA(cudaFree(this->host_data_.r0ab));                 // free the r0ab array
    CHECK_CUDA(cudaFree(this->host_data_.rcov));                 // free the rcov array
    CHECK_CUDA(cudaFree(this->host_data_.r2r4));                 // free the r2r4 array
    CHECK_CUDA(cudaFree(this->host_data_.grid_start_indices));   // free the grid start indices array
    CHECK_CUDA(cudaFree(this->host_data_.coordination_numbers)); // free the coordination numbers array
    CHECK_CUDA(cudaFree(this->host_data_.dCN_dr));               // free the dCN/dr array
    CHECK_CUDA(cudaFree(this->host_data_.dE_dCN));               // free the dE/dCN array
    CHECK_CUDA(cudaFree(this->host_data_.energy));               // free the energy array
    CHECK_CUDA(cudaFree(this->host_data_.forces));               // free the forces array
    CHECK_CUDA(cudaFree(this->host_data_.stress));               // free the stress array
    CHECK_CUDA(cudaFree(this->device_data_));                    // free the device data pointer
} // Device_Buffer destructor

__host__ Device_Buffer::Device_Buffer(Device_Buffer &&other) noexcept
    : device_data_(other.device_data_), host_data_(other.host_data_)
{
    other.device_data_ = nullptr;                        // transfer ownership of the data pointer
    memset(&other.host_data_, 0, sizeof(device_data_t)); // reset the other host data to 0
} // move constructor

__host__ Device_Buffer &Device_Buffer::operator=(
    Device_Buffer &&other) noexcept
{
    if (this != &other)
    {
        // Free existing resources of *this* object FIRST
        CHECK_CUDA(cudaFree(this->host_data_.atom_types));
        CHECK_CUDA(cudaFree(this->host_data_.atoms));
        CHECK_CUDA(cudaFree(this->host_data_.c6_ab_ref));
        CHECK_CUDA(cudaFree(this->host_data_.r0ab));
        CHECK_CUDA(cudaFree(this->host_data_.rcov));
        CHECK_CUDA(cudaFree(this->host_data_.r2r4));
        CHECK_CUDA(cudaFree(this->host_data_.grid_start_indices));
        CHECK_CUDA(cudaFree(this->host_data_.coordination_numbers));
        CHECK_CUDA(cudaFree(this->host_data_.dCN_dr));
        CHECK_CUDA(cudaFree(this->host_data_.dE_dCN));
        CHECK_CUDA(cudaFree(this->host_data_.energy));
        CHECK_CUDA(cudaFree(this->host_data_.forces));
        CHECK_CUDA(cudaFree(this->host_data_.stress));
        CHECK_CUDA(cudaFree(this->device_data_));
        // Transfer ownership from other to this
        this->device_data_ = other.device_data_;
        this->host_data_ = other.host_data_;
        // Null out other's pointers to prevent double free by its destructor
        other.device_data_ = nullptr;
        memset(&other.host_data_, 0, sizeof(device_data_t)); // Zero out all pointers in other.host_data_
    }
    return *this;
}

__host__ void Device_Buffer::set_atoms(
    uint16_t *elements,
    real_t coords[][3],
    uint64_t length)
{
    // check that then length doesn't exceed current length
    if (length > this->host_data_.num_atoms)
    {
        fprintf(stderr, "Error: length %zu exceeds the current length %zu\n",
                length, this->host_data_.num_atoms);
        exit(EXIT_FAILURE);
    }
    // update to host and device data
    this->host_data_.num_atoms = length;                                                                          // set the number of atoms in the system in host_data
    CHECK_CUDA(cudaMemcpy(this->device_data_, &this->host_data_, sizeof(device_data_t), cudaMemcpyHostToDevice)); // copy the host data to device
    // check that all elements are within scope
    Unique_Elements unique_elements(elements, length); // create the unique elements object
    for (uint64_t i = 0; i < length; ++i)
    {
        if (elements[i] >= MAX_ELEMENTS)
        {
            throw std::runtime_error("Error: element exceeds maximum allowed value");
        }
        unique_elements.find(elements[i]); // check that the element is in the unique elements array. if not found, it will crash.
    }
    // set the atoms in the device data
    atom_t *h_atoms = (atom_t *)malloc(length * sizeof(atom_t));
    if (h_atoms == NULL)
    {
        throw std::runtime_error("Error: failed to allocate host memory for atoms");
    }
    debug("Setting atoms: \n");
    for (uint64_t i = 0; i < length; ++i)
    {
        h_atoms[i].element = elements[i];
        h_atoms[i].original_index = i;
        h_atoms[i].home_grid_cell = 0; // to be filled in later
        h_atoms[i].x = coords[i][0];
        h_atoms[i].y = coords[i][1];
        h_atoms[i].z = coords[i][2];
        debug("Atom %zu: %d %f %f %f\n", i, h_atoms[i].element, h_atoms[i].x, h_atoms[i].y, h_atoms[i].z);
    }
    CHECK_CUDA(cudaMemcpy(this->host_data_.atoms, h_atoms, length * sizeof(atom_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    free(h_atoms); // free the host atoms array
} // set atoms

__host__ void Device_Buffer::set_cell(real_t cell[3][3])
{
    // set the cell in the device data
    debug("Setting cell: \n");
    for (uint16_t i = 0; i < 3; ++i)
    {
        for (uint16_t j = 0; j < 3; ++j)
        {
            this->host_data_.cell[i][j] = cell[i][j];
            debug("%f ", this->host_data_.cell[i][j]); // print the cell matrix
        }
        debug("\n");
    }

    real_t CN_cutoff = this->host_data_.coordination_number_cutoff;
    real_t cutoff = this->host_data_.cutoff;
    // construct number of grid cells in each direction
    double inversed_cell_matrix[3][3];
    // we hypothesize that the CN cutoff and dispersion cutoff is close,
    // so only using the larger one to determine the grid size doesn't affect performace too much.
    WorkloadDistributionType distribution_type = CELL_LIST;
    double larger_cutoff = CN_cutoff > cutoff ? CN_cutoff : cutoff; // the larger cutoff value among CN cutoff and disp cutoff
    matrix_inverse<real_t, double>(this->host_data_.cell, inversed_cell_matrix);
    for (uint16_t i = 0; i < 3; ++i)
    {
        // calculate the norm of reciprocal lattice vector, note that in host_data_.cell, cell vectors are stored in rows
        double vec_norm = std::sqrt(
            inversed_cell_matrix[i][0] * inversed_cell_matrix[i][0] +
            inversed_cell_matrix[i][1] * inversed_cell_matrix[i][1] +
            inversed_cell_matrix[i][2] * inversed_cell_matrix[i][2]);
        double perpendicular_height = 1 / vec_norm;
        uint64_t num_grid_cell = (uint64_t)std::ceil(perpendicular_height / larger_cutoff);
        this->host_data_.num_grid_cells[i] = num_grid_cell;
        if (num_grid_cell <= 2)
        {
            // if any direction has less than 3 grid cells, we have to use all iterate
            // for directions with only 1 grid cells, the cell list method may miss some interactions
            // for directions with 2 grid cells, the cell list method will double-count some interactions
            distribution_type = ALL_ITERATE;
        }
    }
    this->host_data_.workload_distribution_type = distribution_type;
    // construct supercell information
    calculate_cell_repeats(cell, larger_cutoff, this->host_data_.max_cell_bias);
    debug("max_cell_bias: %zu %zu %zu\n",
          this->host_data_.max_cell_bias[0],
          this->host_data_.max_cell_bias[1],
          this->host_data_.max_cell_bias[2]);
    CHECK_CUDA(cudaMemcpy(this->device_data_, &this->host_data_, sizeof(device_data_t), cudaMemcpyHostToDevice)); // copy the host data to device
    CHECK_CUDA(cudaDeviceSynchronize());                                                                          // synchronize the device
} // set cell

__host__ void Device_Buffer::clear()
{
    CHECK_CUDA(cudaMemset(host_data_.coordination_numbers, 0, host_data_.num_atoms * sizeof(real_t))); // clear the coordination numbers
    CHECK_CUDA(cudaMemset(host_data_.dCN_dr, 0, host_data_.num_atoms * 3 * sizeof(real_t)));           // clear the dCN/dr
    CHECK_CUDA(cudaMemset(host_data_.dE_dCN, 0, host_data_.num_atoms * sizeof(real_t)));               // clear the dE/dCN
    CHECK_CUDA(cudaMemset(host_data_.energy, 0, host_data_.num_atoms * sizeof(real_t)));               // clear the energy
    CHECK_CUDA(cudaMemset(host_data_.forces, 0, host_data_.num_atoms * 3 * sizeof(real_t)));           // clear the forces
    CHECK_CUDA(cudaMemset(host_data_.stress, 0, 9 * sizeof(real_t)));                                  // clear the stress
    CHECK_CUDA(cudaMemcpy(device_data_, &host_data_, sizeof(device_data_t), cudaMemcpyHostToDevice));  // copy the host data to device
    CHECK_CUDA(cudaDeviceSynchronize());                                                               // synchronize the device
} // clear the device buffer

__host__ void Device_Buffer::construct_grids()
{
    // for debug
    // this->host_data_.workload_distribution_type = ALL_ITERATE;
    // CHECK_CUDA(cudaMemcpy(this->device_data_, &this->host_data_, sizeof(device_data_t), cudaMemcpyHostToDevice));

    // print the workload distribution type
    debug("Workload distribution type: %d\n", this->host_data_.workload_distribution_type);
    // if the workload distribution type is ALL_ITERATE, return directly
    if (this->host_data_.workload_distribution_type == ALL_ITERATE)
    {
        return;
    }
    uint64_t num_atoms = this->host_data_.num_atoms;

    // construct atoms
    // now we need to figure out which grid each atom belongs to
    // and sort the atoms according to the grid indices
    // we use counting sort to achieve this
    uint64_t total_grids = host_data_.num_grid_cells[0] *
                           host_data_.num_grid_cells[1] *
                           host_data_.num_grid_cells[2];
    uint64_t *grid_indices = (uint64_t *)malloc(num_atoms * sizeof(uint64_t)); // array of the grid index of each atom
    memset(grid_indices, 0, num_atoms * sizeof(uint64_t));
    uint64_t *grid_counts = (uint64_t *)malloc(total_grids * sizeof(uint64_t)); // array of the counts of atoms in each grid
    memset(grid_counts, 0, total_grids * sizeof(uint64_t));

    // calculate grid indices of each atom and count atoms per grid
    double inv_cell[3][3]; // inverse of the cell matrix
    matrix_inverse<real_t, double>(this->host_data_.cell, inv_cell);

    // Allocate temporary storage for wrapped coordinates
    double(*wrapped_coords)[3] = (double(*)[3])malloc(num_atoms * sizeof(double[3]));
    if (wrapped_coords == NULL)
    {
        throw std::runtime_error("Error: failed to allocate memory for wrapped_coords");
    }

    // copy atoms and atom types from device to host
    atom_t *original_atoms = (atom_t *)malloc(num_atoms * sizeof(atom_t));
    CHECK_CUDA(cudaMemcpy(original_atoms, this->host_data_.atoms, num_atoms * sizeof(atom_t), cudaMemcpyDeviceToHost));
    uint64_t *original_atom_types = (uint64_t *)malloc(num_atoms * sizeof(uint64_t));
    CHECK_CUDA(cudaMemcpy(original_atom_types, this->host_data_.atom_types, num_atoms * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    for (uint64_t i = 0; i < num_atoms; ++i)
    {
        // transform the coordinates to fractional coordinates
        double frac[3] = {0.0, 0.0, 0.0};
        for (uint8_t j = 0; j < 3; ++j)
        {
            frac[j] = inv_cell[0][j] * original_atoms[i].x + inv_cell[1][j] * original_atoms[i].y + inv_cell[2][j] * original_atoms[i].z;
        }
        // calculate grid indices and handle periodic boundary conditions
        uint64_t grid_idx[3];
        for (uint8_t j = 0; j < 3; ++j)
        {
            double wrapped_frac = frac[j] - std::floor(frac[j]); // wrap to [0, 1)
            frac[j] = wrapped_frac;                              // update fractional coordinate to wrapped value
            grid_idx[j] = (uint64_t)(wrapped_frac * host_data_.num_grid_cells[j]);
            if (grid_idx[j] == host_data_.num_grid_cells[j])
            {
                grid_idx[j] = 0; // handle the edge case where coord == 1.0
            }
        }
        // convert fractional coordinates back to Cartesian
        // Note: cell vectors are stored in rows, so cell[i] is the i-th lattice vector
        // Store in temporary buffer instead of modifying input coords
        wrapped_coords[i][0] = frac[0] * this->host_data_.cell[0][0] +
                               frac[1] * this->host_data_.cell[1][0] +
                               frac[2] * this->host_data_.cell[2][0];
        wrapped_coords[i][1] = frac[0] * this->host_data_.cell[0][1] +
                               frac[1] * this->host_data_.cell[1][1] +
                               frac[2] * this->host_data_.cell[2][1];
        wrapped_coords[i][2] = frac[0] * this->host_data_.cell[0][2] +
                               frac[1] * this->host_data_.cell[1][2] +
                               frac[2] * this->host_data_.cell[2][2];
        grid_indices[i] = grid_idx[0] +
                          grid_idx[1] * host_data_.num_grid_cells[0] +
                          grid_idx[2] * host_data_.num_grid_cells[0] * host_data_.num_grid_cells[1];
        assert(grid_indices[i] < total_grids);
        grid_counts[grid_indices[i]] += 1;
    }
    // calculate the starting index of each grid in the sorted array
    uint64_t *h_grid_start_index = (uint64_t *)malloc(total_grids * sizeof(uint64_t)); // starting index of each grid
    h_grid_start_index[0] = 0;
    for (uint64_t i = 1; i < total_grids; ++i)
    {
        h_grid_start_index[i] = h_grid_start_index[i - 1] + grid_counts[i - 1];
    }
    // sort atoms using counting sort
    atom_t *h_atoms = (atom_t *)malloc(num_atoms * sizeof(atom_t));
    uint64_t *h_atom_types = (uint64_t *)malloc(num_atoms * sizeof(uint64_t)); // rearranged atom type array
    if (h_atoms == NULL || h_atom_types == NULL)
    {
        throw std::runtime_error("Error: failed to allocate host memory for atoms or atom types");
    }
    uint64_t *current_position = (uint64_t *)malloc(total_grids * sizeof(uint64_t));
    memcpy(current_position, h_grid_start_index, total_grids * sizeof(uint64_t));
    for (uint64_t i = 0; i < num_atoms; ++i)
    {
        uint64_t grid_idx = grid_indices[i];
        uint64_t pos = current_position[grid_idx];
        assert(pos < num_atoms);
        h_atoms[pos].original_index = i; // store the original index
        h_atoms[pos].element = original_atoms[i].element;
        h_atoms[pos].x = wrapped_coords[i][0];
        h_atoms[pos].y = wrapped_coords[i][1];
        h_atoms[pos].z = wrapped_coords[i][2];
        h_atoms[pos].home_grid_cell = grid_idx; // store the grid cell index
        h_atom_types[pos] = original_atom_types[i]; // rearranged atom type
        assert(grid_idx < total_grids);
        current_position[grid_idx] += 1;
    }
    // copy data to device
    CHECK_CUDA(cudaMemcpy(this->host_data_.atom_types, h_atom_types, num_atoms * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(this->host_data_.atoms, h_atoms, num_atoms * sizeof(atom_t), cudaMemcpyHostToDevice));
    // the size of `grid_start_indices` might vary, so we need to reallocate the memory
    if (this->host_data_.grid_start_indices != nullptr)
    {
        CHECK_CUDA(cudaFree(this->host_data_.grid_start_indices)); // free the old memory
    }
    uint64_t *d_grid_start_indices;
    CHECK_CUDA(cudaMalloc((void **)&d_grid_start_indices, sizeof(uint64_t) * total_grids));
    this->host_data_.grid_start_indices = d_grid_start_indices; // update the pointer
    CHECK_CUDA(cudaMemcpy(this->host_data_.grid_start_indices, h_grid_start_index, total_grids * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(this->device_data_, &this->host_data_, sizeof(device_data_t), cudaMemcpyHostToDevice)); // copy the host data to device

    // cleanup
    free(grid_indices);
    free(grid_counts);
    free(h_grid_start_index);
    free(current_position);
    free(original_atoms);
    free(original_atom_types);
    free(h_atoms);
    free(h_atom_types);
    free(wrapped_coords);
}