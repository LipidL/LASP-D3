#include <assert.h>

#include <cmath>

#include "d3_buffer.cuh"
#include "d3_internal.h"
#include "d3_types.h"
#include "constants_include.h"

// Calculate inverse of a 3x3 matrix
void matrix_inverse(const real_t mat[3][3], real_t inv[3][3]) {
    // Calculate determinant
    real_t det = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) -
                 mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) +
                 mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);

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
        norms[i] = sqrt(mat[i][0] * mat[i][0] + mat[i][1] * mat[i][1] +
                        mat[i][2] * mat[i][2]);
    }
}

// Equivalent to torch.ceil(cutoff * inv_distances).long()
void calculate_cell_repeats(real_t cell[3][3], real_t cutoff,
                            size_t max_cell_bias[3]) {
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
        max_cell_bias[i] = ((int)(cutoff * norms[i]) + 1) * 2 +
                           1;  // the number of repeats need to be timed by 2
                               // due to two directions, and add 1 due to the
                               // central unit (no translation at all)
    }
}

// implementations for UniqueElements class
Unique_Elements::Unique_Elements(uint16_t* elements, uint16_t length) {
    uint16_t* all_elements = (uint16_t*)malloc(MAX_ELEMENTS * sizeof(uint16_t));
    memset(all_elements, 0,
           MAX_ELEMENTS * sizeof(uint16_t));  // initialize all elements to 0
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
        all_elements[elements[i]] = 1;  // mark the element as present
    }
    this->elements_ = (uint16_t*)malloc(this->num_elements * sizeof(uint16_t));
    uint16_t index = 0;
    for (uint16_t i = 0; i < MAX_ELEMENTS; ++i) {
        if (all_elements[i] == 1) {
            this->elements_[index] = i;
            index++;
        }
    }
    assert_(index == this->num_elements);  // check if the number of unique
                                           // elements is correct
    free(all_elements);                    // free the temporary array
}  // Unique_Elements constructor
Unique_Elements::~Unique_Elements() {
    free(this->elements_);  // free the unique elements array
}  // Unique_Elements destructor
uint16_t Unique_Elements::find(uint16_t element) {
    for (uint16_t i = 0; i < this->num_elements; ++i) {
        if (this->elements_[i] == element) {
            return i;  // return the index of the element
        }
    }
    fprintf(stderr, "Error: element %d not found\n", element);
    exit(EXIT_FAILURE);
}  // find the index of the element in the unique elements array, if not found,
   // it will add the element to the array and return the index
uint16_t Unique_Elements::operator[](uint16_t index) {
    if (index >= this->num_elements) {
        fprintf(stderr, "Error: index %d is out of range\n", index);
        exit(EXIT_FAILURE);
    }
    return this->elements_[index];  // return the element at the given index
}  // operator to access the element at the given index

/* implementation for Device_Buffer class */
__host__ Device_Buffer::Device_Buffer(real_t coords[][3], uint16_t* elements,
                                      real_t cell[3][3], uint64_t length,
                                      real_t cutoff, real_t CN_cutoff,
                                      DampingType damping_type, FunctionalType functional_type) {
    memset(&this->host_data_, 0,
           sizeof(device_data_t));  // initialize the host data to 0
    this->device_data_ = nullptr;  // initialize the device data pointer to null
    Unique_Elements unique_elements(
        elements, length);  // create the unique elements object
    {
        /* construct elements */
        this->host_data_.num_atoms = length;  // number of atoms in the system
        this->host_data_.num_elements =
            unique_elements
                .num_elements;  // number of unique elements in the system

        uint64_t* h_atom_types = (uint64_t*)malloc(sizeof(uint64_t) * length);
        for (uint64_t i = 0; i < length; ++i) {
            h_atom_types[i] = unique_elements.find(elements[i]);
        }

        uint64_t* d_atom_types;
        CHECK_CUDA(
            cudaMalloc((void**)&d_atom_types, sizeof(uint64_t) * length));
        CHECK_CUDA(cudaMemcpy(d_atom_types, h_atom_types,
                              sizeof(uint64_t) * length,
                              cudaMemcpyHostToDevice));
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
        atom_t* h_atoms = (atom_t*)malloc(length * sizeof(atom_t));
        if (h_atoms == NULL) {
            fprintf(stderr,
                    "Error: failed to allocate memory for atoms on host");
            exit(EXIT_FAILURE);
        }
        for (uint64_t i = 0; i < length; ++i) {
            h_atoms[i].element = elements[i];
            h_atoms[i].x = coords[i][0];
            h_atoms[i].y = coords[i][1];
            h_atoms[i].z = coords[i][2];
        }
        atom_t* d_atoms;
        CHECK_CUDA(cudaMalloc((void**)&d_atoms, length * sizeof(atom_t)));
        CHECK_CUDA(cudaMemcpy(d_atoms, h_atoms, length * sizeof(atom_t),
                              cudaMemcpyHostToDevice));
        this->host_data_.atoms =
            d_atoms;    // set the atoms pointer in device data
        free(h_atoms);  // free the host atoms array
    }

    /* construct constants */
    uint16_t num_elements = unique_elements.num_elements;
    {
        /* c6ab_ref array */
        this->host_data_.c6_stride_1 =
            num_elements * NUM_REF_C6 * NUM_REF_C6 * NUM_C6AB_ENTRIES;
        this->host_data_.c6_stride_2 =
            NUM_REF_C6 * NUM_REF_C6 * NUM_C6AB_ENTRIES;
        this->host_data_.c6_stride_3 = NUM_REF_C6 * NUM_C6AB_ENTRIES;
        this->host_data_.c6_stride_4 = NUM_C6AB_ENTRIES;
        real_t* h_c6ab_ref =
            (real_t*)malloc(num_elements * num_elements * NUM_REF_C6 *
                            NUM_REF_C6 * NUM_C6AB_ENTRIES * sizeof(real_t));
        for (uint16_t i = 0; i < num_elements; ++i) {
            for (uint16_t j = 0; j < num_elements; ++j) {
                uint16_t element_i = unique_elements[i];
                uint16_t element_j = unique_elements[j];
                for (uint16_t k = 0; k < NUM_REF_C6; ++k) {
                    for (uint16_t l = 0; l < NUM_REF_C6; ++l) {
                        uint64_t index = this->host_data_.c6_stride_1 * i +
                                         this->host_data_.c6_stride_2 * j +
                                         this->host_data_.c6_stride_3 * k +
                                         this->host_data_.c6_stride_4 * l;
                        for (uint16_t m = 0; m < NUM_C6AB_ENTRIES; ++m) {
                            h_c6ab_ref[index + m] =
                                c6ab_ref[element_i - 1][element_j - 1][k][l][m];
                        }
                    }
                }
            }
        }
        real_t* d_c6ab_ref;
        CHECK_CUDA(cudaMalloc((void**)&d_c6ab_ref, num_elements * num_elements *
                                                       NUM_REF_C6 * NUM_REF_C6 *
                                                       NUM_C6AB_ENTRIES *
                                                       sizeof(real_t)));
        CHECK_CUDA(cudaMemcpy(d_c6ab_ref, h_c6ab_ref,
                              num_elements * num_elements * NUM_REF_C6 *
                                  NUM_REF_C6 * NUM_C6AB_ENTRIES *
                                  sizeof(real_t),
                              cudaMemcpyHostToDevice));
        this->host_data_.c6_ab_ref = d_c6ab_ref;
        free(h_c6ab_ref);
    }  // c6ab_ref array
    {
        /* r0ab array */
        real_t* h_r0ab =
            (real_t*)malloc(num_elements * num_elements * sizeof(real_t));
        for (uint16_t i = 0; i < num_elements; ++i) {
            for (uint16_t j = 0; j < num_elements; ++j) {
                uint16_t element_i = unique_elements[i];
                uint16_t element_j = unique_elements[j];
                h_r0ab[i * num_elements + j] = r0ab[element_i - 1][element_j - 1];
            }
        }
        real_t* d_r0ab;
        CHECK_CUDA(cudaMalloc((void**)&d_r0ab,
                              num_elements * num_elements * sizeof(real_t)));
        CHECK_CUDA(cudaMemcpy(d_r0ab, h_r0ab,
                              num_elements * num_elements * sizeof(real_t),
                              cudaMemcpyHostToDevice));
        this->host_data_.r0ab = d_r0ab;
        free(h_r0ab);
    }  // r0ab array
    {
        /* rcov array */
        real_t* h_rcov = (real_t*)malloc(num_elements * sizeof(real_t));
        for (uint16_t i = 0; i < num_elements; ++i) {
            h_rcov[i] = rcov[unique_elements[i] - 1];
        }
        real_t* d_rcov;
        CHECK_CUDA(cudaMalloc((void**)&d_rcov, num_elements * sizeof(real_t)));
        CHECK_CUDA(cudaMemcpy(d_rcov, h_rcov, num_elements * sizeof(real_t),
                              cudaMemcpyHostToDevice));
        this->host_data_.rcov = d_rcov;
        free(h_rcov);
    }  // rcov array
    {
        /* r2r4 array */
        real_t* h_r2r4 = (real_t*)malloc(num_elements * sizeof(real_t));
        for (uint16_t i = 0; i < num_elements; ++i) {
            h_r2r4[i] = r2r4[unique_elements[i] - 1];
        }
        real_t* d_r2r4;
        CHECK_CUDA(cudaMalloc((void**)&d_r2r4, num_elements * sizeof(real_t)));
        CHECK_CUDA(cudaMemcpy(d_r2r4, h_r2r4, num_elements * sizeof(real_t),
                              cudaMemcpyHostToDevice));
        this->host_data_.r2r4 = d_r2r4;
        free(h_r2r4);
    }  // r2r4 array
    {
        /* construct supercell information */
        this->host_data_.coordination_number_cutoff = CN_cutoff;
        this->host_data_.cutoff = cutoff;
        real_t larger_cutoff = CN_cutoff > cutoff ? CN_cutoff : cutoff;
        calculate_cell_repeats(cell, larger_cutoff,
                               this->host_data_.max_cell_bias);
        debug("max_cell_bias: %zu %zu %zu\n", this->host_data_.max_cell_bias[0],
              this->host_data_.max_cell_bias[1],
              this->host_data_.max_cell_bias[2]);
    }  // cupercell information
    {
        /* construct other fields */
        this->host_data_.damping_type = damping_type;
        this->host_data_.functional_type = functional_type;
        this->host_data_.functional_params =
            FUNCTIONAL_PARAMS[functional_type];  // set the functional parameters
        real_t* coordination_numbers;
        CHECK_CUDA(
            cudaMalloc((void**)&coordination_numbers, length * sizeof(real_t)));
        CHECK_CUDA(
            cudaMemset(coordination_numbers, 0, length * sizeof(real_t)));
        this->host_data_.coordination_numbers = coordination_numbers;
        this->host_data_.status = COMPUTE_SUCCESS;  // set the status to normal
        real_t* dE_dCN;
        CHECK_CUDA(cudaMalloc((void**)&dE_dCN, length * sizeof(real_t)));
        CHECK_CUDA(cudaMemset(dE_dCN, 0, length * sizeof(real_t)));
        this->host_data_.dE_dCN = dE_dCN;
        real_t* energy;
        CHECK_CUDA(cudaMalloc((void**)&energy, length * sizeof(real_t)));
        CHECK_CUDA(cudaMemset(energy, 0, length * sizeof(real_t)));
        this->host_data_.energy = energy;
        real_t* forces;
        CHECK_CUDA(cudaMalloc((void**)&forces, length * 3 * sizeof(real_t)));
        CHECK_CUDA(cudaMemset(forces, 0, length * 3 * sizeof(real_t)));
        this->host_data_.forces = forces;
        real_t* stress;
        CHECK_CUDA(cudaMalloc((void**)&stress, 9 * sizeof(real_t)));
        CHECK_CUDA(cudaMemset(stress, 0, 9 * sizeof(real_t)));
        this->host_data_.stress = stress;
    }
    /* copy the data to device */
    device_data_t* d_data;
    CHECK_CUDA(cudaMalloc((void**)&d_data, sizeof(device_data_t)));
    CHECK_CUDA(cudaMemcpy(d_data, &this->host_data_, sizeof(device_data_t),
                          cudaMemcpyHostToDevice));
    this->device_data_ = d_data;  // set the data pointer in the class
}  // Device_Buffer constructor
__host__ Device_Buffer::~Device_Buffer() {
    CHECK_CUDA(
        cudaFree(this->host_data_.atom_types));    // free the atom types array
    CHECK_CUDA(cudaFree(this->host_data_.atoms));  // free the atoms array
    CHECK_CUDA(
        cudaFree(this->host_data_.c6_ab_ref));    // free the c6ab_ref array
    CHECK_CUDA(cudaFree(this->host_data_.r0ab));  // free the r0ab array
    CHECK_CUDA(cudaFree(this->host_data_.rcov));  // free the rcov array
    CHECK_CUDA(cudaFree(this->host_data_.r2r4));  // free the r2r4 array
    CHECK_CUDA(cudaFree(
        this->host_data_
            .coordination_numbers));  // free the coordination numbers array
    CHECK_CUDA(cudaFree(this->host_data_.dE_dCN));  // free the dE/dCN array
    CHECK_CUDA(cudaFree(this->host_data_.energy));  // free the energy array
    CHECK_CUDA(cudaFree(this->host_data_.forces));  // free the forces array
    CHECK_CUDA(cudaFree(this->host_data_.stress));  // free the stress array
    CHECK_CUDA(cudaFree(this->device_data_));  // free the device data pointer
}  // Device_Buffer destructor
__host__ Device_Buffer::Device_Buffer(Device_Buffer&& other) noexcept
    : device_data_(other.device_data_), host_data_(other.host_data_) {
    other.device_data_ = nullptr;  // transfer ownership of the data pointer
    memset(&other.host_data_, 0,
           sizeof(device_data_t));  // reset the other host data to 0
}  // move constructor
__host__ Device_Buffer& Device_Buffer::operator=(
    Device_Buffer&& other) noexcept {
    if (this != &other) {
        // Free existing resources of *this* object FIRST
        CHECK_CUDA(cudaFree(this->host_data_.atom_types));
        CHECK_CUDA(cudaFree(this->host_data_.atoms));
        CHECK_CUDA(cudaFree(this->host_data_.c6_ab_ref));
        CHECK_CUDA(cudaFree(this->host_data_.r0ab));
        CHECK_CUDA(cudaFree(this->host_data_.rcov));
        CHECK_CUDA(cudaFree(this->host_data_.r2r4));
        CHECK_CUDA(cudaFree(this->host_data_.coordination_numbers));
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
        memset(
            &other.host_data_, 0,
            sizeof(
                device_data_t));  // Zero out all pointers in other.host_data_
    }
    return *this;
}
__host__ void Device_Buffer::set_atoms(uint16_t* elements, real_t coords[][3],
                                       uint64_t length) {
    /* check that then length doesn't exceed current length */
    if (length > this->host_data_.num_atoms) {
        fprintf(stderr, "Error: length %zu exceeds the current length %zu\n",
                length, this->host_data_.num_atoms);
        exit(EXIT_FAILURE);
    }
    /* update to host and device data*/
    this->host_data_.num_atoms =
        length;  // set the number of atoms in the system in host_data
    CHECK_CUDA(
        cudaMemcpy(this->device_data_, &this->host_data_, sizeof(device_data_t),
                   cudaMemcpyHostToDevice));  // copy the host data to device
    /* check that all elements are within scope */
    Unique_Elements unique_elements(
        elements, length);  // create the unique elements object
    for (uint64_t i = 0; i < length; ++i) {
        if (elements[i] >= MAX_ELEMENTS) {
            /* check that no element number exceed MAX_LEMENTS */
            fprintf(stderr, "Error: element %d is out of range\n", elements[i]);
            exit(EXIT_FAILURE);
        }
        unique_elements.find(
            elements[i]);  // check that the element is in the unique elements
                           // array. if not found, it will crash.
    }
    /* set the atoms in the device data */
    atom_t* h_atoms = (atom_t*)malloc(length * sizeof(atom_t));
    if (h_atoms == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for atoms on host");
        exit(EXIT_FAILURE);
    }
    debug("Setting atoms: \n");
    for (uint64_t i = 0; i < length; ++i) {
        h_atoms[i].element = elements[i];
        h_atoms[i].x = coords[i][0];
        h_atoms[i].y = coords[i][1];
        h_atoms[i].z = coords[i][2];
        debug("Atom %zu: %d %f %f %f\n", i, h_atoms[i].element, h_atoms[i].x,
              h_atoms[i].y, h_atoms[i].z);
    }
    CHECK_CUDA(cudaMemcpy(this->host_data_.atoms, h_atoms,
                          length * sizeof(atom_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    free(h_atoms);  // free the host atoms array
}  // set atoms
__host__ void Device_Buffer::set_cell(real_t cell[3][3]) {
    /* set the cell in the device data */
    debug("Setting cell: \n");
    for (uint16_t i = 0; i < 3; ++i) {
        for (uint16_t j = 0; j < 3; ++j) {
            this->host_data_.cell[i][j] = cell[i][j];
            debug("%f ", this->host_data_.cell[i][j]);  // print the cell matrix
        }
        debug("\n");
    }
    /* the cell size is changed, so the max_cell_bias will also change */
    calculate_cell_repeats(
        cell, this->host_data_.cutoff,
        this->host_data_.max_cell_bias);  // calculate the new max_cell_bias
    debug("max_cell_bias: %zu %zu %zu\n", this->host_data_.max_cell_bias[0],
          this->host_data_.max_cell_bias[1], this->host_data_.max_cell_bias[2]);
    CHECK_CUDA(
        cudaMemcpy(this->device_data_, &this->host_data_, sizeof(device_data_t),
                   cudaMemcpyHostToDevice));  // copy the host data to device
    CHECK_CUDA(cudaDeviceSynchronize());      // synchronize the device
}  // set cell
__host__ void Device_Buffer::clear() {
    CHECK_CUDA(
        cudaMemset(host_data_.coordination_numbers, 0,
                   host_data_.num_atoms *
                       sizeof(real_t)));  // clear the coordination numbers
    CHECK_CUDA(
        cudaMemset(host_data_.dE_dCN, 0,
                   host_data_.num_atoms * sizeof(real_t)));  // clear the dE/dCN
    CHECK_CUDA(
        cudaMemset(host_data_.energy, 0, sizeof(real_t)));  // clear the energy
    CHECK_CUDA(cudaMemset(
        host_data_.forces, 0,
        host_data_.num_atoms * 3 * sizeof(real_t)));  // clear the forces
    CHECK_CUDA(cudaMemset(host_data_.stress, 0,
                          9 * sizeof(real_t)));  // clear the stress
    CHECK_CUDA(
        cudaMemcpy(device_data_, &host_data_, sizeof(device_data_t),
                   cudaMemcpyHostToDevice));  // copy the host data to device
    CHECK_CUDA(cudaDeviceSynchronize());      // synchronize the device
}  // clear the device buffer