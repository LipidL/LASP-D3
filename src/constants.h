#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

// d3 system parameters
#define NUM_ELEMENTS 95
#define NUM_REF_C6 5 // each element have at most 5 reference points for C6 computation
#define NUM_C6AB_ENTRIES 3 // first entry: C6; second entry: CN of first element; third entry: CN of second element

// constant values for d3 system
float c6ab_ref[NUM_ELEMENTS][NUM_ELEMENTS][NUM_REF_C6][NUM_REF_C6][NUM_C6AB_ENTRIES]; // table of reference C6 values, -1 means the entry is invalid
float r0ab[NUM_ELEMENTS][NUM_ELEMENTS]; // table of reference cutoff radii between elements
float rcov[NUM_ELEMENTS]; // table of covalent radii of each element
float r2r4[NUM_ELEMENTS]; // table of r2/r4 values of each element, needed for computing C8 from C6.

// type definitions
typedef float real_t; // type for real numbers

/**
 * @brief function to initialize the parameters
 * @note this function reads data from params.bin file and initializes the c6ab_ref, r0ab, rcov, and r2r4 arrays
 */
void init_params() {
    // construct c6ab_ref, r0ab, rcov and r2r4 arrays
    FILE *file = fopen("params.bin", "rb");
    if (!file) {
        fprintf(stderr, "Error: failed to open params.bin\n");
        exit(EXIT_FAILURE);
    }

    if (!c6ab_ref || !r0ab || !rcov || !r2r4) {
        fprintf(stderr, "Error: failed to allocate memory for constants\n");
        fclose(file);
        free(c6ab_ref);
        free(r0ab);
        free(rcov);
        free(r2r4);
        exit(EXIT_FAILURE);
    }

    // construct c6ab_ref
    uint32_t num_dimensions;
    fread(&num_dimensions, sizeof(uint32_t), 1, file);

    if (num_dimensions != 5) {
        fprintf(stderr, "Error: unexpected number of dimensions in params.bin\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    uint32_t dimensions[5];
    fread(dimensions, sizeof(uint32_t), 5, file);

    if (dimensions[0] != NUM_ELEMENTS || dimensions[1] != NUM_ELEMENTS || 
        dimensions[2] != NUM_REF_C6 || dimensions[3] != NUM_REF_C6 || 
        dimensions[4] != NUM_C6AB_ENTRIES) {
        fprintf(stderr, "Error: unexpected dimensions in params.bin\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    fread(c6ab_ref, sizeof(real_t), NUM_ELEMENTS * NUM_ELEMENTS * NUM_REF_C6 * NUM_REF_C6 * NUM_C6AB_ENTRIES, file);

    // read r0ab
    fread(&num_dimensions, sizeof(uint32_t), 1, file);
    if (num_dimensions != 2) {
        fprintf(stderr, "Error: unexpected number of dimensions in params.bin\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fread(dimensions, sizeof(uint32_t), 2, file);
    if (dimensions[0] != NUM_ELEMENTS || dimensions[1] != NUM_ELEMENTS) {
        fprintf(stderr, "Error: unexpected dimensions in params.bin\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    // read r0ab
    fread(r0ab, sizeof(real_t), NUM_ELEMENTS * NUM_ELEMENTS, file);

    // read rcov
    fread(&num_dimensions, sizeof(uint32_t), 1, file);
    if (num_dimensions != 1) {
        fprintf(stderr, "Error: unexpected number of dimensions in params.bin\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fread(dimensions, sizeof(uint32_t), 1, file);
    if (dimensions[0] != NUM_ELEMENTS) {
        fprintf(stderr, "Error: unexpected dimensions in params.bin\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    // read rcov
    fread(rcov, sizeof(real_t), NUM_ELEMENTS, file);

    // read r2r4
    fread(&num_dimensions, sizeof(uint32_t), 1, file);
    if (num_dimensions != 1) {
        fprintf(stderr, "Error: unexpected number of dimensions in params.bin\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fread(dimensions, sizeof(uint32_t), 1, file);
    if (dimensions[0] != NUM_ELEMENTS) {
        fprintf(stderr, "Error: unexpected dimensions in params.bin\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    // read r2r4
    fread(r2r4, sizeof(real_t), NUM_ELEMENTS, file);

    fclose(file);
}

typedef struct atom {
    size_t element; // element type of the atom
    real_t x, y, z; // coordinates in Cartesian space
} atom_t;

typedef struct result {
    real_t energy, fx, fy, fz;
} result_t;

// structure to hold the reference C6 values for a specific system
// only relevant elements are included to save memory
// the user must not construct a c6ab_ref_t object directly, but use the c6ab_ref_init function instead
// the user must also guarantee that the data is freed after use
typedef struct c6ab_ref {
    real_t *data; // size: num_elements * num_elements * NUM_REF_C6 * NUM_REF_C6 * NUM_C6AB_ENTRIES
    size_t num_elements;

    size_t stride1, stride2, stride3, stride4; // strides for each dimension

    // helper accessor methods
    __host__ __device__ inline real_t get(size_t i1, size_t i2, size_t i3, size_t i4, size_t i5) const {
        assert(i1 < num_elements); // i1 must be less than num_elements
        assert(i2 < num_elements); // i2 must be less than num_elements
        assert(i3 < NUM_REF_C6); // i3 must be less than NUM_REF_C6
        assert(i4 < NUM_REF_C6); // i4 must be less than NUM_REF_C6
        assert(i5 < NUM_C6AB_ENTRIES); // i5 must be less than NUM_C6AB_ENTRIES
        assert(data != nullptr); // data must not be null
        return data[i1 * stride1 + i2 * stride2 + i3 * stride3 + i4 * stride4 + i5];
    }
} c6ab_ref_t;

/**
 * @brief Initialize the c6ab_ref_t object on device.
 * 
 * @param num_elements Number of elements in the system.
 * @param atom_types Array of atomic numbers for the system.
 * @return c6ab_ref_t Pointer to initialized c6ab_ref_t object on device.
 * 
 * @note The user must also guarantee that atom_types is a valid array of size num_elements.
 * @note The data constructed is a device pointer and cannot be used at host side.
 */
__host__ inline c6ab_ref_t* c6ab_ref_init(size_t num_elements, size_t *atom_types) {
    c6ab_ref_t h_ref;
    real_t *h_data = (real_t *)malloc(num_elements * num_elements * NUM_REF_C6 * NUM_REF_C6 * NUM_C6AB_ENTRIES * sizeof(real_t));
    if (!h_data) {
        fprintf(stderr, "Error: failed to allocate memory for c6ab_ref_t\n");
        exit(EXIT_FAILURE);
    }
    h_ref.stride1 = num_elements * NUM_REF_C6 * NUM_REF_C6 * NUM_C6AB_ENTRIES;
    h_ref.stride2 = NUM_REF_C6 * NUM_REF_C6 * NUM_C6AB_ENTRIES;
    h_ref.stride3 = NUM_REF_C6 * NUM_C6AB_ENTRIES;
    h_ref.stride4 = NUM_C6AB_ENTRIES;
    h_ref.num_elements = num_elements;
    // the large loop is time consuming, but is expected to called only once at initialization
    for (size_t i = 0; i < num_elements; ++i) {
        for (size_t j = 0; j < num_elements; ++j) {
            size_t element_i = atom_types[i];
            size_t element_j = atom_types[j];
            for (size_t k = 0; k < NUM_REF_C6; ++k) {
                for (size_t l = 0; l < NUM_REF_C6; ++l) {
                    size_t index = i * h_ref.stride1 + j * h_ref.stride2 + k * h_ref.stride3 + l * h_ref.stride4;
                    h_data[index] = c6ab_ref[element_i][element_j][k][l][0]; // C6 value
                    h_data[index + 1] = c6ab_ref[element_i][element_j][k][l][1]; // CN of first element
                    h_data[index + 2] = c6ab_ref[element_i][element_j][k][l][2]; // CN of second element
                }
            }
        }
    }
    // allocate memory for data on device
    real_t *d_data;
    cudaMalloc((void **)&d_data, num_elements * num_elements * NUM_REF_C6 * NUM_REF_C6 * NUM_C6AB_ENTRIES * sizeof(real_t));
    // copy data to device
    cudaMemcpy(d_data, h_data, num_elements * num_elements * NUM_REF_C6 * NUM_REF_C6 * NUM_C6AB_ENTRIES * sizeof(real_t), cudaMemcpyHostToDevice);
    // constrct a c6ab_ref_t object on device
    h_ref.data = d_data; // point to device data
    c6ab_ref_t *d_ref;
    cudaMalloc((void **)&d_ref, sizeof(c6ab_ref_t));
    // copy the c6ab_ref_t object to device
    cudaMemcpy(d_ref, &h_ref, sizeof(c6ab_ref_t), cudaMemcpyHostToDevice);
    // free the host data
    free(h_data);
    return d_ref;
}

// structure to hold all the constants for a specific system
// only relevant elements are included to save memory
// the user must not construct a d3_constant_t object directly, but use the d3_constant_init function instead
// the user must also guarantee that the data is freed after use
typedef struct d3_constant {
    size_t num_elements; // number of elements in the system
    size_t *atom_types; // array of atom types for the system
    real_t **r0ab; // table of reference cutoff radii between elements
    real_t *rcov; // table of covalent radii of each element
    real_t *r2r4; // table of r2/r4 values of each element, needed for computing C8 from C6.
    c6ab_ref_t *c6ab_ref; // reference C6 values for the system
} d3_constant_t;

/**
 * @brief Initialize the d3_constant_t object on device.
 * 
 * @param num_elements Number of elements in the system.
 * @param atom_types Array of atom types for the system.
 * @return d3_constant_t Initialized d3_constant_t object.
 * 
 * @note The user must guarantee that the data is freed after use.
 * @note The user must also guarantee that atom_types is a valid array of size num_elements.
 */
__host__ inline d3_constant_t* d3_constant_init(size_t num_elements, size_t *atom_types) {
    d3_constant_t constants;
    constants.num_elements = num_elements;
    printf("num_elements: %zu\n", num_elements);
    size_t *h_atom_types = (size_t *)malloc(num_elements * sizeof(size_t));
    for (size_t i = 0; i < num_elements; ++i) {
        h_atom_types[i] = atom_types[i];
    }    
    // initiate atom_types array on device
    size_t *d_atom_types;
    cudaMalloc((void **)&d_atom_types, num_elements * sizeof(size_t));
    cudaMemcpy(d_atom_types, h_atom_types, num_elements * sizeof(size_t), cudaMemcpyHostToDevice);
    // free the host atom_types array
    free(h_atom_types);
    constants.atom_types = d_atom_types; // point to device data

    real_t **h_r0ab = (real_t **)malloc(num_elements * sizeof(real_t *));
    for (size_t i = 0; i < num_elements; ++i) {
        h_r0ab[i] = (real_t *)malloc(num_elements * sizeof(real_t));
        for (size_t j = 0; j < num_elements; ++j) {
            h_r0ab[i][j] = r0ab[atom_types[i]][atom_types[j]];
        }
    }
    // initialize r0ab array on device
    real_t **d_r0ab;
    cudaMalloc((void **)&d_r0ab, num_elements * sizeof(real_t *));
    for (size_t i = 0; i < num_elements; ++i) {
        real_t *d_r0ab_i;
        cudaMalloc((void **)&d_r0ab_i, num_elements * sizeof(real_t));
        cudaMemcpy(d_r0ab_i, h_r0ab[i], num_elements * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_r0ab[i]), &d_r0ab_i, sizeof(real_t *), cudaMemcpyHostToDevice);
        free(h_r0ab[i]); // free the host r0ab[i] array
    }
    free(h_r0ab);
    constants.r0ab = d_r0ab; // point to device data

    real_t *h_rcov = (real_t *)malloc(num_elements * sizeof(real_t));
    for (size_t i = 0; i < num_elements; ++i) {
        h_rcov[i] = rcov[atom_types[i]];
    }
    // initialize rcov array on device
    real_t *d_rcov;
    cudaMalloc((void **)&d_rcov, num_elements * sizeof(real_t));
    cudaMemcpy(d_rcov, h_rcov, num_elements * sizeof(real_t), cudaMemcpyHostToDevice);
    // free the host rcov array
    free(h_rcov);
    constants.rcov = d_rcov; // point to device data
    real_t *h_r2r4 = (real_t *)malloc(num_elements * sizeof(real_t));
    for (size_t i = 0; i < num_elements; ++i) {
        printf("r2r4 for %zu: %f\n", atom_types[i], r2r4[atom_types[i]]);
        h_r2r4[i] = r2r4[atom_types[i]];
    }
    // initialize r2r4 array on device
    real_t *d_r2r4;
    cudaMalloc((void **)&d_r2r4, num_elements * sizeof(real_t));
    cudaMemcpy(d_r2r4, h_r2r4, num_elements * sizeof(real_t), cudaMemcpyHostToDevice);
    // free the host r2r4 array
    free(h_r2r4);
    constants.r2r4 = d_r2r4; // point to device data
    // initialize c6ab_ref_t object
    constants.c6ab_ref = c6ab_ref_init(num_elements, atom_types);
    // construct a d3_constant_t object on device
    d3_constant_t *d_constants;
    cudaMalloc((void **)&d_constants, sizeof(d3_constant_t));
    // copy the d3_constant_t object to device
    cudaMemcpy(d_constants, &constants, sizeof(d3_constant_t), cudaMemcpyHostToDevice);
    return d_constants; // return the device pointer to the d3_constant_t object
}
