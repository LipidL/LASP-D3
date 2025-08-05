#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// constant values for d3 system
float c6ab_ref[NUM_ELEMENTS][NUM_ELEMENTS][NUM_REF_C6][NUM_REF_C6]
              [NUM_C6AB_ENTRIES];  // table of reference C6 values, -1 means the
                                   // entry is invalid
float r0ab[NUM_ELEMENTS]
          [NUM_ELEMENTS];  // table of reference cutoff radii between elements
float rcov[NUM_ELEMENTS];  // table of covalent radii of each element
float r2r4[NUM_ELEMENTS];  // table of r2/r4 values of each element, needed for
                           // computing C8 from C6.

/**
 * @brief function to initialize the parameters
 * @note this function reads data from params.bin file and initializes the
 * c6ab_ref, r0ab, rcov, and r2r4 arrays
 */
inline void init_params() {
    // construct c6ab_ref, r0ab, rcov and r2r4 arrays
    FILE* file = fopen("params.bin", "rb");
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
        fprintf(stderr,
                "Error: unexpected number of dimensions in params.bin\n");
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

    fread(c6ab_ref, sizeof(real_t),
          NUM_ELEMENTS * NUM_ELEMENTS * NUM_REF_C6 * NUM_REF_C6 *
              NUM_C6AB_ENTRIES,
          file);

    // read r0ab
    fread(&num_dimensions, sizeof(uint32_t), 1, file);
    if (num_dimensions != 2) {
        fprintf(stderr,
                "Error: unexpected number of dimensions in params.bin\n");
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
        fprintf(stderr,
                "Error: unexpected number of dimensions in params.bin\n");
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
        fprintf(stderr,
                "Error: unexpected number of dimensions in params.bin\n");
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

#endif  // CONSTANTS_H