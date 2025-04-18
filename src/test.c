#include <stdio.h>
#include "d3.h"

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
    // initialize parameters
    init_params();
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
    real_t *result = compute_dispersion_energy(atoms, 10, cell, cutoff_radius, CN_cutoff_radius);
    printf("energy: %f\n", result[0]);
    for (int i = 1; i <= 10; ++i) {
        real_t force_x = result[0 + i * 3];
        real_t force_y = result[1 + i * 3];
        real_t force_z = result[2 + i * 3];
        printf("force[%d]: %f %f %f\n", i, force_x, force_y, force_z);
    }
    return 0;
}