#include <stdio.h>
#include <stdlib.h>
#include "d3.h"

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
    printf("Computing dispersion energy for %zu atoms...\n", sizeof(atoms)/sizeof(atoms[0]));
    // initialize parameters
    init_params();
    printf("Computing dispersion energy...\n");
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
    printf("energy: %f eV\n", energy); // convert to eV
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