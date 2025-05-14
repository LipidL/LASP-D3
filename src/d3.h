#ifndef D3_H
#define D3_H

#include <stdint.h>

#ifndef real_t
typedef float real_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

void init_params();

// Main function to compute dispersion energy
void compute_dispersion_energy(
    real_t atoms[][3], // array of atom positions
    uint16_t *elements, // array of atomic numbers
    uint64_t num_atoms, // number of atoms
    real_t cell[3][3], // cell parameters
    real_t cutoff_radius, // cutoff radius for dispersion energy calculation
    real_t CN_cutoff_radius, // cutoff radius for coordination number calculation
    real_t *energy, // pointer to store the computed energy
    real_t *force, // pointer to store the computed forces
    real_t *stress // pointer to store the computed stress    
);

#ifdef __cplusplus
}
#endif

#endif // D3_H