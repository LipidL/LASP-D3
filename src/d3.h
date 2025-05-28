#ifndef D3_H
#define D3_H

#include <stdint.h>

#ifndef real_t
typedef float real_t;
#endif

/* define type for d3 handle */
typedef void D3Handle_t;

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
    uint64_t max_neighbors, // maximum number of neighbors to consider
    real_t *energy, // pointer to store the computed energy
    real_t *force, // pointer to store the computed forces
    real_t *stress // pointer to store the computed stress    
);

D3Handle_t *init_d3_handle( 
    uint16_t *elements,
    uint64_t max_length, 
    real_t cutoff_radius,
    real_t coordination_number_cutoff,
    uint64_t max_neigbors
);

void set_atoms(D3Handle_t *handle, real_t *coords, uint16_t *elements, uint64_t length);

void set_cell(D3Handle_t *handle, real_t cell[3][3]);

void free_d3_handle(D3Handle_t *handle);

void clear_d3_handle(D3Handle_t *handle);

void compute_dispersion_energy_from_handle(
    D3Handle_t *handle,
    real_t *energy,
    real_t *force,
    real_t *stress
);

#ifdef __cplusplus
}
#endif

#endif // D3_H