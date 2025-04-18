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
real_t* compute_dispersion_energy(
    real_t atoms[][4], 
    uint64_t length, 
    real_t cell[3][3],
    real_t cutoff_radius,
    real_t coordination_number_cutoff);

#ifdef __cplusplus
}
#endif

#endif // D3_H