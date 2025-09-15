#ifndef D3_H
#define D3_H

#include <stdint.h>

#include "d3_types.h"

/* define type for d3 handle */
typedef void D3Handle_t;

/* status code for compute status */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief the function is used to compute the dispersion energy of the system
 * using the D3 potential.
 *
 * @param atoms the array of atoms' coordination in the system. The array is of size num_atoms * 3.
 * @param elements the array of elements in the system. The array is of size num_atoms.
 * @param num_atoms number of atoms in the system.
 * @param cell cell matrix.
 * @param cutoff_radius cutoff radius when calculating d3 energy and force.
 * @param CN_cutoff_radius cutoff radius when calculating coordination number.
 * @param damping_type type of damping function.
 * @param functional_type type of functional to use.
 * @param energy pointer to energy result.
 * @param force array of force result. size: 3 * num_atoms.
 * @param stress array of stress result. size: 9.
 * 
 * @note the function is NOT thread safe, and should be called from a single
 * thread.
 */
void compute_dispersion_energy(
    real_t atoms[][3],
    uint16_t* elements,
    uint64_t num_atoms,
    real_t cell[3][3],
    real_t cutoff_radius,
    real_t CN_cutoff_radius,
    DampingType damping_type,
    FunctionalType functional_type,
    real_t* energy,
    real_t* force,
    real_t* stress
);

/**
 * @brief this function is used to init a handle for d3 energy/force/stress calculation.
 * @param elements the array of elements in the system. The array is of size max_length.
 * @param max_length maximum number of atoms that can be set in the handle.
 * @param cutoff_radius cutoff radius when calculating d3 energy and force.
 * @param coordination_number_cutoff cutoff radius when calculating coordination number.
 * @param damping_type type of damping function.
 * @param functional_type type of functional to use.
 *
 * @return a handle to the D3 potential.
 *
 * @note if you need to call calculate d3 for multiple times where the
 * structures are similar in element composition, you would better use this
 * handle.
 * this function initializes the handle in heap area, so you need to free
 * it after use.
 * if the element composition might vary during the simulation, you need to
 * specify all the possible elements in the `elements` parameter
 */
D3Handle_t* init_d3_handle(
    uint16_t* elements, 
    uint64_t num_elements,
    uint64_t max_length,
    real_t cutoff_radius,
    real_t coordination_number_cutoff,
    DampingType damping_type,
    FunctionalType functional_type
);
/**
 * @brief this function is used to set the coordinates and elements of the atoms
 * in the system.
 * 
 * @param handle the handle to the D3 potential.
 * @param coords the array of atoms' coordinates in the system. size: 3 * length
 * @param elements the array of elements in the system. size: length
 * @param length number of atoms to be set.
 *
 * @note the coordinates and elements should be in the same order as the atoms
 * in the system.
 * the number of atoms should not exceed the maximum number of atoms
 * specified in the init_d3_handle function, or the system will crash.
 */
void set_atoms(
    D3Handle_t* handle, 
    real_t* coords, 
    uint16_t* elements,
    uint64_t length
);

/**
 * @brief this function is used to set the cell matrix of the system.
 *
 * @param handle the handle to the D3 potential.
 * @param cell the cell matrix of the system. size: 3 * 3
 */

void set_cell(D3Handle_t* handle, real_t cell[3][3]);
/**
 * @brief this function is used to free the handle after use.
 *
 * @param handle the handle to be freed.
 */

void free_d3_handle(D3Handle_t* handle);
/**
 * @brief this function is used to clear the intermediate data in the handle.
 *
 * @param handle the handle to the D3 potential.
 *
 * @note you need to call this function before reusing the handle.
 */

void clear_d3_handle(D3Handle_t* handle);
/**
 * @brief this function is used to compute the dispersion energy of the system
 * using the D3 potential without return value.
 * @param handle the handle to the D3 potential.
 * @param energy pointer to the energy value to be computed.
 * @param force pointer to the force values to be computed.
 * @param stress pointer to the stress values to be computed.
 */
void compute_dispersion_energy_from_handle(
    D3Handle_t* handle, 
    real_t* energy,
    real_t* force, 
    real_t* stress
);

/**
 * @brief this function is used to compute the dispersion energy of the system
 * using the D3 potential.
 * @param handle the handle to the D3 potential.
 * @param energy pointer to the energy value to be computed.
 * @param force pointer to the force values to be computed.
 * @param stress pointer to the stress values to be computed.
 * 
 * @return uint16_t status code indicating the result of the computation.
 */
uint16_t compute_dispersion_energy_from_handle_status(
    D3Handle_t* handle,
    real_t* energy,
    real_t* force,
    real_t* stress
);

#ifdef __cplusplus
}
#endif

#endif  // D3_H