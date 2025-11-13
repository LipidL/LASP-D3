#ifndef D3_TYPES_H
#define D3_TYPES_H

#include <stdint.h>

typedef float real_t;

typedef struct atom {
    uint16_t element;  // element type of the atom
    uint64_t original_index; // original index of the atom in the input structure
    real_t x, y, z;    // coordinates in Cartesian space
} atom_t;

typedef enum {
    ZERO_DAMPING = 0,  // original DFT-D3 damping
    BJ_DAMPING = 1,    // Becke-Johnson damping
} DampingType;

typedef enum {
    PBE0 = 0,
    PBE = 1,
    B3LYP = 2,
    BLYP = 3,
    BP86 = 4,
    REVPBE = 5,
    CUSTOM = 99  // Special value for custom parameters
} FunctionalType;

typedef enum {
    ALL_ITERATE = 0, // all atoms iterate over all other atoms over all supercell images
    CELL_LIST = 1 // use cell list to accelerate the search for neighboring atoms
} WorkloadDistributionType;

typedef struct {
    real_t s6;
    real_t s8;
    real_t sr6; // SR_6 parameter for zero damping
    real_t sr8; // SR_8 parameter for zero damping
    real_t a1;  // a1 parameter for BJ damping
    real_t a2;  // a2 parameter for BJ damping
} functional_params_t;

static const functional_params_t FUNCTIONAL_PARAMS[] = {
    {1.0f, 0.722f, 1.217f, 1.0f, 0.40085597f, 5.02928789f},  // PBE0
    {1.0f, 0.777f, 1.277f, 1.0f, 0.38574991f, 4.80688534f},  // PBE
    {1.0f, 1.706f, 1.314f, 1.0f, 0.40868035f, 4.53807137f},  // B3LYP
    {1.0f, 2.022f, 1.243f, 1.0f, 0.44488865f, 4.09330090f},  // BLYP
    {1.0f, 1.838f, 1.221f, 1.0f, 0.43645861f, 4.92406854f},  // BP86
    {1.0f, 0.989f, 0.953f, 1.0f, 0.53634900f, 3.07261485f},  // REVPBE
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}   // CUSTOM (default)
};
typedef struct device_data {
    uint64_t num_atoms;     // number of atoms in the system
    uint64_t num_elements;  // number of unique elements in the system
    uint64_t* atom_types;  // array of atom types, size: num_atoms. the entries is not the atomic number, but the index of the corresponding entry in constants.
    atom_t* atoms;         // array of atom data, sorted so atoms in the same grid are together. size: num_atoms.
    real_t* c6_ab_ref;  // size: num_elements*num_elements*NUM_REF_C6*NUM_REF_C6*NUM_C6AB_ENTRIES
    uint64_t c6_stride_1, c6_stride_2, c6_stride_3, c6_stride_4;    // strides for c6ab array
    real_t* r0ab;       // size: num_elements*num_elements
    real_t* rcov;       // size: num_elements
    real_t* r2r4;       // size: num_elements
    real_t cell[3][3];  // cell matrix, specify the three vectors of the cell
    WorkloadDistributionType workload_distribution_type;  // the workload distribution type used for the calculation
    uint64_t max_cell_bias[3];  // the maximum bias of the cell in each direction, this must be an odd number (because of symmetry)
    uint64_t num_grid_cells[3];  // number of grid cells in each direction
    uint64_t *grid_start_indices;  // starting indices of each grid cell in the atoms array, size: num_grid_cells[0]*num_grid_cells[1]*num_grid_cells[2]
    real_t coordination_number_cutoff;  // the cutof radius for CN computation
    real_t cutoff;  // the cutoff radius for the dispersion energy calculation
    DampingType damping_type;  // the damping type used for the calculation
    FunctionalType functional_type;  // the functional type used for the
                                         // calculation
    functional_params_t functional_params;  // parameters for the functional
    // some intermediate variables, not initialized but used during computation
    real_t* coordination_numbers;  // array of coordination numbers, length:
                                   // num_atoms.
    uint16_t status;  // status of the calculation process, 0:normal, 0b01: neighbor list overflow detected
    real_t* dCN_dr;  // dCN/dr for each atom, length: num_atoms*3 (x, y and z).
    real_t* dE_dCN;   // dE/dCN for each atom, length: num_atoms.
    real_t* energy;   // energy of the system, length: num_atoms.
    real_t* forces;   // forces on each atom, length: 3*num_atoms.
    real_t* stress;   // stress tensor, length: 9
} device_data_t;

#endif  // D3_TYPES_H