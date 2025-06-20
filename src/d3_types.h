#ifndef D3_TYPES_H
#define D3_TYPES_H

#include <stdint.h>

typedef float real_t;

typedef struct atom {
    uint16_t element; // element type of the atom
    real_t x, y, z; // coordinates in Cartesian space
} atom_t;

typedef struct neighbor {
    uint64_t index; // index of the neighbor atom
    atom_t atom; // atom data of the neighbor atom
    real_t distance; // distance to the neighbor atom
    real_t dCN_dr; // dCN_ab/dr_ab * 1/r for the neighbor atom
} neighbor_t;

typedef struct device_data {
    uint64_t num_atoms; // number of atoms in the system
    uint64_t num_elements; // number of unique elements in the system
    // uint64_t *unique_elements; // array of unique elements in the system, length: num_elements
    uint64_t *atom_types; // array of atom types, length: num_atoms. the entries is not the atomic number, but the index of the corresponding entry in constants.
    atom_t *atoms; // array of atom data
    real_t *c6_ab_ref; // size: num_elements*num_elements*NUM_REF_C6*NUM_REF_C6*NUM_C6AB_ENTRIES
    uint64_t c6_stride_1, c6_stride_2, c6_stride_3, c6_stride_4; // strides for c6ab array
    real_t *r0ab; // size: num_elements*num_elements
    real_t *rcov; // size: num_elements
    real_t *r2r4;// size: num_elements
    real_t cell[3][3]; // cell matrix, specify the three vectors of the cell
    uint64_t max_cell_bias[3]; // the maximum bias of the cell in each direction, this must be an odd number (because of symmetry)
    neighbor_t *neighbors; // array of neighbors, size: num_atoms * max_neigbors
    neighbor_t *CN_neighbors; //array of neighbors within CN cutoff, size: num_atoms * max_neighbors
    real_t coordination_number_cutoff; // the cutof radius for CN computation
    real_t cutoff; // the cutoff radius for the dispersion energy calculation
    /* some intermediate variables, not initialized but used during computation*/
    real_t *coordination_numbers; // array of coordination numbers, length: num_atoms.
    uint64_t *num_neighbors; // array of maximum number of neighbors for each atom, length: num_atoms.
    uint64_t *num_CN_neighbors;
    uint64_t max_neighbors;
    uint16_t status; // status of the calculation process, 0:normal, 0b01: neighbor list overflow detected
    real_t *dE_dCN; // dE/dCN for each atom, length: num_atoms.
    real_t *energy; // energy of the system, length: 1
    real_t *forces; // forces on each atom, length: 3*num_atoms.
    real_t *stress; // stress tensor, length: 9
} device_data_t;

#endif // D3_TYPES_H