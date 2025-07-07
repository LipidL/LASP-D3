#include "../src/d3.h"
#include <gtest/gtest.h>

class D3TestSmall : public testing::Test {
    protected:
    D3TestSmall() {
        uint16_t tmp_elements[10] = {6, 6, 6, 8, 1, 1, 1, 1, 1, 1}; // C, O, H
        for (size_t i = 0; i < 10; ++i) {
            elements[i] = tmp_elements[i];
        }

        max_length = 10;
        cutoff_radius = 40.0f;
        coordination_number_cutoff = 40.0f;
        max_neighbors = 10000;

        real_t tmp_atoms[10][3] = {
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
        for (size_t i = 0; i < 10; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                atoms[i][j] = tmp_atoms[i][j];
            }
        }
        
        for (size_t i = 0; i < 3; ++i) {
            cell[i][i] = 20.0f; // Simple cubic cell
        }

    }

    uint16_t elements[10];
    uint64_t max_length;
    real_t cutoff_radius;
    real_t coordination_number_cutoff;
    uint64_t max_neighbors;
    real_t atoms[10][3];
    real_t cell[3][3];
};

TEST_F(D3TestSmall, BasicTest) {
    D3Handle_t *handle = init_d3_handle(elements, max_length, cutoff_radius, coordination_number_cutoff, max_neighbors);
    ASSERT_NE(handle, nullptr) << "Failed to initialize D3 handle";
    set_atoms(handle, (real_t *)atoms, elements, max_length);
    set_cell(handle, cell);
    clear_d3_handle(handle);
    free_d3_handle(handle);
}

TEST_F(D3TestSmall, EnergyNegativeTest) {
    real_t energy = 0.0f;
    real_t force[30] = {0}; // 10 atoms * 3 components
    real_t stress[9] = {0}; // 3x3 stress tensor
    compute_dispersion_energy((real_t (*)[3])atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, max_neighbors, &energy, force, stress);
    ASSERT_LT(energy, 0.0f) << "Expected negative energy, got " << energy;
}