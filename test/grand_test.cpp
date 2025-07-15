#include "grand_test.h"
#include "../src/d3.h"

#include <gtest/gtest.h>

class D3Test : public testing::TestWithParam<TestConfig> {
  protected:
    D3Test() {
        TestConfig config = GetParam();
        max_length = config.elements.size();
        cutoff_radius = 40.0f; // Example value, adjust as needed
        coordination_number_cutoff = 40.0f; // Example value, adjust as needed
        elements = (uint16_t *)malloc(config.elements.size() * sizeof(uint16_t));
        atoms = (real_t *)malloc(max_length * 3 * sizeof(real_t));
        for (size_t i = 0; i < max_length; ++i) {
            elements[i] = config.elements[i];
            atoms[i * 3 + 0] = config.atoms[i][0];
            atoms[i * 3 + 1] = config.atoms[i][1];
            atoms[i * 3 + 2] = config.atoms[i][2];
        }
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                cell[i][j] = config.cell[i][j];
            }
        }
        force = (real_t *)malloc(max_length * 3 * sizeof(real_t)); // 3 components per atom
        assert(force != nullptr);
        for (size_t i = 0; i < max_length * 3; ++i) {
            force[i] = 0.0f; // Initialize force to zero
        }
    }

    ~D3Test() {
        free(elements);
        free(atoms);
        free(force);
    }

    uint16_t *elements;
    real_t energy = 0.0f;
    real_t *atoms;
    real_t *force;
    real_t stress[9] = {0.0f}; // 3x3 stress tensor
    real_t cell[3][3];
    uint64_t max_length;
    real_t cutoff_radius;
    real_t coordination_number_cutoff;
};

TEST_P(D3Test, ResultMatch) {
    real_t compute_energy = 0.0f;
    real_t *compute_force = (real_t *)malloc(max_length * 3 * sizeof(real_t)); // 3 components per atom
    real_t compute_stress[9] = {0.0f}; // 3x3 stress tensor
    ASSERT_NE(compute_force, nullptr) << "Failed to allocate memory for compute force";
    compute_dispersion_energy((real_t (*)[3])atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, 10000, &compute_energy, compute_force, compute_stress);
    // Conversion factors
    const real_t eV_to_hartree = 1.0 / 27.2114;
    const real_t angstrom_to_bohr = 1.0 / 0.529177;
    const real_t eV_per_ang_to_hartree_per_bohr = eV_to_hartree * angstrom_to_bohr;
    const real_t eV_per_ang3_to_hartree_per_bohr3 = eV_per_ang_to_hartree_per_bohr * angstrom_to_bohr * angstrom_to_bohr;

    // Convert computed values from eV to hartree units
    real_t converted_energy = compute_energy * eV_to_hartree;
    EXPECT_NEAR(converted_energy, energy, 1e-5);
    
    for (size_t i = 0; i < max_length * 3; ++i) {
        real_t converted_force = compute_force[i] * eV_per_ang_to_hartree_per_bohr;
        EXPECT_NEAR(converted_force, force[i], 1e-5) << "Force mismatch at index " << i;
    }
    
    for (size_t i = 0; i < 9; ++i) {
        real_t converted_stress = compute_stress[i] * eV_per_ang3_to_hartree_per_bohr3;
        EXPECT_NEAR(converted_stress, stress[i], 1e-5)
            << "Stress mismatch at index " << i;
    }
    
    free(compute_force);
}

INSTANTIATE_TEST_SUITE_P(
    FullTest,
    D3Test,
    testing::ValuesIn(test_configs),
    [](const testing::TestParamInfo<TestConfig>& info) {
        return info.param.test_name;
    }
);