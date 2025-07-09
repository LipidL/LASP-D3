#include "../src/d3.h"
#include <gtest/gtest.h>
#include <vector>
#include <assert.h>

struct TestConfig {
    std::string test_name;
    std::vector<uint16_t> elements;
    uint64_t max_length;
    real_t cutoff_radius;
    real_t coordination_number_cutoff;
    uint64_t max_neighbors;
    std::vector<std::array<real_t, 3>> atoms;
    real_t cell[3][3];
};

void PrintTo(const TestConfig &config, std::ostream *os) {
    *os << "TestConfig: " << config.test_name << ", max_length: " << config.max_length
        << ", cutoff_radius: " << config.cutoff_radius
        << ", coordination_number_cutoff: " << config.coordination_number_cutoff
        << ", max_neighbors: " << config.max_neighbors;
} // PrintTo

class D3Test : public testing::TestWithParam<TestConfig> {
    protected:
    D3Test() {
        TestConfig config = GetParam();
        elements = (uint16_t *)malloc(config.max_length * sizeof(uint16_t));
        atoms = (real_t *)malloc(config.max_length * 3 * sizeof(real_t));
        for (size_t i = 0; i < config.max_length; ++i) {
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
        force = (real_t *)malloc(config.max_length * 3 * sizeof(real_t)); // 3 components per atom
        assert(force != nullptr);
        for (size_t i = 0; i < config.max_length * 3; ++i) {
            force[i] = 0.0f; // Initialize force to zero
        }
        max_length = config.max_length;
        cutoff_radius = config.cutoff_radius;
        coordination_number_cutoff = config.coordination_number_cutoff;
        max_neighbors = config.max_neighbors;
    } // D3Test

    ~D3Test() {
        free(elements);
        free(atoms);
        free(force);
    } // ~D3Test

    real_t *atoms;
    uint16_t *elements;
    uint64_t max_length;
    real_t cutoff_radius;
    real_t coordination_number_cutoff;
    uint64_t max_neighbors;
    real_t cell[3][3];
    real_t energy;
    real_t *force;
    real_t stress[9];

};

TEST_P(D3Test, HandleOperations) {
    D3Handle_t *handle = init_d3_handle(elements, max_length, cutoff_radius, coordination_number_cutoff, max_neighbors);
    EXPECT_NE(handle, nullptr) << "Failed to initialize D3 handle";
    set_atoms(handle, atoms, elements, max_length);
    set_cell(handle, cell);
    clear_d3_handle(handle);
    free_d3_handle(handle);
} // HandleOperations

TEST_P(D3Test, EnergyNegative) {
    compute_dispersion_energy((real_t (*)[3])atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, max_neighbors, &energy, force, stress);
    EXPECT_LT(energy, 0.0f) << "Expected negative energy, got " << energy;
} // EnergyNegative

TEST_P(D3Test, ResultStable) {
    // Run the dispersion energy calculation multiple times and verify stability
    const int num_runs = 64; // Number of runs to check stability

    real_t energy_tolerance = 1e-5; // Tolerance for energy comparison
    real_t force_tolerance = 1e-5; // Tolerance for force comparison
    real_t stress_tolerance = 1e-5; // Tolerance for stress comparison
    
    // Perform initial run
    compute_dispersion_energy((real_t (*)[3])atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, max_neighbors, &energy, force, stress);

    // Perform multiple runs
    for (int run = 0; run < num_runs; ++run) {
        real_t current_energy = 0.0f;
        real_t *current_force = (real_t *)malloc(max_length * 3 * sizeof(real_t)); // 3 components per atom
        EXPECT_NE(current_force, nullptr) << "Failed to allocate memory for current force";
        for (size_t i = 0; i < max_length * 3; ++i) {
            current_force[i] = 0.0f; // Initialize force to zero
        }
        real_t current_stress[9] = {0}; // 3x3 stress tensor
        compute_dispersion_energy((real_t (*)[3])atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, max_neighbors, &current_energy, current_force, current_stress);

        // Check stability of results
        EXPECT_NEAR(current_energy, energy, energy_tolerance) 
            << "Energy not stable between initial and run " << run;

        for (size_t i = 0; i < max_length * 3; ++i) {
            EXPECT_NEAR(current_force[i], force[i], force_tolerance)
                << "Force not stable between initial and run " << run << " at index " << i;
        }

        for (size_t i = 0; i < 9; ++i) {
            EXPECT_NEAR(current_stress[i], stress[i], stress_tolerance)
                << "Stress not stable between initial and run " << run << " at index " << i;
        }
        // Clean up current force
        free(current_force);
    }
} // ResultStable

TEST_P(D3Test, TestSumToZero) {
    real_t tolerance = 1e-5; // Tolerance for sum to zero check
    compute_dispersion_energy((real_t (*)[3])atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, max_neighbors, &energy, force, stress);
    real_t force_sum[3] = {0.0f};
    // Perform sum over forces
    for (size_t i = 0; i < max_length; ++i) {
        force_sum[0] += force[i * 3 + 0];
        force_sum[1] += force[i * 3 + 1];
        force_sum[2] += force[i * 3 + 2];
    }
    // Check if the sum of forces is close to zero
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(force_sum[i], 0.0f, tolerance) 
            << "Force sum not zero for component " << i << ", got " << force_sum[i];
    }
}

TEST_P(D3Test, StressSymmetry) {
    real_t tolerance = 1e-5; // Tolerance for stress symmetry check
    compute_dispersion_energy((real_t (*)[3])atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, max_neighbors, &energy, force, stress);
    
    // Check stress symmetry: stress[i][j] should equal stress[j][i]
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(stress[i * 3 + j], stress[j * 3 + i], tolerance)
                << "Stress symmetry broken at indices (" << i << ", " << j << ")";
        }
    }
}

TEST_P(D3Test, NumericForceMatch) {
    // Perform the dispersion energy calculation
    compute_dispersion_energy((real_t (*)[3])atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, max_neighbors, &energy, force, stress);

    // Prepare atoms for numerical differentiation
    real_t *tmp_atoms = (real_t *)malloc(max_length * 3 * sizeof(real_t));
    EXPECT_NE(tmp_atoms, nullptr) << "Failed to allocate memory for temporary atoms";
    for (size_t i = 0; i < max_length * 3; ++i) {
        tmp_atoms[i] = atoms[i]; // Copy original atom positions
    }

    real_t tolerance = 1e-3f; // Tolerance for numerical differentiation
    real_t delta = 1e-4f; // Perturbation size for numerical differentiation (increased from 1e-5f)
    for(size_t atom_idx = 0; atom_idx < max_length; ++atom_idx) {
        for(size_t component = 0; component < 3; ++component) {
            // Prepare for numerical differentiation
            real_t *dummy_force = (real_t *)malloc(max_length * 3 * sizeof(real_t)); // Temporary force for numerical differentiation
            EXPECT_NE(dummy_force, nullptr) << "Failed to allocate memory for dummy force";
            for (size_t i = 0; i < max_length * 3; ++i) {
                dummy_force[i] = 0.0f;
            }
            real_t dummy_stress[9] = {0}; // 3x3 stress tensor
            
            // Reset tmp_atoms to original positions
            for (size_t i = 0; i < max_length * 3; ++i) {
                tmp_atoms[i] = atoms[i];
            }
            
            // Forward displacement
            tmp_atoms[atom_idx * 3 + component] += delta; 
            real_t energy_plus = 0.0f;
            compute_dispersion_energy((real_t (*)[3])tmp_atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, max_neighbors, &energy_plus, dummy_force, dummy_stress);
            
            // Reset tmp_atoms to original positions
            for (size_t i = 0; i < max_length * 3; ++i) {
                tmp_atoms[i] = atoms[i];
            }
            
            // Backward displacement
            tmp_atoms[atom_idx * 3 + component] -= delta;
            real_t energy_minus = 0.0f;
            compute_dispersion_energy((real_t (*)[3])tmp_atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, max_neighbors, &energy_minus, dummy_force, dummy_stress);

            // Central difference formula for the derivative: f'(x) = (f(x + delta) - f(x - delta)) / (2 * delta)
            real_t numerical_force = -(energy_plus - energy_minus) / (2 * delta);
            // Check numeric force against computed force
            EXPECT_NEAR(numerical_force, force[atom_idx * 3 + component],
                        tolerance) << "Force mismatch for atom " << atom_idx
                                    << ", component " << component
                                    << ": expected " << force[atom_idx * 3 + component]
                                    << ", got " << numerical_force;
            // Clean up dummy force
            free(dummy_force);
        }
    }
    
    // Clean up tmp_atoms
    free(tmp_atoms);
}

TestConfig small_system = {
    "SmallSystem",
    {6, 6, 6, 8, 1, 1, 1, 1, 1, 1}, // C, O, H
    10, // max_length
    40.0f, // cutoff_radius
    40.0f, // coordination_number_cutoff
    10000, // max_neighbors
    {
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
    }, // atoms
    {
        {20.0f, 0.0f, 0.0f},
        {0.0f, 20.0f, 0.0f},
        {0.0f, 0.0f, 20.0f}
    } // cell
};

// Instantiate the test suite with all system configurations
INSTANTIATE_TEST_SUITE_P(
    SmokeTest,
    D3Test,
    ::testing::Values(small_system),
    [](const testing::TestParamInfo<TestConfig>& info) {
        return info.param.test_name; // Use the test name as the test case name
    }
);
