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
    DampingType damping_type;
    FunctionalType functional_type;
    std::vector<std::array<real_t, 3>> atoms;
    real_t cell[3][3];
};

void PrintTo(const TestConfig &config, std::ostream *os) {
    *os << "TestConfig: " << config.test_name << ", max_length: " << config.max_length
        << ", cutoff_radius: " << config.cutoff_radius
        << ", coordination_number_cutoff: " << config.coordination_number_cutoff
        << ", damping_type: " << config.damping_type;
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
        damping_type = config.damping_type;
        functional_type = config.functional_type;
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
    DampingType damping_type;
    FunctionalType functional_type;
    real_t cell[3][3];
    real_t energy;
    real_t *force;
    real_t stress[9];

};

TEST_P(D3Test, HandleOperations) {
    D3Handle_t *handle = init_d3_handle(elements, max_length, max_length, cutoff_radius, coordination_number_cutoff, damping_type, functional_type);
    EXPECT_NE(handle, nullptr) << "Failed to initialize D3 handle";
    set_atoms(handle, atoms, elements, max_length);
    set_cell(handle, cell);
    clear_d3_handle(handle);
    free_d3_handle(handle);
} // HandleOperations

TEST_P(D3Test, EnergyNegative) {
    compute_dispersion_energy((real_t (*)[3])atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, damping_type, functional_type, &energy, force, stress);
    EXPECT_LT(energy, 0.0f) << "Expected negative energy, got " << energy;
} // EnergyNegative

TEST_P(D3Test, ResultStable) {
    // Run the dispersion energy calculation multiple times and verify stability
    const int num_runs = 64; // Number of runs to check stability

    real_t energy_tolerance = 1e-7; // Tolerance for energy comparison
    real_t force_tolerance = 1e-5; // Tolerance for force comparison
    real_t stress_tolerance = 1e-5; // Tolerance for stress comparison
    
    // Perform initial run
    compute_dispersion_energy((real_t (*)[3])atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, damping_type, functional_type, &energy, force, stress);

    // Perform multiple runs
    for (int run = 0; run < num_runs; ++run) {
        real_t current_energy = 0.0f;
        real_t *current_force = (real_t *)malloc(max_length * 3 * sizeof(real_t)); // 3 components per atom
        ASSERT_NE(current_force, nullptr) << "Failed to allocate memory for current force";
        for (size_t i = 0; i < max_length * 3; ++i) {
            current_force[i] = 0.0f; // Initialize force to zero
        }
        real_t current_stress[9] = {0}; // 3x3 stress tensor
        compute_dispersion_energy((real_t (*)[3])atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, damping_type, functional_type, &current_energy, current_force, current_stress);

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

TEST_P(D3Test, ForceSumToZero) {
    real_t tolerance = 1e-5; // Tolerance for sum to zero check
    compute_dispersion_energy((real_t (*)[3])atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, damping_type, functional_type, &energy, force, stress);
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
    compute_dispersion_energy((real_t (*)[3])atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, damping_type, functional_type, &energy, force, stress);
    
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
    compute_dispersion_energy((real_t (*)[3])atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, damping_type, functional_type, &energy, force, stress);

    // Prepare atoms for numerical differentiation
    real_t *tmp_atoms = (real_t *)malloc(max_length * 3 * sizeof(real_t));
    ASSERT_NE(tmp_atoms, nullptr) << "Failed to allocate memory for temporary atoms";
    for (size_t i = 0; i < max_length * 3; ++i) {
        tmp_atoms[i] = atoms[i]; // Copy original atom positions
    }

    real_t tolerance = 1e-3f; // Tolerance, relatively large because numerical differentiation can bring significant errors
    real_t delta = 1e-3f; // Perturbation size, relatively large to cover instability in energy computation
    for(size_t atom_idx = 0; atom_idx < max_length; ++atom_idx) {
        for(size_t component = 0; component < 3; ++component) {
            // Prepare for numerical differentiation
            real_t *dummy_force = (real_t *)malloc(max_length * 3 * sizeof(real_t)); // Temporary force for numerical differentiation
            ASSERT_NE(dummy_force, nullptr) << "Failed to allocate memory for dummy force";
            for (size_t i = 0; i < max_length * 3; ++i) {
                dummy_force[i] = 0.0f;
            }
            real_t dummy_stress[9] = {0}; // 3x3 stress tensor
            
            // Reset tmp_atoms to original positions
            for (size_t i = 0; i < max_length * 3; ++i) {
                tmp_atoms[i] = atoms[i];
            }
            
            // Forward perturbation
            tmp_atoms[atom_idx * 3 + component] += delta; 
            real_t energy_plus = 0.0f;
            compute_dispersion_energy((real_t (*)[3])tmp_atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, damping_type, functional_type, &energy_plus, dummy_force, dummy_stress);
            
            // Reset tmp_atoms to original positions
            for (size_t i = 0; i < max_length * 3; ++i) {
                tmp_atoms[i] = atoms[i];
            }
            
            // Backward perturbation
            tmp_atoms[atom_idx * 3 + component] -= delta;
            real_t energy_minus = 0.0f;
            compute_dispersion_energy((real_t (*)[3])tmp_atoms, elements, max_length, cell, cutoff_radius, coordination_number_cutoff, damping_type, functional_type, &energy_minus, dummy_force, dummy_stress);

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

TEST_P(D3Test, SupercellConsistency) {
    // Define supercell dimensions (nx, ny, nz)
    int supercell_dims[3] = {2, 2, 2};
    int supercell_volume = supercell_dims[0] * supercell_dims[1] * supercell_dims[2];
    
    // Calculate energy, force, and stress for the original system
    real_t original_energy = 0.0f;
    real_t *original_force = (real_t *)malloc(max_length * 3 * sizeof(real_t));
    ASSERT_NE(original_force, nullptr);
    for (size_t i = 0; i < max_length * 3; ++i) {
        original_force[i] = 0.0f;
    }
    real_t original_stress[9] = {0.0f};
    
    compute_dispersion_energy(
        (real_t (*)[3])atoms, 
        elements, 
        max_length, 
        cell, 
        cutoff_radius, 
        coordination_number_cutoff, 
        damping_type, 
        functional_type, 
        &original_energy, 
        original_force, 
        original_stress
    );
    
    // Create a supercell with dimensions defined by supercell_dims
    uint64_t supercell_length = max_length * supercell_volume;
    uint16_t *supercell_elements = (uint16_t *)malloc(supercell_length * sizeof(uint16_t));
    real_t *supercell_atoms = (real_t *)malloc(supercell_length * 3 * sizeof(real_t));
    real_t *supercell_force = (real_t *)malloc(supercell_length * 3 * sizeof(real_t));
    real_t supercell_cell[3][3];
    real_t supercell_energy = 0.0f;
    real_t supercell_stress[9] = {0.0f};
    
    ASSERT_NE(supercell_elements, nullptr);
    ASSERT_NE(supercell_atoms, nullptr);
    ASSERT_NE(supercell_force, nullptr);
    
    // Copy original atoms and create duplicated atoms with shifted positions
    size_t atom_index = 0;
    for (int x = 0; x < supercell_dims[0]; ++x) {
        for (int y = 0; y < supercell_dims[1]; ++y) {
            for (int z = 0; z < supercell_dims[2]; ++z) {
                for (size_t i = 0; i < max_length; ++i) {
                    supercell_elements[atom_index] = elements[i];
                    // Position = original position + x*cell[0] + y*cell[1] + z*cell[2]
                    supercell_atoms[atom_index * 3 + 0] = atoms[i * 3 + 0] + 
                                                          x * cell[0][0] + 
                                                          y * cell[1][0] + 
                                                          z * cell[2][0];
                    supercell_atoms[atom_index * 3 + 1] = atoms[i * 3 + 1] + 
                                                          x * cell[0][1] + 
                                                          y * cell[1][1] + 
                                                          z * cell[2][1];
                    supercell_atoms[atom_index * 3 + 2] = atoms[i * 3 + 2] + 
                                                          x * cell[0][2] + 
                                                          y * cell[1][2] + 
                                                          z * cell[2][2];
                    atom_index++;
                }
            }
        }
    }
    
    // Initialize forces to zero
    for (size_t i = 0; i < supercell_length * 3; ++i) {
        supercell_force[i] = 0.0f;
    }
    
    // Create the supercell lattice vectors
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            supercell_cell[i][j] = cell[i][j] * supercell_dims[i];
        }
    }
    
    // Calculate energy, force, and stress for the supercell
    compute_dispersion_energy(
        (real_t (*)[3])supercell_atoms, 
        supercell_elements, 
        supercell_length, 
        supercell_cell, 
        cutoff_radius, 
        coordination_number_cutoff, 
        damping_type, 
        functional_type, 
        &supercell_energy, 
        supercell_force, 
        supercell_stress
    );
    
    // Check if energy scales with system size
    // For a NxNxN supercell, we expect energy to be roughly N^3 times the original
    real_t expected_energy_ratio = static_cast<real_t>(supercell_volume);
    real_t energy_ratio = supercell_energy / original_energy;
    EXPECT_NEAR(energy_ratio, expected_energy_ratio, 0.2f * expected_energy_ratio) 
        << "Energy doesn't scale properly with supercell size. "
        << "Original: " << original_energy << ", Supercell: " << supercell_energy;
    
    // Check if forces on equivalent atoms are similar
    // Compare atoms at the same relative positions in different unit cells
    real_t force_tolerance = 1e-5f;
    for (int x = 0; x < supercell_dims[0]; ++x) {
        for (int y = 0; y < supercell_dims[1]; ++y) {
            for (int z = 0; z < supercell_dims[2]; ++z) {
                if (x == 0 && y == 0 && z == 0) continue; // Skip first unit cell (reference)
                
                for (size_t i = 0; i < max_length; ++i) {
                    size_t ref_idx = i;
                    size_t test_idx = i + (x * supercell_dims[1] * supercell_dims[2] + 
                                          y * supercell_dims[2] + z) * max_length;
                    
                    for (size_t j = 0; j < 3; ++j) {
                        EXPECT_NEAR(
                            supercell_force[ref_idx * 3 + j], 
                            supercell_force[test_idx * 3 + j], 
                            force_tolerance
                        ) << "Force mismatch on equivalent atoms: ref atom " << ref_idx 
                          << ", test atom " << test_idx << ", component " << j;
                    }
                }
            }
        }
    }
    
    // Check if stress tensor components scale appropriately
    real_t stress_tolerance = 1e-5f;
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_NEAR(
            supercell_stress[i], 
            original_stress[i], 
            stress_tolerance
        ) << "Stress component mismatch at index " << i;
    }
    
    // Clean up
    free(supercell_elements);
    free(supercell_atoms);
    free(supercell_force);
    free(original_force);
}

TestConfig generate_crystal() {
    // Create a BCC iron crystal (2x2x2 supercell)
    const int atomic_number_Fe = 26; // Iron
    const real_t lattice_constant = 2.87f; // Typical BCC Fe lattice constant in Angstroms
    
    // Create a 2x2x2 supercell
    const int nx = 2, ny = 2, nz = 2;
    const int total_atoms = nx * ny * nz * 2; // 2 atoms per unit cell in BCC
    
    // Elements vector - all iron
    std::vector<uint16_t> elements(total_atoms, atomic_number_Fe);
    
    // Cell dimensions
    real_t cell[3][3] = {
        {nx * lattice_constant, 0.0f, 0.0f},
        {0.0f, ny * lattice_constant, 0.0f},
        {0.0f, 0.0f, nz * lattice_constant}
    };
    
    // Create positions for atoms
    std::vector<std::array<real_t, 3>> atoms;
    atoms.reserve(total_atoms);
    
    // Generate atom positions for the BCC supercell
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                // Corner atom
                atoms.push_back({
                    i * lattice_constant, 
                    j * lattice_constant, 
                    k * lattice_constant
                });
                
                // Body-centered atom
                atoms.push_back({
                    (i + 0.5f) * lattice_constant,
                    (j + 0.5f) * lattice_constant,
                    (k + 0.5f) * lattice_constant
                });
            }
        }
    }
    
    return {
        "BCC_Fe_Crystal",
        elements,
        total_atoms,  // max_length
        40.0f,        // cutoff_radius
        40.0f,        // coordination_number_cutoff
        ZERO_DAMPING, // damping_type
        PBE0, // functional_type
        atoms,
        {
            {cell[0][0], cell[0][1], cell[0][2]},
            {cell[1][0], cell[1][1], cell[1][2]},
            {cell[2][0], cell[2][1], cell[2][2]}
        }
    };
}

TestConfig generate_surface() {
    TestConfig crystal = generate_crystal();
    // Lengthen the cell to create a surface
    crystal.cell[2][2] *= 1.2f; // Increase the z dimension to create a surface
    crystal.test_name = "BCC_Fe_Surface";
    return crystal;
}

TestConfig small_system = {
    "SmallSystem",
    {6, 6, 6, 8, 1, 1, 1, 1, 1, 1}, // C, O, H
    10, // max_length
    40.0f, // cutoff_radius
    40.0f, // coordination_number_cutoff
    ZERO_DAMPING, // damping_type
    PBE0, // functional_type
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

TestConfig crystal_system = generate_crystal(); // Generate BCC iron crystal
TestConfig surface_system = generate_surface(); // Generate surface from the crystal

TestConfig init_system_BJ(TestConfig system) {
    TestConfig bj_system = system;
    bj_system.test_name += "_BJ";
    bj_system.damping_type = BJ_DAMPING;
    return bj_system;
}

TestConfig small_system_BJ = init_system_BJ(small_system);
TestConfig crystal_system_BJ = init_system_BJ(crystal_system);
TestConfig surface_system_BJ = init_system_BJ(surface_system);


// Instantiate the test suite with all system configurations
INSTANTIATE_TEST_SUITE_P(
    SmokeTest,
    D3Test,
    ::testing::Values(
        small_system, 
        crystal_system,
        surface_system,
        small_system_BJ,
        crystal_system_BJ,
        surface_system_BJ
    ),
    [](const testing::TestParamInfo<TestConfig>& info) {
        return info.param.test_name; // Use the test name as the test case name
    }
);
