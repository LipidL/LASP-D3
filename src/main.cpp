#include "d3.h"
#include "parser.cpp"
#include <optional>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <filesystem>

int main(int argc, char *argv[]) {
    auto files = std::vector<std::string>();
    bool use_atm = false;
    if (argc > 1) {
        // Process command line arguments
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];

            if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: dft_d3 filename" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  -h, --help     Display this help message" << std::endl;
                return 0;
            } else if (arg == "--use-atm") {
                use_atm = true;
            } else if (std::filesystem::exists(arg)) {
                files.push_back(arg);
            } else {
                std::cerr << "Unknown argument or file does not exist: " << arg << std::endl;
                return 1;
            }
        }
    } else {
        std::cerr << "No input file provided. Use -h or --help for usage." << std::endl;
        return EXIT_FAILURE;
    }
    for (const auto &file : files) {
        parser::ArcParser<float> parser;
        auto structures_optional = parser.parse_file(file);
        if (!structures_optional) {
            std::cerr << "Error parsing file: " << file << std::endl;
            return EXIT_FAILURE;
        }
        auto structures = structures_optional.value();
        std::cout << "Parsed " << structures.size() << " structures from file: " << file << std::endl;
        for (const auto &structure : structures) {
            uint64_t num_atoms = structure.atoms.size();
            uint16_t *elements = (uint16_t *)malloc(sizeof(uint16_t) * num_atoms);
            float *atoms = (float *)malloc(sizeof(float) * num_atoms * 3); // allocate memory for atom positions
            float cell[3][3]; // allocate memory for cell parameters
            for (size_t i = 0; i < num_atoms; ++i) {
                elements[i] = structure.atoms[i].element.atomic_number;
                atoms[i * 3] = structure.atoms[i].position.x;
                atoms[i * 3 + 1] = structure.atoms[i].position.y;
                atoms[i * 3 + 2] = structure.atoms[i].position.z;
            }
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    cell[i][j] = structure.cell.cell[i][j];
                    std::cout << "Cell[" << i << "][" << j << "]: " << cell[i][j] << std::endl;
                }
            }
            std::cout << "Computing dispersion energy for " << num_atoms << " atoms..." << std::endl;
            float cutoff_radius = 60.0f; // cutoff radius in bohr
            float CN_cutoff_radius = 40.0f; // cutoff radius in bohr
            float atm_cutoff_radius = 40.0f; // cutoff radius for ATM term in bohr
            float energy;
            float *force = (float *)malloc(sizeof(float) * num_atoms * 3); // allocate memory for force
            float *stress = (float *)malloc(sizeof(float) * 9); // allocate memory for stress

            // set up handle
            d3_handle_t *handle;
            if (use_atm) {
                handle = init_d3_handle_with_atm(elements, num_atoms, num_atoms, cutoff_radius, CN_cutoff_radius,
                                              atm_cutoff_radius, BJ_DAMPING, PBE);
            } else {
                handle = init_d3_handle(elements, num_atoms, num_atoms, cutoff_radius, CN_cutoff_radius, BJ_DAMPING, PBE);
            }
            if (handle == nullptr) {
                std::cerr << "Failed to initialize D3 handle." << std::endl;
                return EXIT_FAILURE;
            }
            // set atom positions and elements
            set_atoms(handle, atoms, elements, num_atoms);
            set_cell(handle, cell);
            clear_d3_handle(handle); // clear previous data if any
            // Start measuring execution time
            auto start_time = std::chrono::high_resolution_clock::now();
            compute_dispersion_energy_from_handle(handle, &energy, force, stress);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = end_time - start_time;
            std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;
            std::cout << std::fixed << std::setprecision(20) << "Energy: " << energy << " eV"
                        << std::endl; // convert to eV
            float force_sum[3] = {0.0f, 0.0f, 0.0f};
            for (size_t i = 0; i < num_atoms; ++i) {
                float force_x = force[0 + i * 3];
                float force_y = force[1 + i * 3];
                float force_z = force[2 + i * 3];
                force_sum[0] += force_x;
                force_sum[1] += force_y;
                force_sum[2] += force_z;
                std::cout << std::fixed << std::setprecision(20) << "Force[" << i << "]: " << force_x << " "
                            << force_y << " " << force_z << std::endl;
            }
            std::cout << "Force sum: " << force_sum[0] << " " << force_sum[1] << " " << force_sum[2] << std::endl;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    std::cout << "Stress[" << i << "][" << j << "]: " << stress[i * 3 + j] << std::endl;
                }
            }
        }

    }
    return EXIT_SUCCESS;
}