#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include "parser.cpp"
#include "d3.h"

int main(int argc, char* argv[])
{
    if (argc > 1) {
        // Process command line arguments
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            
            if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: dft_d3 filename" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  -h, --help     Display this help message" << std::endl;
                return 0;
            }
            
            parser::ArcParser<float> parser;
            auto structures_optional = parser.parse_file(arg);
            if (!structures_optional) {
                std::cerr << "Error parsing file: " << arg << std::endl;
                return 1;
            }
            auto structures = structures_optional.value();
            for (const auto& structure : structures) {
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
                float energy;
                float *force = (float *)malloc(sizeof(float) * num_atoms * 3); // allocate memory for force
                float *stress = (float *)malloc(sizeof(float) * 9); // allocate memory for stress
                // Start measuring execution time
                auto start_time = std::chrono::high_resolution_clock::now();
                compute_dispersion_energy((float (*)[3])atoms, elements, num_atoms, cell, cutoff_radius, CN_cutoff_radius, ZERO_DAMPING, PBE0, &energy, force, stress);
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_time = end_time - start_time;
                std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;
                std::cout << std::fixed << std::setprecision(9) << "Energy: " << energy << " eV" << std::endl; // convert to eV
                float force_sum[3] = {0.0f, 0.0f, 0.0f};
                for (size_t i = 0; i < num_atoms; ++i) {
                    float force_x = force[0 + i * 3];
                    float force_y = force[1 + i * 3];
                    float force_z = force[2 + i * 3];
                    force_sum[0] += force_x;
                    force_sum[1] += force_y;
                    force_sum[2] += force_z;
                    std::cout << std::fixed << std::setprecision(9) << "Force[" << i << "]: " << force_x << " " << force_y << " " << force_z << std::endl;
                }
                std::cout << std::fixed << std::setprecision(9) << "Force sum: " << force_sum[0] << " " << force_sum[1] << " " << force_sum[2] << std::endl;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        std::cout << std::fixed << std::setprecision(9) << "Stress[" << i << "][" << j << "]: " << stress[i * 3 + j] << std::endl;
                    }
                }
            }
        }
    } else {
        std::cerr << "No input file provided. Use -h or --help for usage." << std::endl;
        return 1;
    }
    return EXIT_SUCCESS;
}