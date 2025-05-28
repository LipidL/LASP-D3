#include <iostream>
#include <chrono>
#include <ctime>
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
            auto structure = structures[0]; // only handle first structure
            std::cout << "Parsed structure with energy: " << structure.energy << std::endl;
            size_t num_atoms = structure.atoms.size();
            std::cout << "Number of atoms: " << num_atoms << std::endl;
            uint16_t *elements = (uint16_t *)malloc(sizeof(uint16_t) * num_atoms);
            float *atoms = (float *)malloc(sizeof(float) * num_atoms * 3); // allocate memory for atom positions
            float angstron_to_bohr = 1 / 0.52917726f; // angstron to bohr conversion factor
            for (size_t i = 0; i < num_atoms; ++i) {
                elements[i] = structure.atoms[i].element.atomic_number;
                atoms[i * 3] = structure.atoms[i].position.x * angstron_to_bohr; // convert to bohr
                atoms[i * 3 + 1] = structure.atoms[i].position.y * angstron_to_bohr;
                atoms[i * 3 + 2] = structure.atoms[i].position.z * angstron_to_bohr;
            }
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    structure.cell.cell[i][j] *= angstron_to_bohr; // convert to bohr   
                    std::cout << "Cell[" << i << "][" << j << "]: " << structure.cell.cell[i][j] << std::endl;
                }
            }
            std::cout << "Computing dispersion energy for " << num_atoms << " atoms..." << std::endl;
            // initialize parameters
            init_params();
            // compute dispersion energy
            float cutoff_radius = 46.4758f; // cutoff radius in bohr
            float CN_cutoff_radius = 46.4758f; // cutoff radius in bohr
            float energy;
            float *force = (float *)malloc(sizeof(float) * num_atoms * 3); // allocate memory for force
            float *stress = (float *)malloc(sizeof(float) * 9); // allocate memory for stress
            // Start measuring execution time
            D3Handle_t *handle = init_d3_handle(elements, num_atoms, cutoff_radius, CN_cutoff_radius);
            for (size_t i = 0; i < 100000; ++i){
                auto start_time = std::chrono::high_resolution_clock::now();
                // compute_dispersion_energy((float (*)[3])atoms, elements, num_atoms, structure.cell.cell, cutoff_radius, CN_cutoff_radius, &energy, force, stress);

                atoms[0] += i * 0.01f; // modify atoms for testing
                set_atoms(handle, atoms, elements, num_atoms);
                set_cell(handle, structure.cell.cell);
                clear_d3_handle(handle);
                compute_dispersion_energy_from_handle(handle, &energy, force, stress);
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_time = end_time - start_time;
                std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;
                std::cout << "Energy: " << energy << " eV" << std::endl; // convert to eV
            }
            free_d3_handle(handle);
            std::cout << "Energy: " << energy << " eV" << std::endl;
            float force_sum[3] = {0.0f, 0.0f, 0.0f};
            for (size_t i = 0; i < num_atoms; ++i) {
                float force_x = force[0 + i * 3];
                float force_y = force[1 + i * 3];
                float force_z = force[2 + i * 3];
                force_sum[0] += force_x;
                force_sum[1] += force_y;
                force_sum[2] += force_z;
                std::cout << "Force[" << i << "]: " << force_x << " " << force_y << " " << force_z << std::endl;
            }
            std::cout << "Force sum: " << force_sum[0] << " " << force_sum[1] << " " << force_sum[2] << std::endl;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    std::cout << "Stress[" << i << "][" << j << "]: " << stress[i * 3 + j] << std::endl;
                }
            }


        }
    } else {
        std::cerr << "No input file provided. Use -h or --help for usage." << std::endl;
        return 1;
    }
    return EXIT_SUCCESS;
}