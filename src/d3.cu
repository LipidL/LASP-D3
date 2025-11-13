#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <ctime>
#include <exception>

#include "d3.h"
#include "d3_buffer.cuh"
#include "d3_kernel.cuh"


D3Handle_t* init_d3_handle(
    uint16_t* elements, 
    uint64_t length_elements,
    uint64_t max_length,
    real_t cutoff_radius,
    real_t coordination_number_cutoff,
    DampingType damping_type, 
    FunctionalType functional_type
) {
    real_t* coords = (real_t*)malloc(max_length * 3 * sizeof(real_t));  // allocate memory for coordinates
    if (coords == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for coordinates in init_d3_handle\n");
        return NULL;
    }
    real_t cell[3][3] = {10};   // initialize the cell matrix
    try {
        Device_Buffer* buffer =
        new Device_Buffer((real_t(*)[3])coords, elements, length_elements, cell, max_length,
                          cutoff_radius, coordination_number_cutoff,
                          damping_type, functional_type);   // create a buffer to hold the data
        free(coords);  // free the coordinates array
        return (D3Handle_t*)buffer; // return the pointer to the handle
    } catch(const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        free(coords);  // free the coordinates array
        return NULL;
    }
    
}


void set_atoms(
    D3Handle_t* handle,
    real_t* coords, 
    uint16_t* elements,
    uint64_t length) {
    if (handle == NULL) {
        fprintf(stderr, "Error: handle is NULL in set_atoms\n");
        return;
    }
    Device_Buffer* buffer = (Device_Buffer*)handle; // cast the handle to Device_Buffer
    // Convert coordinates from Angstrom to Bohr
    real_t angstrom_to_bohr = 1.0f / 0.52917726f;
    real_t (*bohr_coords)[3] = (real_t(*)[3])malloc(length * 3 * sizeof(real_t));
    if (bohr_coords == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for coordinates in set_atoms\n");
        return;
    }
    for (uint64_t i = 0; i < length; ++i) {
        bohr_coords[i][0] = coords[i * 3] * angstrom_to_bohr;
        bohr_coords[i][1] = coords[i * 3 + 1] * angstrom_to_bohr;
        bohr_coords[i][2] = coords[i * 3 + 2] * angstrom_to_bohr;
    }
    try {
        buffer->set_atoms(elements, bohr_coords, length);
        free(bohr_coords);
        return;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        free(bohr_coords);
        return;
    }
}


void set_cell(D3Handle_t* handle, real_t cell[3][3]) {
    if (handle == NULL) {
        fprintf(stderr, "Error: handle is NULL in set_cell\n");
        return;
    }
    Device_Buffer* buffer = (Device_Buffer*)handle; // cast the handle to Device_Buffer
    // Convert cell from Angstrom to Bohr
    real_t angstrom_to_bohr = 1.0f / 0.52917726f;
    real_t bohr_cell[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            bohr_cell[i][j] = cell[i][j] * angstrom_to_bohr;
        }
    }
    try {
        buffer->set_cell(bohr_cell);
        return;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return;
    }
}


void clear_d3_handle(D3Handle_t* handle) {
    if (handle == NULL) {
        fprintf(stderr, "Error: handle is NULL in clear_d3_handle\n");
        return;
    }
    Device_Buffer* buffer = (Device_Buffer*)handle; // cast the handle to Device_Buffer
    try {
        buffer->clear();
        return;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return;
    }
}


void free_d3_handle(D3Handle_t* handle) {
    if (handle == NULL) {
        fprintf(stderr, "Error: handle is NULL in free_d3_handle\n");
        return;
    }
    Device_Buffer* buffer = (Device_Buffer*)handle; // cast the handle to Device_Buffer
    try {
        delete buffer;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return;
    }
}


uint16_t compute_dispersion_energy_from_handle_status(D3Handle_t* handle,
                                                      real_t* energy,
                                                      real_t* force,
                                                      real_t* stress) {
    if (handle == NULL) {
        fprintf(stderr, "Error: handle is NULL in compute_dispersion_energy_from_handle\n");
        return 1;
    }
    try {

    
        // create a CUDA stream
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        Device_Buffer* buffer =
            (Device_Buffer*)handle;  // cast the handle to Device_Buffer
        // print debug information about cell
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                debug("cell[%d][%d] = %f\n", i, j,
                    buffer->get_host_data().cell[i][j]);  // print the cell matrix
            }
        }
        // launch the kernel
        uint64_t length = buffer->get_host_data().num_atoms;  // get the number of atoms in the system
        debug("launching coordination_number_kernel, size: %zu, %d\n", length, MAX_BLOCK_SIZE);
        // calculate coordination number
        coordination_number_kernel<<<length, MAX_BLOCK_SIZE, 0, stream>>>(buffer->get_device_data());
    #ifdef DEBUG
        // print some debug information
        print_coordination_number_kernel<<<1, 1, 0, stream>>>(buffer->get_device_data());
    #endif
        debug("launching two_body_kernel, size: %zu, %d\n", length, MAX_BLOCK_SIZE);
        // calculate energy and two-body part of force
        two_body_kernel<<<length, MAX_BLOCK_SIZE, 0, stream>>>(buffer->get_device_data());
        real_t* atomic_energy = (real_t*)malloc(length * sizeof(real_t));  // allocate memory for atomic energy
        CHECK_CUDA(cudaMemcpyAsync(atomic_energy, buffer->get_host_data().energy, length * sizeof(real_t), cudaMemcpyDeviceToHost, stream));  // copy the energy from device to host memory
        CHECK_CUDA(cudaStreamSynchronize(stream));  // synchronize the stream to ensure the first two kernels are finished
        // calculate the three-body part of force
        debug("launching three_body_kernel, size: %zu, %d\n", length, MAX_BLOCK_SIZE);
        three_body_kernel<<<length, MAX_BLOCK_SIZE, 0, stream>>>(buffer->get_device_data());
        // perform energy accumulation
        double energy_sum = 0.0;  // use high precision at CPU side
        /* perform reduction to get the total energy */
        for (uint64_t i = 0; i < length; ++i) {
            energy_sum += atomic_energy[i];
        }
        *energy = energy_sum;  // set the energy value
        free(atomic_energy);   // free the atomic energy array

        // copy the force and stress back to host side
        cudaMemcpyAsync(force, buffer->get_host_data().forces, length * 3 * sizeof(real_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(stress, buffer->get_host_data().stress, 9 * sizeof(real_t), cudaMemcpyDeviceToHost, stream);

        uint16_t status;
        // check for compute status
        device_data_t* device_data = buffer->get_device_data();
        cudaMemcpyAsync(&status, &(device_data->status), sizeof(uint16_t), cudaMemcpyDeviceToHost, stream);  // copy the compute status back to host memory

        CHECK_CUDA(cudaStreamSynchronize(stream));  // synchronize the stream to ensure all operations are finished

        // convert bohr to angstron, hartree to eV
        real_t angstron_to_bohr = 1 / 0.52917726f;  // angstron to bohr conversion factor
        real_t hartree_to_eV = 27.211396641308f;  // hartree to eV conversion factor
        *energy *= -hartree_to_eV;  // convert energy to eV and negate it
        for (uint64_t i = 0; i < length; ++i) {
            // convert force from hartree/bohr to eV/angstrom
            force[i * 3 + 0] *= hartree_to_eV * angstron_to_bohr;
            force[i * 3 + 1] *= hartree_to_eV * angstron_to_bohr;
            force[i * 3 + 2] *= hartree_to_eV * angstron_to_bohr;
        }
        for (uint64_t i = 0; i < 3; ++i) {
            for (uint64_t j = 0; j < 3; ++j) {
                stress[i * 3 + j] *= hartree_to_eV * powf(angstron_to_bohr, 3);  // convert stress to from hartree/bohr^3 to eV/angstron^3
            }
        }
        CHECK_CUDA(cudaStreamDestroy(stream));  // destroy the stream
        return status;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}

void compute_dispersion_energy_from_handle(D3Handle_t* handle, real_t* energy,
                                           real_t* force, real_t* stress) {
    uint16_t status = compute_dispersion_energy_from_handle_status( handle, energy, force, stress);  // compute the dispersion energy from the handle
    if (status != COMPUTE_SUCCESS) {
        fprintf(stderr, "Error: compute_dispersion_energy_from_handle failed with status %d\n",status);
    }
}


__host__ void compute_dispersion_energy(real_t coords[][3], uint16_t* elements,
                                        uint64_t length, real_t cell[3][3],
                                        real_t cutoff_radius,
                                        real_t coordination_number_cutoff,
                                        DampingType damping_type, 
                                        FunctionalType functional_type,
                                        real_t* energy,
                                        real_t* force, real_t* stress) {
    // compute dispersion energy
    try {
        D3Handle_t* handle = init_d3_handle(elements, length, length, cutoff_radius, coordination_number_cutoff, damping_type, functional_type);
        set_atoms(handle, (real_t*)coords, elements, length);
        set_cell(handle, cell);
        clear_d3_handle(handle);
        compute_dispersion_energy_from_handle(handle, energy, force, stress);
        free_d3_handle(handle);
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
    }
}

#ifndef BUILD_LIBRARY
int main() {
    // example usage of the compute_dispersion_energy function
    real_t atoms[10][3] = {
        {5.1372f, 5.5512f, 10.1047f}, {4.5169f, 6.1365f, 11.3604f},
        {6.1937f, 4.4752f, 10.2703f}, {4.7872f, 5.9358f, 8.9937f},
        {6.7474f, 4.3475f, 9.3339f},  {5.6975f, 3.5214f, 10.5181f},
        {6.8870f, 4.7006f, 11.0939f}, {4.8579f, 5.6442f, 12.2774f},
        {3.4204f, 6.0677f, 11.2935f}, {4.7678f, 7.2075f, 11.4098f}};
    uint16_t elements[10] = {6, 6, 6, 8, 1,
                             1, 1, 1, 1, 1};  // atomic numbers of the atoms
    real_t angstron_to_bohr =
        1 / 0.52917726f;  // angstron to bohr conversion factor
    for (uint64_t i = 0; i < 10; ++i) {
        atoms[i][0] *= angstron_to_bohr;  // convert to bohr
        atoms[i][1] *= angstron_to_bohr;  // convert to bohr
        atoms[i][2] *= angstron_to_bohr;  // convert to bohr
    }
    // fill the atoms array with Po element
    debug("Computing dispersion energy for %zu atoms...\n", sizeof(atoms) / sizeof(atoms[0]));
    real_t cell[3][3] = {
        {200.0f, 0.0f, 0.0f}, {0.0f, 20.0f, 0.0f}, {0.0f, 0.0f, 20.0f}};
    for (uint64_t i = 0; i < 3; ++i) {
        for (uint64_t j = 0; j < 3; ++j) {
            cell[i][j] *= angstron_to_bohr;  // convert to bohr
        }
    }
    real_t CN_cutoff_radius = 40.0f;  // cutoff radius in bohr
    real_t cutoff_radius = 94.8683f;
    real_t energy;
    real_t* force =
        (real_t*)malloc(sizeof(real_t) * 10 * 3);  // allocate memory for force
    real_t* stress =
        (real_t*)malloc(sizeof(real_t) * 9);  // allocate memory for stress
    compute_dispersion_energy(atoms, elements, 10, cell, cutoff_radius,
                              CN_cutoff_radius, ZERO_DAMPING, PBE0, &energy, force, stress);
    debug("energy: %f eV\n", energy);
    real_t force_sum[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 10; ++i) {
        real_t force_x = force[0 + i * 3];
        real_t force_y = force[1 + i * 3];
        real_t force_z = force[2 + i * 3];
        force_sum[0] += force_x;
        force_sum[1] += force_y;
        force_sum[2] += force_z;
        debug("force[%d]: %.13f %.13f %.13f\n", i, force_x, force_y, force_z);
    }
    debug("force sum: %.13f %.13f %.13f\n", force_sum[0], force_sum[1],
          force_sum[2]);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            debug("stress[%d][%d]: %.13f\n", i, j, stress[i * 3 + j]);
        }
    }
    return 0;
}
#endif