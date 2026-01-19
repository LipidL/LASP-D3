#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <ctime>
#include <exception>

#include "d3.h"
#include "d3_buffer.cuh"
#include "d3_kernel.cuh"

d3_handle_t *init_d3_handle(uint16_t *elements, uint64_t length_elements, uint64_t max_length, real_t cutoff_radius,
                            real_t coordination_number_cutoff, damping_type_t damping_type,
                            functional_t functional_type) {
    try {
        Device_Buffer *buffer = new Device_Buffer(elements, length_elements, max_length, cutoff_radius,
                                                  coordination_number_cutoff, damping_type,
                                                  functional_type); // create a buffer to hold the data
        return (d3_handle_t *)buffer; // return the pointer to the handle
    } catch (const std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return NULL;
    }
}

void set_atoms(d3_handle_t *handle, real_t *coords, uint16_t *elements, uint64_t length) {
    if (handle == NULL) {
        fprintf(stderr, "Error: handle is NULL in set_atoms\n");
        return;
    }
    Device_Buffer *buffer = (Device_Buffer *)handle; // cast the handle to Device_Buffer

    // Convert coordinates from Angstrom to Bohr
    real_t angstrom_to_bohr = 1.0f / 0.52917726f;
    real_t(*bohr_coords)[3] = (real_t(*)[3])malloc(length * 3 * sizeof(real_t));
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
    } catch (const std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        free(bohr_coords);
        return;
    }
}

void set_cell(d3_handle_t *handle, real_t cell[3][3]) {
    if (handle == NULL) {
        fprintf(stderr, "Error: handle is NULL in set_cell\n");
        return;
    }
    Device_Buffer *buffer = (Device_Buffer *)handle; // cast the handle to Device_Buffer
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
    } catch (const std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return;
    }
}

void clear_d3_handle(d3_handle_t *handle) {
    if (handle == NULL) {
        fprintf(stderr, "Error: handle is NULL in clear_d3_handle\n");
        return;
    }
    Device_Buffer *buffer = (Device_Buffer *)handle; // cast the handle to Device_Buffer
    try {
        buffer->clear();
        return;
    } catch (const std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return;
    }
}

void free_d3_handle(d3_handle_t *handle) {
    if (handle == NULL) {
        fprintf(stderr, "Error: handle is NULL in free_d3_handle\n");
        return;
    }
    Device_Buffer *buffer = (Device_Buffer *)handle; // cast the handle to Device_Buffer
    try {
        CHECK_CUDA(cudaDeviceSynchronize()); // Ensure all CUDA operations complete before freeing
        // Clear any pending CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Warning: Clearing CUDA error state before freeing handle: %s\n", cudaGetErrorString(err));
        }
        delete buffer;
    } catch (const std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return;
    }
}

uint16_t compute_dispersion_energy_from_handle_status(d3_handle_t *handle, real_t *energy, real_t *force,
                                                      real_t *stress) {
    if (handle == NULL) {
        fprintf(stderr, "Error: handle is NULL in compute_dispersion_energy_from_handle\n");
        return 1;
    }
    try {
        // create a CUDA stream
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        Device_Buffer *buffer = (Device_Buffer *)handle; // cast the handle to Device_Buffer
        buffer->clear(); // clear previous intermediate results
        // construct grid cells for neighbor list (if applicable)
        buffer->construct_grids();
        // print debug information about cell
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                debug("cell[%d][%d] = %f\n", i, j,
                      buffer->get_host_data().cell[i][j]); // print the cell matrix
            }
        }
        // launch the kernel
        uint64_t length = buffer->get_host_data().num_atoms; // get the number of atoms in the system
        debug("launching coordination_number_kernel, size: %zu, %d\n", length, MAX_BLOCK_SIZE);
        // calculate coordination number
        coordination_number_kernel<<<length, MAX_BLOCK_SIZE, 0, stream>>>(buffer->get_device_data());
        CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors
        // print some debug information
#ifdef DEBUG
        print_coordination_number_kernel<<<1, 1, 0, stream>>>(buffer->get_device_data());
        CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors
#endif
        // calculate ATM interaction
        debug("launching atm_kernel, size: %zu, %d\n", length, MAX_BLOCK_SIZE);
        atm_kernel_new<<<length, MAX_BLOCK_SIZE, 0, stream>>>(buffer->get_device_data());
        CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors
        debug("launching two_body_kernel, size: %zu, %d\n", length, MAX_BLOCK_SIZE);

        // calculate energy and two-body part of force
        two_body_kernel<<<length, MAX_BLOCK_SIZE, 0, stream>>>(buffer->get_device_data());
        CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

        real_t *atomic_energy = (real_t *)malloc(length * sizeof(real_t)); // allocate memory for atomic energy
        CHECK_CUDA(cudaMemcpyAsync(atomic_energy, buffer->get_host_data().energy, length * sizeof(real_t),
                                   cudaMemcpyDeviceToHost, stream)); // copy the energy from device to host memory
        CHECK_CUDA(
            cudaStreamSynchronize(stream)); // synchronize the stream to ensure the first two kernels are finished
        // calculate the three-body part of force
        debug("launching three_body_kernel, size: %zu, %d\n", length, MAX_BLOCK_SIZE);
        three_body_kernel<<<length, MAX_BLOCK_SIZE, 0, stream>>>(buffer->get_device_data());
        CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors
        // perform energy accumulation
        double energy_sum = 0.0; // use high precision at CPU side
        /* perform reduction to get the total energy */
        for (uint64_t i = 0; i < length; ++i) {
            energy_sum += atomic_energy[i];
        }
        *energy = energy_sum; // set the energy value
        free(atomic_energy); // free the atomic energy array

        // copy the force and stress back to host side
        cudaMemcpyAsync(force, buffer->get_host_data().forces, length * 3 * sizeof(real_t), cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(stress, buffer->get_host_data().stress, 9 * sizeof(real_t), cudaMemcpyDeviceToHost, stream);

        uint16_t status;
        // check for compute status
        device_data_t *device_data = buffer->get_device_data();
        cudaMemcpyAsync(&status, &(device_data->status), sizeof(uint16_t), cudaMemcpyDeviceToHost,
                        stream); // copy the compute status back to host memory

        CHECK_CUDA(cudaStreamSynchronize(stream)); // synchronize the stream to ensure all operations are finished

        // convert bohr to angstron, hartree to eV
        real_t angstron_to_bohr = 1 / 0.52917726f; // angstron to bohr conversion factor
        real_t hartree_to_eV = 27.211396641308f; // hartree to eV conversion factor
        *energy *= -hartree_to_eV; // convert energy to eV and negate it
        for (uint64_t i = 0; i < length; ++i) {
            // convert force from hartree/bohr to eV/angstrom
            force[i * 3 + 0] *= hartree_to_eV * angstron_to_bohr;
            force[i * 3 + 1] *= hartree_to_eV * angstron_to_bohr;
            force[i * 3 + 2] *= hartree_to_eV * angstron_to_bohr;
        }
        for (uint64_t i = 0; i < 3; ++i) {
            for (uint64_t j = 0; j < 3; ++j) {
                // printf("debug: stress[%zu][%zu] before conversion = %f\n", i, j, stress[i * 3 + j]);
                stress[i * 3 + j] *=
                    hartree_to_eV * powf(angstron_to_bohr, 3); // convert stress to from hartree/bohr^3 to eV/angstron^3
            }
        }
        CHECK_CUDA(cudaStreamDestroy(stream)); // destroy the stream
        return status;
    } catch (const std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}

void compute_dispersion_energy_from_handle(d3_handle_t *handle, real_t *energy, real_t *force, real_t *stress) {
    uint16_t status = compute_dispersion_energy_from_handle_status(
        handle, energy, force, stress); // compute the dispersion energy from the handle
    if (status != COMPUTE_SUCCESS) {
        fprintf(stderr, "Error: compute_dispersion_energy_from_handle failed with status %d\n", status);
    }
}

__host__ void compute_dispersion_energy(real_t coords[][3], uint16_t *elements, uint64_t length, real_t cell[3][3],
                                        real_t cutoff_radius, real_t coordination_number_cutoff,
                                        damping_type_t damping_type, functional_t functional_type, real_t *energy,
                                        real_t *force, real_t *stress) {
    // compute dispersion energy
    try {
        d3_handle_t *handle = init_d3_handle(elements, length, length, cutoff_radius, coordination_number_cutoff,
                                             damping_type, functional_type);
        set_atoms(handle, (real_t *)coords, elements, length);
        set_cell(handle, cell);
        clear_d3_handle(handle);
        compute_dispersion_energy_from_handle(handle, energy, force, stress);
        free_d3_handle(handle);
    } catch (const std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
    }
}