#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <chrono>
#include <ctime>

#include "d3.h"
#include "d3_buffer.cuh"
#include "d3_kernel.cuh"

/**
 * @brief this function is used to init a handle for d3 energy/force/stress calculation.
 * 
 * @note if you need to call calculate d3 for multiple times where the structures are similar in element composition, you would better use this handle.
 * @note this function initializes the handle in heap area, so you need to free it after use.
 * @note if the number of atoms will vary during your simulation, you need to assign the largest possible number of atoms as the `length` parameter.
 * @note if the elements will vary during your simulation, you need to include all possible elements in the `elements` parameter.
 */
D3Handle_t *init_d3_handle( 
    uint16_t *elements,
    uint64_t max_length, 
    real_t cutoff_radius,
    real_t coordination_number_cutoff,
    uint64_t max_neighbors
) {
    real_t *coords = (real_t *)malloc(max_length * 3 * sizeof(real_t)); // allocate memory for coordinates
    real_t cell[3][3] = {10}; // initialize the cell matrix
    Device_Buffer *buffer = new Device_Buffer((real_t (*)[3])coords, elements, cell, max_length, cutoff_radius, coordination_number_cutoff, max_neighbors); // create a buffer to hold the data
    return (D3Handle_t *)buffer; // return the handle
}

/**
 * @brief this function is used to set the coordinates and elements of the atoms in the system.
 * @note the coordinates and elements should be in the same order as the atoms in the system.
 * @note the number of atoms should not exceed the maximum number of atoms specified in the init_d3_handle function, or the system will crash.
 */
void set_atoms(D3Handle_t *handle, real_t *coords, uint16_t *elements, uint64_t length) {
    Device_Buffer *buffer = (Device_Buffer *)handle; // cast the handle to Device_Buffer
    // Convert coordinates from Angstrom to Bohr
    real_t angstrom_to_bohr = 1.0f/0.52917726f;
    real_t (*bohr_coords)[3] = (real_t (*)[3])malloc(length * 3 * sizeof(real_t));
    for (uint64_t i = 0; i < length; ++i) {
        bohr_coords[i][0] = coords[i*3] * angstrom_to_bohr;
        bohr_coords[i][1] = coords[i*3+1] * angstrom_to_bohr;
        bohr_coords[i][2] = coords[i*3+2] * angstrom_to_bohr;
    }
    buffer->set_atoms(elements, bohr_coords, length);
    free(bohr_coords);
}

/**
 * @brief this function is used to set the cell matrix of the system.
 */
void set_cell(D3Handle_t *handle, real_t cell[3][3]) {
    Device_Buffer *buffer = (Device_Buffer *)handle; // cast the handle to Device_Buffer
    // Convert cell from Angstrom to Bohr
    real_t angstrom_to_bohr = 1.0f/0.52917726f;
    real_t bohr_cell[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            bohr_cell[i][j] = cell[i][j] * angstrom_to_bohr;
        }
    }
    buffer->set_cell(bohr_cell); // set the cell in the buffer
}

/**
 * @brief this function is used to clear the intermediate data in the handle.
 * @note you need to call this function before using the handle again.
 */
void clear_d3_handle(D3Handle_t *handle) {
    Device_Buffer *buffer = (Device_Buffer *)handle; // cast the handle to Device_Buffer
    buffer->clear(); // clear the buffer
    
}

/**
 * @brief this function is used to free the handle after use.
 */
void free_d3_handle(D3Handle_t *handle) {
    Device_Buffer *buffer = (Device_Buffer *)handle; // cast the handle to Device_Buffer
    delete buffer; // free the buffer
}

/**
 * @brief this function is used to compute the dispersion energy of the system using the D3 potential.
 * @param handle the handle to the D3 potential.
 * @param energy pointer to the energy value to be computed.
 * @param force pointer to the force values to be computed.
 * @param stress pointer to the stress values to be computed.
 * @return uint16_t status code indicating the result of the computation.
 */
uint16_t compute_dispersion_energy_from_handle_status(
    D3Handle_t *handle,
    real_t *energy,
    real_t *force,
    real_t *stress
) {
    /* create a CUDA stream */
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    Device_Buffer *buffer = (Device_Buffer *)handle; // cast the handle to Device_Buffer
    /* print debug information about cell */
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            debug("cell[%d][%d] = %f\n", i, j, buffer->get_host_data().cell[i][j]); // print the cell matrix
        }
    }
    // launch the kernel
    uint64_t length = buffer->get_host_data().num_atoms; // get the number of atoms in the system
    debug("launching coordination_number_kernel, size: %zu, %d\n", length, MAX_BLOCK_SIZE);
    coordination_number_kernel<<<length, MAX_BLOCK_SIZE, 0, stream>>>(buffer->get_device_data()); // launch the kernel to compute the coordination numbers
    #ifdef DEBUG
    print_coordination_number_kernel<<<1,1,0, stream>>>(buffer->get_device_data()); // print the coordination numbers for debugging
    CHECK_CUDA(cudaDeviceSynchronize()); // synchronize the device to ensure all threads are finished
    #endif
    debug("launching two_body_kernel, size: %zu, %d\n", length, MAX_BLOCK_SIZE);
    two_body_kernel<<<length, MAX_BLOCK_SIZE, 0, stream>>>(buffer->get_device_data());
    debug("launching three_body_kernel, size: %zu, %d\n", length, MAX_BLOCK_SIZE);
    three_body_kernel<<<length, MAX_BLOCK_SIZE, 0, stream>>>(buffer->get_device_data());

    cudaMemcpyAsync(force, buffer->get_host_data().forces, length * 3 * sizeof(real_t), cudaMemcpyDeviceToHost, stream); // copy the forces back to host memory
    cudaMemcpyAsync(energy, buffer->get_host_data().energy, sizeof(real_t), cudaMemcpyDeviceToHost, stream); // copy the energy back to host memory
    cudaMemcpyAsync(stress, buffer->get_host_data().stress, 9 * sizeof(real_t), cudaMemcpyDeviceToHost, stream); // copy the stress back to host memory
    
    uint16_t status;
    /* check for compute status */
    device_data_t *device_data = buffer->get_device_data();
    cudaMemcpyAsync(&status, &(device_data->status), sizeof(uint16_t), cudaMemcpyDeviceToHost, stream); // copy the compute status back to host memory

    CHECK_CUDA(cudaStreamSynchronize(stream)); // synchronize the stream to ensure all operations are finished

    real_t angstron_to_bohr = 1/0.52917726f; // angstron to bohr conversion factor
    real_t hartree_to_eV = 27.211396641308f; // hartree to eV conversion factor
    *energy *= -hartree_to_eV; // convert energy to eV and negate it
    for (uint64_t i = 0; i < length; ++i) {
        /* convert force from hartree/bohr to eV/angstron */
        force[i * 3 + 0] *= hartree_to_eV * angstron_to_bohr;
        force[i * 3 + 1] *= hartree_to_eV * angstron_to_bohr;
        force[i * 3 + 2] *= hartree_to_eV * angstron_to_bohr;
    }
    for (uint64_t i = 0; i < 3; ++i) {
        for (uint64_t j = 0; j < 3; ++j) {
            stress[i * 3 + j] *= hartree_to_eV * powf(angstron_to_bohr,3); // convert stress to from hartree/bohr^3 to eV/angstron^3
        }
    }
    CHECK_CUDA(cudaStreamDestroy(stream)); // destroy the stream
    return status;
}

void compute_dispersion_energy_from_handle(
    D3Handle_t *handle,
    real_t *energy,
    real_t *force,
    real_t *stress
) {
    uint16_t status = compute_dispersion_energy_from_handle_status(handle, energy, force, stress); // compute the dispersion energy from the handle
    if (status != COMPUTE_SUCCESS) {
        fprintf(stderr, "Error: compute_dispersion_energy_from_handle failed with status %d\n", status);
    }
}

/**
 * @brief the function is used to compute the dispersion energy of the system using the D3 potential.
 * 
 * @param atoms the array of atoms in the system. The array is of size num_atoms * 4, where the first entry is atomic number and the last 3 entries are the coordinates of the atom.
 * @param length the number of atoms in the system.
 * @note the function is not thread safe, and should be called from a single thread.
 */
__host__ void compute_dispersion_energy(
    real_t coords[][3], 
    uint16_t *elements,
    uint64_t length, 
    real_t cell[3][3],
    real_t cutoff_radius,
    real_t coordination_number_cutoff,
    uint64_t max_neighbors,
    real_t *energy,
    real_t *force,
    real_t *stress
    ) {
    // initialize parameters
    init_params();
    // compute dispersion energy
    // Start measuring execution time
    D3Handle_t *handle = init_d3_handle(elements, length, cutoff_radius, coordination_number_cutoff, max_neighbors);
    auto start_time = std::chrono::high_resolution_clock::now();
    set_atoms(handle, (real_t *)coords, elements, length);
    set_cell(handle, cell);
    clear_d3_handle(handle);
    compute_dispersion_energy_from_handle(handle, energy, force, stress);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    debug("Elapsed time: %.6f seconds\n", elapsed_time.count());
    free_d3_handle(handle);
}

#ifndef BUILD_LIBRARY
int main()
{
    // example usage of the compute_dispersion_energy function
    real_t atoms[10][3] = {
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
    uint16_t elements[10] = {6, 6, 6, 8, 1, 1, 1, 1, 1, 1}; // atomic numbers of the atoms
    real_t angstron_to_bohr = 1/0.52917726f; // angstron to bohr conversion factor
    for(uint64_t i = 0; i < 10; ++i) {
        atoms[i][0] *= angstron_to_bohr; // convert to bohr
        atoms[i][1] *= angstron_to_bohr; // convert to bohr
        atoms[i][2] *= angstron_to_bohr; // convert to bohr
    }
    // fill the atoms array with Po element
    debug("Computing dispersion energy for %zu atoms...\n", sizeof(atoms)/sizeof(atoms[0]));
    // initialize parameters
    init_params();
    debug("Computing dispersion energy...\n");
    real_t cell[3][3] = {
        {200.0f, 0.0f, 0.0f},
        {0.0f, 20.0f, 0.0f},
        {0.0f, 0.0f, 20.0f}
    };
    for (uint64_t i = 0; i < 3; ++i) {
        for (uint64_t j = 0; j < 3; ++j) {
            cell[i][j] *= angstron_to_bohr; // convert to bohr
        }
    }
    real_t CN_cutoff_radius = 40.0f; // cutoff radius in bohr
    real_t cutoff_radius = 94.8683f;
    real_t energy;
    real_t *force = (real_t *)malloc(sizeof(real_t) * 10 * 3); // allocate memory for force
    real_t *stress = (real_t *)malloc(sizeof(real_t) * 9); // allocate memory for stress
    compute_dispersion_energy(atoms, elements, 10, cell, cutoff_radius, CN_cutoff_radius, 5000, &energy, force, stress);
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
    debug("force sum: %.13f %.13f %.13f\n", force_sum[0], force_sum[1], force_sum[2]);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            debug("stress[%d][%d]: %.13f\n", i, j, stress[i * 3 + j]);
        }
    }
    return 0;
}
#endif