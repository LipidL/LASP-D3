#ifndef D3_BUFFER_CUH
#define D3_BUFFER_CUH

#include <stdio.h>
#include "d3_internal.h"

void calculate_cell_repeats(real_t cell[3][3], real_t cutoff, size_t max_cell_bias[3]); 

/**
 * @brief class to store unique elements in the system
 * @note the user must not construct a Unique_Elements object directly, but use the Unique_Elements constructor instead
 */
class Unique_Elements {
public:
    uint16_t num_elements; // number of unique elements in the system
    Unique_Elements(uint16_t *elements, uint16_t length); // Unique_Elements constructor
    ~Unique_Elements(); // Unique_Elements destructor
    uint16_t find(uint16_t element); // find the index of the element in the unique elements array, if not found, it will add the element to the array and return the index
    uint16_t operator[](uint16_t index); // operator to access the element at the given index

private:
    uint16_t *elements_; // array of unique elements in the system
}; // Unique_Elements

/**
 * @brief class on host to store the device data
 * @note the data is stored in device memory, so no host access is allowed
 */
class Device_Buffer {
public:
    __host__ Device_Buffer(real_t coords[][3], uint16_t *elements, real_t cell[3][3], uint64_t length, real_t cutoff, real_t CN_cutoff, uint64_t max_neighbors); // Device_Buffer constructor
    __host__ ~Device_Buffer(); // Device_Buffer destructor

    /* disable copying */
    Device_Buffer(const Device_Buffer&) = delete; // disable copy constructor
    Device_Buffer& operator=(const Device_Buffer&) = delete; // disable copy assignment operator

    /* enable moving */
    __host__ Device_Buffer(Device_Buffer&& other) noexcept; // move constructor
    __host__ Device_Buffer& operator=(Device_Buffer&& other) noexcept; // move assignment operator
    __host__ device_data_t* get_device_data() {
        return this->device_data_; // return the device data pointer
    } // get device data pointer
    __host__ device_data_t get_host_data() {
        return this->host_data_; // return the host data
    } // get host data

    __host__ void set_atoms(uint16_t *elements, real_t coords[][3], uint64_t length); // set atoms

    __host__ void set_cell(real_t cell[3][3]); // set cell

    __host__ void clear(); // clear the device data
    private:
    device_data_t *device_data_; // pointer to the device data
    device_data_t host_data_;
}; // Device_Buffer

#endif // D3_BUFFER_CUH