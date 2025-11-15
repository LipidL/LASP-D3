#ifndef D3_BUFFER_CUH
#define D3_BUFFER_CUH

#include <stdio.h>

#include "d3_internal.h"
#include "d3_types.h"

void calculate_cell_repeats(
    real_t cell[3][3], 
    real_t cutoff, 
    size_t max_cell_bias[3]
);

/**
 * @brief class to store unique elements in the system
 * @note the user must not construct a Unique_Elements object directly, but use
 * the Unique_Elements constructor instead
 */
class Unique_Elements {
   public:
    uint16_t num_elements;  // number of unique elements in the system
    /**
     * @brief constructor of `Unique_Elements` class
     * @param elements the array of elements in the system
     * @param length the length of the elements array
     * 
     * @throws std::runtime_error if any element exceeds MAX_ELEMENTS
     */
    Unique_Elements(uint16_t* elements, uint64_t length);
    /**
     * @brief destructor of `Unique_Elements` class
     */
    ~Unique_Elements();
    /**
     * @brief find the index of the element in the unique elements array.
     * if not found, the new element will be inserted and the index will be returned
     * @return the index of the element in the unique elements array
     */
    uint16_t find(uint16_t element);
    /**
     * @brief access the element at the given index
     * @return the element at the given index
     */
    uint16_t operator[](uint16_t index);

   private:
    uint16_t* elements_;  // array of unique elements in the system
};  // Unique_Elements

/**
 * @brief class on host to store the device data
 * @note the data is stored in device memory, so no host access is allowed
 */
class Device_Buffer {
   public:
   /**
    * @brief constructor of `Device_Buffer` class
    */
    __host__ Device_Buffer(
        real_t coords[][3], 
        uint16_t* elements, 
        uint64_t length_elements, 
        real_t cell[3][3], 
        uint64_t length, 
        real_t cutoff, 
        real_t CN_cutoff, 
        DampingType damping_type, 
        FunctionalType functional_type);
        
    /**
     * @brief destructor of `Device_Buffer` class
     */
    __host__ ~Device_Buffer();

    /* disable copying */
    Device_Buffer(const Device_Buffer&) = delete;  // disable copy constructor
    Device_Buffer& operator=(const Device_Buffer&) = delete;  // disable copy assignment operator

    /* enable moving */
    __host__ Device_Buffer(Device_Buffer&& other) noexcept;  // move constructor
    __host__ Device_Buffer& operator=(Device_Buffer&& other) noexcept;  // move assignment operator
    /**
     * @brief get the pointer to the device side data
     * @note it points to the data at device side, so don't dereference it at host side
     */
    __host__ device_data_t* get_device_data() {
        return this->device_data_;
    }
    /**
     * @brief get the pointer to the host side data
     * @note the host side data only contain a copy of the device side data at initialization.
     * You can't find any calculation result inside.
     */
    __host__ device_data_t get_host_data() {
        return this->host_data_;
    }

    /**
     * @brief set the atom informations, like elements, coordinations
     * @note the elements should be within the range of elements specified when initialization
     * the number of atoms should also be within the maximum value specified when initialization
     * @param elements the array of elements, length: num_atoms
     * @param coords the array of coordinates, length: num_atoms * 3
     * @param length the number of atoms
     */
    __host__ void set_atoms(
        uint16_t* elements, 
        real_t coords[][3], 
        uint64_t length
    );

    /**
     * @brief set the cell matrix of the system to be calculated
     */
    __host__ void set_cell(real_t cell[3][3]);

    /**
     * @brief clear intermediate results produced during calculation
     * @note normally, you should always call it before start calculation
     */
    __host__ void clear();  // clear the device data

    /**
     * @brief divide the atoms into grid cells for neighbor list construction
     * @note you should always call it before start calculation
     */
    __host__ void construct_grids();  // construct grid cells
   private:
    device_data_t* device_data_;  // pointer to the device data
    device_data_t host_data_;
};  // Device_Buffer

#endif  // D3_BUFFER_CUH