#include "constants_include.h"
#include "d3_buffer.cuh"
#include "d3_internal.h"

/* global parameters */
/* cuda kernel launch parameters */
#define MAX_BLOCK_SIZE 256 // number of threads per block
#define MAX_LOCAL_NEIGHBORS 256 // the maximum neighbor of one thread
#define FLT_MAX 3.402823466e+38F // maximum float value, used to initialize distances
#define FLT_MIN 1.175494e-38F // minimum positive float value, used to avoid division by zero

/*
constants used in the simulation
these constants are from Grimme, S., Antony, J., Ehrlich, S. & Krieg, H. The
Journal of Chemical Physics 132, 154104 (2010).
*/
#define K1 16.0f
#define K2 1.33333f
#define K3 4.0f
#define ALPHA_N(N) (N + 8.0f)

__global__ void coordination_number_kernel(device_data_t *data);
__global__ void print_coordination_number_kernel(device_data_t *data);
__global__ void two_body_kernel(device_data_t *data);
__global__ void atm_kernel(device_data_t *data);
__global__ void atm_kernel_single(device_data_t *data);
__global__ void three_body_kernel(device_data_t *data);