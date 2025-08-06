#ifndef D3_INTERNAL_H
#define D3_INTERNAL_H

#define MAX_ELEMENTS 118

#define COMPUTE_SUCCESS 0b00
#define COMPUTE_NEIGHBOR_LIST_OVERFLOW 0b01

// macros for debugging
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", __func__, \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
        }                                                                \
        cudaError_t error = cudaGetLastError();                          \
        if (error != cudaSuccess) {                                      \
            printf("CUDA Error: %s\n", cudaGetErrorString(error));       \
        }                                                                \
    } while (0)

#ifdef DEBUG
#define debug(...) printf(__VA_ARGS__)
#define assert_(...) assert(__VA_ARGS__)
#else
#define debug(...)
#define assert_(...)
#endif

#endif  // D3_INTERNAL_H