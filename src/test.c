#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "d3.h"

typedef struct {
    int thread_id;
    real_t energy;
    real_t *force;
    real_t *stress;
} ThreadData;

void* compute_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int thread_id = data->thread_id;
    
    printf("Thread %d starting computation...\n", thread_id);
    
    // Set up atoms coordinates
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
    uint16_t elements[10] = {6, 6, 6, 8, 1, 1, 1, 1, 1, 1};
    real_t angstrom_to_bohr = 1/0.52917726f;
    
    // Convert coordinates to bohr
    for(uint64_t i = 0; i < 10; ++i) {
        for(int j = 0; j < 3; ++j) {
            atoms[i][j] *= angstrom_to_bohr;
        }
    }
    
    // Initialize parameters (testing thread safety)
    init_params();
    
    // Set up cell parameters
    real_t cell[3][3] = {
        {200.0f, 0.0f, 0.0f},
        {0.0f, 20.0f, 0.0f},
        {0.0f, 0.0f, 20.0f}
    };
    
    real_t CN_cutoff_radius = 46.4758f;
    real_t cutoff_radius = 46.4758f;
    
    // Compute dispersion energy
    compute_dispersion_energy(atoms, elements, 10, cell, cutoff_radius, 
                             CN_cutoff_radius, 10000, &data->energy, data->force, data->stress);
    
    printf("Thread %d completed: energy = %f eV\n", thread_id, data->energy);
    
    return NULL;
}

int main() {
    const int num_threads = 500;
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    
    printf("Starting thread safety test with %d threads...\n", num_threads);
    
    // Create threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].force = (real_t*)malloc(sizeof(real_t) * 10 * 3);
        thread_data[i].stress = (real_t*)malloc(sizeof(real_t) * 9);
        
        if (pthread_create(&threads[i], NULL, compute_thread, &thread_data[i]) != 0) {
            perror("Failed to create thread");
            return 1;
        }
    }
    
    // Join threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Compare results
    printf("\nResults comparison:\n");
    for (int i = 0; i < num_threads; i++) {
        printf("Thread %d: energy = %.10f eV\n", i, thread_data[i].energy);
    }
    
    // Free resources
    for (int i = 0; i < num_threads; i++) {
        free(thread_data[i].force);
        free(thread_data[i].stress);
    }
    
    return 0;
}