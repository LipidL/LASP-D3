#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
int main()
{
    for(uint64_t n = 5; n < 50000; ++n) {
        uint64_t total = n*(n-1)/2;
        for(uint64_t k = 0; k < total; ++k) {
            uint64_t i = floorf((1.0f + sqrtf(1.0f + 8.0f*k))/2.0f);
            if (i*(i-1) > 2*k) {
                --i;
            }
            uint64_t j = k - i*(i-1)/2;
            assert(i != j);
            assert(i < n && j < n);
            assert(i*(i-1)/2 + j == k);
        }
    }
}