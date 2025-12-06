#include <math.h>

__global__ void baseline(float* d_input, float* d_output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    atomicAdd(d_output, d_input[i]);
}