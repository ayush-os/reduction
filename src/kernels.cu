#include <math.h>

#define FULL_MASK 0xffffffff
const int warpSize = 32;
const int threadsPerBlock = 256;
const int numBlocks = 65536;

__global__ void baseline(float *d_input, float *d_output, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  atomicAdd(d_output, d_input[i]);
}

__global__ void reduce(float *d_input, float *d_output, int N) {
  __shared__ float tmp[threadsPerBlock / warpSize];

  float val = 0;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N;
       idx += gridDim.x * blockDim.x) {
    val += d_input[idx];
  }

  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);

  if (threadIdx.x % warpSize == 0)
    tmp[threadIdx.x / warpSize] = val;

  __syncthreads();

  if (threadIdx.x == 0) {
    float sum = 0;
    for (int i = 0; i < (threadsPerBlock / warpSize); i++)
      sum += tmp[i];
    d_output[blockIdx.x] = sum;
  }
}