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

__global__ void smem(float *d_input, float *d_output, int N) {
  __shared__ int tmp[threadsPerBlock / warpSize];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  float val = d_input[i];

  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);

  if (threadIdx.x % warpSize == 0)
    tmp[threadIdx.x / warpSize] = val;

  __syncthreads();

  if (threadIdx.x == 0) {
    int sum = 0;
    for (int i = 0; i < (threadsPerBlock / warpSize); i++)
      sum += tmp[i];
    atomicAdd(d_output, sum);
  }
}