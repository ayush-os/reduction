#include <math.h>

const int threadsPerBlock = 256;
const int numBlocks = 65536;

__global__ void baseline(float *d_input, float *d_output, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  atomicAdd(d_output, d_input[i]);
}

__global__ void smem(float *d_input, float *d_output, int N) {
  __shared__ int tmp[threadsPerBlock];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  tmp[threadIdx.x] = d_input[i];

  __syncthreads();

  for (int stride = threadsPerBlock / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      tmp[threadIdx.x] += tmp[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0)
    atomicAdd(d_output, tmp[threadIdx.x]);
}