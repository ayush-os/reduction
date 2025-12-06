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
  __shared__ int tmp2[threadsPerBlock];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  tmp[threadIdx.x] = d_input[i];

  __syncthreads();

  if (threadIdx.x >= 128)
    return;

  tmp2[threadIdx.x] = tmp[threadIdx.x * 2] + tmp[(threadIdx.x * 2) + 1];
  __syncthreads();

  if (threadIdx.x >= 64)
    return;

  tmp[threadIdx.x] = tmp2[threadIdx.x * 2] + tmp2[(threadIdx.x * 2) + 1];
  __syncthreads();

  if (threadIdx.x >= 32)
    return;

  tmp2[threadIdx.x] = tmp[threadIdx.x * 2] + tmp[(threadIdx.x * 2) + 1];
  __syncthreads();

  if (threadIdx.x >= 16)
    return;

  tmp[threadIdx.x] = tmp2[threadIdx.x * 2] + tmp2[(threadIdx.x * 2) + 1];
  __syncthreads();

  if (threadIdx.x >= 8)
    return;

  tmp2[threadIdx.x] = tmp[threadIdx.x * 2] + tmp[(threadIdx.x * 2) + 1];
  __syncthreads();

  if (threadIdx.x >= 4)
    return;

  tmp[threadIdx.x] = tmp2[threadIdx.x * 2] + tmp2[(threadIdx.x * 2) + 1];
  __syncthreads();

  if (threadIdx.x >= 2)
    return;

  tmp2[threadIdx.x] = tmp[threadIdx.x * 2] + tmp[(threadIdx.x * 2) + 1];
  __syncthreads();

  if (threadIdx.x >= 1)
    return;

  tmp[threadIdx.x] = tmp2[threadIdx.x * 2] + tmp2[(threadIdx.x * 2) + 1];

  atomicAdd(d_output, tmp[threadIdx.x]);
}