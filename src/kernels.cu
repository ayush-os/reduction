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
  __shared__ float tmp[32];

  // Use 4 independent accumulators for ILP
  float val0 = 0, val1 = 0, val2 = 0, val3 = 0;

  float4 *d_input4 = reinterpret_cast<float4 *>(d_input);
  int N4 = N / 4;

  // Process 4 float4s per iteration (16 floats total)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (; idx < N4 - 3; idx += stride * 4) {
    float4 v0 = d_input4[idx];
    float4 v1 = d_input4[idx + stride];
    float4 v2 = d_input4[idx + stride * 2];
    float4 v3 = d_input4[idx + stride * 3];

    val0 += v0.x + v0.y + v0.z + v0.w;
    val1 += v1.x + v1.y + v1.z + v1.w;
    val2 += v2.x + v2.y + v2.z + v2.w;
    val3 += v3.x + v3.y + v3.z + v3.w;
  }

  // Handle remaining float4 chunks
  for (; idx < N4; idx += stride) {
    float4 v = d_input4[idx];
    val0 += v.x + v.y + v.z + v.w;
  }

  // Combine accumulators
  float val = val0 + val1 + val2 + val3;

  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);

  if (threadIdx.x % warpSize == 0)
    tmp[threadIdx.x / warpSize] = val;

  __syncthreads();

  if (threadIdx.x == 0) {
    float sum = 0;
    for (int i = 0; i < (blockDim.x / warpSize); i++)
      sum += tmp[i];
    d_output[blockIdx.x] = sum;
  }
}