#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>

// CUB library
#include <cub/cub.cuh>

// Thrust library
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#define FULL_MASK 0xffffffff

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

// Your optimized kernel
__global__ void yourReduce(float *d_input, float *d_output, int N) {
  __shared__ float tmp[32];

  float val0 = 0, val1 = 0, val2 = 0, val3 = 0;

  float4 *d_input4 = reinterpret_cast<float4 *>(d_input);
  int N4 = N / 4;

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

  for (; idx < N4; idx += stride) {
    float4 v = d_input4[idx];
    val0 += v.x + v.y + v.z + v.w;
  }

  float val = val0 + val1 + val2 + val3;

  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);

  if (threadIdx.x % 32 == 0)
    tmp[threadIdx.x / 32] = val;

  __syncthreads();

  if (threadIdx.x == 0) {
    float sum = 0;
    for (int i = 0; i < (blockDim.x / 32); i++)
      sum += tmp[i];
    d_output[blockIdx.x] = sum;
  }
}

int main() {
  const int N = 1 << 24; // 16,777,216 elements
  const size_t bytes = N * sizeof(float);
  const int WARMUP = 10;
  const int TIMING = 1000;

  std::cout << "=== Reduction Benchmark Comparison ===" << std::endl;
  std::cout << "N = " << N << " elements (" << bytes / (1024.0 * 1024.0)
            << " MB)" << std::endl;
  std::cout << std::endl;

  // Host data
  std::vector<float> h_input(N, 1.0f);
  float expected = static_cast<float>(N);

  // Device allocations
  float *d_input, *d_temp, *d_output;
  checkCudaError(cudaMalloc(&d_input, bytes), "d_input");
  checkCudaError(cudaMalloc(&d_temp, 4096 * sizeof(float)), "d_temp");
  checkCudaError(cudaMalloc(&d_output, sizeof(float)), "d_output");

  checkCudaError(
      cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice),
      "memcpy H2D");

  cudaEvent_t start, stop;
  checkCudaError(cudaEventCreate(&start), "event create");
  checkCudaError(cudaEventCreate(&stop), "event create");

  // ============ YOUR KERNEL ============
  std::cout << "--- Your Optimized Kernel ---" << std::endl;

  // Warmup
  for (int i = 0; i < WARMUP; i++) {
    cudaMemset(d_output, 0, sizeof(float));
    yourReduce<<<2048, 512>>>(d_input, d_temp, N);
    yourReduce<<<1, 256>>>(d_temp, d_output, 2048);
  }
  cudaDeviceSynchronize();

  // Timing
  float total_ms = 0;
  for (int i = 0; i < TIMING; i++) {
    cudaMemset(d_output, 0, sizeof(float));
    cudaEventRecord(start);
    yourReduce<<<2048, 512>>>(d_input, d_temp, N);
    yourReduce<<<1, 256>>>(d_temp, d_output, 2048);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    total_ms += ms;
  }

  float your_avg_us = (total_ms / TIMING) * 1000;
  float result;
  cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Average time: " << your_avg_us << " μs" << std::endl;
  std::cout << "Result: " << result << " (expected: " << expected << ")"
            << std::endl;
  std::cout << "Bandwidth: " << (bytes / 1e9) / (your_avg_us / 1e6) << " GB/s"
            << std::endl;
  std::cout << std::endl;

  // ============ CUB DeviceReduce ============
  std::cout << "--- CUB DeviceReduce::Sum ---" << std::endl;

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // Get required temp storage size
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output,
                         N);
  checkCudaError(cudaMalloc(&d_temp_storage, temp_storage_bytes),
                 "CUB temp storage");

  // Warmup
  for (int i = 0; i < WARMUP; i++) {
    cudaMemset(d_output, 0, sizeof(float));
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input,
                           d_output, N);
  }
  cudaDeviceSynchronize();

  // Timing
  total_ms = 0;
  for (int i = 0; i < TIMING; i++) {
    cudaMemset(d_output, 0, sizeof(float));
    cudaEventRecord(start);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input,
                           d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    total_ms += ms;
  }

  float cub_avg_us = (total_ms / TIMING) * 1000;
  cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Average time: " << cub_avg_us << " μs" << std::endl;
  std::cout << "Result: " << result << " (expected: " << expected << ")"
            << std::endl;
  std::cout << "Bandwidth: " << (bytes / 1e9) / (cub_avg_us / 1e6) << " GB/s"
            << std::endl;
  std::cout << std::endl;

  // ============ Thrust reduce ============
  std::cout << "--- Thrust::reduce ---" << std::endl;

  thrust::device_ptr<float> thrust_input(d_input);

  // Warmup
  for (int i = 0; i < WARMUP; i++) {
    float thrust_result = thrust::reduce(thrust_input, thrust_input + N, 0.0f,
                                         thrust::plus<float>());
  }
  cudaDeviceSynchronize();

  // Timing
  total_ms = 0;
  for (int i = 0; i < TIMING; i++) {
    cudaEventRecord(start);
    float thrust_result = thrust::reduce(thrust_input, thrust_input + N, 0.0f,
                                         thrust::plus<float>());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    total_ms += ms;
  }

  float thrust_avg_us = (total_ms / TIMING) * 1000;
  float thrust_result = thrust::reduce(thrust_input, thrust_input + N, 0.0f,
                                       thrust::plus<float>());

  std::cout << "Average time: " << thrust_avg_us << " μs" << std::endl;
  std::cout << "Result: " << thrust_result << " (expected: " << expected << ")"
            << std::endl;
  std::cout << "Bandwidth: " << (bytes / 1e9) / (thrust_avg_us / 1e6) << " GB/s"
            << std::endl;
  std::cout << std::endl;

  // ============ SUMMARY ============
  std::cout << "=== SUMMARY ===" << std::endl;
  std::cout << "Your kernel:  " << your_avg_us << " μs" << std::endl;
  std::cout << "CUB:          " << cub_avg_us << " μs ("
            << (your_avg_us / cub_avg_us) << "x vs yours)" << std::endl;
  std::cout << "Thrust:       " << thrust_avg_us << " μs ("
            << (your_avg_us / thrust_avg_us) << "x vs yours)" << std::endl;

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_temp);
  cudaFree(d_output);
  cudaFree(d_temp_storage);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}