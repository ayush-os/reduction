#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void baseline(float *d_input, float *d_total_sum, int N);

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main() {
  const int SHIFT = 24;
  const int VECTOR_DIM = 1 << SHIFT; // 16,777,216 elements
  const size_t bytes = VECTOR_DIM * sizeof(float);

  const int WARMUP_RUNS = 10;
  const int TIMING_RUNS = 1000;

  std::cout << "--- Vector Reduction Sum Stable Timing Test (N x N) ---"
            << std::endl;
  std::cout << "Vector Dim N: " << VECTOR_DIM << std::endl;
  std::cout << "Total Array Size: " << VECTOR_DIM << " elements ("
            << (double)bytes / (1024 * 1024 * 1024) << " GB)" << std::endl;

  std::vector<float> h_input(VECTOR_DIM);
  std::vector<float> h_output(1);

  for (int i = 0; i < VECTOR_DIM; ++i) {
    h_input[i] = (float)i;
  }

  float *d_input, *d_output;
  checkCudaError(cudaMalloc(&d_input, bytes), "d_input allocation");
  checkCudaError(cudaMalloc(&d_output, sizeof(float)), "d_output allocation");

  checkCudaError(
      cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice),
      "input copy H->D");

  const int threadsPerBlock = 256;
  const int numBlocks = VECTOR_DIM / threadsPerBlock;

  std::cout << "Grid: " << numBlocks << " blocks, " << threadsPerBlock
            << " threads/block." << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::cout << "Warming up the GPU and Caches (" << WARMUP_RUNS << " runs)..."
            << std::endl;
  for (int i = 0; i < WARMUP_RUNS; ++i) {
    baseline<<<numBlocks, threadsPerBlock>>>(d_output, d_input, VECTOR_DIM);
  }
  cudaDeviceSynchronize();
  checkCudaError(cudaGetLastError(), "warm-up kernel launch");

  float total_milliseconds = 0;
  std::cout << "Starting Stable Timing Loop (" << TIMING_RUNS << " runs)..."
            << std::endl;

  for (int i = 0; i < TIMING_RUNS; ++i) {
    cudaEventRecord(start);

    baseline<<<numBlocks, threadsPerBlock>>>(d_output, d_input, VECTOR_DIM);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds_i = 0;
    cudaEventElapsedTime(&milliseconds_i, start, stop);
    total_milliseconds += milliseconds_i;
  }

  float average_milliseconds = total_milliseconds / TIMING_RUNS;

  std::cout << "\n--- Timing Results ---" << std::endl;
  std::cout << "Total execution time for " << TIMING_RUNS
            << " stable runs: " << total_milliseconds << " ms" << std::endl;
  std::cout << "**Average kernel execution time:** "
            << average_milliseconds * 1000.0f << " us" << std::endl;

  checkCudaError(checkCudaError(h_output.data(), d_output, sizeof(float),
                                cudaMemcpyDeviceToHost),
                 "output copy D->H");

  int expected_value = thrust::reduce(d_input.begin(), d_input.end());

  if (std::abs(h_output[0] - expected_value) < 1e-5) {
    std::cout << "\nVerification Check: **PASSED**" << std::endl;
  } else {
    std::cout << "\nVerification Check: **FAILED**" << std::endl;
  }
  std::cout << "h_output[0]: " << h_output[0]
            << " , expected_value: " << expected_value << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}