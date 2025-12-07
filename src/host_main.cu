#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>

__global__ void baseline(float *d_input, float *d_total_sum, int N);
__global__ void reduce(float *d_input, float *d_total_sum, int N);

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
  int N = VECTOR_DIM;
  const size_t bytes = VECTOR_DIM * sizeof(float);

  const int WARMUP_RUNS = 10;
  const int TIMING_RUNS = 1000;

  std::cout << "--- Vector Reduction Sum Stable Timing Test ---" << std::endl;
  std::cout << "Vector Dim N: " << VECTOR_DIM << std::endl;
  std::cout << "Total Array Size: " << VECTOR_DIM << " elements ("
            << (double)bytes / (1024 * 1024 * 1024) << " GB)" << std::endl;

  std::vector<float> h_input(VECTOR_DIM);
  float h_output_val = 0.0f;

  for (int i = 0; i < VECTOR_DIM; ++i) {
    h_input[i] = 1.0f;
  }

  float *d_input, *d_temp, *d_output;
  checkCudaError(cudaMalloc(&d_input, bytes), "d_input allocation");
  checkCudaError(cudaMalloc(&d_temp, 1024 * sizeof(float)),
                 "d_temp allocation");
  checkCudaError(cudaMalloc(&d_output, sizeof(float)), "d_output allocation");
  checkCudaError(cudaMemset(d_output, 0, sizeof(float)), "d_output memset");

  float *in = d_input;
  float *out = d_temp;

  checkCudaError(
      cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice),
      "input copy H->D");

  const int threadsPerBlock = 256;

  cudaEvent_t start, stop;
  checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
  checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

  std::cout << "Warming up the GPU and Caches (" << WARMUP_RUNS << " runs)..."
            << std::endl;
  for (int i = 0; i < WARMUP_RUNS; ++i) {
    reduce<<<1024, 256>>>(d_input, d_temp, N);
    reduce<<<1, 256>>>(d_temp, d_output, 1024);
  }
  checkCudaError(cudaDeviceSynchronize(), "warm-up device sync");
  checkCudaError(cudaGetLastError(), "warm-up kernel launch");

  float total_milliseconds = 0;
  std::cout << "Starting Stable Timing Loop (" << TIMING_RUNS << " runs)..."
            << std::endl;

  for (int i = 0; i < TIMING_RUNS; ++i) {
    checkCudaError(cudaMemset(d_output, 0, sizeof(float)), "d_output reset");
    checkCudaError(cudaEventRecord(start), "cudaEventRecord start");

    reduce<<<1536, 512>>>(d_input, d_temp, N);
    reduce<<<1, 256>>>(d_temp, d_output, 1536);

    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float milliseconds_i = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds_i, start, stop),
                   "cudaEventElapsedTime");
    total_milliseconds += milliseconds_i;
  }

  float average_milliseconds = total_milliseconds / TIMING_RUNS;

  checkCudaError(cudaMemcpy(&h_output_val, d_output, sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "output copy D->H");

  float expected_value = std::accumulate(h_input.begin(), h_input.end(), 0.0f);

  std::cout << "\n--- Timing Results ---" << std::endl;
  std::cout << "Total execution time for " << TIMING_RUNS
            << " stable runs: " << total_milliseconds << " ms" << std::endl;
  std::cout << "**Average kernel execution time:** "
            << average_milliseconds * 1000 << " us" << std::endl;

  if (std::abs(h_output_val - expected_value) < 1e-5 * expected_value) {
    std::cout << "\nVerification Check: **PASSED**" << std::endl;
  } else {
    std::cout << "\nVerification Check: **FAILED**" << std::endl;
  }
  std::cout << "Kernel Result: " << h_output_val
            << " , Expected Value: " << expected_value << std::endl;

  checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
  checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");
  checkCudaError(cudaFree(d_input), "cudaFree d_input");
  checkCudaError(cudaFree(d_output), "cudaFree d_output");

  return 0;
}