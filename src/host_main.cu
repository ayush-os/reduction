#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
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

  std::cout << "--- Vector Reduction Sum Comparison Test (N=" << VECTOR_DIM
            << ") ---" << std::endl;
  std::cout << "Array Size: " << VECTOR_DIM << " elements ("
            << (double)bytes / (1024 * 1024 * 1024) << " GB)" << std::endl;

  std::vector<float> h_input(VECTOR_DIM);
  for (int i = 0; i < VECTOR_DIM; ++i) {
    h_input[i] = 1.0f;
  }

  float *d_input, *d_output;
  checkCudaError(cudaMalloc(&d_input, bytes), "d_input allocation");
  checkCudaError(cudaMalloc(&d_output, sizeof(float)), "d_output allocation");
  checkCudaError(
      cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice),
      "input copy H->D");

  thrust::device_ptr<float> d_ptr(d_input);

  const int threadsPerBlock = 256;
  const int numBlocks = (VECTOR_DIM + threadsPerBlock - 1) / threadsPerBlock;
  std::cout << "Grid: " << numBlocks << " blocks, " << threadsPerBlock
            << " threads/block." << std::endl;

  cudaEvent_t start, stop;
  checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
  checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

  std::cout << "\nWarming up the GPU and Caches (" << WARMUP_RUNS << " runs)..."
            << std::endl;
  for (int i = 0; i < WARMUP_RUNS; ++i) {
    baseline<<<numBlocks, threadsPerBlock>>>(d_input, d_output, VECTOR_DIM);
    thrust::reduce(d_ptr, d_ptr + VECTOR_DIM);
  }
  checkCudaError(cudaDeviceSynchronize(), "warm-up device sync");
  checkCudaError(cudaGetLastError(), "warm-up kernel launch");

  // #######################################################
  // ## A. TIMING CUSTOM KERNEL
  // #######################################################
  float total_ms_custom = 0;
  std::cout << "\nStarting Stable Timing Loop (CUSTOM KERNEL)..." << std::endl;

  for (int i = 0; i < TIMING_RUNS; ++i) {
    checkCudaError(cudaMemset(d_output, 0, sizeof(float)), "d_output reset");
    checkCudaError(cudaEventRecord(start), "cudaEventRecord start");

    baseline<<<numBlocks, threadsPerBlock>>>(d_input, d_output, VECTOR_DIM);

    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float milliseconds_i = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds_i, start, stop),
                   "cudaEventElapsedTime");
    total_ms_custom += milliseconds_i;
  }

  float h_custom_output = 0.0f;
  checkCudaError(cudaMemcpy(&h_custom_output, d_output, sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "custom output copy D->H");
  float average_ms_custom = total_ms_custom / TIMING_RUNS;

  // #######################################################
  // ## B. TIMING THRUST
  // #######################################################
  float total_ms_thrust = 0;
  std::cout << "\nStarting Stable Timing Loop (THRUST::REDUCE)..." << std::endl;

  for (int i = 0; i < TIMING_RUNS; ++i) {
    checkCudaError(cudaEventRecord(start), "cudaEventRecord start");

    // Thrust Reduction
    float h_thrust_result = thrust::reduce(d_ptr, d_ptr + VECTOR_DIM);

    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float milliseconds_i = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds_i, start, stop),
                   "cudaEventElapsedTime");
    total_ms_thrust += milliseconds_i;
  }

  float h_thrust_final_result = thrust::reduce(d_ptr, d_ptr + VECTOR_DIM);
  float average_ms_thrust = total_ms_thrust / TIMING_RUNS;

  float expected_value = std::accumulate(h_input.begin(), h_input.end(), 0.0f);

  std::cout << "\n--- Timing Results ---" << std::endl;
  std::cout << "Expected Sum: " << expected_value << std::endl;

  std::cout << "\n**CUSTOM KERNEL**" << std::endl;
  std::cout << "Avg. Execution Time: " << average_ms_custom << " ms"
            << std::endl;
  if (std::abs(h_custom_output - expected_value) < 1e-5 * expected_value) {
    std::cout << "Verification Check: PASSED (Result: " << h_custom_output
              << ")" << std::endl;
  } else {
    std::cout << "Verification Check: **FAILED** (Result: " << h_custom_output
              << ")" << std::endl;
    std::cout
        << "NOTE: If this fails, your kernel is likely not implemented "
           "correctly (e.g., using a slow or race-prone method like atomicAdd)."
        << std::endl;
  }

  std::cout << "\n**THRUST::REDUCE**" << std::endl;
  std::cout << "Avg. Execution Time: " << average_ms_thrust << " ms"
            << std::endl;
  if (std::abs(h_thrust_final_result - expected_value) <
      1e-5 * expected_value) {
    std::cout << "Verification Check: PASSED (Result: " << h_thrust_final_result
              << ")" << std::endl;
  } else {
    std::cout << "Verification Check: **FAILED** (Result: "
              << h_thrust_final_result << ")" << std::endl;
  }

  checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
  checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");
  checkCudaError(cudaFree(d_input), "cudaFree d_input");
  checkCudaError(cudaFree(d_output), "cudaFree d_output");

  return 0;
}