# reduction

### Step 0: Baseline - 37.2898 ms execution time

8 bytes * 16,777,216 elems = 134,217,728 bytes / 0.0372898 seconds = 3.6 gb/s - bandwidth

### Step 1: Smem - 179.789 us - 207.4x less from step 0

logic behind this optimization - want to reduce the massive contention of 16 million threads doing an atomicAdd to 1 global var
Now, we have 1 thread per block doing all the work and the others just load. the problem with this approach is that only 65536 / 16777216 threads are doing computation and there is still massive contention from the atomicAdd where 65536 threads are competing to do d_output += sum.

There is also another massive serialization point where we have to wait for thread0 to do all the addition for that block.

(4 bytes * 16777216 elems) + (4 bytes * 65,536 elems (numBlocks)) = 67,371,008 bytes / 0.000179789 seconds = 374 gB/s - bandwidth ~ 104x improvement from step 1

### Step 2: Logarithmic algorithm - 197.926 us - slowdown from step 1

67,371,008 bytes / 0.000197926 seconds ~ 340 gB/s bandwidth ~ slowdown from step 1

the atomicAdd where thread0 from each block meaning 65536 threads are competing to add to d_output is still killing me
the constant __syncthreads() inside the for loop is what probably makes this slower than step 1 plus the fact that as we get later into the iterations of the logarithic reduction more and more threads are just sitting idle

### Step 3: warp aggregated reduction - 181 us - still a slowdown from step 1

67,371,008 bytes / 0.000181 s ~ 372 gB/s bandwidth ~ slowdown from step 1

hitting an amdahl's law point here where the warp level reduction itself is probably incredibly fast but the 65536 threads doing an atomic add to gmem is just murdering me

### step 4: multi-kernel - 108.586 us - 40% reduction from step 3

108.586 us now which is 67,371,008 bytes / 0.000108586 s => 620 gB/s -> 1.67x better than step 3

realizing that we need to get rid of that atomic add, the options we have are multi-kernel or grid stride loop.

The nice thing about multi-kernel is it makes it a recursive problem, where first kernel reduces 16million into an intermediate array of 65k, and then second kernel reduces from 65k to 256, and then 3rd kernel reduces from 256 to 1. This gets rid of the contention

### step 5: increase arithmetic intensity (each thread does more work) - 99.69us - 8.2% faster from step 4

now that i'm this close to the theoretical memory bandwidth i should increase arithmetic intensity to hide the memory access latency

67,112,964 bytes / 0.00009969 s = 673 gB/s

using a grid-stride loop where threads keep getting the next elem by the stride gridDim.x * blockDim.x, accumulating multiple elems into their register then doing the reduction.

This increases the amount of compute each thread does to hide latency access, and it also allowed me to go from 3 kernels to 2 kernels

###  step 6 tuning - 78.83 us - 20.9% faster from step 5

512 blocks × 256 threads  ← 99.5283 μs
1024 blocks × 256 threads ← 99.69 μs
2048 blocks × 256 threads ← 85.3205 μs
1024 blocks × 512 threads ← 83.3416 μs
512 blocks × 512 threads ← 99.039 μs

1536 blocks × 512 threads ← 79.292  μs
2048 blocks × 512 threads ← 78.8283 μs
3072 blocks × 512 threads ← 81.308 μs

going with 2048 blocks × 512 threads so we're now at 78.83 us

### step 7 vectorized loads float4 - 59.6931 us - 24.3% faster than step 6

self explanatory - load 4 floats at once instead of 1 float and handle any remainder

final bandwidth utilization - 67,112,964 bytes / 0.0000596931 s -> 1.124 TB/s where the theoretical max is 1.56 TB/s, meaning we got 72% utilization

### intermediate step of more ILP exposure but failed - basically the same perf as step 7

### step 8 compare with thrust and CUB - holy cow hahaha

N = 16777216 elements (64 MB)

--- My Kernel ---
Average time: 59.6224 μs
Result: 1.67772e+07 (expected: 1.67772e+07)
Bandwidth: 1125.56 GB/s

--- CUB DeviceReduce::Sum ---
Average time: 59.1923 μs
Result: 1.67772e+07 (expected: 1.67772e+07)
Bandwidth: 1133.74 GB/s

--- Thrust::reduce ---
Average time: 86.8957 μs
Result: 1.67772e+07 (expected: 1.67772e+07)
Bandwidth: 772.292 GB/s

=== SUMMARY ===
Your kernel:  59.6224 μs
CUB:          59.1923 μs (1.00727x vs yours)
Thrust:       86.8957 μs (0.686138x vs yours)