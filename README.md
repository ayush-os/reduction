# reduction

### Step 0: Baseline - 37.2898 ms execution time

8 bytes * 16,777,216 elems = 134,217,728 bytes / 0.0372898 seconds = 3.6 gb/s - bandwidth -- MAYBE ACCURATE

### Step 1: Smem - 179.789 us - 207.4x less from step 0

8 bytes * 16,777,216 elems = 134,217,728 bytes / 0.000179789 seconds = 746 gb/s - bandwidth -- INACCURATE

### Step 2: Logarithmic algorithm

### Step 3: warp level
181
### step 4: multi-kernel

108.586 us now which is 134,217,728 bytes / 0.000108586 s => 1.2360500249 TB/s -- INACCURATE

### step 5: increase arithmetic intensity (each thread does more work)

now that i'm this close to the theoretical memory bandwidth i should increase arithmetic intensity to hide the memory access latency

now i'm at 99.69us

134,217,728 bytes / 0.00009969 s = 1.346350968T B/s

1025 global writes, 16,777,216 global reads

67,112,964 bytes / 0.00009969 s = 673 gB/s

###  step 6 tuning

512 blocks × 256 threads  ← 99.5283 μs
1024 blocks × 256 threads ← 99.69 μs
2048 blocks × 256 threads ← 85.3205 μs
1024 blocks × 512 threads ← 83.3416 μs
512 blocks × 512 threads ← 99.039 μs

1536 blocks × 512 threads ← 79.292  μs
2048 blocks × 512 threads ← 78.8283 μs
3072 blocks × 512 threads ← 81.308 μs

going with 2048 blocks × 512 threads so we're now at 78.83 us

### step 7 vectorized loads float4

59.6931 us


### step 8 compare with thrust and CUB

(main) root@C.28568743:/workspace/reduction/build$ ./bin/reduction 
=== Reduction Benchmark Comparison ===
N = 16777216 elements (64 MB)

--- Your Optimized Kernel ---
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
(main) root@C.28568743:/workspace/reduction/build$ 


holy cow hahahahah