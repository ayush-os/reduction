# reduction

### Step 0: Baseline - 37.2898 ms execution time

8 bytes * 16,777,216 elems = 134,217,728 bytes / 0.0372898 seconds = 3.6 gb/s - bandwidth

### Step 1: Smem - 179.789 us - 207.4x less from step 0

8 bytes * 16,777,216 elems = 134,217,728 bytes / 0.000179789 seconds = 746 gb/s - bandwidth

### Step 2: Logarithmic algorithm

### Step 3: warp level
181
### step 4: multi-kernel

108.586 us now which is 134,217,728 bytes / 0.000108586 s => 1.2360500249 TB/s

# step 5: increase arithmetic intensity (each thread does more work)

now that i'm this close to the theoretical memory bandwidth i should increase arithmetic intensity to hide the memory access latency