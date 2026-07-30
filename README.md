# CUDA Reduction: 3.6 GB/s → 1.12 TB/s

*README generated with [Claude Code](https://claude.com/claude-code)*

A sum-reduction over 16.7M floats (64 MB) on an A100, optimized step by step from a one-line `atomicAdd` baseline to a kernel within 0.7% of NVIDIA's own CUB library — including two honest regressions along the way that turned out to be the most useful data points in the whole log.

**Result:** 59.7 µs, **1.12 TB/s — 72% of the A100's theoretical peak HBM bandwidth** — beating Thrust's `reduce` by 1.46× and landing within measurement noise of CUB's `DeviceReduce::Sum`.

## The progression

| Step | Technique | Time | Bandwidth | vs. previous |
|---|---|---|---|---|
| 0 | Baseline — every thread `atomicAdd`s directly to one global output | 37.29 ms | 3.6 GB/s | — |
| 1 | Per-block shared-memory reduction, one `atomicAdd` per block instead of per thread | 179.8 µs | 374 GB/s | **207×** |
| 2 | Textbook logarithmic (tree) reduction | 197.9 µs | 340 GB/s | *regression* |
| 3 | Warp-aggregated reduction (`__shfl_down_sync`, no shared-memory tree) | 181 µs | 372 GB/s | *still a regression vs. step 1* |
| 4 | Multi-kernel reduction — removes the cross-block atomic entirely | 108.6 µs | 620 GB/s | **1.67×** |
| 5 | Grid-stride loop — more work per thread, hides memory latency | 99.7 µs | 673 GB/s | 1.09× |
| 6 | Launch-config tuning (block/thread count sweep) | 78.8 µs | — | 1.27× |
| 7 | Vectorized `float4` loads | **59.7 µs** | **1.12 TB/s** | 1.32× |

### Steps 2–3 were regressions, and that's the important part

The standard "optimize a reduction" playbook says: replace the naive per-thread atomic with a shared-memory tree, then replace the tree with a warp-shuffle reduction. Both are real optimizations *in isolation* — and both were measured slower than the plain step-1 shared-memory version here. The reason wasn't the per-block reduction strategy at all: every block still finished by doing one `atomicAdd` into a single global output, and with 65,536 blocks, *that* remaining atomic — not the intra-block algorithm — was the dominant cost the whole time. Chasing the textbook per-block optimization was optimizing the wrong 5% of the problem.

Step 4 is what actually mattered: restructure the problem so the cross-block atomic never happens at all, by making the reduction recursive instead — one kernel reduces 16.7M elements down to a 65,536-element intermediate array, a second kernel reduces that to 256, a third to 1. Removing the contention point outright bought a bigger win (1.67×) than either "smarter" per-block algorithm had.

### Final kernel

Step 5 onward collapsed back to two kernel launches (grid-stride loops mean each thread block can cover the whole input regardless of block count, so the second reduction pass folds into the same launch shape). The final per-block kernel combines four independent accumulators (ILP), `float4` vectorized loads, and a warp-shuffle-then-shared-memory two-level reduction:

```c
__global__ void reduce(float *d_input, float *d_output, int N) {
  __shared__ float tmp[32];
  float val0 = 0, val1 = 0, val2 = 0, val3 = 0;

  float4 *d_input4 = reinterpret_cast<float4 *>(d_input);
  int N4 = N / 4;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  // 4 independent float4 loads per iteration: vectorized traffic + ILP together
  for (; idx < N4 - 3; idx += stride * 4) {
    float4 v0 = d_input4[idx],           v1 = d_input4[idx + stride];
    float4 v2 = d_input4[idx + stride*2], v3 = d_input4[idx + stride*3];
    val0 += v0.x + v0.y + v0.z + v0.w;   val1 += v1.x + v1.y + v1.z + v1.w;
    val2 += v2.x + v2.y + v2.z + v2.w;   val3 += v3.x + v3.y + v3.z + v3.w;
  }
  for (; idx < N4; idx += stride) {           // remainder
    float4 v = d_input4[idx];
    val0 += v.x + v.y + v.z + v.w;
  }

  float val = val0 + val1 + val2 + val3;
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);   // warp reduction, no smem

  if (threadIdx.x % warpSize == 0) tmp[threadIdx.x / warpSize] = val;
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0;
    for (int i = 0; i < blockDim.x / warpSize; i++) sum += tmp[i];
    d_output[blockIdx.x] = sum;
  }
}
```

Block/thread tuning (step 6) swept launch configurations directly rather than guessing — 2048 blocks × 512 threads won at 78.8 µs, beating both smaller and larger configurations tried on either axis.

## Comparison against NVIDIA's own libraries

Same problem, same GPU, run back to back:

```
--- My Kernel ---            --- CUB DeviceReduce::Sum ---     --- Thrust::reduce ---
59.6224 µs                   59.1923 µs                        86.8957 µs
1125.56 GB/s                 1133.74 GB/s                      772.29 GB/s
```

CUB — a hardware-vendor-tuned library — is faster by well under 1%. Thrust's general-purpose `reduce` is 1.46× slower than both. A hand-written kernel landing within measurement noise of CUB is the actual headline result of this whole exercise: getting there took removing exactly one architectural bottleneck (the cross-block atomic) and three genuinely small tuning passes (grid-stride reuse, launch config, vectorized loads) on top of it — not any single clever trick.
