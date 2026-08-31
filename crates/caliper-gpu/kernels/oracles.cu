// Oracle kernels for caliper's self-checks.
//
// Each kernel's true behaviour is knowable from first principles, so measuring
// it pins one caliper measurement path without caliper having to trust itself.
// The analytic expectations and pass/fail checks live in
// `caliper-core/src/oracles.rs`; this file is the device side.
//
// Build: compiled by the `caliper-gpu` `cuda` feature on a CUDA host (nvcc /
// the `cc` crate). The `launch_*` wrappers are the ABI the Rust launcher calls.
//
// Reference: docs/plan.md, Appendix A.

#include <cstdint>
#include <cuda_runtime.h>

extern "C" {

// --- O1: calibrated duration -------------------------------------------------
// Spin until `target_cycles` of the (locked) SM clock have elapsed. With clocks
// locked, cycles <-> nanoseconds is exact, so p50 per launch == target_ns/1000
// microseconds. host: target_cycles = target_ns * sm_clock_hz / 1e9.
__global__ void o1_busy(unsigned long long target_cycles) {
    unsigned long long t0 = clock64();
    while (clock64() - t0 < target_cycles) {
        __threadfence_block();  // keep the loop from being optimised away
    }
}

void launch_o1_busy(unsigned long long target_cycles, cudaStream_t stream) {
    o1_busy<<<1, 1, 0, stream>>>(target_cycles);
}

// --- O2: streaming triad ---------------------------------------------------
// a[i] = b[i] + s * c[i]. Bytes moved = 3 * n * sizeof(float) (read b, read c,
// write a). achieved_gbps = 3 * n * sizeof(float) / p50_seconds.
__global__ void o2_triad(float* __restrict__ a, const float* __restrict__ b,
                         const float* __restrict__ c, float s, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        a[i] = b[i] + s * c[i];
    }
}

void launch_o2_triad(float* a, const float* b, const float* c, float s, size_t n,
                     int grid, int block, cudaStream_t stream) {
    o2_triad<<<grid, block, 0, stream>>>(a, b, c, s, n);
}

// --- O3: register-resident FMA peak -----------------------------------------
// No memory traffic; 4 independent FMA lanes (ILP). flops = 2 * 4 * iters *
// total_threads. Should hit >= 90% of the documented FP32 FMA peak and classify
// as compute-bound.
__global__ void o3_fma_peak(float* __restrict__ sink, int iters) {
    float x0 = threadIdx.x, x1 = 1.1f, x2 = 2.2f, x3 = 3.3f;
    const float a = 0.9f, b = 1.0001f;
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        x0 = x0 * a + b;
        x1 = x1 * a + b;
        x2 = x2 * a + b;
        x3 = x3 * a + b;
    }
    if (x0 == -1.0f) sink[threadIdx.x] = x0 + x1 + x2 + x3;  // keep it live
}

void launch_o3_fma_peak(float* sink, int iters, int grid, int block,
                        cudaStream_t stream) {
    o3_fma_peak<<<grid, block, 0, stream>>>(sink, iters);
}

// --- O4: single instruction ------------------------------------------------
// Effectively one instruction; p50 per launch == pure launch + teardown. Under
// CUDA-graph replay the per-launch cost drops below 1 us.
__global__ void o4_one_op(int* p) {
    if (threadIdx.x == 0xffff) p[0] = 1;
}

void launch_o4_one_op(int* p, cudaStream_t stream) {
    o4_one_op<<<1, 1, 0, stream>>>(p);
}

// --- O6: throttle bait -----------------------------------------------------
// Sustained high-power FMA across many blocks to trip a lowered power cap. Reuse
// the O3 body with a large iteration count and a full grid.
void launch_o6_throttle_bait(float* sink, int iters, int grid, int block,
                             cudaStream_t stream) {
    o3_fma_peak<<<grid, block, 0, stream>>>(sink, iters);
}

}  // extern "C"
