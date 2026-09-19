// Query hardware execution-model limits for the doc.
// ASCII only on purpose: avoids the Windows codepage pitfalls for nvcc.
//   nvcc -O3 -arch=sm_120 device_query.cu -o device_query.exe

#include <cuda_runtime.h>
#include <cstdio>

#define CK(call)                                                              \
    do {                                                                      \
        cudaError_t e = (call);                                               \
        if (e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %d: %s\n", __LINE__,                  \
                    cudaGetErrorString(e));                                   \
            return 1;                                                         \
        }                                                                     \
    } while (0)

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));

    printf("== device ==\n");
    printf("  name                          %s\n", p.name);
    printf("  compute capability            sm_%d%d\n", p.major, p.minor);
    printf("  SMs                           %d\n", p.multiProcessorCount);

    printf("\n== per-block limits ==\n");
    printf("  warpSize                      %d\n", p.warpSize);
    printf("  maxThreadsPerBlock            %d\n", p.maxThreadsPerBlock);
    printf("  maxBlocksPerMultiprocessor    %d\n", p.maxBlocksPerMultiProcessor);

    printf("\n== per-SM limits ==\n");
    printf("  maxThreadsPerMultiProcessor   %d\n", p.maxThreadsPerMultiProcessor);
    printf("  regsPerMultiprocessor         %d\n", p.regsPerMultiprocessor);
    printf("  sharedMemPerMultiprocessor    %zu bytes\n",
           (size_t)p.sharedMemPerMultiprocessor);
    printf("  sharedMemPerBlockOptin        %zu bytes\n",
           (size_t)p.sharedMemPerBlockOptin);

    printf("\n== derived ==\n");
    int warps = p.maxThreadsPerMultiProcessor / p.warpSize;
    printf("  warps resident per SM         %d\n", warps);
    printf("  threads resident per SM       %d\n",
           p.maxThreadsPerMultiProcessor);
    printf("  threads resident on the GPU   %d\n",
           p.multiProcessorCount * p.maxThreadsPerMultiProcessor);
    printf("  max regs per thread at 100%%   %d\n",
           p.regsPerMultiprocessor / p.maxThreadsPerMultiProcessor);

    printf("\n== clocks / memory ==\n");
    printf("  clockRate (max SM)            %d kHz (%.0f MHz)\n", p.clockRate,
           p.clockRate / 1000.0);
    printf("  memoryClockRate               %d kHz\n", p.memoryClockRate);
    printf("  memoryBusWidth                %d bit\n", p.memoryBusWidth);
    printf("  l2CacheSize                   %d bytes\n", p.l2CacheSize);
    double peak =
        2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;
    printf("  theoretical peak BW           %.0f GB/s\n", peak);

    // How many waves does the N=25M vector-add need?
    printf("\n== waves for N = 25,000,000 (blockDim = 256) ==\n");
    long long resident = (long long)p.multiProcessorCount *
                         p.maxThreadsPerMultiProcessor;
    long long n4 = 25000000LL / 4;
    int blocks_full = (int)((n4 + 255) / 256);
    printf("  resident threads on GPU       %lld\n", resident);
    printf("  float4 blocks if exactly full %d  (%d threads)\n", blocks_full,
           blocks_full * 256);
    printf("  waves, float4 exact grid      %.1f\n",
           (double)blocks_full * 256 / resident);
    long long n = 25000000LL;
    int blocks_naive = (int)((n + 255) / 256);
    printf("  naive blocks if exactly full  %d  (%d threads)\n",
           blocks_naive, blocks_naive * 256);
    printf("  waves, naive exact grid       %.1f\n",
           (double)blocks_naive * 256 / resident);
    printf("  oversubscribed 262140 blocks  %d threads -> %.0f waves\n",
           262140 * 256, 262140.0 * 256 / resident);

    return 0;
}
