// LeetGPU 01 · Vector Addition —— 本地可运行版
//
// 用途：上游 starter.cu 只有骨架（kernel 体是空的，也没有 main），
//       而且官方评测脚本 challenge.py 依赖 CUDA 版 PyTorch——本地 CPU-only 的
//       torch 跑不起来。所以这里自带一个 main：正确性校验 + 带宽实测。
//
// 编译（Windows + MSVC）：
//   nvcc -O3 -arch=sm_120 vector_add.cu -o vector_add.exe
// 运行：
//   ./vector_add.exe              # 默认 N = 25,000,000（题面性能测试的规模）
//   ./vector_add.exe 1000000      # 自定义 N
//
// 两个 Windows 编码坑（都已在本文中规避，改代码时注意别踩回去）：
//
//   1. 本文件必须存为「带 BOM 的 UTF-8」。nvcc 前端在 Windows 上默认按本地代码页
//      （936/GBK）读源码，中文注释会被拆错字节，报一堆 "missing closing quote" /
//      "identifier is undefined"。BOM 让它自动识别 UTF-8。
//
//   2. 运行时输出（printf 的字面量）一律用 ASCII。窄字符串字面量会被编译成
//      *执行字符集*，MSVC 默认就是本地代码页（936），于是 UTF-8 源码里的中文
//      到运行时变成 GBK 字节，控制台/管道里全是乱码；而加 -Xcompiler -utf-8 修正
//      执行字符集，又会反过来让 nvcc 前端解析不了这些字面量。中文留在注释里最省事。
//
// 也可以直接交给官方评测：把下面 solve() 的实现贴回 starter.cu 即可。

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

// 出错就打印并退出。solve 的契约是 void 返回，所以错误只能这么报。
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// 实现 1：朴素版。一个线程一个元素。
// ---------------------------------------------------------------------------
__global__ void vector_add_naive(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 long long N) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {              // ← 尾部保护。N 不是 blockDim 整数倍时靠它
        C[i] = A[i] + B[i];
    }
}

// ---------------------------------------------------------------------------
// 实现 2：float4 向量化版。一次搬 16 字节，访存指令数降到 1/4。
// 前提：三个指针都 16 字节对齐（cudaMalloc / torch caching allocator 都满足）。
// ---------------------------------------------------------------------------
__global__ void vector_add_vec4(const float4* __restrict__ A,
                                const float4* __restrict__ B,
                                float4* __restrict__ C,
                                long long n4) {
    // grid-stride：固定大小的 grid 覆盖任意大的 n4
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n4; i += stride) {
        float4 a = A[i];
        float4 b = B[i];
        C[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

// ---------------------------------------------------------------------------
// solve：与上游 starter 完全一致的签名，可以直接贴回 starter.cu 提交。
// ---------------------------------------------------------------------------
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    const int threads = 256;
    long long n4 = N / 4;           // 能整除 4 的部分走向量化
    long long tail = n4 * 4;        // 余下的 0~3 个元素单独处理

    if (n4 > 0) {
        long long want = (n4 + threads - 1) / threads;
        // 压住 grid 上限，剩余部分靠 grid-stride 循环覆盖。
        // 注意：本题 n4 最多 2.5e7，want 约 2.4e4，远低于 cap——
        // 也就是说 grid-stride 只是通用性保险，这里根本用不上。
        long long cap = 65535LL * 32;
        unsigned blocks = (unsigned)(want > cap ? cap : want);
        vector_add_vec4<<<blocks, threads>>>(
            (const float4*)A, (const float4*)B, (float4*)C, n4);
    }

    // 尾部：最多 3 个元素。用标量 kernel 处理，开销可忽略。
    if (tail < N) {
        long long rest = N - tail;
        unsigned blocks = (unsigned)((rest + threads - 1) / threads);
        vector_add_naive<<<blocks, threads>>>(A + tail, B + tail, C + tail, rest);
    }

    // 评测器调用 solve 后立刻读 C，必须等 kernel 跑完。
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// 正确性校验：小规模 N，覆盖 N 不是 4 的倍数 / 不是 block 整数倍的边界
// ---------------------------------------------------------------------------
static bool check_one(int N, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    std::vector<float> hA(N), hB(N), hC(N, 0.0f);
    for (int i = 0; i < N; ++i) {
        hA[i] = dist(rng);
        hB[i] = dist(rng);
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, N * sizeof(float)));

    solve(dA, dB, dC, N);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        // 每元素只做一次加法，应当逐比特相同；这里仍按题面容差判定
        float ref = hA[i] + hB[i];
        if (fabsf(hC[i] - ref) > 1e-5f + 1e-5f * fabsf(ref)) {
            printf("    N=%d mismatch at %d: got %g, want %g\n", N, i, hC[i], ref);
            ok = false;
            break;
        }
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return ok;
}

static bool test_correctness() {
    // 极小 N（一个 warp 都填不满）、非 4 倍数、非 block 整数倍、整数倍
    const int cases[] = {1, 2, 3, 4, 7, 30, 255, 256, 257, 1000, 4096, 10000, 65537};
    printf("=== correctness ===\n");
    bool all = true;
    for (int n : cases) {
        bool ok = check_one(n, 12345u + n);
        printf("  N=%-7d %s\n", n, ok ? "PASS" : "FAIL");
        all = all && ok;
    }
    return all;
}

// ---------------------------------------------------------------------------
// 性能实测
//
// 三种配置对比，重点是第三种：它演示了「grid-stride + 超大 grid」这个
// 常见错误——线程数远超工作量时，多出来的线程虽然一次循环都不执行，
// 却仍然要被调度，开销相当可观。
// ---------------------------------------------------------------------------
static void launch_naive(const float* A, const float* B, float* C, int N,
                         int blocks) {
    // blocks 参数忽略：标量版总是铺满 grid
    const int threads = 256;
    long long n = ((long long)N + threads - 1) / threads;
    vector_add_naive<<<(unsigned)n, threads>>>(A, B, C, (long long)N);
}

static void launch_vec4(const float* A, const float* B, float* C, int N,
                        int blocks) {
    vector_add_vec4<<<blocks, 256>>>(
        (const float4*)A, (const float4*)B, (float4*)C, N / 4);
}

static void benchmark(int N, double peak_gbps) {
    printf("\n=== benchmark N=%d ===\n", N);
    printf("traffic 300 MB (read A/B %.0f MB each, write C %.0f MB)\n",
           N * 4.0 / 1e6, N * 4.0 / 1e6);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)N * 4));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)N * 4));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)N * 4));
    CUDA_CHECK(cudaMemset(dA, 0, (size_t)N * 4));
    CUDA_CHECK(cudaMemset(dB, 0, (size_t)N * 4));

    long long n4 = N / 4;
    int blocks_full = (int)((n4 + 255) / 256);   // 恰好铺满，与 solve 一致

    typedef void (*LaunchFn)(const float*, const float*, float*, int, int);
    struct Cfg { const char* name; int blocks; LaunchFn fn; };
    Cfg cfgs[] = {
        {"naive scalar",              0,           launch_naive},
        {"float4 full grid",          blocks_full, launch_vec4},
        {"float4 oversubscribed",     65535 * 4,   launch_vec4},
    };

    const double bytes = 3.0 * N * sizeof(float);   // A 读 + B 读 + C 写
    const int warmup = 3, iters = 20;

    for (const Cfg& cfg : cfgs) {
        for (int i = 0; i < warmup; ++i) cfg.fn(dA, dB, dC, N, cfg.blocks);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < iters; ++i) cfg.fn(dA, dB, dC, N, cfg.blocks);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        double per_iter_ms = ms / iters;
        double gbps = bytes / (per_iter_ms * 1e-3) / 1e9;

        printf("  %-24s %8.1f us   %7.1f GB/s   %5.1f%% of peak\n",
               cfg.name, per_iter_ms * 1000.0, gbps, 100.0 * gbps / peak_gbps);

        CUDA_CHECK(cudaEventDestroy(t0));
        CUDA_CHECK(cudaEventDestroy(t1));
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

int main(int argc, char** argv) {
    // 打印设备信息，顺便算出理论峰值带宽
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // 峰值 = 2(DDR) × 显存频率(kHz) × 10^3 × 位宽/8 字节
    double peak_gbps =
        2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;

    printf("device: %s  (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("memory: %.1f GB, %d-bit, theoretical peak ~%.0f GB/s\n",
           prop.totalGlobalMem / 1e9, prop.memoryBusWidth, peak_gbps);

    int N = 25000000;   // 题面性能测试规模
    if (argc > 1) N = atoi(argv[1]);
    if (N <= 0) {
        fprintf(stderr, "N must be a positive integer\n");
        return EXIT_FAILURE;
    }

    bool ok = test_correctness();
    benchmark(N, peak_gbps);

    printf("\n%s\n", ok ? "ALL PASS" : "FAILURES PRESENT");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
