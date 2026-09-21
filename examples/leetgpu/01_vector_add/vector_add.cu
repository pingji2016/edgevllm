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


// ===========================================================================
// 先搞清楚 `<<<grid, block>>>` 和 threadIdx / blockIdx 到底是什么
//
// 这段是理解下面所有 kernel 的前提，建议先读完。
//
// ---------------------------------------------------------------------------
// 1) CUDA 的启动配置天生是「三维」的，但和你的数据维度无关
// ---------------------------------------------------------------------------
//
//     kernel<<<gridDim, blockDim>>>(...)
//     //        └ 网格     └ 每个 block
//
// 展开成 dim3 就是 (x, y, z) 三个分量。比如：
//
//     vector_add_vec4<<<24415, 256>>>(...)
//     // 等价于 gridDim  = dim3(24415, 1, 1)
//     //        blockDim = dim3(256,   1, 1)
//
// 四个内建变量：
//
//     threadIdx.x/y/z   我这个线程，在本 block 里排第几     （0 ~ blockDim-1）
//     blockIdx.x/y/z    我这个 block，在网格里排第几        （0 ~ gridDim-1）
//     blockDim.x/y/z    每个 block 有多少线程
//     gridDim.x/y/z     网格里有多少 block
//
// 【最容易搞混的一点】
//     .x / .y / .z 不是「矩阵的行列」，而是三套互不相干的计数器。
//     你想用几个就用几个，用在哪一维完全由你自己决定。
//
//     本题的 A、B、C 是三个扁平的一维 float 数组，所以只用了 .x，
//     .y 和 .z 保持默认值 1。整个文件里不会出现 .y —— 这不是因为它「不二维」，
//     而是因为这里压根不需要第二套计数器。
//
//     真处理二维矩阵时当然可以用 .y：dim3 block(32, 32) 配 threadIdx.x /
//     threadIdx.y，行号列号更直观。但那是「为了写起来方便」，不是语法要求。
//     很多人处理矩阵照样只用 .x，然后自己 row = i / width、col = i % width。
//
// ---------------------------------------------------------------------------
// 2) 全局线程编号（下面每个 kernel 的第一个表达式）
// ---------------------------------------------------------------------------
//
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//             └─ 我前面有几个完整的 block ─┘   └ 我在本 block 里排第几 ┘
//
// 这就是「我是全体线程里的第几号」。
//
// threadIdx.x 只在本 block 内有意义（0~255），必须加上前面所有 block 的线程数，
// 才能得到一个全局唯一、从 0 连续排到 (总线程数-1) 的编号：
//
//     block 0        block 1        block 2
//     ┌──────────┐   ┌──────────┐   ┌──────────┐
//     │ 0 … 255  │   │ 256 … 511│   │ 512 … 767│
//     └──────────┘   └──────────┘   └──────────┘
//      threadIdx.x    threadIdx.x    threadIdx.x
//      +0*256         +1*256         +2*256
//
// ---------------------------------------------------------------------------
// 3) 整个网格有多少线程
// ---------------------------------------------------------------------------
//
//     gridDim.x * blockDim.x        // ← 这就是下面那个 stride
//
// 「有多少个 block」×「每个 block 多少线程」= 全体线程站成一排有多少人。
//
// 【溢出警告】这个乘法必须先把其中一个操作数转成 64 位：
//
//     long long stride = (long long)gridDim.x * blockDim.x;   // 对
//     long long stride = gridDim.x * blockDim.x;              // 错！
//
// 因为 gridDim.x 上限是 2^31-1、blockDim.x 最大 1024，乘积能到 2^41 量级，
// int 早就炸了。而第二行的写法里，乘法是先按 int 算完、溢出已经发生，
// 再把这个错误的结果转成 long long —— 转了也白转。转换必须加在乘法前面。
// ===========================================================================


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
//
// 这是最直白的写法，也是性能和「优化版」打平的写法（原因见文件末尾的说明）。
// ---------------------------------------------------------------------------
__global__ void vector_add_naive(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 long long N) {
    // 全局线程编号，见文件开头第 2 节。
    // 用 long long 而不是 int：N 可达 1e8，gridDim.x * blockDim.x 也是个大数。
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    // ← 尾部保护。启动的线程数通常是向上取整来的（保证覆盖全部 N 个元素），
    //   所以最后可能多出几个线程。不加这个 if 就会越界读写。
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// ---------------------------------------------------------------------------
// 实现 2：float4 向量化版。一次搬 16 字节，访存指令数降到 1/4。
//
// 前提：三个指针都 16 字节对齐（cudaMalloc / torch caching allocator 都满足）。
// 注意 n4 = N/4 —— 这里的下标单位是「4 个 float 一组」，不是单个 float。
// ---------------------------------------------------------------------------
__global__ void vector_add_vec4(const float4* __restrict__ A,
                                const float4* __restrict__ B,
                                float4* __restrict__ C,
                                long long n4) {

    // =======================================================================
    // grid-stride 循环
    //
    // 一句话：每个线程干完自己那份，就往后跳「全体线程数」这么远，接着干，
    //         直到跳出数组末尾。
    //
    // 为什么需要它？两个场景：
    //   (a) n4 太大，一个线程一个元素的话 grid 会超出硬件上限 → 必须靠循环分片
    //   (b) 你想固定 grid 大小（比如按 SM 数量开），让它适配任意 n4
    //
    // 本题的 n4 最大 2.5e7，两种场景都用不上，纯粹是通用性保险（run 里实际
    // 只循环一次就退出了）。但下面第三个 benchmark 配置会演示：**滥用它是有代价的**。
    // =======================================================================

    // stride = 整个网格的线程总数 = 「全体线程站成一排有多少人」。
    // 为什么要 long long：见文件开头第 3 节的溢出警告，转换必须加在乘法之前。
    long long stride = (long long)gridDim.x * blockDim.x;

    // 循环变量三要素，逐个看：
    //
    //   初始值  blockIdx.x * blockDim.x + threadIdx.x
    //           → 我的全局线程编号，也就是本轮我要处理的元素下标
    //
    //   条件    i < n4
    //           → 越界就不干了。这一条同时兜住了两件事：数组末尾的保护，
    //             以及「线程数比元素数多」时多余线程的立即退出。
    //
    //   步长    i += stride
    //           → 跳过一整格，也就是跳过全体线程的数量
    //
    // 为什么这样能不漏不重？因为全体线程的第 1 轮正好覆盖 [0, stride)，
    // 第 2 轮覆盖 [stride, 2*stride)，依次类推 —— 首尾相接，天然是完美划分。
    //
    //    假设 stride = 8（总共 8 个线程）、数组有 19 个元素：
    //
    //    元素:   0  1  2  3  4  5  6  7 │ 8  9 10 11 12 13 14 15 │ 16 17 18
    //    第1轮: T0 T1 T2 T3 T4 T5 T6 T7 │
    //    第2轮:                        │ T0 T1 T2 T3 T4 T5 T6 T7│
    //    第3轮:                        │                        │ T0 T1 T2
    //            └─── 覆盖 [0,8) ─────┘ └─── 覆盖 [8,16) ───────┘ └ 覆盖 [16,19)
    //
    //     像发牌：每个线程拿第「自己编号 + k × stride」张，k = 0,1,2,…
    //     注意同一轮里 T0~T7 访问的是连续地址 —— 这一点对性能至关重要，
    //     见文件末尾关于合并访问的说明。
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n4; i += stride) {

        float4 a = A[i];        // 一条 128-bit load 指令搬 4 个 float
        float4 b = B[i];        // 同上
        // float4 只是个 struct { float x, y, z, w; }，没有算术运算符，
        // 得手动逐分量相加再拼回去。
        C[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

// ---------------------------------------------------------------------------
// solve：与上游 starter 完全一致的签名，可以直接贴回 starter.cu 提交。
//
// 评测器通过 ctypes 加载 .so 后按这个签名调用，签名改了会直接失败。
// ---------------------------------------------------------------------------
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    const int threads = 256;

    // N 不一定能被 4 整除，所以劈成两段：
    //   [0, tail)     走向量化 float4
    //   [tail, N)     剩下 0~3 个元素，走标量
    long long n4 = N / 4;           // 能整除 4 的部分走向量化
    long long tail = n4 * 4;        // 余下的 0~3 个元素单独处理

    if (n4 > 0) {
        // 向上取整：(n4 + threads - 1) / threads 个 block 刚好铺满 n4 个元素
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
    // 把指针整体前移 tail 个元素，这样新数组的 [0, rest) 就是原来的 [tail, N)。
    if (tail < N) {
        long long rest = N - tail;
        unsigned blocks = (unsigned)((rest + threads - 1) / threads);
        vector_add_naive<<<blocks, threads>>>(A + tail, B + tail, C + tail, rest);
    }

    // 评测器调用 solve 后立刻读 C，必须等 kernel 跑完。
    // kernel 启动是异步的，少了这一句 C 可能还没算完就被读走。
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// 正确性校验：小规模 N，覆盖 N 不是 4 的倍数 / 不是 block 整数倍的边界
//
// 用例列表见 test_correctness()，关键是那几个刁钻值：
//   N=1,2,3      → 走不到 float4 分支，全在 tail 里
//   N=4          → 恰好一个 float4，tail 为空
//   N=255        → 非 4 倍数，且有 tail
//   N=256        → 恰好一个 block
//   N=257        → 多出一个元素，逼出尾部保护
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
    // 先清零：万一 kernel 少写了某个元素，这里能暴露出来（否则读到的是垃圾，可能碰巧对）
    CUDA_CHECK(cudaMemset(dC, 0, N * sizeof(float)));

    solve(dA, dB, dC, N);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        // 每元素只做一次加法，应当逐比特相同；这里仍按题面容差判定
        // （容差取自题面：1e-5 绝对 + 1e-5 相对）
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
// 三行配置 = 两个独立的对照实验，别当成三选一：
//
//   ┌─────────────────────┬──────────┬───────────┬──────────────────────────┐
//   │ 配置                │ 每线程搬 │ block 数  │ 在验证什么               │
//   ├─────────────────────┼──────────┼───────────┼──────────────────────────┤
//   │ naive scalar        │ 4 B      │  97,657   │ 基准                     │
//   │ float4 full grid    │ 16 B     │  24,415   │ vs ①：向量化有用吗？     │
//   │ float4 oversubscrib │ 16 B     │ 262,140   │ vs ②：多开 grid 要钱吗？ │
//   └─────────────────────┴──────────┴───────────┴──────────────────────────┘
//
// 实验 A（① vs ②）只改「一个线程搬几个字节」，grid 都是刚好铺满。
//   结论：打平。因为一个 warp 的 32 个线程访问连续地址时，硬件会把它们合并成
//   128 B 的事务（32×4 B），向量化后是 512 B = 4 个事务——DRAM 实际搬运的字节
//   数一模一样。向量化只省了指令发射数，而本题瓶颈是内存带宽（算术强度
//   0.083 FLOP/Byte），发射根本不是瓶颈。详见 docs/leetgpu/01-vector-add.md。
//
// 实验 B（② vs ③）用同一个 kernel，只改启动的 block 数（×10.7）。
//   结论：慢 23%。因为 n4 只有 6,250,000，24,415 个 block 就把活干完了，
//   多出来的 237,725 个 block 里一个元素都不处理——但它们照样要被调度、
//   占 SM 的 block 槽位、发射 warp、走一遍循环判断再退休。这些开销与
//   「block 里有没有活干」无关。用 device_query.cu 算就是 113 波 vs 1214 波，
//   多出来的 1101 波全是纯空转。
// ---------------------------------------------------------------------------

// 注意这两个 launch_* 的函数签名被统一成 5 个参数，只为了让它们能塞进同一个
// 函数指针数组（见 benchmark 里的 Cfg）。blocks 参数不是每个实现都用得上。
static void launch_naive(const float* A, const float* B, float* C, int N,
                         int blocks) {
    // blocks 参数忽略：标量版总是铺满 grid
    const int threads = 256;
    long long n = ((long long)N + threads - 1) / threads;
    vector_add_naive<<<(unsigned)n, threads>>>(A, B, C, (long long)N);
}

static void launch_vec4(const float* A, const float* B, float* C, int N,
                        int blocks) {
    // 这里 blocks 由调用方给定 —— 实验 A 和 B 的差异全部来自这一个参数
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
    // 内容不影响带宽测试（只读不判断），但别留着未初始化内存
    CUDA_CHECK(cudaMemset(dA, 0, (size_t)N * 4));
    CUDA_CHECK(cudaMemset(dB, 0, (size_t)N * 4));

    // 256 与 launcher 里的 blockDim 对应
    long long n4 = N / 4;
    int blocks_full = (int)((n4 + 255) / 256);   // 恰好铺满，与 solve 一致

    typedef void (*LaunchFn)(const float*, const float*, float*, int, int);
    struct Cfg { const char* name; int blocks; LaunchFn fn; };
    Cfg cfgs[] = {
        // 标量版的 blocks 传 0，反正 launch_naive 会忽略它、自己算铺满的 grid
        {"naive scalar",              0,           launch_naive},

        // 实验 A 的一半：铺满。stride = 24,415 × 256 = 6,250,240 > n4 = 6,250,000，
        // 所以 grid-stride 循环每个线程只跑一轮就退出（i += stride 立刻越界）。
        {"float4 full grid",          blocks_full, launch_vec4},

        // 实验 B 的一半：同一个 kernel，grid 放大到 10.7 倍。
        // stride = 262,140 × 256 = 67,107,840，是 n4 的 10.7 倍。
        // 前 6,250,000 个线程各干一个元素，剩下 6,085 万个线程一进来就
        // 发现 i >= n4、直接退出 —— 什么都没干，但调度一步没少。
        // （65535 是 gridDim.y/z 的上限，这里 ×4 只是为了造一个明显超订的数。）
        {"float4 oversubscribed",     65535 * 4,   launch_vec4},
    };

    // 流量 = 读 A + 读 B + 写 C。算术强度极低，所以这是纯粹的带宽题。
    const double bytes = 3.0 * N * sizeof(float);   // A 读 + B 读 + C 写
    const int warmup = 3, iters = 20;

    for (const Cfg& cfg : cfgs) {
        // 先热身：让 GPU 提到高时钟、把页表/TLB 预热，否则第一次的计时偏慢
        for (int i = 0; i < warmup; ++i) cfg.fn(dA, dB, dC, N, cfg.blocks);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 用 cudaEvent 而不是 CPU 计时：kernel 启动是异步的，
        // CPU 侧的时间戳根本没包住真正的执行时间。event 记录在流里，
        // 是 GPU 自己的时间线。
        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < iters; ++i) cfg.fn(dA, dB, dC, N, cfg.blocks);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        // 连发 20 次取平均，摊平单次抖动。注意这 20 次是背靠背排队的，
        // 所以测的是「稳态吞吐」，不是「第一次要多久」。
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
    //   ×2      GDDR 一个时钟周期传两次数据
    //   ×10^3   kHz → Hz
    //   /8      位 → 字节
    // 算出的是理论值，实测能到 85~88% 就算不错了。
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

    // 先验正确性再测性能：一段算错的快代码没有意义
    bool ok = test_correctness();
    benchmark(N, peak_gbps);

    printf("\n%s\n", ok ? "ALL PASS" : "FAILURES PRESENT");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

// ===========================================================================
// 附：为什么「向量化」在这道题上不带来收益
//
// 直觉上，一个线程搬 16 B 应该比搬 4 B 快 4 倍。但实测两者打平（~761 µs）。
//
// 原因是硬件的「合并访问」（coalescing）：一个 warp 的 32 个线程同时发访存
// 请求时，内存系统会把落在同一个 128 B 段里的地址**合并成一次事务**。
//
//     朴素版：  T0→A[0]  T1→A[1]  ... T31→A[31]    32 × 4 B  = 128 B → 1 个事务
//     float4：  T0→A[0..3] ... T31→A[124..127]     32 × 16 B = 512 B → 4 个事务
//
// 两种写法搬运的字节总数完全一样，事务数也按比例一样。**DRAM 看到的工作量
// 是相同的。**
//
// 那向量化省下了什么？只有指令发射次数（4 条 load + 4 条 store 变成 1 + 1）。
// 而这道题的算术强度只有 0.083 FLOP/Byte —— 每搬 12 字节才做 1 次浮点加法，
// 瓶颈牢牢卡在内存带宽上，发射单元根本不是瓶颈。省下来的发射时间无处体现。
//
// 真正让性能掉下去的，是 benchmark 里第三个配置那种「线程数远超工作量」的
// 写法 —— 加的是纯调度开销，而调度开销会直接吃掉带宽。
//
// 完整推导与实测数据见 docs/leetgpu/01-vector-add.md。
// ===========================================================================
