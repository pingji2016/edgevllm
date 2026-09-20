// 打印本机 GPU 的执行模型与显存属性。
//
// 用途：docs/kernel-opt/02-cuda-basics.md 第 0.5 节「硬件上限」那张表的数字，
//       就是这个程序跑出来的，不是从规格书上抄的。换台机器跑一遍就能得到那台的值。
//
// 编译运行：
//   nvcc -O3 -arch=sm_120 device_query.cu -o device_query.exe
//   或者用 examples/build.sh（它会顺带检查下面第 1 条约束）：
//     cd examples && ./build.sh run leetgpu/01_vector_add
//
// 两个必须遵守的约束（和 vector_add.cu 同源，改代码时别踩回去）：
//
//   1. 本文件含中文，必须存为「带 BOM 的 UTF-8」。nvcc 前端在 Windows 上默认按
//      本地代码页(cp936)读源码，中文注释会被拆错字节，报出 "missing closing
//      quote" / "identifier is undefined" 之类的假错误。examples/build.sh 里
//      的 check_bom 会替你检查这一点。
//
//   2. printf 的字面量一律用 ASCII。窄字符串字面量按「执行字符集」编码，MSVC
//      默认就是本地代码页(936)，UTF-8 源码里的中文到运行时变成 GBK 字节，控制台
//      和管道里全是乱码。所以下面所有标签都是英文，中文只出现在注释里。
//
// ---------------------------------------------------------------------------
// 关于"这些数是不是定死的"
//
// 全部是 cudaGetDeviceProperties 在运行时查出来的，没有一个硬编码 —— 唯一的
// 常数是那个设备号 0，即"查第 0 号 GPU"。但它们分两类，这个区分很实用：
//
//   [架构] 同一个 compute capability 的所有卡完全一样。
//          换一张 sm_120 的卡（5070 / 5080），这些数一个都不变。
//   [板卡] 同一个芯片的不同型号之间也会不同。换个型号就要重新跑这个程序。
//
// 下面逐项标了是哪一类。
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdio>

// 出错就打印行号并返回 1。设备属性查询几乎不会失败，留着是为了多卡 / 驱动异常时
// 能看出是哪一步挂的，而不是拿到一堆 0 还以为正常。
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
    CK(cudaGetDeviceProperties(&p, 0));  // 0 = 第 0 号 GPU。多卡时这里决定查哪张

    // -----------------------------------------------------------------------
    // 设备身份
    // -----------------------------------------------------------------------
    printf("== device ==\n");
    printf("  name                          %s\n", p.name);

    // major.minor 拼成惯例写法：12 和 0 -> "sm_120"。
    // 注意这是两个 %d 直接相邻拼字符串，只对 minor < 10 成立；
    // 将来 minor 上两位数会拼出 sm_1210 这种错值，到时候要改成 %d.%d 再补零。
    printf("  compute capability            sm_%d%d\n", p.major, p.minor);

    // [板卡] 这张卡具体有几个 SM。5060 Ti 是 36；同为 Blackwell 的 5080 更多，
    //        A100（Ampere）是 108。它和下面的"每 SM 线程数"相乘 = 全 GPU 的
    //        驻留线程总量，是算 wave 的分母。
    printf("  SMs                           %d\n", p.multiProcessorCount);

    // -----------------------------------------------------------------------
    // 单个 block 的限制
    // -----------------------------------------------------------------------
    printf("\n== per-block limits ==\n");

    // [架构] 32，从 Volta 至今没变过，也是为什么 blockDim 一定要取 32 的倍数。
    //        warp 是硬件真正调度和执行的最小单位：一个 block 的线程按
    //        threadIdx/32 打包成若干个 warp，32 个线程永远同进同出。
    //        详见 02-cuda-basics.md 第 0.4 节。
    printf("  warpSize                      %d\n", p.warpSize);

    // [架构] 1024。即 blockDim.x * y * z <= 1024。超过这个数启动会直接返回
    //        invalid configuration argument 报错，不会悄悄降级。
    printf("  maxThreadsPerBlock            %d\n", p.maxThreadsPerBlock);

    // [架构] 24。一个 SM 上最多同时驻留多少个 block —— 也就是"槽位"数。
    //        这是"个数"限制，和 block 多大无关：256 线程的 block 最多能占 24 个
    //        槽位。但在本机上真正卡住的不是它（24 个槽位远远够用），而是下面那个
    //        线程数上限：1536/256 = 6 个 block 就把线程数吃满了，根本轮不到 24。
    //        只有用极小 block（比如 32 线程）时，24 这个槽位上限才会先撞上。
    printf("  maxBlocksPerMultiprocessor    %d\n", p.maxBlocksPerMultiProcessor);

    // -----------------------------------------------------------------------
    // 单个 SM 的限制 —— 决定 occupancy 的三个共享资源
    // -----------------------------------------------------------------------
    printf("\n== per-SM limits ==\n");

    // [架构] 1536。本机 occupancy 的硬顶：一个 SM 同时最多住 1536 个线程。
    //        256 线程的 block 就是 6 个（6 x 256 = 1536），第 7 个进不来。
    //        注意不同架构这个数不一样：A100 / H100 是 2048，消费卡历来是 1536。
    printf("  maxThreadsPerMultiProcessor   %d\n", p.maxThreadsPerMultiProcessor);

    // [架构] 65536 —— 这是【一个 SM 的寄存器文件有多大】，单位是寄存器【个数】，
    //        不是线程数，也不是字节数。65536 个 32 位寄存器 = 256 KB，
    //        这就是一个 SM 的寄存器物理总量。
    //
    //        它是所有驻留线程共用的池子：一个线程用 R 个寄存器就占掉 R 份。
    //        所以  这个 SM 最多能住的线程数 = 65536 / R。
    //
    //        反过来说更常用：想让 1536 个线程全住满（occupancy 100%），
    //        每线程的寄存器预算 = 65536 / 1536 = 42.67，取整 42 个。
    //        超过 42 就开始掉档 —— 64 个寄存器只能住 1024 线程（67%），
    //        128 个只能住 512（33%）。这就是 02-cuda-basics.md 第 4 节
    //        「寄存器压力」的硬件根源。
    //        查自己 kernel 用了几个寄存器：nvcc -Xptxas -v
    printf("  regsPerMultiprocessor         %d\n", p.regsPerMultiprocessor);

    // [架构] 102400 字节 = 100 KB。一个 SM 上 shared memory 的总量。
    //        和寄存器一样是驻留线程共享的资源，会参与 occupancy 计算。
    //        用法是在 kernel 里 __shared__ 声明；静态声明有 48 KB 的单独上限，
    //        想用更多要走 cudaFuncSetAttribute 设成 dynamic shared memory。
    printf("  sharedMemPerMultiprocessor    %zu bytes\n",
           (size_t)p.sharedMemPerMultiprocessor);

    // [架构] 101376 字节 = 99 KB。单个 block 通过 opt-in 能申请到的 dynamic
    //        shared memory 上限。比上面的 100 KB 少 1 KB，那 1 KB 被运行时留作
    //        驱动自用（比如某些架构上要放 block 元数据）。
    printf("  sharedMemPerBlockOptin        %zu bytes\n",
           (size_t)p.sharedMemPerBlockOptin);

    // -----------------------------------------------------------------------
    // 上面几个数推出来的量
    // -----------------------------------------------------------------------
    printf("\n== derived ==\n");

    // 1536 / 32 = 48 个 warp 槽位。看 occupancy 时用 warp 作单位比用线程直观，
    // 因为硬件本来就是按 warp 调度的。
    int warps = p.maxThreadsPerMultiProcessor / p.warpSize;
    printf("  warps resident per SM         %d\n", warps);

    printf("  threads resident per SM       %d\n",
           p.maxThreadsPerMultiProcessor);

    // 36 x 1536 = 55296。整个 GPU 同时能装下的线程总数。
    // 这个数是下面算"要跑多少波(wave)"的分母，也是判断"grid 是不是配大了"的基准。
    printf("  threads resident on the GPU   %d\n",
           p.multiProcessorCount * p.maxThreadsPerMultiProcessor);

    // 65536 / 1536 = 42（整数除法，真实值 42.67）。
    // 即：想让 occupancy 达到 100%，每线程最多能用几个寄存器。见上面对
    // regsPerMultiprocessor 的注释。
    printf("  max regs per thread at 100%%   %d\n",
           p.regsPerMultiprocessor / p.maxThreadsPerMultiProcessor);

    // -----------------------------------------------------------------------
    // 时钟与显存 —— 算带宽和峰值算力用
    // -----------------------------------------------------------------------
    printf("\n== clocks / memory ==\n");

    // [板卡] 2587000 kHz = 2587 MHz。注意这是驱动报的【额定】boost 频率，
    //        nvidia-smi 报的 max 可能更高（本机是 3090 MHz）。
    //        算理论算力时用哪个会在结果上差十几个百分点，写文档时要说明用的哪个。
    printf("  clockRate (max SM)            %d kHz (%.0f MHz)\n", p.clockRate,
           p.clockRate / 1000.0);

    // [板卡] 14001000 kHz。显存时钟，下面算带宽用。
    printf("  memoryClockRate               %d kHz\n", p.memoryClockRate);

    // [板卡] 128 bit。显存位宽。5060 Ti 是 128 位，属于比较窄的；
    //        5080 是 256 位，这是两者带宽差距的主要来源（而不是频率）。
    printf("  memoryBusWidth                %d bit\n", p.memoryBusWidth);

    // [板卡] 33554432 字节 = 32 MB。L2 缓存大小。
    //        L2 是比 shared memory 更大、比显存更快的一层，跨 block 共享，
    //        做算子融合 / 提高复用率时是很有价值的目标（见 06-fusion-layout.md）。
    printf("  l2CacheSize                   %d bytes\n", p.l2CacheSize);

    // 峰值带宽 = 2 x 显存频率 x (位宽 / 8)
    //   2 * 14.001e9 Hz * (128/8) Byte = 448 GB/s
    // 那个 2 是因为 GDDR 是双沿传输（时钟上升沿和下降沿都传数据），所以叫
    // "Double Data Rate"。
    // 这是【理论】峰值。本机实测能跑到 394 GB/s，即 88% —— 见 01-vector-add.md，
    // 那里还解释了为什么朴素版和 float4 版都停在这个数上。
    double peak =
        2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;
    printf("  theoretical peak BW           %.0f GB/s\n", peak);

    // -----------------------------------------------------------------------
    // 用 N = 2500 万的 vector add 算"这份工作量要跑多少波"
    //
    // wave 的概念：一个 wave = 整个 GPU 同时能装下的 block 数。
    // 本机 = 36 SM x 6 block = 216 个 block（blockDim = 256 时）。
    // 完整推导和它为什么决定性能，见 02-cuda-basics.md 第 0.6 节。
    // -----------------------------------------------------------------------
    printf("\n== waves for N = 25,000,000 (blockDim = 256) ==\n");

    // 55296，见上面 derived 一节。所有 wave 的分母。
    long long resident = (long long)p.multiProcessorCount *
                         p.maxThreadsPerMultiProcessor;

    // float4 版一个线程一次处理 4 个元素，所以"工作项"只有 2500万/4 = 625 万个。
    long long n4 = 25000000LL / 4;

    // 向上取整：(6250000 + 255) / 256 = 24415。这就是"贴着工作量配"的 grid 大小。
    // 加 255 是向上取整的惯用写法，等价于 ceil(n / 256)。
    int blocks_full = (int)((n4 + 255) / 256);

    printf("  resident threads on GPU       %lld\n", resident);

    // 24415 个 block，每个都有活干，没有一个空转。
    printf("  float4 blocks if exactly full %d  (%d threads)\n", blocks_full,
           blocks_full * 256);

    // 24415 x 256 / 55296 = 113 波。这是这份工作量在理想 grid 下要跑的波数。
    printf("  waves, float4 exact grid      %.1f\n",
           (double)blocks_full * 256 / resident);

    // 朴素版一个线程只处理 1 个元素，所以工作项是 2500 万个，block 数翻 4 倍。
    long long n = 25000000LL;
    int blocks_naive = (int)((n + 255) / 256);
    printf("  naive blocks if exactly full  %d  (%d threads)\n",
           blocks_naive, blocks_naive * 256);

    // 97657 x 256 / 55296 = 452 波。注意：波数是 float4 版的 4 倍，
    // 但 01-vector-add.md 实测两者耗时完全一样（760.8 vs 760.9 微秒）。
    // 这就是那句结论的依据 —— 【wave 数本身不花时间】，能装下的并行度才是关键。
    printf("  waves, naive exact grid       %.1f\n",
           (double)blocks_naive * 256 / resident);

    // 超订版：65535 x 4 = 262140 个 block。其中只有 24415 个真的有活干，
    // 剩下 237725 个是纯空的。
    //
    // 但空 block 并不是免费的 —— 它照样要被调度、占 SM 的 block 槽位、发射
    // 8 个 warp、让每个 warp 走一遍循环条件判断、然后退休。这些都要花时间，
    // 而且和 block 里有没有活干无关。
    // 1214 - 113 = 1101 波，全是这种纯空转。
    // 这就是 01-vector-add.md 里那 22% 损耗的由来（933.1 vs 760.9 微秒）。
    printf("  oversubscribed 262140 blocks  %d threads -> %.0f waves\n",
           262140 * 256, 262140.0 * 256 / resident);

    return 0;
}
