# 02 · CUDA 算子优化基本功

这一篇的东西在任何 GPU 算子上都成立。**LLM 里 80% 的 kernel 性能问题，最后都落到下面这几条上。**

## 1. 合并访存（Memory Coalescing）—— 最重要，没有之一

GPU 的显存事务以 32 字节 / 128 字节为单位。一个 warp 的 32 个线程如果访问连续的地址，硬件会合并成 1~4 个事务；如果地址散乱，最坏情况变成 32 个事务，**带宽直接掉 32 倍**。

```cuda
// 反例：线程沿行方向走，相邻线程的地址隔了一整个 N
for (int i = tid; i < M; i += blockDim.x)
    out[i * N] = in[i * N];          // 每个线程访问不同 cache line

// 正例：线程沿列方向走，相邻线程访问连续地址
for (int j = tid; j < N; j += blockDim.x)
    out[i * N + j] = in[i * N + j];  // 一个 warp 覆盖 128 字节
```

**判断方法**：想象 warp 的第 0 号线程和第 1 号线程的地址差。等于元素大小 → 完美；等于一行长度 → 灾难。

### 向量化访存

一次搬 128 bit（`float4` / `int4` / `uint4`）而不是 32 bit，能把访存指令数降到 1/4，通常能显著提升带宽利用率：

```cuda
// 前提：指针 16 字节对齐，且元素数能被 4 整除
const float4* in4 = reinterpret_cast<const float4*>(in);
float4 v = in4[idx];
```

vLLM 里几乎所有 elementwise kernel 都做了这个 —— 见 `csrc/layernorm_kernels.cu` 里的 `vectorized_load` 分支。

**坑**：不对齐会直接崩（misaligned address），所以要有 fallback 标量路径：

```cuda
if (is_aligned && n % 4 == 0) { kernel_vectorized<<<...>>>(...); }
else                          { kernel_scalar<<<...>>>(...); }
```

### 结构体数组 vs 数组结构体（AoS / SoA）

```cuda
struct AoS { float x, y, z; };   // 访问 .x 时把 .y .z 也搬进来了，浪费 2/3 带宽
struct SoA { float *x, *y, *z; }; // 各自连续，按需搬运
```

KV cache 就是典型的 SoA：K 和 V 分开存，因为 attention 里不同阶段访问模式不同。

## 2. 共享内存与 Bank Conflict

共享内存被划分成 32 个 bank（每个 4 字节）。同一 warp 内两个线程访问**同一个 bank 的不同地址** → 冲突，请求被串行化。

```cuda
__shared__ float tile[32][32];

// 转置访问：tid 读 tile[tid][j] —— 步长为 32，所有线程落在同一个 bank
float v = tile[threadIdx.x][j];        // 32-way conflict，慢 32 倍

// padding 一行，步长变成 33，天然错开
__shared__ float tile[32][32 + 1];
float v = tile[threadIdx.x][j];        // 无冲突
```

**要点**：

- 步长是 2 的幂 → 几乎必然冲突，加个 padding 就好
- 广播（所有线程读同一地址）**不算**冲突，硬件会自动广播
- 从 vLLM / CUTLASS 抄 kernel 时，别乱改 shared memory 的 padding

### 什么时候不值得用 shared memory

单纯 elementwise 或 reduction，用 shared memory 反而多一次写读。**寄存器 + warp shuffle 更快。**

## 3. Warp Shuffle：不做 block 级同步的归约

`__shfl_down_sync` 在寄存器之间直接搬数据，不需要经过 shared memory，也就没有 `__syncthreads()`。

```cuda
// warp 内归约：5 步搞定 32 个数
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// block 级归约：每个 warp 先在寄存器里归约，只有 warp 之间才用 shared memory
__shared__ float shm[32];
float v = warp_reduce_sum(local);
if (threadIdx.x % 32 == 0) shm[threadIdx.x / 32] = v;
__syncthreads();
if (threadIdx.x < 32) v = warp_reduce_sum(shm[threadIdx.x]);
```

vLLM 的 RMSNorm 就是这么做的（`csrc/layernorm_kernels.cu`），整体是纯带宽瓶颈，把归约开销压到最低才有意义。

## 4. Occupancy 与寄存器压力

occupancy = 实际活跃 warp 数 / 硬件上限。它决定**用多少并行度来掩盖访存延迟**。

限制因素通常是三个，取最紧的那个：

- 寄存器 / 线程（A100 每 SM 65536 个寄存器）
- shared memory / block
- 每 SM 最大 block / warp 数

```cuda
// 用 -Xptxas -v 看实际用了多少寄存器
// 例子：255 寄存器/线程 → 每个 SM 只能跑 256 线程 → 25% occupancy

__launch_bounds__(256, 4)   // 告诉编译器：每 SM 至少 4 个 block，请把寄存器压到 64 以内
```

**并不是 occupancy 越高越好**：

- 带宽瓶颈的 kernel → occupancy 很关键（需要有足够 warp 来隐藏访存延迟）
- 计算密集、寄存器复用充分的 kernel（如 GEMM）→ 50% 甚至更低往往更快，因为减少寄存器反而会牺牲数据复用

经验：**先用 occupancy 解释现象，再决定是否要提。**

## 5. 软件流水线与异步拷贝

核心思想：**当前迭代在算的时候，下一轮的数据已经在搬了。**

```cuda
// Ampere+：cp.async 让数据从显存直接进 shared memory，不占寄存器
__pipeline_memcpy_async(&smem[next], &gmem[...], sizeof(float4));
__pipeline_commit();
__pipeline_wait_prior(0);

// Hopper+：TMA (cp.async.bulk) 更好，一条指令搬整个 tile，还能做 swizzle
```

Double buffering 的典型结构：

```cuda
load(0);
for (int k = 0; k < K; k += TILE) {
    load(k + TILE);      // 异步发起下一块
    compute(k);          // 同时算当前块
    wait(k + TILE);
}
```

这是所有高性能 GEMM / FlashAttention 的骨架，见 [03-gemm.md](03-gemm.md)。

## 6. 其它真正常用的技巧

### 用 `__restrict__` 和 `const`

帮助编译器判断没有指针别名，从而放心地做重排和缓存。

```cuda
__global__ void add(const float* __restrict__ a,
                    const float* __restrict__ b,
                          float* __restrict__ c)
```

### 分支发散

同一个 warp 里不同线程走不同分支 → 两条路都执行。用「计算代替分支」通常更快：

```cuda
// 差
if (x > 0) y = x; else y = 0;

// 好
y = fmaxf(x, 0.f);
```

### 网格步长循环（Grid-Stride Loop）

比「一个线程一个元素」更灵活，且天然支持任意尺寸输入：

```cuda
__global__ void kernel(float* a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) { /* ... */ }
}
```

好处：可以刻意把 grid 大小设成 SM 数量的整数倍，做出「持久化 kernel」的效果，省掉反复启动。

### Kernel Launch 开销

一次 kernel 启动大约 3~10 微秒。decode 阶段每个 token 可能要跑几百个小 kernel，**启动开销会变成主要成本**。两个对策：

1. 用 CUDA Graph 把整个前向捕获成一张图，一次提交 → 省掉 CPU 端逐次 launch（vLLM 的 `enforce_eager=False` 走的就是这条路）
2. 融合算子，减少 kernel 数量（见 [06-fusion-layout.md](06-fusion-layout.md)）

## 检查清单

写/改完一个 kernel，按顺序过一遍：

- [ ] 相邻线程访存地址连续吗？能不能向量化到 128 bit？
- [ ] 用到 shared memory 的话，有没有 2 的幂步长导致的 bank conflict？
- [ ] `-Xptxas -v` 看寄存器数 → occupancy 是不是低到无法隐藏延迟？
- [ ] 带宽利用率（`实际字节/耗时 ÷ Peak BW`）到 70% 了吗？没到就是访存没弄好
- [ ] 有没有可以用 `__restrict__`、`fmaxf`、shuffle 消掉的东西？
- [ ] 用 CUDA Graph 了吗？kernel 是不是碎得太多？
