# 09 · Triton 与 CUDA：两种 GPU 编程模型

这个仓库的代码片段一直是两种混着写的，但没人解释过它们的分界线在哪。本篇补上。

先摆一个反差，它是本篇的全部动机：

- 同一条 vector add，CUDA 写 8 行、Triton 写 7 行，**两者都撞同一个 DRAM 带宽天花板**
  （[01-vector-add](../leetgpu/01-vector-add.md) 实测 394 GB/s = 峰值的 88%）。
- 但在 vLLM 里，有人把 MoE 的一个 kernel 从 Triton 换回手写 CUDA，小 batch 下**快了 1.74 倍**
  （[PR #53693](https://github.com/vllm-project/vllm/pull/53693)）。

同一个语言，一边是「几乎白送」，一边是「被吊打」。原因是它们**抽象掉的层级不同**，
而这个层级恰好决定了你能表达什么、不能表达什么。

---

## 1. 区别的根：你写的代码作用在哪个层级

回到 [02-cuda-basics.md 第 0 节](02-cuda-basics.md) 那四层结构：

```
grid → block → warp → thread
```

**CUDA 让你写 `thread` 那一层。** 源码里的每条语句描述的是「一个线程做什么」，
32 个线程怎么凑成一个 warp、warp 之间怎么调度，是硬件的事。

**Triton 让你写 `block` 那一层，而且是张量形式的。** 源码里的每个变量不是一个标量，
而是一个**长度为 BLOCK_SIZE 的向量**；整段代码描述的是「一个 block 对一整块数据做什么」。
哪个元素落到哪个线程、同一个 warp 内部怎么交换数据，是**编译器**的事。

这个区别的一个直接后果：**Triton 里根本没有 `threadIdx`。**

```cuda
// CUDA：你得自己算"我是第几个线程"
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

```python
# Triton：没有 threadIdx，offs 是一整个向量
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#      └ 标量         └ 长度为 BLOCK_SIZE 的向量，自动逐元素广播
```

后面所有的差异——能做和不能做——都是从这一条推出来的。

---

## 2. 同一条 vector add，两种写法

### 2.1 CUDA 版

取自本仓库 [examples/leetgpu/01_vector_add/vector_add.cu](../../examples/leetgpu/01_vector_add/vector_add.cu)：

```cuda
__global__ void vector_add_naive(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 long long N) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {              // ← 尾部保护，这是你自己写的
        C[i] = A[i] + B[i];
    }
}
// 启动：vector_add_naive<<<cdiv(N, 256), 256>>>(A, B, C, N);
```

### 2.2 Triton 版

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(A, B, C, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                          # ← 相当于 blockIdx.x
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N                                 # ← 尾部保护，一个布尔向量
    a = tl.load(A + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)
    tl.store(C + offs, a + b, mask=mask)

# 启动：一个 program 一个 block
grid = (triton.cdiv(N, BLOCK_SIZE),)
vector_add_kernel[grid](A, B, C, N, BLOCK_SIZE=1024, num_warps=4)
```

注意 `a + b` —— 这是**两个向量相加**，不是两个 float 相加。`mask` 也不是一个
per-thread 的 if，而是一整个布尔向量，由编译器翻译成谓词化的 load/store。

### 2.3 骨架对照

| 概念 | CUDA | Triton |
| --- | --- | --- |
| block 编号 | `blockIdx.x` | `tl.program_id(0)` |
| block 大小 | `blockDim.x`（**运行时**启动参数） | `BLOCK_SIZE`，必须 `tl.constexpr`（**编译期常量**） |
| 线程编号 | `threadIdx.x`（显式） | **不存在**，编译器分配 |
| 一个 block 的线程数 | `blockDim.x` | `num_warps × 32`（启动参数，不是 `tl.constexpr`） |
| 尾部保护 | `if (i < N)` | `mask = offs < N` |
| 循环 | `for` | `for`，但循环体作用在张量上 |
| 共享内存 | `__shared__ float s[32][33];` 手动管理 | **没有显式语法**，编译器按数据复用自动分配 |
| 编译时机 | AOT（离线 `nvcc`） | JIT（首次调用时按具体参数编译） |

两处最容易被忽略的差异：

**`BLOCK_SIZE` 是编译期常量。** 改它要重新编译（JIT 缓存按参数分桶）。所以它不能像
CUDA 的 `blockDim` 那样「运行时看着形状再决定」——一个 kernel 想要多套 block 尺寸，
要么靠 `@triton.autotune` 声明候选，要么在 Python 侧分派。

**索引宽度。** `tl.arange` 默认是 int32，`pid * BLOCK_SIZE` 也是 int32。
上面那段代码在 N > 2³¹（约 21 亿）时会溢出，而 CUDA 版里我们特意用了 `long long`。
要修的话是 `offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`。

---

## 3. Triton 替你做了哪些事

这五件事在 CUDA 里都是「不写就错、写错就慢」，在 Triton 里是默认行为。

| 事情 | CUDA 里你得手写 | Triton 里 |
| --- | --- | --- |
| **合并访存** | 保证相邻线程访问相邻地址，否则带宽掉一个数量级 | 由 layout 推断保证，你写 `offs = pid*BLOCK + arange(...)` 就是这个形状 |
| **Shared memory 分配** | `__shared__`、手工算容量、算容量是否超限 | 编译器看数据复用决定要不要用、用多少 |
| **Bank conflict 规避** | 自己写 `[32][33]` 这种 padding（见 02 第 2 节） | 编译器选 padding，通常比手写的好 |
| **软件流水线** | 手写双缓冲：`cp.async` + 两组 buffer + barrier 轮转（见 02 第 5 节） | `num_stages=3`，一个参数 |
| **Tensor Core** | 手写 `mma` 的 fragment 布局（见 [03-gemm.md](03-gemm.md)） | `tl.dot(a, b)` |
| **调参** | 自己写 benchmark 脚本扫参数 | `@triton.autotune` 扫 `BLOCK_SIZE` / `num_warps` / `num_stages` |

**这解释了第 2 节里那个「白送」**：vector add 是纯带宽受限的（你的实测已经证明了——
朴素版和 float4 版跑出一样的 394 GB/s，因为瓶颈在 DRAM 不在指令发射）。
Triton 自动生成的代码天然是合并访存的，也就自然撞上同一个天花板。
**这类 kernel 上 Triton 没有性能代价，只有代码量的优势。**

反过来说：**如果某个 kernel 的瓶颈是 Triton 编译器不管的东西，它就不免费了。**
这就是下一节。

---

## 4. Triton 的边界在哪

「编译器替你决定」的另一面是「你决定不了」。以下是硬边界，不是调参能绕过的：

| 做不到的事 | 为什么 |
| --- | --- |
| **线程级的分支与发散控制** | 没有 `threadIdx`。你能按元素值做选择（`tl.where`），但不能让不同的 lane 走真正不同的代码路径、拿不同的寄存器分配 |
| **Warp 级原语** | 没有 `__shfl_sync` / `__ballot_sync` / `__match_any_sync`。归约只能用 `tl.sum` / `tl.max` 这种整块形式，做不了手写的 warp 内 scan |
| **显式 shared memory** | 没有 `extern __shared__`，没有手工 layout。想把自己算好的中间结果摆在 shared 里给邻居读——表达不了 |
| **内联汇编 / 特定指令** | 没有 `asm volatile`。`tl.inline_asm_elementwise` 存在，但只作用于逐元素，够不着 `cp.async.bulk` / TMA 这类有状态的东西 |
| **协作组 / grid 同步** | 和 CUDA 一样没有 grid 级同步，但 CUDA 至少有 cooperative groups 这个正规路径 |
| **数据依赖的循环边界** | 循环边界要能被编译器静态推断（`tl.range` + 运行时边界可以，但能力受限）。指针追逐、不规则稀疏这类负载基本没法写 |
| **调试** | 没有 cuda-gdb 单步你的 Triton 源码，没有 printf 式断点（`tl.device_print` 有限）。看 SASS 时你看到的是编译器的意图，不是你的 |

还有一类**移动的边界**：Triton 每个版本都在往里加东西（TMA descriptor、warp
specialization 都在路上）。所以「Triton 不能做 X」这句话有保质期，查最新文档为准。

---

## 5. 什么时候 Triton 会输：一个真实案例

vLLM 的 MoE W4A16 路径上，代码一度从手写 CUDA kernel 迁到了 Triton 版
（[PR #44120](https://github.com/vllm-project/vllm/pull/44120)），结果**小 batch 性能崩了**。
[PR #53693](https://github.com/vllm-project/vllm/pull/53693) 又把运行时按 token 数分派的
逻辑加回来才修好。RTX 3090 上的数据：

| 输入 token 数 | CUDA W4A16 | Triton | 谁快 |
| --- | --- | --- | --- |
| 1 | **0.272 ms** | 0.473 ms | CUDA，1.74× |
| 8 | — | — | CUDA，约 1.77× |
| 64 | — | — | CUDA，约 1.59× |
| 192 以上 | — | — | **Triton** |

为什么小 batch 上手写 CUDA 能赢这么多？**因为那时候没有并行度可藏。**

decode 阶段 batch=1 时，整个 GPU 上只有极少量有效工作。Triton 生成的代码是按
「一个规整的 tile」来做的——它的 block 划分、流水线深度、寄存器预算是针对**吃满 GPU**
的场景调出来的。工作量为空或极小时，这些假设全部落空，而手写 kernel 可以把
「几乎没活干」这条路径专门优化掉（比如提前退出、缩小 tile、跳过流水线）。

**这就是选型的第一性判断：**

> Triton 的优势来自「编译器比你更懂常规形状的优化」。
> 当负载是**常规且规模足够**时，这个优势是纯赚；
> 当负载**小到撑不起并行度**、或者**形状不规整**时，编译器的假设反而成了负担。

---

## 6. vLLM 里到底谁用谁

不是「迁到 Triton」或「留在 CUDA」的单向叙事，而是按 kernel 特性分工：

| kernel | 在哪 | 为什么 |
| --- | --- | --- |
| **Paged attention**（`triton_unified_attention`、`triton_decode_attn_stage1/2`、`triton_paged_prefix_prefill`） | **Triton** | 整个 attention backend 用 Triton 写，同一份源码跑 NVIDIA / AMD / Intel；在 ROCm 上是默认，在 NVIDIA 上是 FlashAttention / FlashInfer 不可用时的回退，也是 fp32 场景的唯一选择 |
| **KV cache 读写**（`reshape_and_cache`、`swap_blocks`） | `csrc/` CUDA | 纯搬运，且要和 block table 这种非张量结构打交道 |
| **RMSNorm / SiLU-and-mul / RoPE / top-k** | `csrc/` CUDA | 元素级或小归约，形状固定，手写没有额外成本 |
| **量化 GEMM**（Marlin、GPTQ、AWQ、W4A16） | 两者都有，**按 batch 大小运行时分派** | 就是第 5 节那个案例：小 batch 走 CUDA，大 batch 走 Triton |
| **Fused MoE** | 两者都有 | 同上 |

**规律很清楚：Triton 赢在「形状规整 + 规模够大 + 需要跨平台」；CUDA 赢在
「纯搬运 / 元素级 / 小 batch / 需要和非常规数据结构打交道」。**

顺带一个值得记住的工程结论：**vLLM 的 Triton 版 attention 不是「性能备选」，
而是「可移植性主力」。** 它的存在让 vLLM V1 能在 AMD 和 Intel 上跑起来——
这本身就说明了 Triton 的核心价值：**用一份代码覆盖多种硬件**，而不只是少写点代码。

---

## 7. 在你这台机器上跑起来

现实提醒：**这篇文章写的时候，你这台机器跑不了 Triton。** 和
[01-vector-add](../leetgpu/01-vector-add.md) 里踩的坑同源——那里是官方评测脚本
依赖 CUDA 版 PyTorch，而你的 torch 是 CPU-only 的：

```
torch 2.6.0+cpu | built for CUDA None
cuda available: False
```

Triton 依赖 CUDA 版 PyTorch，所以先得有它。而且**Windows 上还多一层坑**：

| 事实 | 说明 |
| --- | --- |
| 官方 Triton **只有 Linux 版** | Windows 要用社区 fork [triton-windows](https://github.com/triton-lang/triton-windows)（已迁到 `triton-lang` 组织下，`3.2.0.post11` 起上 PyPI，pip 能直接装到） |
| **sm_120 需要 Triton ≥ 3.3** | 同时要 PyTorch ≥ 2.7、CUDA ≥ 12.8（你的 CUDA 12.8 ✓） |
| PyTorch 和 Triton 版本必须配对 | 2.7→3.3 / 2.8→3.4 / 2.9→3.5 / 2.10、2.11→3.6。装错版本会静默降级 PyTorch，整套环境就废了 |
| PyTorch **stable** 曾经不含 sm_120 kernel | 2026 年的多份报告显示 RTX 50 系要装 nightly（`--pre ... /nightly/cu130`）。装之前先确认当前 stable 是否已覆盖 sm_120 |

所以你这台机器要跑 Triton，大致是：

```bash
# 1. 换掉 CPU-only 的 torch，装能覆盖 sm_120 的版本（stable 不行就 nightly）
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130

# 2. 装配套的 triton-windows（版本号必须跟上面的 torch 对上，见上表）
pip install -U "triton-windows<3.7"

# 3. 跑第 2.2 节那段 vector_add，对比你自己的 CUDA 版 394 GB/s
```

**建议在 WSL2 里做**，理由和 01 里那组 WSL2 实测数据一样：官方 Triton 原生支持
Linux，不用赌 fork 的版本配对，出问题时的报错也是社区里有人见过的。
你这台机器 WSL2 里已经有 CUDA 12.8，够用了。

---

## 8. 怎么选

**默认选 Triton**，除非命中下面任一条：

- [ ] 负载**小到撑不起并行度**（decode 的 batch=1、小 n），且实测过手写更快 → CUDA
- [ ] kernel 是**纯数据搬运**（copy / reshape / cache 读写），没有计算要优化 → CUDA
- [ ] 需要**线程级分支发散**、warp shuffle、手写 scan → CUDA
- [ ] 需要**手工摆布 shared memory** 的 layout → CUDA
- [ ] 需要**特定指令**（TMA、`cp.async.bulk`、特定 mma 形态） → CUDA
- [ ] 要接**非常规数据结构**（block table、指针追逐、稀疏） → CUDA
- [ ] 需要**跨平台**（AMD / Intel） → **Triton**，这条反过来压过上面几条
- [ ] 团队里**没人会写 CUDA** → Triton，能跑起来的 90% 胜过写不出来的 100%

**最重要的一条：先测量。** 第 5 节那个 1.74 倍的案例，是在迁移**已经发生、性能已经
回归之后**才被测出来的。凭直觉猜「Triton 更现代所以更慢/更快」在这两个方向上都会错。

---

## 参考

- [vLLM Triton backend deep dive](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-03-04-vllm-triton-backend-deep-dive.md)
- [vLLM PR #53693 · 恢复 WNA16 的 CUDA 分派](https://github.com/vllm-project/vllm/pull/53693)
- [triton-lang/triton-windows](https://github.com/triton-lang/triton-windows)
- [Triton 官方文档](https://triton-lang.org/main/getting-started/tutorials/index.html)

**相关篇章：** [02-cuda-basics.md](02-cuda-basics.md)（执行模型，第 1 节的基础）、
[03-gemm.md](03-gemm.md)（手工 mma 的复杂度，对照 `tl.dot`）、
[08-profiling.md](08-profiling.md)（怎么测出第 5 节那种回归）。
