# 03 · GEMM 与矩阵乘优化

LLM 里 90% 的 FLOPs 和 90% 的访存都发生在矩阵乘上。这一篇分三种形态讲：
**大 GEMM（prefill）**、**GEMV / 瘦 GEMM（decode）**、**Batched GEMM（MoE、多 LoRA）**。

## 1. 为什么朴素 GEMM 慢

朴素实现每个输出元素都要读一整行 A 和一整列 B：

```text
算术强度 I = 2MNK / (2MNK·... ) ≈ O(1)   —— 每个乘加都要两次全局访存
```

A100 的拐点是 153，我们在 1 附近，**慢了 150 倍**。

优化的全部目的就一句话：**把数据搬进片上（寄存器 / shared memory），然后反复用。**

## 2. 分块（Tiling）：经典的三层结构

```text
全局显存 (HBM)
   ↓  tile A[BM, BK] + B[BK, BN]        ← 合并访存，一次搬一大块
Shared Memory (SMEM)
   ↓  每个线程取一小块到寄存器           ← 复用率在这里产生
Register (每个线程持有 A_frag[TM, TK], B_frag[TK, TN])
   ↓
累加 C[TM, TN]
```

三个层级的尺寸是**调参的核心**：

| 参数 | 典型值 (A100, FP16) | 作用 |
| --- | --- | --- |
| `BM × BN` (block tile) | 128 × 128 或 256 × 128 | 决定 SMEM 占用和 L2 复用 |
| `BK` | 32 / 64 | 决定流水线深度，越大越省访存但 SMEM 吃紧 |
| `TM × TN` (每线程) | 8 × 8 或 8 × 4 | 决定寄存器复用率，太小则访存占比高 |

每搬一个字节能做多少次乘加：

```text
复用率 = (2 · BM · BN · BK) / ((BM + BN) · BK · 2 bytes)   # FP16
       = BM·BN / (BM + BN)    次/元素
```

128×128 的 tile 里，每个元素被复用 **64 次** —— 这就是为什么分块能带来数量级的提升。

Block tile 越大越好吗？不是。BM×BN 受限于：

- 寄存器总量（128×128 的累加器，256 线程下每线程 64 个寄存器，只剩一半可用）
- SMEM 容量（A100 每 SM 164 KB）

**常见坑：改成 256×256 后反而变慢** —— 因为寄存器溢出（register spilling），局部变量被扔回本地内存。

## 3. Tensor Core：用 `mma` 指令

从 Volta 开始，矩阵乘的最小单位是一条 warp 级指令：

```cuda
// A100 (sm80)，FP16 输入 FP32 累加
// m16n8k16：一个 warp 计算 16×8 的输出，K=16
__device__ void mma_m16n8k16(float* c, const half* a, const half* b) {
    uint32_t const* A = reinterpret_cast<uint32_t const*>(a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(b);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]));
}
```

要点：

- **每一个 Tensor Core 指令都要求特定的寄存器排布**（fragment layout）。手动拼这些 layout 极其痛苦，所以实践中都用 CUTLASS 或 Triton。
- FP16 输入的 K 维度必须是 8 的倍数（`m16n8k16` 的 k=16，实际按 k=8 切两次更好用）。
- 累加器必须是 FP32 —— **FP16 累加的误差会在 K=4096 时爆掉**。

### 用 Triton 写，性价比最高

```python
import triton, triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'GROUP_M': 8},
                      num_stages=4, num_warps=8),
        triton.Config({'BM': 64,  'BN': 64,  'BK': 32, 'GROUP_M': 8},
                      num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    # Swizzle：让同一批 block 落到同一列，提高 L2 命中率
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(a, b, acc)              # ← 自动映射到 Tensor Core
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

`num_stages=4` 就是 Triton 帮你做的 4 级软件流水线（等价于手写 double/triple buffering）。

**关键收获：`GROUP_M` swizzle。** 不做这个的话，block 按行优先调度，同一时刻活跃的 block 会在 B 矩阵上重复读同一块 L2 —— 加上 swizzle 后 L2 命中率通常能从 60% 提到 90%+。

## 4. Split-K：当 M 和 N 都很小时

decode 阶段 `M` 很小（batch × 1），`K` 很大（hidden dim）。此时 block 数量 = `(M/BM) × (N/BN)` 可能只有几十个，**填不满 GPU**。

解决：沿 K 方向切，每个 block 算部分和，最后用 atomic 或第二个 kernel 归约。

```text
grid = (M/BM) × (N/BN) × SPLIT_K
每个 block 累加 K/SPLIT_K 段 → 写 partial[split_k] → 归约
```

代价：

- 要写额外的 partial buffer（`M × N × SPLIT_K × 4` 字节），**这本身是访存开销**，带宽紧张时得不偿失
- 需要确定性的归约顺序时（vLLM 要求 batch 不变时输出可复现），不能用 atomicAdd，得用单独 kernel

经验：只有当 `SM 利用率 < 30%` 时拆分才值得。**vLLM 里 decode 的 GEMM 一般不拆 K，而是改用下面说的 GEMV 路径。**

## 5. GEMV / 瘦 GEMM：decode 的真正形态

decode 时 `M = batch_size`（1~64），而且它压根不是算力问题 —— 见 [01-roofline.md](01-roofline.md)。

此时优化目标变成：**让权重的每一个字节只被读一次，且读满带宽。**

```cuda
// 一个 block 负责输出的一小段，每个线程读权重的一部分
// 关键：权重按行连续读 → 天然合并访存
__global__ void gemv(const half* __restrict__ W,   // [N, K]
                     const half* __restrict__ x,   // [K]
                           float* __restrict__ y,  // [N]
                     int N, int K) {
    int row = blockIdx.x;
    float acc = 0.f;
    const half2* W2 = reinterpret_cast<const half2*>(W + row * K);
    const half2* x2 = reinterpret_cast<const half2*>(x);
    for (int i = threadIdx.x; i < K / 2; i += blockDim.x)
        acc += __half2float(__low2half(W2[i])) * __half2float(__low2half(x2[i]))
             + __half2float(__high2half(W2[i])) * __half2float(__high2half(x2[i]));
    // warp/block 归约后写 y[row]
}
```

真实场景要用 **Marlin**（vLLM `csrc/quantization/marlin/`）：

- 权重按 Tensor Core 需要的 fragment layout **预先重排**，运行时零开销
- 反量化和 `mma` 融合在一条流水线里，SMEM 里只放还没反量化的 INT4
- 在 A100 上跑 W4A16，decode 吞吐相比 FP16 cuBLAS 能到 **3~4 倍**

## 6. Batched GEMM（MoE / 多 LoRA）

MoE 层里每个专家只处理一部分 token，形状不规整：

```python
# 朴素做法：循环跑 N 次 cuBLAS —— 每次 launch 都有开销，小 batch 时完全被吞掉
for expert_id in range(num_experts):
    tokens = hidden[expert_id == chosen]     # 形状每次都不同
    out[expert_id == chosen] = tokens @ W[expert_id].T

# 更好的做法：一次 grouped GEMM
# CUTLASS 的 GroupedGemm / Triton 的 persistent grouped matmul
# 把 (M_e, N, K) 的每组任务「拼」到一个 grid 里，block 通过 scheduler 动态领任务
```

要点：

- **Triton 里用「persistent kernel + 原子计数器领任务」**（`tl.atomic_add` 抢 tile id）比静态切分更均衡
- MoE 的 `M_e` 分布极不均匀（有的专家几百个 token，有的 0 个），静态切分会导致长尾
- vLLM 的 `fused_moe_kernel` 就是这么做的，参考 `vllm/model_executor/layers/fused_moe/`

## 7. 一张表总结该用哪个

| 场景 | M 范围 | 瓶颈 | 方案 |
| --- | --- | --- | --- |
| Prefill GEMM | 512–8192 | 算力 | cuBLAS / CUTLASS / Triton + Tensor Core |
| Decode GEMM（FP16） | 1–64 | 带宽 | cuBLAS (gemv 路径) 或手写 GEMV |
| Decode GEMM（量化） | 1–64 | 带宽 | **Marlin / GPTQ / AWQ 专用 kernel** |
| MoE | 不定 | 算力 + 负载均衡 | Grouped GEMM + persistent kernel |
| 多 LoRA / 多任务 | 小 | 带宽 | Batched GEMM + 权重常驻 SMEM |

## 检查清单

- [ ] 算过理论时间上限了吗？（`FLOPs/Peak` 与 `Bytes/BW` 取大者）
- [ ] Block tile 的 L2 swizzle（`GROUP_M`）加了吗？
- [ ] K 循环用 `num_stages` / double buffering 了吗？
- [ ] 累加器是 FP32 吗？（FP16 累加在长 K 上精度会崩）
- [ ] M 小的时候，SM 占用率是不是低到该改 GEMV / Split-K？
- [ ] TF32 有没有显式开启？`torch.backends.cuda.matmul.allow_tf32 = True`（默认关闭，这是很多人「FP32 慢 8 倍」的原因）
