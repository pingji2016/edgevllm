# 06 · 算子融合与内存布局

带宽瓶颈下，**最省时间的办法是根本不搬这一趟**。算子融合和内存布局是两把主要的刀。

## 1. 为什么融合有效：用一个例子算清楚

RMSNorm 后接一个 Linear：

```python
# 不融合
h = rms_norm(x)        # 读 x (N, 4096)，写 h (N, 4096)
y = linear(h, W)       # 读 h

# 流量（BF16，N=1）
rms_norm:  读 8 KB + 写 8 KB = 16 KB
linear:    读 8 KB + 读权重 33 MB
```

看起来 `rms_norm` 只占 0.05%？**在 batch=1 时权重 33 MB 是主导，确实只有 0.05%。**

现在换成 batch=256：

```text
rms_norm:  读 2 MB + 写 2 MB = 4 MB
linear:    读 2 MB + 权重 33 MB = 35 MB
总流量 39 MB，其中 rms_norm 贡献 4 MB ≈ 10%
```

再叠上 MLP 里的 `silu_and_mul`、残差 `add`，**一堆占 3~10% 的小算子加起来就是 30%+**。

> 融合的价值不在单个算子有多快，而在**消掉中间张量的写+读**。

## 2. 该融合什么：判断标准

一个 elementwise 算子值得融合，当且仅当：

```text
省下的 = 2 × (中间张量大小) 字节的读写
代价   = 多占的寄存器 / 更复杂的 kernel
```

**排序原则**（收益从高到低）：

| 融合对象 | 典型收益 | 说明 |
| --- | --- | --- |
| 归一化 + 量化 | 高 | 省掉一次全量读写，且隐藏量化开销 |
| 残差 add + 归一化 | 高 | 每个 block 都跑，次数多 |
| 激活 + 门控（SiLU+Mul） | 中高 | MLP 里两个大张量合并成一个 |
| RoPE + KV cache 写入 | 中高 | 省掉 transpose 的中间张量 |
| 两个相邻 elementwise | 中 | 收益小，除非张量很大 |
| GEMM + bias + 激活 | 中 | cuBLAS 的 epilogue 已经能做一部分 |

**不该融合的**：

- 融合后寄存器溢出、occupancy 掉太狠 → 得不偿失
- 融合跨越了不同的并行维度（比如 GEMM 的 block 划分和 layernorm 的 block 划分不匹配）
- 只是为了「少一个 kernel launch」（3 微秒），却让 kernel 慢 20%

## 3. 具体案例

### 案例 1：`fused_add_rms_norm`

vLLM 的 `csrc/layernorm_kernels.cu`：把「残差相加」和「RMSNorm」合成一个 kernel，并且**顺便把残差结果写回**（下一层的输入需要它）。

```cuda
template <typename scalar_t>
__global__ void fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,          // 既是输入也是输出
    scalar_t* __restrict__ residual,       // 原地更新残差
    const scalar_t* __restrict__ weight,
    float eps, int hidden_size) {

    const int row = blockIdx.x;
    __shared__ float s_variance;

    // 1) 同时做残差加法和平方和
    float variance = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = to_float(residual[row * hidden_size + i]);
        x += to_float(input[row * hidden_size + i]);
        residual[row * hidden_size + i] = from_float(x);   // ← 残差就地更新，不另开 kernel
        variance += x * x;
    }
    variance = blockReduceSum(variance);      // warp shuffle + 少量 SMEM
    s_variance = rsqrtf(variance / hidden_size + eps);

    // 2) 归一化并写回 input
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        input[row * hidden_size + i] =
            from_float(to_float(residual[row * hidden_size + i]) * s_variance * to_float(weight[i]));
}
```

一箭三雕：省了一次残差写的 kernel、省了读回残差的流量、`weight` 从 SMEM/常量缓存读（广播，不吃带宽）。

### 案例 2：`silu_and_mul`

Gated MLP 里 `gate_proj` 和 `up_proj` 的输出要相乘：

```python
# 不融合
a = silu(gate_out)      # 读 4N，写 4N
b = a * up_out          # 读 4N + 4N，写 4N
# 总流量 20N bytes

# 融合
out = silu_and_mul(gate_out, up_out)   # 读 4N + 4N，写 4N
# 总流量 12N bytes → 省 40%
```

vLLM 里对应 `vllm/model_executor/layers/activation.py` 的 `SiluAndMul`，底层调 `csrc/activation_kernels.cu`。

### 案例 3：RoPE + KV cache 写入

```text
朴素：rotate → transpose → reshape → scatter 进 cache   （4 个 kernel，2 次中间写读）
融合：一个 kernel 里算 cos/sin、旋转、直接写到 slot_mapping 指定的位置
```

见 `csrc/pos_encoding_kernels.cu` 的 `rotary_embedding` 与 `csrc/cache_kernels.cu` 的 `reshape_and_cache`。

**注意 vLLM 里有两种 RoPE 布局**：`NORMAL`（`[..., head_dim]`）和 `NEOX`（`[..., head_dim/2]` 两半交错）。搞错会导致模型输出乱码但 shape 全对 —— 排查时优先查这个。

## 4. 内存布局：看不见的性能杀手

### 布局决定访存模式

同一个张量，`[B, H, N, D]` 和 `[B, N, H, D]` 在 attention kernel 里的合并访存表现完全不同：

```text
[B, H, N, D]:  一个 head 内的 N 个 token 不连续 → attention 要按 stride 跳着读
[B, N, H, D]:  同一 token 的所有 head 连续 → decode 时（Q 只有 1 行）读起来更顺
```

**这就是 vLLM 里 `reshape_and_cache` 存在的全部理由** —— 把 `[num_tokens, num_heads, head_size]` 的 KV 写进按 block 组织的 cache。

### KV cache 的两种布局

```text
vLLM 默认（cache_kernels.cu）:
  key_cache   : [num_blocks, block_size, num_kv_heads, head_size]
  value_cache : [num_blocks, block_size, num_kv_heads, head_size]
  → 一个 block 内的 16 个 token 连续，适合 attention 顺序扫描 KV

xformers 风格:
  [num_blocks, num_kv_heads, head_size/x, block_size, x]
  → 为了匹配 FlashAttention 的 SMEM 加载模式
```

选哪个取决于 attention kernel。**改布局时一定要同时改 kernel，否则会静默地读错数据**（shape 检查抓不到）。

### 布局转换的成本

```python
x = x.transpose(1, 2)      # 只是改 stride，零开销
y = x.contiguous()         # 真实的数据搬移，全量读写！
```

**`.contiguous()` 在关键路径上等于一次额外的全量访存。** 让它发生在权重加载阶段（离线），而不是每次前向。

排查方法：

```python
# 找出所有触发 copy 的地方
torch.autograd.profiler.profile(use_cuda=True)  # 看 aten::copy_
```

## 5. 用 CUDA Graph 消掉 launch 开销

decode 时每个 token 要跑几百个小 kernel，**每个 3~10 微秒的 launch 开销累积起来能到毫秒级**。

```python
# vLLM 里（简化）
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_output = model(static_input)   # 捕获整张计算图

# 之后每次只需要：
static_input.copy_(real_input)
graph.replay()                            # 一次提交，CPU 开销接近 0
```

要求与坑：

- **所有张量地址必须固定** —— 所以要预分配 static input/output buffer 并 copy 进去（这次 copy 本身有开销，但远小于几百次 launch）
- **控制流必须静态** —— Python 里的 `if batch_size > 4:` 会导致捕获到错误的分支
- **要用独立的显存池** —— `torch.cuda.graph_pool_handle()`
- 通常**按 batch size 分桶**捕获多张图（比如 batch ∈ {1,2,4,8}），运行时选最接近的那张

vLLM 里对应 `vllm/worker/model_runner.py` 的 `capture_model` / `CUDAGraphRunner`。

## 6. 融合的极端：写一个 fused MLP kernel

理论上可以把 `gate_proj + up_proj + silu + mul + down_proj` 全塞进一个 kernel：

```text
好处：中间激活完全不落显存
代价：需要一个 block 同时持有 gate W 和 up W 的 tile → SMEM 放不下
      → 只能靠 L2 缓存，可控性差
```

**现实做法**：把 `gate_proj` 和 `up_proj` 的权重拼成一个大矩阵 `[2*d_ff, d_model]`，**一次 GEMM 出两半**，再融合 `silu_and_mul`：

```python
# 权重加载时拼好（离线，零运行时开销）
W_combined = torch.cat([gate_proj.weight, up_proj.weight], dim=0)   # [2*d_ff, d]

# 前向
gate_up = x @ W_combined.T          # 一次 GEMM，N 翻倍 → 算术强度更好
out = silu_and_mul(gate_up)         # 一次 elementwise，融合
```

这一招在 vLLM 里叫 `MergedColumnParallelLinear`。**它同时改善了两个东西**：GEMM 的 N 更大（效率更高）、少一个 kernel。

## 检查清单

- [ ] 跑过 profiler，看小算子加起来占了多少吗？（用 [08-profiling.md](08-profiling.md)）
- [ ] 关键路径上有 `.contiguous()` / `.transpose()` 吗？
- [ ] 残差、norm、激活这几处，有没有合成一个 kernel？
- [ ] 能拼的权重（gate/up、QKV）在加载时拼好了吗？
- [ ] RoPE 的布局（NORMAL vs NEOX）和模型匹配吗？
- [ ] 用 CUDA Graph 了吗？static buffer 的 copy 开销测过吗？
- [ ] 融合后 occupancy 掉了吗？掉太多的话，不融合可能更快。
