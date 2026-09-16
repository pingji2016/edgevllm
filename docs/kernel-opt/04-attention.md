# 04 · Attention 算子优化

Attention 是 LLM 里唯一一个**序列长度平方复杂度**的算子，也是优化收益最大的地方。

```text
O = softmax(Q Kᵀ / √d) V
```

## 1. 朴素实现的三个开销

```python
S = Q @ K.transpose(-1, -2)     # [B, H, N, N] ← 写回显存
P = softmax(S, dim=-1)          # 读 S，写 P
O = P @ V                       # 读 P
```

对于 `N=4096, H=32, B=1`：

- `S` 和 `P` 各占 `32 × 4096² × 2 bytes = 1 GB`
- 中间张量要**写一次读一次**，共 4 GB 的额外流量
- 对比：Q/K/V 本身只有 3 × 32 × 4096 × 128 × 2 = 100 MB

**流量放大 40 倍。** 这就是 FlashAttention 要解决的问题。

## 2. FlashAttention：不把 S 写出来

核心思路：**分块 + online softmax**，让 `S` 只在 SMEM/寄存器里短暂存在。

### Online softmax 的推导

普通 softmax 需要先看到整行才能算 max 和 sum。分块之后看不到整行，于是维护一个**滚动最大值 `m`** 和**滚动归一化因子 `l`**：

```text
初始化：m = -inf, l = 0, O = 0

对每个 KV 块 j：
    S_j     = Q_i K_jᵀ / √d                    # [BM, BN]
    m_new   = max(m, rowmax(S_j))
    P̃_j     = exp(S_j - m_new)
    l       = exp(m - m_new) · l + rowsum(P̃_j)  # 修正旧的和
    O       = exp(m - m_new) · O + P̃_j V_j      # 修正旧的输出
    m       = m_new

最后：O = O / l
```

那个 `exp(m - m_new)` 修正因子是全部关键 —— **每次最大值变大，之前累积的结果都要按比例缩回去**。

### 为什么它是对的

数学上这与先算完整 softmax 再乘 V 完全等价，误差只来自浮点舍入。`m` 取的是运行最大值，所以 `exp` 的参数恒 ≤ 0，**不会溢出**。

### 收益

| | 朴素 | FlashAttention |
| --- | --- | --- |
| HBM 流量 | O(N²) | O(N²·d²/M) —— 当 `M ≫ d²` 时显著降低 |
| 额外显存 | O(N²) | **O(N)**，不随序列平方增长 |
| 实测加速 | 1× | 2–4×（N=2K, d=128） |

`M` 是 SMEM 大小。以 A100（164 KB SMEM）为例，`d=128` 时 `d² × 4 bytes = 64 KB`，SMEM 里能放下 2 个 KV 块。

### Triton 版骨架

```python
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, Out,
              stride_qz, stride_qh, stride_qm, stride_qk,
              N_CTX, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    q_offset = off_hz * stride_qh + start_m * BLOCK_M * stride_qm
    Q_block = tl.load(Q + q_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        K_block = tl.load(K + ...)
        qk = tl.dot(Q_block, tl.trans(K_block)) * sm_scale

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)          # ← 修正因子

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]           # ← 旧结果缩放
        acc = tl.dot(p.to(V.dtype.element_ty), V_block, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(Out + ..., acc.to(Out.dtype.element_ty))
```

**注意 `acc = acc * alpha[:, None]` 这一步**，这是最容易被漏掉、漏了还会「看起来能跑但结果微妙地错」的地方。

## 3. PagedAttention：KV cache 的内存管理

vLLM 的立身之本。传统做法给每个序列预分配 `max_seq_len` 的连续显存：

```text
问题 1：内部碎片 —— 一个请求只用了 100 个 token，却占了 2048 的空间
问题 2：外部碎片 —— 连续大块分配，显存利用率可能只有 20–40%
问题 3：无法共享 —— 并行采样 / beam search 里前缀明明一样，却各存一份
```

PagedAttention 按操作系统的分页思路，把 KV cache 切成固定大小的 **block**（vLLM 默认 16 个 token 一块）：

```text
逻辑视图（每个序列看来是连续的）        物理视图（显存里散落）
seq A: [0,1,2,3][4,5,6,7][8,9,...]  →  block_table = [7, 12, 3, ...]
seq B: [0,1,2,3][4,5,6,7]          →  block_table = [7, 12]   ← 共享前缀！
```

收益：

- 显存浪费从 ~60% 降到 **< 4%**（只有最后一个 block 可能半满）
- 前缀共享：`fork` 时直接把 block_table 复制一份，写的时候再 copy-on-write
- 显存省下来 → 能塞更多请求 → **吞吐直接翻倍**

代价：

- 访存不再连续，需要 gather。kernel 里要做 block table 查表：

```cuda
// paged attention 的访存模式：逻辑位置 → 物理位置
int block_idx  = logical_pos / BLOCK_SIZE;
int block_off  = logical_pos % BLOCK_SIZE;
int physical   = block_table[seq_id * max_blocks + block_idx];
// 然后从 kv_cache[physical][block_off][head] 取数
```

- 因为 block 内 16 个 token 是连续的，**合并访存仍然成立** —— 这是选 16 而不是 1 的原因

> 实践教训：block size 太小（如 1）访存会退化；太大（如 128）碎片又会回来。16 是权衡后的默认值，vLLM 也支持 8/32/64。

## 4. GQA / MQA：从源头减少 KV cache

```text
MHA：  Q 头 = K 头 = V 头 = 64     每 token 存 64×128×2×2 = 32 KB
GQA：  Q 头 = 64, KV 头 = 8        每 token 存 8×128×2×2  = 4 KB    ← 8× 降低
MQA：  Q 头 = 64, KV 头 = 1        每 token 存 1×128×2×2  = 0.5 KB  ← 64× 降低
```

Llama-2-70B 用的就是 GQA（64 Q 头 / 8 KV 头）。**这是算法层面的优化，比任何 kernel 技巧都有效。**

kernel 实现上要注意：多个 Q 头共享同一个 KV 头，所以 **`kv_head = q_head // (num_q_heads / num_kv_heads)`**，可以让同一个 block 内的不同 warp 处理不同 Q 头、读同一份 K/V —— 天然的 SMEM 复用。

## 5. Decode 阶段的 Attention

decode 时 `Q` 只有 1 行（每个序列），`K/V` 有几千行。这退化成一个 **GEMV + 归约**：

```text
分数 = Q · Kᵀ        # [1, seq_len] —— 与整个 KV cache 做点积
O    = softmax(分数) · V
```

特征：

- 是**极端 memory bound**：要读完整条 KV cache（`seq_len × KV头 × d × 2 × 2 bytes`），而 FLOPs 只有 `2 × seq_len × d`
- 所以 **KV cache 量化收益巨大**：FP16 → FP8 直接省一半带宽（见 [05-quantization.md](05-quantization.md)）
- 同时处理多个序列时，可以把它拼成一个 block-diagonal 的 batched GEMM，但更简单的做法是**每个序列一个 block，按 KV 长度负载均衡调度**

### Flash-Decoding：长序列下更好的并行

当 `seq_len = 128K`、batch = 1 时，只有一个 Q 行，GPU 上只跑得起来极少的 block —— **并行度枯竭**。

Flash-Decoding 沿 KV 维度切分：

```text
1. 把 KV 切成 S 段，每段独立算 partial (m, l, O)
2. 用类似 online softmax 的公式合并 S 个 partial
3. 输出 O = Σ exp(m_s - M) · O_s / Σ exp(m_s - M) · l_s
```

这样并行度从 1 变成 `S × num_heads`。

vLLM 里对应的实现是 `csrc/attention/attention_kernels.cu` 的 `partition` 参数（`--max-num-partitions`）。

## 6. 其它实用技巧

### Causal Mask：少算一半

因果掩码下，位置 `i` 只看 `j ≤ i` 的部分。**分块后大约一半的块完全被 mask 掉**，直接跳过：

```python
for start_n in range(0, start_m * BLOCK_M + BLOCK_M, BLOCK_N):   # ← 上界不用到 N_CTX
    # 只算需要的块
    if start_n + BLOCK_N <= start_m * BLOCK_M:
        ...  # 整块都不需要 mask，走 fast path（不生成 mask 张量）
    else:
        ...  # 边界块，带 mask
```

prefill 阶段这一条能省将近 50% 的算力。

### 滑窗 / 稀疏 Attention

Mistral 的 sliding window、StreamingLLM 的 attention sink，都是把 `N²` 降到 `N·w`。代价是实现复杂度激增，收益取决于任务。

### 别忘了 `√d` 缩放

`sm_scale = 1/√head_dim`。漏了不会崩，但数值会随 `d` 增大迅速变大，softmax 直接饱和成 one-hot。

### KV cache 写入用 fused kernel

`reshape_and_cache` 把「reshape + 写 KV cache」合成一个 kernel：

```cuda
// vllm/csrc/cache_kernels.cu
// key    : [num_tokens, num_heads, head_size]
// slot   : 该 token 在 KV cache 中的物理位置（由 block_table 算出）
// 一个线程写一个 head_size 元素，保证合并访存
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping, ...)
```

拆成 `view` + `index_copy` 会多两次全量读写，融合后省掉。

## 检查清单

- [ ] 用 FlashAttention 了吗？（`vllm` 默认走 FlashAttention / FlashInfer）
- [ ] causal mask 下有没有跳过整块被 mask 的 tile？
- [ ] 分块循环里维护了 `m` / `l` 吗？`acc * alpha` 那步没漏吧？
- [ ] GQA 的 KV 头复用，有没有让多个 Q 头共享 SMEM 里的 K/V？
- [ ] decode 长序列时并行度够吗？需要 flash-decoding 式的 KV 切分吗？
- [ ] KV cache 是 FP16 还是量化过的？长上下文场景下这是最大的带宽项。
- [ ] block size 是 16 还是别的？改过之后测过吞吐吗？
