# 01 · Roofline 模型与性能瓶颈判断

优化的第一步不是写代码，是回答一个问题：**这个算子到底受限于算力还是受限于带宽？**
Roofline 模型就是回答这个问题的工具。

## 1. 算术强度（Arithmetic Intensity）

```text
I = FLOPs / Bytes        # 单位：FLOP/Byte
```

- `FLOPs`：这次计算总共要做多少次浮点运算
- `Bytes`：为了完成计算，必须从显存（HBM）搬进搬出多少字节

把 `I` 画在横轴上，纵轴是实际达到的性能，你会得到一条「屋顶线」：

```text
性能
(FLOPS)  ┌─────────────── 算力天花板 (Peak FLOPS)
         │          ______________
         │        /
         │      /  ← 带宽斜线 (Peak BW)
         │    /
         │  /
         └──┴──────────────────────────→ 算术强度 I
          I*          (拐点)
```

拐点：

```text
I* = Peak FLOPS / Peak BW
```

| 设备 | 带宽 | FP16 TC 算力 | 拐点 I* (FP16) | FP32 拐点 |
| --- | --- | --- | --- | --- |
| A100-SXM-80G | 2039 GB/s | 312 TFLOP/s | ≈ 153 | ≈ 9.6 |
| H100-SXM | 3350 GB/s | 989 TFLOP/s | ≈ 295 | ≈ 20 |
| RTX 4090 | 1008 GB/s | 330 TFLOP/s | ≈ 327 | ≈ 41 |
| Jetson Orin AGX 64G | 204 GB/s | 42 TFLOP/s (稀疏) | ≈ 200 | ≈ 16 |

**落在斜线左侧 = 带宽瓶颈（memory bound），右侧 = 算力瓶颈（compute bound）。**

优化的方向完全不同：

- 带宽瓶颈 → 减少字节数（量化、融合、提高复用），继续压榨算力毫无意义
- 算力瓶颈 → 上 Tensor Core、减少无效 FLOPs（稀疏化、算法改进）

## 2. LLM 推理的两个阶段，瓶颈完全相反

### Prefill（处理 prompt）

一次处理 N 个 token，矩阵乘是 `[N, d] × [d, d_out]`，N 通常几百到几千。

```text
算术强度 ≈ 2·N·d·d_out / (2·N·d + 2·d·d_out)  ≈ N   (当 N ≪ d_out 时)
```

N=512 时 I ≈ 512，远大于拐点 → **compute bound**，Tensor Core 利用率是核心指标。

### Decode（逐 token 生成）

batch=1 时每个权重只用一次：

```text
I = 2 FLOP / 2 Byte = 1 FLOP/Byte    (FP16)
```

比拐点低两个数量级 → **极端 memory bound**。此时：

```text
理论最高 tokens/s ≈ Peak_BW / 模型权重大小(字节)
```

70B 模型 FP16 权重 140 GB，A100 上：`2039 / 140 ≈ 14.5 tokens/s`，这就是单卡上限，**跟算力一点关系都没有**。

这条公式解释了几件事：

- W4A16 量化把 140 GB 压到 36 GB，理论上能到 56 tokens/s —— vLLM 里 marlin kernel 的价值就在这
- 增大 batch 能把权重摊薄，是小幅提升吞吐的主要手段
- 在边缘设备（带宽可能只有 20–200 GB/s）上，**量化不是可选项，是必需品**

## 3. 快速估算：眼见为实的数字

做优化前先算一遍「理论上限」，能省掉大量无用功。

```python
def gpu_limit_flops(flops, peak_tflops):
    return flops / (peak_tflops * 1e12) * 1e3   # ms

def gpu_limit_mem(bytes_moved, bw_gbps):
    return bytes_moved / (bw_gbps * 1e9) * 1e3  # ms

def gemm_flops(M, N, K):
    return 2 * M * N * K

def gemm_bytes(M, N, K, elem=2):
    return elem * (M*K + K*N + M*N)   # 输入 + 权重 + 输出

# Llama-2-7B, decode, batch=1, 一个 linear 层 (4096 -> 4096)
F = gemm_flops(1, 4096, 4096)          # 33.5 MFLOP
B = gemm_bytes(1, 4096, 4096)          # 33.6 MB
print(gpu_limit_flops(F, 312e0))       # 0.0001 ms  算力根本不是问题
print(gpu_limit_mem(B, 2039))          # 0.0165 ms  实际耗时由这里决定
```

**结论写在代码里：算力只要 0.1 微秒，搬数据要 16.5 微秒。差 165 倍。**

## 4. 判断方法（不需要跑 profiler）

拿到一个 kernel，用下面三步 5 分钟内就能定位瓶颈：

1. **算算术强度** —— 有多少 FLOPs，必须搬多少字节。
2. **和拐点比** —— 远小于拐点 → 别再想 Tensor Core 了；远大于 → 别再想合并访存了。
3. **量实际时间** —— `实际时间 vs 理论带宽时间`：
   - 接近 1.0 → 已经打满带宽，只能靠减少字节数（量化 / 融合）
   - 只有 0.2~0.4 → 访存没优化好，去查合并访存、向量化、缓存命中率

经验阈值：**实际带宽利用率 < 60% 时，先修访存；> 80% 时，只考虑减少字节数或改算法。**

## 5. 常见误判

| 现象 | 误判 | 真相 |
| --- | --- | --- |
| GEMM 只有 5% 峰值算力 | "Tensor Core 没调好" | batch=1，根本是 GEMV，算力指标无意义 |
| 加了 `float4` 反而变慢 | "向量化没用" | 对齐没处理，或者寄存器压力上去导致 occupancy 掉 |
| Attention 加 FlashAttention 没提升 | "FlashAttention 被吹过头了" | seq_len 太短（<512），本来就没到 memory bound |
| 换了更快的库但端到端没变 | "库没用" | 该 kernel 只占端到端 3%，Amdahl 定律 |

> Amdahl 定律提醒：**优化前先看 profiler 的耗时占比**。优化一个占 3% 的算子，就算快 10 倍，端到端也只快 2.7%。

## 下一步

- 带宽瓶颈的通用修法 → [02-cuda-basics.md](02-cuda-basics.md)
- 算力瓶颈的通用修法 → [03-gemm.md](03-gemm.md)
- 边缘设备上的实际数字 → [07-edge.md](07-edge.md)
