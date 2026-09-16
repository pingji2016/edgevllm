# 05 · 量化与低精度算子

在 [01-roofline.md](01-roofline.md) 的结论下：**decode 阶段 100% 带宽瓶颈 → 减少字节数 = 直接加速**。
量化是端侧部署里性价比最高的一招，没有之一。

## 1. 先算清楚收益

```text
BF16/FP16 权重： 2.0 bytes/参数
INT8     权重： 1.0 bytes/参数           → 2.0×
FP8 (E4M3) 权重：1.0 bytes/参数          → 2.0×
INT4 + group=128：0.5 + 16/128 = 0.516   → 3.9×
INT4 + group=32 ：0.5 + 16/32  = 1.0     → 2.0×  ← group 太小会把收益吃掉！
```

**注意这个反直觉的结论**：group size 越小，精度越好，但 scale 的存储开销越来越大。
`group_size=32` 的 INT4 和 INT8 占用一样多 —— 那就别费劲了，直接用 INT8。

**7B 模型 decode 的理论上限（A100，2039 GB/s）：**

| 精度 | 权重占用 | 理论 tokens/s |
| --- | --- | --- |
| FP16 | 14 GB | 145 |
| INT8 | 7 GB | 291 |
| INT4 (g=128) | 3.6 GB | 566 |

实际会打折扣（KV cache、activation 也要搬运），但数量级关系成立。

## 2. 量化方案的谱系

### 按「量化什么」分

| 方案 | 权重 | 激活 | 典型代表 | 适用 |
| --- | --- | --- | --- | --- |
| W16A16 | FP16 | FP16 | 原始 | 基线 |
| W8A8 | INT8 | INT8 | SmoothQuant | 大 batch、算力受限 |
| W4A16 | INT4 | FP16 | **GPTQ / AWQ** | **decode 带宽受限 ← 最常用** |
| W4A8 | INT4 | FP8 | 混合 | 少数场景 |
| W8A8-FP8 | FP8 | FP8 | 原生 FP8 模型 | H100 / 新版 GPU |

**关键权衡：量化激活比量化权重难得多。**

理由是 activation 里存在**离群值通道**（outlier channels）——某些维度上的数值比其他大 100 倍以上。权重的分布则温和得多（近似高斯）。所以：

- 只想吃带宽红利（decode）→ 只量化权重（W4A16）
- 想同时吃算力红利（prefill，可以用 INT8 Tensor Core）→ 必须处理激活的离群值

### 三种主流 PTQ 算法的直觉

**GPTQ** —— 逐层做，用二阶信息（Hessian / 逆 Hessian）决定「量化一个权重后，怎么调整剩下的权重来补偿误差」。数学上是 OBQ 的高效版本。对 W4A16 效果稳定。

**AWQ** —— 观察到一个事实：**只有约 1% 的权重通道是「重要的」**（由激活的幅度决定）。对这些通道乘以一个 scale 保护起来，再量化：

```text
W' = W · diag(s),   X' = X · diag(s)⁻¹
```

`s` 由激活的 per-channel 幅度算出。等价变换，不改变数学结果，但让重要的通道落在量化误差更小的区间。

**SmoothQuant** —— 同一个思路，但目的是 W8A8。把激活的量化难度「迁移」到权重上：

```text
Y = (X · diag(s)⁻¹) · (diag(s) · W)
     └─ 激活好量化了 ─┘   └─ 权重稍微难一点，但能忍 ─┘
```

`s_j = max(|X_j|)^α / max(|W_j|)^(1-α)`，`α` 一般取 0.5。

## 3. 反量化 GEMM 的两种写法

W4A16 的 GEMM 长这样：

```python
Y = (dequant(W_int4, scale, zero) ) @ X       # X 是 FP16
```

### 写法 A：先反量化到 FP16，再做标准 GEMM

```cuda
// dequant_kernel: [N, K/2] int4 → [N, K] half
// 然后再调 cuBLAS
```

**问题**：反量化后的 FP16 权重必须写回显存再读回来 → 流量翻倍，**把量化省下的带宽又还回去了**。

即使融合成一个 kernel 不落显存，权重的 SMEM 占用也变成 2 倍，占用率下降。

### 写法 B：SMEM 里保持低精度，在寄存器里反量化（推荐）

```text
HBM:  int4 权重 (0.5 bytes/param)
  ↓ cp.async 异步搬进 SMEM
SMEM: 仍然是 int4            ← 省一半 on-chip 空间，能放更大的 tile
  ↓ 取到寄存器
寄存器: int4 → half → mma    ← 反量化只在寄存器里发生，零额外带宽
Tensor Core: mma → FP32 累加
```

这就是 **Marlin** 的做法。它的几个关键设计：

1. **权重预重排（pre-shuffle）**：离线把权重按 Tensor Core 需要的 fragment layout 重排好，运行时零 shuffle 开销
2. **零开销反量化**：`(int4 & 0xF) * scale - zero*scale` 编译成几条整数指令，跟 `mma` 的流水线重叠
3. **per-group scale 的广播**：group=128 时，一个 scale 要被 128 个元素复用，放在 SMEM 广播即可

实测（A100，Llama-2-7B decode）：**相比 FP16 cuBLAS 约 3~4×**，接近 3.9× 的理论上限。

> 想读源码：`vllm/csrc/quantization/marlin/`，文件很长，建议先看 `marlin_template.h` 里 kernel 的 main loop。

## 4. FP8：新一代 GPU 的性价比之选

FP8 有两种格式，用途不同：

| 格式 | 指数位 | 尾数位 | 表示范围 | 用途 |
| --- | --- | --- | --- | --- |
| E4M3 | 4 | 3 | ±448 | **权重 / 激活**（需要精度） |
| E5M2 | 5 | 2 | ±57344 | 梯度（需要范围） |

推理里基本只用 **E4M3**。

好处：**Hopper / Ada 的 FP8 Tensor Core 吞吐是 FP16 的 2 倍**，同时带宽减半 —— 算力和带宽两头都赚。代价是需要 per-tensor 或 per-token 的 scale：

```python
# per-tensor（最简，精度一般）
q = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn)

# per-token（动态，精度好很多，推荐）
scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
```

**坑**：`per-token` 的 scale 是运行时算的，会把一个 elementwise kernel 塞进关键路径里。要跟 GEMM 融合（vLLM 里是 `scaled_fp8_quant`），否则 kernel launch 和多一次读写就吃掉收益。

## 5. KV Cache 量化

长上下文场景下，**KV cache 的带宽开销会超过权重**：

```text
Llama-2-7B, seq=8192, batch=1:
  权重流量:  14 GB（每 token 一遍）
  KV 流量:   2 × 32层 × 32头 × 128 × 8192 × 2 bytes = 4.3 GB
```

序列再长一点，KV 就成了主导项。所以 vLLM 支持 KV cache 独立量化：

```bash
--kv-cache-dtype fp8        # E4M3，显存和带宽都减半
```

实现要点：

- **不要对 Q 量化**（只有一行，量化收益小、误差影响大）
- scale 按 head 维度计算即可，per-token 粒度没有必要
- attention kernel 里读 KV 后立即转 FP16，融合在 `paged_attention` 内部

注意：KV cache 量化**会累积误差**（每个 token 都基于量化过的历史），长生成任务上要实测 perplexity。

## 6. 精度损失怎么评估

不要只看 perplexity。按顺序做：

1. **单层输出对比**：`torch.allclose(fp16_out, quant_out, atol=1e-2)`，看误差从哪一层开始爆
2. **困惑度**：WikiText-2，涨幅 < 0.5 通常可接受
3. **任务指标**：GSM8K / MMLU，**这是最终判据** —— 有些量化方案 perplexity 好看但推理能力掉得厉害
4. **长上下文**："lost in the middle" 现象在量化后可能加剧

经验：W4A16 + group=128 用 AWQ/GPTQ，一般掉 1~3 个点的 MMLU；W8A8 的损失通常可忽略。

## 7. vLLM 里的对应关系

```bash
# 权重
--quantization awq           # W4A16，社区模型最多
--quantization gptq_marlin   # W4A16，GPTQ + Marlin kernel（比 gptq 快）
--quantization fp8           # W8A8-FP8，需要 Hopper/Ada
--quantization compressed-tensors   # 统一入口，支持 W4A8 等

# KV cache
--kv-cache-dtype fp8

# 显存不够时
--gpu-memory-utilization 0.9
--max-model-len 4096         # 显存不足先降这个
```

**选型建议**：

- 端侧 / 单卡显存紧张 → **AWQ 或 GPTQ-Marlin（W4A16）**，收益最大
- H100 且要压 prefill 延迟 → **FP8**
- 要跑在 INT8 算力强的 NPU 上 → **SmoothQuant (W8A8)**

## 检查清单

- [ ] 算过量化后 `理论 tokens/s = 带宽 / 权重大小` 吗？提升符合预期吗？
- [ ] group size 是多少？如果 ≤ 32，收益可能还不如直接用 INT8
- [ ] 反量化发生在寄存器里，还是绕了一圈显存？后者等于白量化
- [ ] 权重有没有按 kernel 需要的 layout 预重排？
- [ ] FP8 的 per-token scale 有没有跟 GEMM 融合？
- [ ] KV cache 量化了吗？长上下文场景下它可能比权重更值钱。
- [ ] 评估过真实任务指标（不只是 perplexity）吗？
