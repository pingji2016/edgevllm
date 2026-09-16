# 07 · 边缘设备优化实践

前面几篇讲的技巧在数据中心 GPU 上成立，但**边缘设备的约束条件完全不同**，照搬会踩坑。
这一篇讲清楚差在哪、以及怎么适配。

## 1. 边缘设备的真实数字

先破除一个误解：**厂商标称的 TOPS 基本没有参考价值**（往往是 INT8 + 稀疏 + 理论峰值 + 理想功耗）。
从「测出来的内存带宽」入手，才是判断能不能跑的可靠办法。

| 设备 | 内存带宽 | 典型可用显存/内存 | FP16 算力（实际） | 备注 |
| --- | --- | --- | --- | --- |
| Jetson Orin Nano 8G | ~68 GB/s | 8 GB 统一 | ~5 TFLOPS | sm_87，有 Tensor Core |
| Jetson Orin NX 16G | ~102 GB/s | 16 GB 统一 | ~12 TFLOPS | |
| Jetson AGX Orin 64G | ~205 GB/s | 64 GB 统一 | ~25 TFLOPS | 标称 275 TOPS(INT8 稀疏) |
| RTX 4060 Laptop | 256 GB/s | 8 GB | 120 TFLOPS | 有独显的笔记本 |
| Apple M4 Pro | ~273 GB/s | 统一内存 | — | Metal，非 CUDA |
| RK3588 | ~12 GB/s (CPU 侧) | 8–32 GB | NPU 6 TOPS(INT8) | 无 CUDA |
| Raspberry Pi 5 | ~17 GB/s | 4–16 GB | 无 | 纯 CPU |

### 立刻可用的结论

**Orin AGX 的带宽是 A100 的 1/10。** 套用 [01-roofline.md](01-roofline.md) 的公式：

```text
Llama-2-7B, FP16 权重 14 GB, Orin AGX 205 GB/s:
    理论上限 = 205 / 14 ≈ 14.6 tokens/s

换成 INT4 (g=128) 权重 3.6 GB:
    理论上限 = 205 / 3.6 ≈ 57 tokens/s     ← 4 倍差距
```

**在边缘设备上，量化不是优化项，是能否可用的分水岭。**

Orin Nano 8G 更极端：

```text
FP16 7B：14 GB > 8 GB 总内存        → 根本装不下
INT4  7B：3.6 GB + KV cache ~1 GB   → 刚好能跑，约 15 tokens/s
```

## 2. 统一内存（Unified Memory）带来的变化

Jetson 的 CPU 和 GPU 共享同一块 LPDDR，没有 PCIe。这带来三个和独显不同的现象：

**好处**：不需要 `cudaMemcpy`，`torch.tensor(...).cuda()` 零拷贝（页锁定内存直接映射）。

**坑 1：带宽是共享的。** 你在跑推理时，CPU 侧的 dataloader、日志、tokenizer 也在吃同一块带宽。**在边缘设备上，减少 CPU 侧的内存动作跟优化 kernel 一样重要。**

**坑 2：`cudaMalloc` 会触发页锁定（pinned）内存分配，很慢。** 训练/推理循环里动态分配张量会导致周期性卡顿。

```python
# 避免：每步都分配
for _ in range(N):
    out = torch.empty(...)          # 可能触发 cudaMalloc + 同步

# 推荐：预分配复用
buf = torch.empty(max_shape, ...)
```

**坑 3：显存和内存是一回事，OOM 会杀掉整个系统**，不只是你的进程。所以：

```bash
--gpu-memory-utilization 0.85    # 边缘设备别设 0.9，留出系统余量
```

## 3. 功耗与热：为什么 benchmark 不可复现

边缘设备的持续频率远低于峰值频率。**同一份代码跑两次性能差 30% 是常态**，因为：

- DVFS 根据功耗预算动态调频
- 温度上来了就降频
- 电池设备还会受剩余电量影响

**采样前先锁频**（Jetson）：

```bash
sudo nvpmodel -m 0          # 最大性能模式
sudo jetson_clocks          # 锁定最高频率
tegrastats                  # 实时看功耗、温度、频率

# 观察指标：GR3D_FREQ（GPU 频率）、VDD_IN（整机功耗 mW）
```

**报告性能时必须同时报告功耗。** 在边缘场景下 `tokens/s/W` 往往比 `tokens/s` 更重要 —— 一个快 2 倍但功耗高 3 倍的方案，在电池设备上是倒退。

也别忘了**给设备足够的散热**。开发板上裸跑和装在金属外壳里，持续吞吐可能差一倍。

## 4. 边缘场景的典型优化路线

按收益排序，逐条试：

### ① 权重 INT4 量化（收益最大）

```bash
# AWQ 是目前端侧最实用的选择
vllm serve Qwen2.5-7B-Instruct-AWQ \
  --quantization awq \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
```

### ② KV cache 量化 / 压缩

长上下文时 KV 会超过权重：

```bash
--kv-cache-dtype fp8
```

Orin 是 sm_87，支持 FP8 存储（虽然 Turing 之前没有 FP8 算力，但作为存储格式仍能省一半带宽，转换在 kernel 里做 FP16 运算）。

更激进的做法（需要改模型）：**滑窗 attention** + **attention sink**，把 KV 长度从 O(N) 压到 O(w)。

### ③ 限制上下文长度

这是最有效也最容易被忽视的一条。`--max-model-len 2048` 对比 `8192`，KV cache 显存直接省 4 倍，attention 的带宽开销也小 4 倍。

**先问清楚：业务真的需要 32K 上下文吗？**

### ④ Chunked Prefill

边缘设备上 prefill 会严重抢占资源，导致正在生成的请求卡顿（TTFT 尖刺）。

```bash
--enable-chunked-prefill --max-num-batched-tokens 512
```

把长 prompt 切成小块，和 decode 请求混在一个 batch 里。**代价是 prompt 处理的绝对延迟略增，收益是延迟曲线平滑。**

### ⑤ Prefix Caching

多轮对话 / 固定 system prompt 的场景：

```bash
--enable-prefix-caching
```

共享前缀的 KV 只算一次。在客服机器人这类固定前缀的场景里，TTFT 能降 50% 以上。

### ⑥ CUDA Graph

边缘设备 CPU 通常也很弱（Orin 的 CPU 单核性能约等于手机）。

**decode 阶段每个 token 要跑几百个 kernel，如果 CPU 端 launch 开销是 20 微秒/kernel，那 300 个 kernel 就是 6 毫秒 —— 比 GPU 计算本身还慢。**

```bash
# vLLM 默认开启；排查问题时用 --enforce-eager 关掉对比
```

如果发现关掉 CUDA Graph 后反而快了，说明你的批大小是动态的（每次都要重新捕获），需要改成静态分桶。

### ⑦ Speculative Decoding

带宽瓶颈下，**用算力换带宽**是划算的：小模型草拟 k 个 token，大模型一次验证。

```text
标准 decode:  每个 token 读一遍 3.6 GB 权重 → 57 tokens/s 上限
投机解码:     一次读权重验证 k 个 token → 带宽利用率提升近 k 倍
```

边缘设备算力有富余（相对带宽而言），这个交换尤其划算。但要注意接受率 —— 接受率低于 60% 时收益会被草拟开销吃掉。

## 5. NPU 异构：能碰，但别指望省事

RK3588、Snapdragon、地平线这类平台的 NPU 有 INT8 算力，但约束很硬：

| 约束 | 影响 |
| --- | --- |
| **静态 shape** | KV cache 长度动态变化 → 每次都要重新编译，或用 padding 浪费算力 |
| **算子支持有限** | FlashAttention、PagedAttention 这类动态访存算子基本没有 |
| **INT8/INT4 为主** | 精度和多精度混合受限 |
| **工具链割裂** | RKNN / QNN / TensorRT，各自一套量化流程 |

现实做法（按复杂度递增）：

1. **全 GPU/CPU**：Jetson 上 vLLM 直接跑，最省事，性能已够很多场景
2. **NPU 跑 vision encoder，GPU 跑 LLM**：多模态场景下分工明确，收益实在
3. **NPU 跑 prefill，GPU 跑 decode**：理论收益大（prefill 是算力密集，decode 是带宽密集），但要在两个 runtime 之间搬 KV cache，**跨设备带宽往往把收益吃光**，只在同片内存（如 RK3588）上才值得试

> 判断标准：如果搬 KV cache 的字节数 × 2 > 省下的计算时间 × 带宽，就不值得。

## 6. 一个端侧部署的配置模板

```bash
vllm serve Qwen2.5-3B-Instruct-AWQ \
  --quantization awq \
  --kv-cache-dtype fp8 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 512 \
  --enable-prefix-caching \
  --block-size 16 \
  --swap-space 2 \
  --disable-log-requests
```

调参顺序（每步都测）：

1. `--max-model-len`：能降就降，收益最大
2. `--gpu-memory-utilization`：0.8 → 0.9 之间找稳定点
3. `--max-num-seqs`：并发太高会 OOM，太低吞吐上不去
4. `--block-size`：8 / 16 / 32 都试一遍（改这个要重测吞吐）
5. `--max-num-batched-tokens`：控制 prefill 对 decode 的干扰

## 7. 编译与适配

在 Jetson 上从源码构建 vLLM：

```bash
# 确认架构：Orin 系列是 8.7，Xavier 是 7.2
export TORCH_CUDA_ARCH_LIST="8.7"
export MAX_JOBS=4                 # Orin 内存有限，并发编译会 OOM

pip install -e . --no-build-isolation
```

常见问题：

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 编译 OOM | `MAX_JOBS` 太大 | 降到 2~4，或加 swap |
| `no kernel image is available` | 架构没编进去 | 检查 `TORCH_CUDA_ARCH_LIST` |
| 跑起来显存不够 | 统一内存被系统占用 | 降 `--gpu-memory-utilization` |
| 首次 token 特别慢 | kernel 首次编译 / 权重加载 | 忽略前几次，或预热 |
| FlashAttention 挂了 | 某些版本对 sm_87 支持不全 | 退回 xformers 后端 |

## 检查清单

- [ ] 算过 `带宽 / 权重大小` 的理论上限了吗？实测离它多远？
- [ ] 量化了吗？（端侧不做量化基本没有讨论的必要）
- [ ] 锁频了吗？报性能时带功耗了吗？
- [ ] `max-model-len` 是不是开得比需要的大？
- [ ] KV cache 量化了吗？长上下文时它比权重更吃带宽。
- [ ] 用 tegrastats / nvidia-smi 看过有没有周期性降频吗？
- [ ] 报告的是 `tokens/s` 还是 `tokens/s/W`？端侧后者更重要。
