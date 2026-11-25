# Roofline 实测 + LLM LoRA 微调（详解版）

> 机器 `pingji-MS-7B89` · RTX 5060 Ti 16 GB（sm_120，36 SM，**448 GB/s**）· Ubuntu 26.04
> 实测日期：2026-11-25 · ncu 2025.3.1 / torch 2.15.0+cu132 / transformers 5.23
>
> 本文分两部分：
> **上半**讲 roofline 怎么看、怎么算、这台机器实测落在哪；
> **下半**讲 LoRA 到底在改什么、为什么 5 M 参数就够、以及训练循环里每一步在干什么。

---

# 第一部分：Roofline

## 1.1 屋顶线到底在说什么

一个 kernel 的耗时，永远被两个上限夹住：

```
算力上限：  t ≥ FLOPs / Peak_FLOPS
带宽上限：  t ≥ Bytes / Peak_BW
实际耗时：  t ≥ max(上面两个)
```

把两个上限画在同一张"横轴=算术强度、纵轴=性能"的图上，就是两条斜线/水平线拼成的屋顶。

**算术强度（Arithmetic Intensity, AI）**：

```
AI = FLOPs / Bytes       单位：FLOP/byte
```

它回答一个问题：**每从显存搬 1 字节，能做多少次浮点运算？**

**判据**（这是全文最有用的一行）：

```
临界 AI = Peak_FLOPS / Peak_BW
```

- `AI > 临界值` → 算力受限（优化方向是更好的 GEMM/kernel）
- `AI < 临界值` → 带宽受限（优化方向是少搬字节：量化、融合、减少读写）

本机：Peak_BW = 448 GB/s；Peak_FLOPS 用实测的 50 TFLOPS（bf16）来算：

```
临界 AI = 50e12 / 448e9 ≈ 112 FLOP/byte
```

**AI < 112 的算子，讨论 TFLOPS 没有意义；AI > 112 的算子，讨论带宽没有意义。**

## 1.2 每个算子的算术强度（手算，可复核）

| 算子 | FLOPs | 搬运字节 | AI (FLOP/byte) | 判定 |
| --- | --- | --- | ---: | --- |
| GEMM 4096³ bf16 | 2n³ = 1.37e11 | 3n²×2 = 1.0e8 | **1365** | 算力受限 |
| GEMM 8192³ bf16 | 2n³ = 1.10e12 | 3n²×2 = 4.0e8 | **2731** | 算力受限 |
| add（n=2.68 亿） | n = 2.68e8 | 3n×2 = 1.61e9 | **0.17** | 带宽受限 |
| copy | 0 | 2n×2 = 1.07e9 | **0** | 纯带宽 |
| GEMV（decode） | 2n² = 3.4e7 | n²×2 = 3.4e7 | **1.0** | 带宽受限 |
| RMSNorm | ~3n = 1e8 | 3n×2 = 2.0e8 | **0.5** | 带宽受限 |
| SDPA 8K causal | 2.75e11 | ~1.3e8 | **~2000** | 算力受限 |

注意两个极端：

- **GEMM 的 AI 高达 1365**：把 4096² 的矩阵读进来，能做 4096 次乘加，
  所以它必然卡在算力，带宽利用率只有 23.8% 是**正常且必须**的。
- **add 的 AI = 0.17**：读两个数、写一个数，只做 1 次加法。
  它在屋顶线的右侧最左边，**再强的 GPU 也只能跑满带宽**。

## 1.3 实测结果与屋顶线对照

torch 侧 20 次平均（ncu 只用来读 %-of-peak）：

| 算子 | 耗时 | 实测吞吐 | 理论天花板 | 达成率 | ncu Compute | ncu DRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GEMM 4096³ bf16 | 2.77 ms | **49.6 TFLOPS** | 50 TFLOPS* | ~99% | 48.9% | 23.8% |
| GEMM 8192³ bf16 | 21.88 ms | **50.3 TFLOPS** | 50 TFLOPS* | ~100% | — | — |
| **add** | 4.06 ms | **396 GB/s** | 448 GB/s | **88%** | 5.1% | **90.2%** |
| copy | 4.24 ms | 253 GB/s | 448 GB/s | 56% | — | — |
| SDPA causal 8K | 9.21 ms | 29.8 TFLOPS | 50 TFLOPS* | 60% | 69.3%† | 41.3%† |
| RMSNorm（朴素） | 1.83 ms | **110 GB/s** | 448 GB/s | **25%** | — | — |
| GEMV 4096²（decode） | 0.106 ms | **318 GB/s** | 448 GB/s | **71%** | — | — |

\* 把 cuBLAS 实测的 50 TFLOPS 当作本机 bf16 的"可用算力天花板"（不是标称值，见 1.5）
† 这一行是 ncu 抓到单个子 kernel 的占比（`--launch-count 1` 的局限），不是整个 op

### 读表要点

1. **`add` 的 DRAM 90.2%、Compute 5.1%** —— 教科书级的带宽受限。它的 AI=0.17，
   离临界值 112 差了三个数量级，所以"优化计算"毫无意义。
2. **`GEMM` 的 DRAM 23.8%、Compute 48.9%** —— AI=1365，带宽根本不是瓶颈；
   但 Compute 只有 48.9% 说明**算力也没吃满**（见 1.5）。
3. **`RMSNorm` 只有 110 GB/s** —— 这是"朴素 PyTorch 实现"的典型：`pow` → `mean` → `rsqrt`
   → 乘法，每一步都是一次全量读写，4 次 pass 把带宽浪费在重复搬运上。
   融合 kernel（一次读完、一次写完）理论上能到 3–4 倍。
4. **`GEMV` 318 GB/s（71%）** —— 这就是 LLM 解码的真实形态：每生成一个 token，
   把权重完整读一遍。318/448=71%，与前面推理实验里 8B Q4 的 365 GB/s、1.7B F16 的 402 GB/s 相互印证。

## 1.4 为什么说这台机器"带宽友好、算力吃亏"

| 指标 | 本机实测 | 说明 |
| --- | ---: | --- |
| 带宽利用率（最好情况） | **90%** | add 算子 |
| 算力利用率（最好情况） | **~50%** | cuBLAS bf16 |

同一块芯片上，带宽能榨到 90%，算力只能到 50%，差了一倍。
这直接决定了优化方向：**任何"少读字节"的技巧（量化、KV 压缩、投机、融合）都值得做；
任何"堆算力"的技巧（更大 tile、更深流水、Tensor Core 特化）收益都受限。**

## 1.5 为什么 GEMM 只有 ~50 TFLOPS

标称的 bf16 张量算力（GB206，36 SM）大约在 90–100 TFLOPS 量级（**估算**），
实测只到一半。可能原因（按可能性排序）：

1. **cuBLAS 对 sm_120 的内核选择不充分**：Blackwell 消费卡（sm_120）与数据中心 Blackwell
   （sm_100/103）内核不同，cuBLAS 可能回退到较保守的 tile 配置。
2. **wave quantization**：36 个 SM 不是常见的 2 的幂，4096³ 分块后最后一批 wave 利用率下降。
   用 8192³ 测同样只到 50.3，说明不是偶发。
3. **累加精度**：PyTorch 默认 bf16 matmul 用 fp32 累加；若换成 fp16 输入 + fp16 累加（精度更差）
   通常能更快，但这属于"用精度换速度"。
4. **功耗/温度墙**：桌面卡的 boost 时钟会随温度回落，ncu 显示 SM 频率 2.40 GHz（标称 2.57 GHz），
   已降了约 7%。

**结论**：在这张卡上，"算力上限"应取实测的 ~50 TFLOPS，而不是标称值；
所有 roofline 判据都用实测值算才靠谱。

## 1.6 ncu 指标怎么读（SpeedOfLight 段）

| 指标 | 含义 | 什么时候看 |
| --- | --- | --- |
| `Compute (SM) Throughput` | 各计算流水线的最高占用率（含 FMA、ALU、Tensor） | 判断算力是否吃满 |
| `Memory Throughput` | 存储子系统（含 L1/L2/DRAM）的最高占用率 | 判断是否被访存拖住 |
| `DRAM Throughput` | 只算 DRAM 的部分 | 判断"真·带宽受限" |
| `L2 Cache Throughput` | L2 带宽占用 | 判断是否被 L2 卡住 |
| `Duration` / `Elapsed Cycles` | 绝对时间 / 周期数 | 换算吞吐 |

**经验判据**：

- `DRAM ≈ Memory ≈ 90%` 且 `Compute < 20%` → **真·带宽受限**（如 add）
- `Compute ≈ 90%` 且 `DRAM < 40%` → **真·算力受限**（理想 GEMM 应该长这样）
- 两者都在 40–70% → **混合受限或无谓开销**（可能 launch 配置差、occupancy 不足）

**权限坑**：ncu 默认不允许普通用户读性能计数器：

```
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

两种解法：`sudo ncu ...`（本文做法），或在 modprobe 配置里设
`NVreg_RestrictProfilingToAdminUsers=0` 后重启驱动。

---

# 第二部分：LLM LoRA 微调

## 2.1 LoRA 在数学上改了什么

全参微调要更新一个 d×k 的权重矩阵 W 的全部 d·k 个元素。
LoRA 假设**微调带来的变化量 ΔW 是低秩的**：

```
W' = W + ΔW,    ΔW = B · A
其中 A ∈ R^{r×k}, B ∈ R^{d×r}, r ≪ min(d, k)
```

前向时：

```
h = W x + (α / r) · B (A x)
```

- `A`：把 k 维输入压到 r 维（降维）
- `B`：把 r 维还原到 d 维（升维）
- `α / r`：缩放，保证换 r 时有效学习率大致不变（本文 α=8, r=8 → 缩放 1）

**为什么 B 初始化为 0**：训练开始时 `ΔW = BA = 0`，模型输出与原模型**完全一致**。
这保证了"微调不会一上来就把模型搞坏"，也解释了为什么 LoRA 很少需要 warmup。

**为什么参数量这么小**：Qwen3-0.6B 的 hidden=1024，一个 `q_proj` 是 1024×1024=1.05M 参数；
LoRA r=8 只加 8×1024 + 1024×8 = 16.4K 参数（1.6%）。七个投影层加起来：

```
5.05 M / 600 M ≈ 0.84%
```

## 2.2 本机的 LoRA 配置与实测

| 项 | 值 | 理由 |
| --- | --- | --- |
| 基座 | Qwen3-0.6B（bf16） | 与 vlm-server 共存，显存够 |
| 数据 | `sft_t2t_mini.jsonl` 前 4000 条 | 中文多轮指令，格式现成 |
| 目标 | 只训**最后一条** assistant 回复 | 最贴近"单轮指令微调"的产品形态 |
| LoRA | r=8, α=8, dropout 0.05 | 小 rank 防过拟合，α=r 是常用默认 |
| 目标模块 | q/k/v/o + gate/up/down（7 处） | 只挂 q/v 效果明显更差；全挂是标准做法 |
| 超参 | bs=2, 累积 4, seq 512, lr 1e-4 | 等效 batch 8 |

**实测**：

| 指标 | 值 |
| --- | ---: |
| 可训练参数 | **5.05 M**（0.84%） |
| 显存峰值 | **3,610 MiB** |
| 单步耗时 | **1,191 ms**（含 4 次前向/反向） |
| 单步 token 数 | bs 2 × seq 512 × 累积 4 = **4,096 token** |
| **训练吞吐** | **≈ 3,440 token/s** |
| 3000 步预估 | **约 1 小时** |

对照本系列早前测的 minimind 30M 全参预训练（85,000 token/s）：
**小模型全参训练比大模型 LoRA 快 25 倍**，因为后者的每一层都在做真实的矩阵乘。

## 2.3 训练循环里每一步在做什么

```
取出 k 条样本（bs=2）
  │
  ├─ tokenize：prompt 部分 label 置 -100（不算 loss），只对 answer 计算
  │
  ├─ 前向：model(input_ids, attention_mask, labels)
  │     └─ 内部：embedding → 28 层 Transformer → lm_head → CE loss
  │
  ├─ 反向：只有 LoRA 的 A/B 矩阵拿到梯度（基座 requires_grad=False）
  │     └─ 梯度检查点：不保存中间激活，反向时重算 → 显存从 O(层数) 降到 O(1)
  │
  └─ 累积 4 次后：clip_grad_norm(1.0) → AdamW.step() → zero_grad
```

三个容易写错的点（都踩过）：

| 坑 | 现象 | 正确做法 |
| --- | --- | --- |
| prompt 不 mask | loss 主要在"预测用户提问"，模型学会复读问题 | labels 里 prompt 段填 -100 |
| 忘了 `enable_input_require_grads()` | 梯度检查点 + LoRA 报 `element 0 of tensors does not require grad` | 基座冻结后必须开这个开关 |
| 保留了 `reasoning_content` | 训练目标是思维链而不是答案 | 只取 `content` 字段 |

## 2.4 效果：60 步改变了什么

| 提问 | base（原模型） | LoRA（60 步） |
| --- | --- | --- |
| 你是谁？ | 我是AI助手，专注于帮助用户解决问题和提供支持。 | 我是AI助手，专注于提供帮助与解答。 |
| 什么是显存带宽？ | 显存带宽是指显卡中用于存储和处理图形数据的带宽，直接影响图形渲染性能。 | 显存带宽是指显卡中用于存储和传输数据的带宽，直接影响显卡的性能和效率。 |
| 把"把文件发给我"改礼貌 | 当然可以，以下是几种更礼貌的表达方式：1… 2… 3… | **请将文件发给我。** |
| 写回文判断函数 | 当然可以！下面是一个使用 Python 写的判断回文串的函数示例：… | 你可以用 Python 写一个判断回文串的函数，如下所示：… |

**读法**：

- **风格迁移非常明显**：LoRA 版去掉了"当然可以！以下是几种…"的客套，直接给答案。
  这与 SFT 数据（多为一句短答）的风格一致。
- **知识注入不明显**：两个版本对"显存带宽"的定义都对，但都没有超出小模型的常识范围。
- **60 步的 loss（1.85 → 1.58）不能当收敛证据**：bs=2 时单步 loss 噪声 ≈ 0.5，
  信号被淹没。要判断效果必须用固定 eval 集（如 100 条留出样本的 loss 或人工打分）。

## 2.5 和 SDXL LoRA 的对比（同为 LoRA，差在哪）

| 项 | SDXL LoRA | Qwen3-0.6B LoRA |
| --- | ---: | ---: |
| 基座参数 | 2.6 B（UNet） | 0.6 B |
| 可训练参数 | 11.61 M | 5.05 M |
| 单步耗时 | 676 ms | 1,191 ms |
| 显存 | 7.0 GB | 3.6 GB |
| 300 步/3000 步 | 3.4 分钟 | 1 小时 |
| 学到的东西 | 画风/主体 | 说话风格 |

**有意思的对比**：SDXL 的基座大 4 倍，但单步更快——因为它只跑一次前向
（扩散模型每个 step 是一次 UNet 前向 + 一次反向），而 LLM 的 seq 长、要算 7 个投影层，
且梯度累积 4 次。

---

# 第三部分：两项合起来的结论

| 观察 | 含义 |
| --- | --- |
| 带宽 90% / 算力 50% | 所有"省字节"的技巧值得做，"堆算力"的技巧收益受限 |
| 朴素 RMSNorm 110 GB/s（25%） | 融合 kernel 有 3–4 倍空间——kernel-opt 的现实落点 |
| GEMV 318 GB/s（71%） | 与 LLM decode 的 365–402 GB/s 闭环，"每 token 读一遍权重"是硬约束 |
| LoRA 0.6B：3.6 GB / 5 M 参数 / 1 小时 | 端侧"人设/风格微调"是分钟到小时级的事 |

---

# 第四部分：复现

```bash
# --- Roofline ---
# ncu 需要 sudo 才能读性能计数器
sudo /usr/local/cuda/bin/ncu --section SpeedOfLight \
  --target-processes all --launch-skip 3 --launch-count 1 \
  ~/apps/bench/yolo-venv/bin/python ~/apps/infer-research/roof_kernels.py add

# torch 侧吞吐（TFLOPS / GB/s）
~/apps/bench/yolo-venv/bin/python ~/apps/infer-research/bench_ops.py

# --- LLM LoRA ---
STEPS=60 BS=2 ACC=4 SEQ=512 ~/apps/bench/yolo-venv/bin/python ~/apps/infer-research/lora_llm.py
~/apps/bench/yolo-venv/bin/python ~/apps/infer-research/lora_ab.py
```

产物：`~/apps/infer-research/roofline/`（ncu 与 torch 结果 json）、
`~/apps/infer-research/lora_llm/`（adapter + result.json + ab.json）
