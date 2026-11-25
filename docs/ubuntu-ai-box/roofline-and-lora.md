# Roofline 实测 + LLM LoRA 微调

> 机器 `pingji-MS-7B89` · RTX 5060 Ti 16 GB（sm_120，36 SM）· Ubuntu 26.04
> 实测日期：2026-11-25 · ncu 2025.3.1 / torch 2.15.0+cu132

---

## 一、Roofline 实测（Nsight Compute + torch 计时）

### 1.1 环境准备

`ncu` 其实已经随 CUDA 13 装在 `/usr/local/cuda/bin/ncu`（不在 PATH 里）。默认无权限：

```
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

用 `sudo ncu` 即可（或把 `NVreg_RestrictProfilingToAdminUsers=0` 写进 modprobe 配置后重启驱动）。

### 1.2 实测结果

torch 侧计时（20 次平均）与 ncu 的 %-of-peak 对照：

| 算子 | 耗时 | 实测吞吐 | ncu Compute(SM) | ncu DRAM | 性质 |
| --- | ---: | ---: | ---: | ---: | --- |
| GEMM 4096³ bf16 | 2.77 ms | **49.6 TFLOPS** | 48.9% | 23.8% | 偏算力 |
| GEMM 8192³ bf16 | 21.88 ms | **50.3 TFLOPS** | — | — | 偏算力 |
| add（1.5 GB 流量） | 4.06 ms | **396 GB/s** | 5.1% | **90.2%** | 纯带宽 |
| copy（1 GB 流量） | 4.24 ms | 253 GB/s | — | — | 纯带宽（实现差） |
| SDPA causal 8K | 9.21 ms | **29.8 TFLOPS** | 69.3%* | 41.3%* | 混合 |
| RMSNorm 8192×4096 | 1.83 ms | **110 GB/s** | — | — | 纯带宽（未融合） |
| GEMV 4096²（decode 形态） | 0.106 ms | **318 GB/s** | — | — | 纯带宽 |

\* SDPA 的行是 ncu 抓到单个子 kernel 的占比，不是整个 op 的（`--launch-count 1` 的局限）。

### 1.3 四条结论

1. **带宽类能吃到 88–90% 峰值**（add 396 GB/s / 448）。与本系列前面测到的
   「F16 量化模型 decode 有效带宽 402 GB/s、8B Q4 达到 365 GB/s」完全一致——
   这张卡的带宽是"能给到"的。
2. **算力类只到 ~50 TFLOPS**。对照 GB206 的标称 bf16 张量算力（约 90–100 TFLOPS，估算），
   cuBLAS 在 sm_120 上只发挥了一半左右。**这张卡是"带宽友好、算力吃亏"**。
3. **朴素实现的差距极大**：PyTorch 直接写的 RMSNorm 只有 **110 GB/s**（25% 带宽），
   而 add 能到 90%。这正是算子融合（fused rmsnorm）存在的意义——同一块数据要读写三次。
4. **decode 形态的 GEMV 是 318 GB/s（71%）**，比大批 GEMM 的 23.8% DRAM 利用率高得多——
   说明 LLM 解码瓶颈确实在"每 token 读一遍权重"，与前面所有推理实验的结论闭环。

---

## 二、LLM LoRA 微调（Qwen3-0.6B）

### 2.1 配置

| 项 | 值 |
| --- | --- |
| 基座 | Qwen3-0.6B（bf16） |
| 数据 | 本地 `sft_t2t_mini.jsonl`，取 4000 条 → 可用 **3932** 条 |
| 目标 | 只训最后一条 assistant 回复（prompt 部分 label = -100） |
| LoRA | r=8, alpha=8, dropout 0.05，挂在 **q/k/v/o + gate/up/down 七处** |
| 超参 | bs=2, 梯度累积 4, seq 512, lr 1e-4, 梯度检查点开 |

### 2.2 训练实测

| 指标 | 实测 |
| --- | ---: |
| 可训练参数 | **5.05 M**（占 0.6B 的 0.84%） |
| 显存峰值 | **3,610 MiB**（vlm-server 仍占 6.4 GB 的情况下） |
| 单步耗时 | **1,191 ms**（含 4 次前向/反向，即 ≈300 ms/微批） |
| 3000 步预估 | **约 1 小时** |
| loss | 1.85 → 1.81 → 0.50 → 1.58（bs=2 抖动大，趋势向下） |

> 对比 SDXL LoRA（676 ms/step @512²）：**LLM LoRA 反而更慢**，
> 因为 LLM 的 seq 更长、且要算 7 个投影层；但显存只要 3.6 GB。

### 2.3 A/B 对比（60 步，关掉 Qwen3 的 thinking）

| 提问 | base（原模型） | LoRA（60 步） |
| --- | --- | --- |
| 你是谁？ | 我是AI助手，专注于帮助用户解决问题和提供支持。 | 我是AI助手，专注于提供帮助与解答。 |
| 什么是显存带宽？ | 显存带宽是指显卡中用于存储和处理图形数据的带宽，直接影响图形渲染性能。 | 显存带宽是指显卡中用于存储和传输数据的带宽，直接影响显卡的性能和效率。 |
| 把"把文件发给我"改礼貌 | 当然可以，以下是几种更礼貌的表达方式：1. 请将文件发给我。2. 请您将文件发给我。3. 请将文件发给我，谢谢。 | 请将文件发给我。 |
| 写回文判断函数 | 当然可以！下面是一个使用 Python 写的判断回文串的函数示例：…（带解释） | 你可以用 Python 写一个判断回文串的函数，如下所示：…（更简洁） |

**结论：**

1. **60 步就够改变"说话风格"**：LoRA 版明显更简洁、直接，少了"当然可以！以下是几种…"的客套
   ——这正是 SFT 数据的风格（短答为主）。
2. **但还不足以注入知识**：返利/事实类回答与原模型差别不大。
3. **bs=2 + 60 步的 loss 不能当收敛指标**（batching 噪声比信号大），要判断效果必须用固定 eval 集。

### 2.4 踩坑

1. **Qwen3 默认开 thinking**：不传 `enable_thinking=False`，64 个 token 全花在 `<think>` 上，
   base 看起来"什么都不会"——这会得出完全错误的 A/B 结论。
2. `enable_input_require_grads()` 必须加，否则 LoRA + 梯度检查点会因为输入不参与求导而报错。
3. 数据里 `reasoning_content` 字段要丢（只保留 `content`），否则训练目标里会混入思维链。

---

## 三、两项合起来的结论

| 观察 | 含义 |
| --- | --- |
| 带宽 90%、算力 50% | 这台机器上**所有"省带宽"的技巧都值得做**，所有"堆算力"的技巧都不划算 |
| 朴素 RMSNorm 只有 110 GB/s | 融合 kernel 有 3–4 倍空间，这是 kernel-opt 的现实落点 |
| LoRA 0.6B 只要 3.6 GB / 5 M 参数 | 端侧"人设/风格微调"在 16 GB 卡上是分钟级的事情 |

---

## 四、复现

```bash
# roofline（需要 sudo 才能读性能计数器）
sudo /usr/local/cuda/bin/ncu --section SpeedOfLight \
  ~/apps/bench/yolo-venv/bin/python ~/apps/infer-research/roof_kernels.py add
~/apps/bench/yolo-venv/bin/python ~/apps/infer-research/bench_ops.py

# LLM LoRA
STEPS=60 ~/apps/bench/yolo-venv/bin/python ~/apps/infer-research/lora_llm.py
~/apps/bench/yolo-venv/bin/python ~/apps/infer-research/lora_ab.py
```
