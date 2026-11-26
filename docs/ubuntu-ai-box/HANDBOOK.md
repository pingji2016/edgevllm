# 实战手册：16 GB 卡上跑本地 AI 的判据与坑

> 这是整套文档的**入口**。23 篇实测记录按主题拆在各自文件里，
> 这里只给**结论、判据和决策树**——遇到具体问题再点进对应文档。

**刚入门、术语看不懂？先看 [`PRIMER.md`](PRIMER.md)（零基础版，每章配 3 分钟实验）**

**想要推导过程和完整 runbook？看 [`MANUAL.md`](MANUAL.md)**
（本文只给结论；MANUAL 讲为什么：带宽/算力上限、显存账本、
投机收益条件、量化误差、训练梯度、评测方法）

机器：RTX 5060 Ti 16 GB（sm_120）· R5 3600 · 32 GB DDR4-3200 · Ubuntu 26.04
所有数字都是在**这台机器**上实测的，不是论文引用。

---

## 1. 一页速查

### 1.1 吞吐（Qwen3-1.7B）

| 配置 | 单流 | 32 并发 | 64 并发 |
| --- | ---: | ---: | ---: |
| bf16 + CUDA Graph | 112.4 tok/s | 2,934 | — |
| **GPTQ-Int8 + CUDA Graph** | **171.9** | 4,159 | 6,578 |
| + FP8 KV | 173.8 | **4,409** | — |
| + ngram 投机（单流最优） | **238.3** | 2,498 ⚠️ | 2,293 ⚠️ |

### 1.2 上下文（Qwen3-1.7B，needle 命中率）

| 长度 | 2K | 4K | 8K | 16K | 24K | 32K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 准确率 | 100% | 100% | 100% | 78% | 67% | **11%** |
| prefill | 0.2 s | 0.3 s | 0.6 s | 1.5 s | 2.8 s | **5.0 s** |

KV 占用 **112 KB/token**：16K = 1.84 GB，32K = 3.67 GB（16 GB 卡上 32K 只能 2.25 路并发）。

### 1.3 检索（我们自己的 19 篇文档，1,726 切片）

| 方案 | R@1 | R@3 | R@5 |
| --- | ---: | ---: | ---: |
| bge-small 稠密 | 42% | 67% | 67% |
| bge-small + rerank | 58% | 75% | 75% |
| **bge-m3 + rerank** | **75%** | **83%** | **92%** |
| + BM25/RRF 混合 | 75% | 83% | 92%（**+0**） |

### 1.4 训练

| 方法 | 起点 | 结果 | 耗时 |
| --- | ---: | ---: | ---: |
| GRPO（1.7B，三位×两位） | 47% | **60%（+13 pp）** | 20 min |
| RFT 拒绝采样微调 | 50.5% | 52.5%（**噪声内**） | 13 min |
| 测试时投票 N=32 | 48% | 47%（**−1 pp**） | 32× 推理 |
| **蒸馏 1.7B→0.6B** | 30.5% | **46.0%（+15.5 pp）** | **1.3 min** |

---

## 2. 决策树

### 2.1 选引擎

```
要部署服务 / 高并发？
├─ 是 → vLLM（V1 引擎，FlashInfer 后端）
└─ 否
   ├─ 只有 GGUF / 要极致省显存 → llama.cpp
   └─ 要改模型 / 做实验       → transformers + PEFT
```

> SGLang 在本机装不上（sm_120 缺 wheel），别浪费时间。

### 2.2 选量化

```
显存放得下 bf16 权重？
├─ 放得下 → GPTQ-Int8（无损 + 1.53× 提速，为什么不呢）
└─ 放不下
   ├─ 任务依赖逐位精确（数学/代码/数值抽取）→ 降模型规模，别上 4bit
   └─ 任务只要求格式/分类/摘要 → NF4 或 GGUF Q4_K_M
```

**关键数字**：Int8 准确率无损；NF4 在算术上 **−6.7 pp**（z=3.02，显著），
但在 JSON 和 8K 检索上**没有退化**。

### 2.3 选上下文

```
必须一次看完整文档？
├─ 是
│  ├─ ≤8K  → 直接塞，100% 可靠
│  ├─ 8–16K → 可以，但准确率开始掉（78%）
│  └─ >16K → 别赌：静默错误率 30–90%
└─ 否 → 检索后塞 4K（准确率更高、延迟低一个数量级）
```

### 2.4 并发与投机

```
并发是多少？
├─ 1–4  → 可以开 ngram 投机（1.4–2.6× 单流）
├─ 8    → 持平，随便
└─ ≥16  → 一定关投机（吞吐掉到 1/3）
```

### 2.5 训练方法怎么选

```
当前任务准确率?
├─ 0–5%    → 先 SFT/蒸馏（RL 会塌缩到众数）
├─ 20–50%  → GRPO / RLVR（在错题上给负信号）
├─ 50–80%  → GRPO；想再稳可加过程奖励
└─ 80%+    → 别训了，改解码策略或加工具调用
```

**不要用**：RFT（正确样本 p≈1 → 梯度≈0，实测 +2 pp 噪声内）；
**不要用**：多数投票（错误系统性时 −1 pp，32× 成本）。

### 2.6 RAG 组件

```
查询 → bge-m3 稠密 top-20 → bge-reranker 重排 top-3 → 返回原文
```

- **不要加 BM25**（实测 +0 pp）；
- 句子级定位时必须拼 **【章节标题 + 表头】**（否则表格行的语义丢失）；
- 让模型**指位置**、程序**抄原文**（1.7B 自己复述数字会错）。

---

## 3. 十条最值钱的判据

| # | 判据 | 出处 |
| ---: | --- | --- |
| 1 | **永远不要 `enforce_eager=True`**：CUDA Graph 单流 +245% | `stack-best-config.md` |
| 2 | **投机是延迟优化，不是吞吐优化**：8 并发是交叉点 | `stack-best-config.md` |
| 3 | **投机的收益条件：接受率 > 草稿成本比** | `draft-spec-decode.md` |
| 4 | **先测 `oracle@N`，不涨就别投票** | `test-time-scaling.md` |
| 5 | **决定准确率的是错题**：只喂正确样本（RFT）几乎不动边界 | `rft-vs-grpo.md` |
| 6 | **量化伤"逐位精确"，不伤"格式/检索"** | `quant-quality.md` |
| 7 | **标称 32K ≠ 可用 32K**：安全线 16K，保守 8K | `long-context-needle.md` |
| 8 | **强嵌入 + 重排 > 混合检索**（本语料 BM25 +0） | `hybrid-retrieval.md` |
| 9 | **表格行是不可独立的最小语义单元**，必须带表头 | `rag-locate.md` |
| 10 | **对比实验必须固定解码方式**（批量 vs 逐条差 3.5 pp） | `rft-vs-grpo.md` |

---

## 4. 常见坑（按症状查）

| 症状 | 原因 | 解法 |
| --- | --- | --- |
| vLLM 比预期慢 3 倍 | 开了 `enforce_eager=True` | 去掉它 |
| 模型只输出 `<think>` 不回答 | Qwen3 思考模式 | `apply_chat_template(..., enable_thinking=False)`；或强制 `assistant\n<answer>` 前缀 |
| ASR 计时快得离谱 | `transcribe()` 返回生成器 | 必须消费完才算解码 |
| 同一模型两次评测差 3–4 pp | 批量 vs 逐条解码 | 固定 batch 与解码方式 |
| RAG 检索对、答案错 | 模型自己复述数字 | 改定位式（返回原文） |
| 表格里的数字总检索错 | 切片切掉了表头 | 每行拼【章节+表头】 |
| 4bit 后算术变差 | 量化伤精细数值 | 换 Int8 |
| 并发一高吞吐暴跌 | 开了投机 | 8 并发以上关掉 |
| 长上下文答错但很自信 | 超过安全线，抓了邻近数字 | 降到 ≤16K 或改检索 |
| RL 训练无梯度 | 组内奖励全相同 | 加部分分（距离塑形） |
| RL 百步内输出塌成常数 | 无 KL 约束 | 加 KL(0.05) 或熵正则 |
| 小模型 RFT 没用 | 正确样本 p≈1 | 改用 GRPO |
| 自定义评测集"全错" | 题目退化（只覆盖少数和） | 用均匀/均衡题集，并报告熵 |
| 跑大模型 OOM | `vlm-server` 常驻占 6.4 GB | 先 `sudo systemctl stop vlm-server` |
| 推送 GitHub 失败 | 443 抖动 / token 只读 | 重试 3–6 次；token 要 Contents: RW |

---

## 5. 文档地图

### 5.1 环境与硬件

| 文档 | 一句话 |
| --- | --- |
| `environment.md` | 完整环境清单（驱动/Python/Java/代理/模型），14 节 |
| `disk-safety.md` | 双系统机器"NVMe 只读"的红线与自查 |
| `roofline-and-lora.md` | ncu 实测：add 396 GB/s（DRAM 90%）、GEMM 50 TFLOPS（SM 49%） |

### 5.2 推理加速

| 文档 | 一句话 |
| --- | --- |
| `inference-acceleration.md` | 量化七档 / KV 压缩 / 投机解码，含负结果 |
| `stack-best-config.md` | 全优化叠加：单流 238 vs 64 并发 6,578；ngram 交叉点 8 |
| `quant-quality.md` | 量化对**准确率**：Int8 无损、NF4 −6.7 pp |
| `draft-spec-decode.md` | 0.6B→1.7B 草稿投机负收益，接受率 = 成本比 |
| `long-context-needle.md` | 16K 安全线、32K 静默崩 |

### 5.3 服务与实时链路

| 文档 | 一句话 |
| --- | --- |
| `serving-and-alignment.md` | 多 LoRA −42%、前缀缓存 4K 省 49%/16K 省 89%、DPO |
| `realtime-services.md` | 流式 ASR 110–130 ms、Qwen3-VL 0.41 s/帧、SAM2 18 ms |
| `e2e-and-opt.md` | 语音助手 3.33 s → 0.493 s（6.8×）；CUDA Graph 物理账 |

### 5.4 多模态与生成

| 文档 | 一句话 |
| --- | --- |
| `deepseek-ocr.md` | 单页 7.26 s、去标签精度 83.5%、两个 vLLM 坑 |
| `minicpm-o.md` | 9B 全模态跑通；Linux 补丁 `F_NOCACHE`；TTS 不可懂 |
| `tts-kokoro.md` | Kokoro 82M：RTF 0.014（70× 实时）、K3 上 2.08 |
| `capability-internals.md` | Wav2Lip / InsightFace / SDXL-LoRA 原理与公式 |
| `camera-ai-filter.md` | 摄像头逐帧 AI 滤镜 |
| `3d-gaussian-splatting.md` | 3D 高斯泼溅 |

### 5.5 RAG

| 文档 | 一句话 |
| --- | --- |
| `rag-over-docs.md` | 自指 RAG：1726 切片、R@1 42%、四类失败 |
| `rag-locate.md` | 定位式：模型指位置、程序抄原文；表格上下文 4→6/12 |
| `hybrid-retrieval.md` | bge-m3 + 重排 R@1 75%；BM25 +0（伪需求） |

### 5.6 训练与对齐

| 文档 | 一句话 |
| --- | --- |
| `minimind-training.md` | 30M 从零：预训练 2.8 h、SFT 1.8 h |
| `rlvr-small-model.md` | 30M 塌缩到众数；1.7B 同配方 47%→60% |
| `test-time-scaling.md` | 投票 −1 pp（错误系统性） |
| `rft-vs-grpo.md` | RFT +2 pp（梯度≈0）；GRPO +13 pp |

---

## 6. 环境与复现

| 用途 | 路径 |
| --- | --- |
| 通用 bench（torch 2.15 / transformers 5.2） | `~/apps/bench/yolo-venv` |
| vLLM 引擎（0.34 + FlashInfer） | `~/apps/infer-engines/.venv` |
| 实验脚本 | `~/apps/infer-research`、`~/apps/infer-engines` |
| 训练/RL | `~/apps/minimind` |
| 文档仓库 | `~/apps/edgevllm/docs/ubuntu-ai-box` |

**跑大模型前必做**：

```bash
echo 123456 | sudo -S systemctl stop vlm-server    # 常驻占 6.4 GB
# …实验…
echo 123456 | sudo -S systemctl start vlm-server
```

**vLLM 环境变量**：

```bash
export PATH=~/apps/infer-engines/.venv/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 7. 已知未解问题

| 问题 | 现状 |
| --- | --- |
| CosyVoice2 音质异常 | 排除 4 个因素后锁定声学栈实现问题，需 CUDA 12.x 或官方镜像 |
| 两步推理 `(a+b)×c` | 1.7B 只有 5%，结果奖励救不了，需过程奖励或工具调用 |
| 20+ 篇 K3 文档主题稀释 | 检索算法已到顶，需文档治理（合并/加元数据） |
| RWKV-7 | 权重与 `rwkv 0.8.32` args 不匹配，未跑通 |
| SGLAang / TensorRT-LLM / MIGraphX | sm_120 环境不划算，已验证放弃 |
