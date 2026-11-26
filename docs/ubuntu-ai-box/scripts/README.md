# 脚本与复现

本目录两个脚本对应正文 [minimind-training.md](../minimind-training.md) 第 6 节（分布式验证）。

它们**不属于 minimind 仓库**，通过 `MINIMIND_HOME` / `MINIMIND` 指向本地 minimind 目录即可，
不需要把脚本复制进去。

| 脚本 | 作用 | 需要 |
| --- | --- | --- |
| [`ddp_cpu_check.py`](ddp_cpu_check.py) | CPU 双进程验证 DDP 正确性（5 项检查） | 2 核以上 CPU，**不需要 GPU** |
| [`run_ddp_2gpu.sh`](run_ddp_2gpu.sh) | 多卡训练启动器（含 lr √N 缩放） | ≥2 张 GPU |

---

## 1. `ddp_cpu_check.py`

```bash
MINIMIND_HOME=~/apps/minimind \
OMP_NUM_THREADS=4 torchrun --nproc_per_node 2 --standalone ddp_cpu_check.py
```

本机实测输出（12 核 CPU，30.03 M 参数模型）：

```
world_size              : 2 进程（gloo）
模型参数量              : 30.03 M（hidden=512, 8 层）
每进程 batch            : 2 × seq 128，梯度累积 2
等效 batch              : 8 条样本/步
--------------------------------------------------------------
A1 梯度跨进程最大差     : 0.000e+00   (应≈0)
A2 DDP梯度 vs 单进程    : 1.118e-08   (应<1e-5)
A3 loss(本进程)         : 8.8881
A4 loss(参考批次)       : 8.8686
--------------------------------------------------------------
B step1 参数最大差        : 0.000e+00    loss=8.8446  1.22s
B step2 参数最大差        : 0.000e+00    loss=8.8834  1.16s
B step3 参数最大差        : 0.000e+00    loss=8.8534  1.07s
--------------------------------------------------------------
C  不用DDP时参数最大差   : 2.001e-03   (应>>0，证明检测有效)
==============================================================
单步耗时(CPU, 2进程)    : 1.15s
```

**怎么读这几行：**

- `A2` 是核心：DDP 的 AllReduce 求平均，数值上等于「把 2 张卡的 4 条样本拼成一个 batch=4 求梯度」。
  1e-8 的残差只是 float32 累加顺序不同。
- `B` 的三行全 0，说明经过 AdamW 更新 + 梯度裁剪 + 梯度累积之后，两个进程**没有悄悄跑偏**。
- `C` 是反证：故意不包 DDP，参数差立刻到 2e-3 —— 证明前面那些 0 不是「检测器坏了」。

---

## 2. `run_ddp_2gpu.sh`

```bash
MINIMIND=~/apps/minimind bash run_ddp_2gpu.sh            # 预训练，2 卡
MINIMIND=~/apps/minimind TASK=sft bash run_ddp_2gpu.sh   # SFT
MINIMIND=~/apps/minimind NPROC=4 BATCH=32 bash run_ddp_2gpu.sh
```

脚本会先打印等效 batch 和缩放后的 lr，再执行 `torchrun`：

| 变量 | 默认 | 说明 |
| --- | ---: | --- |
| `NPROC` | 2 | 卡数 |
| `BATCH` | 24 | **每卡** batch |
| `ACC` | 8 | 梯度累积步数 |
| `BASE_LR` | 1e-4 | 单卡 lr，脚本自动按 √NPROC 缩放 |
| `SEQ` | 512 | `--max_seq_len` |
| `TASK` | pretrain | `pretrain` / `sft` |
| `MINIMIND` | `~/apps/minimind` | minimind 仓库位置 |

---

## 3. 前置条件：训练脚本要能选 gloo

minimind 原始代码把分布式写死成 GPU，CPU 上会直接报错：

```python
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
args.device = f"cuda:{local_rank}"
```

改成按设备自动选择（**GPU 路径行为完全不变**）：

```python
def distributed_uses_cuda():
    """分布式是否走 GPU：无可用 GPU，或显式设置 DDP_CPU=1 时退回 CPU + gloo。"""
    return torch.cuda.is_available() and os.environ.get("DDP_CPU", "0") != "1"
```

用环境变量显式切换，而不是只靠 `torch.cuda.is_available()` —— 有显卡的机器上后者永远为真，
想跑 CPU 验证就永远切不过去。

改完用真实训练脚本验证（`pretrain_smoke200.jsonl` 是从真实语料切出来的 200 条小样本）：

```bash
head -200 ~/apps/minimind/dataset/pretrain_t2t_mini.jsonl \
  > ~/apps/minimind/dataset/pretrain_smoke200.jsonl
cd ~/apps/minimind/trainer
DDP_CPU=1 OMP_NUM_THREADS=4 ../.venv/bin/torchrun --nproc_per_node 2 --standalone \
  train_pretrain.py --device cpu --hidden_size 512 --batch_size 4 --max_seq_len 128 \
  --epochs 1 --data_path ../dataset/pretrain_smoke200.jsonl --num_workers 0 \
  --save_dir ../out_smoke
```

本机实测：200 条样本 / 25 步，loss 8.85 → 8.27，29.9 s（857 tok/s）。
同配置单卡 GPU（nccl）为 13.0 s（1,968 tok/s）—— 这组对比只说明两条路径都通，
seq 128 / batch 4 时启动开销占了大头，不代表真实速度比。

---

## 4. 推理加速实验脚本（路线 A/B/C）

对应 [inference-acceleration.md](../inference-acceleration.md)。四个脚本顶部有硬编码的模型路径
（本机 `~/.cache/modelscope/...`），换机器需要改。

| 脚本 | 内容 | 依赖 |
| --- | --- | --- |
| `routeA_quant.py` | transformers + bitsandbytes 对比 BF16 / INT8 / NF4 | transformers、bitsandbytes |
| `routeB_kv.py` | kvpress 的 7 种 KV 压缩策略 × 多档上下文（大海捞针） | transformers、kvpress |
| `routeC_spec.py` | 投机解码：0.6B 草稿 + 1.7B 靶（bf16 草稿） | transformers |
| `routeC_spec_nf4.py` | 同上，草稿换成 NF4 4-bit | transformers、bitsandbytes |

```bash
python routeB_kv.py          # 模型路径见脚本顶部常量
CTX_LENS=2000,8000,16000 python routeB_kv.py
```

> 量化梯度（GGUF 七档）与投机解码的 llama.cpp 那一组用的是命令行工具，
> 完整命令见 [inference-acceleration.md](../inference-acceleration.md) 的「复现命令」一节。

---

## 5. 第二～六轮实验脚本

对应 [inference-acceleration.md](../inference-acceleration.md) 的 D–R 节。全部硬编码本机路径，
换机器需改脚本顶部的模型路径常量。

| 脚本 | 实验 | 依赖 |
| --- | --- | --- |
| `constrained_decode.py` | 约束解码（xgrammar 强制 JSON Schema）vs 自由生成 | transformers、xgrammar |
| `flex_compiled.py` | FlexAttention 滑窗 mask vs SDPA（含 torch.compile） | torch ≥ 2.5 |
| `wanda.py` | Wanda 剪枝（\|W\|×‖X‖ 打分，稀疏度扫描） | transformers |
| `quarot2.py` | QuaRot 旋转量化（Hadamard + 折叠 RMSNorm + RTN） | transformers |
| `r3_fp8_kivi.py` | torchao FP8 三种配置 / KIVI 式 KV 量化 | transformers、torchao |
| `vllm_test.py` | vLLM 单流与 8 并发基线 | vllm |
| `vllm_extra.py` | vLLM 并发曲线 + ngram 投机 | vllm |
| `kv_fp8.py` | vLLM KV cache fp8 对比 | vllm |
| `prefill_curve.py` | 长上下文 prefill 曲线（**必须关前缀缓存**） | vllm |
| `gptq_test.py` | GPTQ-Int8 权重 + ngram 投机叠加 | vllm |
| `r7_vllm.py` | vLLM 原生结构化输出（guided decoding）对比 | vllm |
| `decode_curve.py` | 长上下文 decode 曲线（用 T256−T1 摊薄 prefill） | vllm |
| `k3_ngram.sh` | K3 上用 llama-server 测 ngram 投机 | K3 + SpacemiT llama.cpp |

> vLLM 相关脚本需要 `ninja` 在 PATH 中、并设 `CUDA_HOME=/usr/local/cuda`
> （CUDA 13 的头文件在 `targets/x86_64-linux/include`，
> 否则 fp8/量化路径会以 `FileNotFoundError: 'ninja'` 或找不到 `cuda_runtime.h` 失败）。\n
---

## 6. 实时链路脚本（流式 ASR / VLM / SAM2 / CosyVoice）

对应 [realtime-services.md](../realtime-services.md)。

| 脚本 | 用途 |
| --- | --- |
| `stream_asr.py` | 滚动窗口流式 ASR，测更新延迟与 RTF |
| `vl_direct.sh` | 打常驻 vlm-server（8092），逐帧测 VLM 延迟 |
| `sam2_test.py` | SAM2 hiera-tiny 点提示分割，测单帧耗时 |
| `cosy_asr.py` | ASR 回听工具（字符级相似度判卷） |
| `cosy_diag2.py` | CosyVoice2 分环节诊断（精度/前端/LLM） |

---

## 7. Roofline 与训练脚本

对应 [roofline-and-lora.md](../roofline-and-lora.md)。

| 脚本 | 用途 |
| --- | --- |
| `roof_kernels.py` | 被测 kernel（gemm / copy / add / attn / rmsnorm） |
| `ncu_roofline.sh` | 用 sudo ncu 抓 SpeedOfLight 段（无权限会报 ERR_NVGPUCTRPERM） |
| `bench_ops.py` | torch 侧计时，算真实 TFLOPS / GB/s |
| `lora_llm.py` | Qwen3-0.6B LoRA 微调（含数据构造与 A/B） |
| `lora_ab.py` | base vs LoRA 对比（记得 enable_thinking=False） |

---

## 8. 服务化与对齐脚本

对应 [serving-and-alignment.md](../serving-and-alignment.md)。

| 脚本 | 用途 |
| --- | --- |
| `vllm_lora_prefix.py` | vLLM 多 LoRA 服务开销 + 前缀缓存初测 |
| `prefix_curve.py` | 前缀缓存精确曲线（按 token 数构造 prompt，3 次取最小） |
| `make_dpo_data.py` | 从 SFT 数据合成 DPO 偏好对（rejected = 张冠李戴） |
| `eval_dpo.py` | DPO 前后 margin / 胜率对比 |
| `tier2_de.py` | 视频多帧理解 + silero-vad 端点检测 |

---

## 9. 补充闭环脚本

| 脚本 | 用途 |
| --- | --- |
| `decode_32k.py` | 32K 上下文 decode（需先停 vlm-server，`gpu_memory_utilization=0.8`） |
| `vad_asr.py` | VAD 分段 vs 固定窗口的 ASR 对比 |
| `asr_sizes.py` | tiny/base/small/medium 尺寸扫描 |
| `frame_scan.py` | 视频帧数 2/4/8 扫描 |
| `lora_formal_eval.py` | LoRA 留出集 eval loss + 长度统计 + 定性 |
| `vram_budget.py` | 多模型共存显存画像（含 cu12/cu13 冲突验证） |

---

## 10. DeepSeek-OCR 脚本

对应 [deepseek-ocr.md](../deepseek-ocr.md)。

| 脚本 | 用途 |
| --- | --- |
| `dsocr_prep.sh` | 下载权重 + 生成 A4 测试文档图与参考文本 |
| `dsocr_default.py` | 单页 OCR（**必须 `moe_backend="triton"`，且不传 `mm_processor_kwargs`**） |

---

## 11. MiniCPM-o 4.5 脚本

| 脚本 | 用途 |
| --- | --- |
| `fix_build_omni.sh` | 给 llama.cpp-omni 打 F_NOCACHE 补丁并重新编译 |
| `run_mco2.sh` | 视觉问答（必须 `--no-mmproj-offload`） |

---

## 12. TTS 脚本（Kokoro）

| 脚本 | 用途 |
| --- | --- |
| `kokoro_tts.py` | Kokoro-82M 中文 TTS（`SKIP_ASR=1` 生成、单独进程做 ASR 验收） |

---

## 13. 端到端与优化脚本

| 脚本 | 用途 |
| --- | --- |
| `voice_agent.py` | VAD→ASR→LLM→TTS 全链路（ASR 走子进程绕 cu12/cu13 冲突） |
| `cudagraph_test.py` | vLLM eager vs CUDA Graph 对照 |
| `k3_tts_run.sh` | 在 K3 上跑 Kokoro（stub misaki + wave 写音频） |

---

## 14. RAG 脚本

| 脚本 | 用途 |
| --- | --- |
| `rag_build.py` | 文档切片（按标题 + 700 字滑窗）+ bge-small-zh 向量化 |
| `rag_eval.py` | 12 问召回率 + 带上下文生成（**记得 enable_thinking=False**） |

---

## 15. 叠加与定位式 RAG

| 脚本 | 用途 |
| --- | --- |
| `stack_all.py` | 6 配置叠加矩阵（eager/graph/fp8kv/ngram × GPTQ/bf16） |
| `ngram_crossover.py` | ngram 投机的并发交叉点（1/4/8/16/32/64） |
| `rag_locate.py` | 直接回答 / 单选定位 / 句子级 reranker 三路对比 |
| `rag_locate2.py` | 上下文增强消融（裸句 / +章节标题 / +表头）+ 交叉验证路由 |

---

## 16. RLVR（可验证奖励）

| 脚本 | 用途 |
| --- | --- |
| `probe_rlvr.py` | 基线探测：模型困在 `<think>`、0/40 格式正确 |
| `grpo_math.py` | GRPO 训练（强制 `<answer>` 前缀 + 距离塑形奖励；`--kl` 开 KL；`--control random` 随机奖励对照） |
| `eval_grpo.py` | 三套题评测（nat / bal / nat9）+ 预测分布与熵 |
| `verify_kl.py` | 3 种子复核均衡题集准确率 |
| `grpo_qwen.py` | Qwen3-1.7B 上的 LoRA-GRPO（三位×两位，300 步） |
| `verify_qwen_rl.py` | 配对验证：同一组新题对比基座 vs LoRA（47%→60%） |
| `probe_qwen_math.py` / `probe_difficulty.py` / `probe_hard*.py` | 难度带探测（找到 52.5% 的三位×两位） |

---

## 17. 测试时扩展

| 脚本 | 用途 |
| --- | --- |
| `tta_sc.py` | self-consistency：200 题 × 32 采样，N=1..32 多数投票 + oracle |
| `tta_why.py` | 机制分析：答案集中度、高温实验 |

---

## 18. 长上下文

| 脚本 | 用途 |
| --- | --- |
| `needle.py` | needle-in-a-haystack：2K–32K × 3 深度，测准确率/prefill/KV |
| `needle_mid.py` | 20K/24K/28K 加密扫描（定位崩溃点） |

---

## 19. 量化与任务准确率

| 脚本 | 用途 |
| --- | --- |
| `quant_quality.py` | vLLM 内 bf16 vs GPTQ-Int8：算术/JSON/needle |
| `quant_quality_hf.py` | HF 内 bf16 vs NF4（含峰值显存） |
| `quant_paired.py` | 300 题配对检验 + McNemar（量化评估必做） |

---

## 20. 草稿模型投机

| 脚本 | 用途 |
| --- | --- |
| `draft_spec.py` | Qwen3-0.6B 当 1.7B 的 draft：1/8/32 并发 + 接受率 |
| `draft_diag.py` | 诊断：读取 vLLM 投机指标、k=3/5 对比 |
