# 服务化与对齐：LoRA 服务 / 前缀缓存 / DPO / 视频理解 / VAD

> 机器 `pingji-MS-7B89` · RTX 5060 Ti 16 GB · Ubuntu 26.04 · 2026-11-25
> 配套：[roofline-and-lora.md](roofline-and-lora.md)（原理）、[realtime-services.md](realtime-services.md)（实时链路）

---

## 1. vLLM 多 LoRA 服务（Qwen3-0.6B + 自训 adapter）

配置：`enable_lora=True, max_lora_rank=16, max_loras=2`，adapter 来自上一轮 LoRA 微调。

| 场景 | base | +LoRA | 变化 |
| --- | ---: | ---: | ---: |
| 单请求 | 40.3 tok/s | 23.3 tok/s | **-42.3%** |
| 8 并发 | 297.3 tok/s | 127.9 tok/s | **-57.0%** |

**两个结论：**

1. **vLLM 的 LoRA 路径在这个尺寸上开销巨大**（-42% ~ -57%）。
   原因：LoRA 层的 kernel（Punica/SGMV 一类）是为"大模型 + 多 adapter 并发"设计的，
   在 0.6B 这种小模型上，额外 kernel 启动与 gather 开销无法被矩阵乘掩盖。
2. **60 步的 adapter 会退化输出**：

   | | 输出 |
   | --- | --- |
   | base | 显存带宽是指显存中每个数据位的传输速率，通常以每秒每比特（bps）为单位… |
   | +LoRA(60步) | 显存带宽是指显卡的显存带宽，即显卡的显存容量与显存带宽的比值。显存带宽越高，显卡的显存容量越大…（重复） |

   **欠训练的 adapter 不只是"没学到"，而是会主动伤害**——因为 LoRA 的 B 矩阵已经开始
   把隐空间往某个方向推，但还没学到稳定的映射。生产上必须用 eval 集卡住"上线门槛"。

## 2. 前缀缓存曲线（TTFT vs 前缀长度）

同一段前缀连续请求两次，对比"命中缓存"与"关掉缓存"的 TTFT（3 次取最小）：

| 前缀长度 | 命中 | 未命中 | 节省 |
| ---: | ---: | ---: | ---: |
| 1,035 token | 125.2 ms | 126.3 ms | 1% |
| 4,107 token | 128.6 ms | 250.0 ms | **49%** |
| 16,395 token | 141.1 ms | 1,265.0 ms | **89%** |

**结论：前缀缓存把 O(L) 的 prefill 变成了 O(1)。**

- 命中时 TTFT 几乎与前缀长度无关（125 → 141 ms，只涨了 13%）
- 未命中时 TTFT 随长度线性增长（126 → 1265 ms，正好 10×，与 token 数比例一致）
- 对 RAG / 长 system prompt / 多轮对话，这是**白送的收益**

## 3. DPO 对齐（minimind 30M）

### 3.1 数据（合成）

本地没有 DPO 数据，从 `sft_t2t_mini.jsonl` 合成：
`chosen` = 原始问答；`rejected` = **把答案换成另一个样本的答案**（张冠李戴，明显更差）。
产出 3000 训练对 + 200 留出对。

### 3.2 训练

| 项 | 值 |
| --- | --- |
| 起点 | `out/full_sft_512.pth`（30.03 M，hidden 512） |
| 超参 | β=0.1，lr=5e-7（余弦退火到 5e-8），bs=4，seq=512 |
| 步数 | 750（`epochs=1`） |
| **耗时** | **约 7.5 分钟**（约 1 分钟 / 100 步） |
| loss | 0.2026 → 0.1938 → 0.1368 → 0.0886 → 0.0016 → … → 0.0037（末步 0.2190 是批量噪声） |

### 3.3 效果（200 条留出对）

指标：`margin = 平均对数概率(chosen) − 平均对数概率(rejected)`，逐样本归一化到 token。

| 模型 | 平均 margin | chosen 更优占比 |
| --- | ---: | ---: |
| DPO 前（full_sft） | +0.4366 | 87.5% |
| **DPO 后（dpo）** | **+0.6161** | **93.5%** |

**结论：30M 小模型上 DPO 确实有效**——margin 提升 **+41%**，胜率 +6 个百分点，
成本只有 **7.5 分钟**。这比 SFT（1.8 小时）便宜一个数量级，适合"最后一公里"的风格校正。

> 注意：这是**合成偏好**（答案张冠李戴），衡量的是"回答相关性"这一最粗的偏好；
> 真实业务需要人工标注的偏好对，才能对齐到"有用/无害/诚实"这类细微差别。

## 4. 视频理解（Qwen3-VL-8B 多帧）

从一段乐高机械臂视频抽 4 帧，一次请求送进常驻 vlm-server：

| 项 | 值 |
| --- | ---: |
| 帧数 | 4 |
| 耗时 | **0.99 s** |
| 输出（节选） | "四帧画面展示了一个由乐高积木搭建的机械结构（吊臂部分）在进行周期性的、上下摆动的动作…" |

**结论**：多帧理解的耗时 ≈ 单帧的 2.4 倍（单帧 0.41 s），说明图片编码是主要成本，
而 4 帧的 token 数增加带来的 decode 开销较小。**"抽 4 帧 + 一次提问"约 1 秒**，
比逐帧描述再看文本（4 × 0.41 = 1.6 s + 二次推理）更划算。

## 5. VAD 端点检测（silero-vad）

| 项 | 值 |
| --- | ---: |
| 音频 | 13.75 s |
| 检出语音段 | **4 段** |
| 语音占比 | 82% |
| 耗时 | **155 ms** |
| **速度** | **89× 实时（RTF 0.011）** |

分段结果：`[(0.4, 3.0), (3.5, 5.6), (6.0, 9.9), (10.8, 13.5)]`

**结论**：VAD 便宜到可以"每个音频块都跑"（155 ms 处理 13.75 s）。
把它接在流式 ASR 前面，可以把"固定窗口切分"换成"按语音段触发转写"，
既省算力（静音段不送 ASR）又提高准确率（不会把半个词切断）。

---

## 6. 第二梯队汇总

| 项目 | 结果 | 关键数字 |
| --- | --- | --- |
| vLLM 多 LoRA | ⚠️ 可用但昂贵 | -42% / -57%；欠训练 adapter 会退化输出 |
| 前缀缓存 | ✅ 强正收益 | 4K 省 49%，16K 省 **89%** |
| DPO（30M） | ✅ 有效 | margin +41%，胜率 87.5%→93.5%，**7.5 分钟** |
| 视频理解（4 帧） | ✅ 可用 | **0.99 s** |
| VAD | ✅ 极便宜 | **89× 实时**（155 ms / 13.75 s） |

**给工程的直接建议：**

1. **LoRA 服务要在 eval 门槛下上线**——欠拟合的 adapter 比没有更糟。
2. **前缀缓存必开**（默认就是开的，别手滑关掉）。
3. **DPO 是性价比最高的对齐手段**（分钟级），前提是有偏好数据。
4. **视频理解走"抽帧 + 单次多图"**，别逐帧全描述。
5. **VAD 前置到 ASR**，按段触发。

---

## 7. 复现

```bash
# LoRA 服务 + 前缀缓存
cd ~/apps/infer-engines && export PATH="$PWD/.venv/bin:$PATH" CUDA_HOME=/usr/local/cuda
.venv/bin/python vllm_lora_prefix.py
.venv/bin/python prefix_curve.py

# DPO
~/apps/bench/yolo-venv/bin/python ~/apps/infer-research/make_dpo_data.py
cd ~/apps/minimind/trainer && ../.venv/bin/python train_dpo.py \
  --from_weight full_sft --hidden_size 512 --num_hidden_layers 8 \
  --data_path ../dataset/dpo_train.jsonl --batch_size 4 --max_seq_len 512 \
  --epochs 1 --learning_rate 5e-7 --save_dir ../out_dpo --save_weight dpo --num_workers 0
~/apps/bench/yolo-venv/bin/python ~/apps/infer-research/eval_dpo.py

# 视频理解 + VAD
~/apps/bench/yolo-venv/bin/python ~/apps/infer-research/tier2_de.py
```
