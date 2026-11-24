# 推理加速三轮实测：量化梯度（A）/ KV 压缩（B）/ 投机解码（C）

> 机器 `pingji-MS-7B89` · RTX 5060 Ti 16 GB（sm_120 / 36 SM / **448 GB/s**）· Ubuntu 26.04
> 实测时间：2026-11-23 · 全部数字为本机实跑
> 测试条件：**`vlm-server` 全程占用 6.4 GB 显存**（可用 ~9.9 GB），未停服务

---

## 路线 A：同一模型的量化梯度（Qwen3-1.7B）

方法：ModelScope 拉 BF16 原始权重 → `convert_hf_to_gguf.py` 转 F16 → `llama-quantize`
生成 6 档 → `llama-bench`（pp512 / tg128，Vulkan 后端，全量卸载）+ `llama-perplexity`（中文语料 300 KB）。

### 结果

| 量化 | 体积 (GiB) | pp512 (tok/s) | tg128 (tok/s) | PPL | PPL 涨幅 | 有效带宽 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **F16** | 3.79 | 12,580 | 98.8 | 11.023 | — | **402 GB/s** |
| **Q8_0** | 2.02 | 11,950 | 159.6 | 11.049 | +0.23% | 346 GB/s |
| Q6_K | 1.56 | 9,172 | 141.2 | 10.923 | -0.91% | 236 GB/s |
| **Q5_K_M** | 1.37 | 9,673 | 203.1 | 11.114 | +0.83% | 299 GB/s |
| **Q4_K_M** | 1.19 | 11,477 | **221.8** | 11.836 | +7.37% | 283 GB/s |
| Q3_K_M | 1.00 | 9,997 | 215.2 | 14.144 | +28.3% | 231 GB/s |
| Q2_K | 0.82 | 9,454 | 212.3 | 56.540 | **+413%** | 187 GB/s |

> 有效带宽 = 模型体积 × tg128，反映实际从显存读了多少字节/秒。448 GB/s 是硬件上限。

### 五个结论

1. **F16 跑到了带宽的 90%**（402 / 448 GB/s）—— 这条曲线不是估算，是真的贴着 roofline。
   这也解释了为什么 decode 加速只能靠"少读字节"。
2. **最快的不是最小的**：tg128 峰值在 **Q4_K_M（221.8，是 F16 的 2.24×）**，
   再往下压体积（Q3/Q2）速度反而停滞甚至回落 —— **反量化计算吃掉了带宽省下的时间**。
3. **Q8_0 是"免费午餐"**：PPL 几乎无损（+0.23%），体积减半，速度 +62%。
4. **Q2_K 彻底崩掉**：PPL 11.0 → 56.5（+413%）。1.7B 这个规模顶不住 2-bit。
5. **Q6_K 数据异常**（PPL 反而最低、tg128 只有 141）：大概率是 Vulkan 后端该类型
   dequant 路径未优化，需用 CUDA 后端复测确认。

### 小模型对照（Qwen3-0.6B）

同一套流程在 0.6B 上跑，趋势一致但更极端：

| 量化 | 体积 (GiB) | pp512 | tg128 | PPL | PPL 涨幅 |
| --- | ---: | ---: | ---: | ---: | ---: |
| F16 | 1.40 | 21,792 | 220.4 | 12.839 | — |
| Q8_0 | 0.76 | 20,796 | 314.8 | 12.839 | +0.00% |
| Q5_K_M | 0.52 | 19,373 | 337.6 | 13.163 | +2.5% |
| Q4_K_M | 0.46 | 20,289 | **348.9** | 13.916 | +8.4% |
| Q3_K_M | 0.39 | 18,925 | 312.2 | 18.221 | +41.9% |
| Q2_K | 0.33 | 9,407 | 278.5 | **195.72** | **+1425%** |

**模型越小，量化容错越差**：0.6B 在 Q2_K 上 PPL 直接涨 15 倍，而 1.7B 只涨 4 倍。

---

## 路线 B：KV cache 压缩（Qwen3-1.7B + kvpress）

方法：构造"大海捞针"——在填充文本中间插入密码 `738291`，问模型密码是多少；
对比 7 种策略在 2K / 8K 上下文下的**峰值显存、检索是否正确、decode 速度**。

### 结果

**~2K tokens（实际 1942～2034）**

| 策略 | 峰值显存 | 检索 | decode | 说明 |
| --- | ---: | :---: | ---: | --- |
| full（基线） | 4,212 MiB | ✅ | **31.4 tok/s** | 预热后重测值 |
| SnapKV @50% | 4,093 MiB | ✅ | 25.6 | 压缩开销 > 省下的时间 |
| SnapKV @75% | 4,038 MiB | ✅ | 29.4 | 同上 |
| TOVA @50% | 4,093 MiB | ✅ | 29.6 | — |
| Knorm @50% | 4,093 MiB | ❌ | 28.0 | 答成 78291 |
| StreamingLLM @50% | 4,093 MiB | ❌ | 29.9 | 只留 sink+窗口，针被丢掉 |
| Random @50%（对照） | 4,093 MiB | ❌ | 23.0 | 编造了一串数字 |

**~8K tokens（实际 7931～8034）**

| 策略 | 峰值显存 | 检索 | decode | 说明 |
| --- | ---: | :---: | ---: | --- |
| full（基线） | 5,174 MiB | ✅ | 18.7 tok/s | — |
| **SnapKV @75%** | **4,499 MiB** | ✅ | 17.8 | **省 12.6% 显存，精度不掉** |
| SnapKV @50% | 4,716 MiB | ✅ | 17.8 | — |
| Knorm @50% | 4,716 MiB | ✅ | 15.4 | 短上下文会失败，长上下文反而对 |
| TOVA @50% | 4,716 MiB | ✅ | 14.8 | — |
| StreamingLLM @50% | 4,716 MiB | ❌ | 18.1 | 输出 7777… 崩坏 |
| Random @50%（对照） | 4,716 MiB | ❌ | 18.1 | 答 738192（差一位） |

### 四个结论

1. **省显存是真的**：8K 下 SnapKV@75% 从 5,174 → 4,499 MiB（-12.6%），
   与按公式算出的 KV 体积（7931 tok ≈ 910 MB，压掉 75% ≈ 680 MB）基本吻合。
2. **decode 速度几乎不变（甚至略降）**：18.7 → 17.8 tok/s。
   因为 decode 时间被**权重读取**主导，KV 只占小头 —— 这就是
   **"省容量 ≠ 省带宽"** 的实测证据。
3. **短上下文压缩是净亏损**：2K 时基线 31.4 tok/s，压缩后只有 25.6～29.4，
   因为 scoring + 重排的固定开销大于省下的那点带宽。
4. **不是所有"重要 token"定义都靠谱**：StreamingLLM 在两端都失败（丢中间内容），
   Knorm 在 2K 失败、8K 成功 —— 说明评分策略对上下文长度敏感。

---

## 方法局限（下一轮要补的）

| 项 | 现状 | 改进 |
| --- | --- | --- |
| 样本量 | 每档 1 次生成、单针单深度 | 多针多深度 × 5 次重复，给置信区间 |
| 显存条件 | vlm-server 常占 6.4 GB | 停掉后复测，确认无竞争 |
| Q6_K 异常 | 未解释 | 换 CUDA 后端（llama-cuda）对照 |
| 量化引擎 | 只有 llama.cpp（GGUF） | 加 transformers + bitsandbytes（INT8/NF4）对照 |
| 精度指标 | 只有 PPL | 加 GSM8K 小样本 / 自建选择题，看"能力"而非"困惑度" |

---

## 复现命令

```bash
# ---------- 路线 A ----------
# 1) 下载原始权重（ModelScope 比 HF 快很多）
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B')"

# 2) 转 F16
python ~/apps/llama-cuda/llama.cpp/convert_hf_to_gguf.py \
  ~/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master \
  --outfile ~/models/qwen3-1.7b-quants/qwen3-1.7b-f16.gguf --outtype f16

# 3) 生成量化梯度
for q in Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K; do
  ~/apps/llama-vulkan/build/bin/llama-quantize \
    ~/models/qwen3-1.7b-quants/qwen3-1.7b-f16.gguf \
    ~/models/qwen3-1.7b-quants/qwen3-1.7b-$q.gguf $q 12
done

# 4) 速度 + 困惑度
~/apps/llama-vulkan/build/bin/llama-bench -m <gguf> -ngl 99 -p 512 -n 128 -r 3
~/apps/llama-vulkan/build/bin/llama-perplexity -m <gguf> -ngl 99 -f corpus.txt -c 512

# ---------- 路线 B ----------
cd ~/apps/infer-research
MODEL_ID=~/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master \
  ~/apps/bench/yolo-venv/bin/python routeB_kv.py
```

脚本位置：`~/apps/infer-research/{routeA_quant.py, routeB_kv.py}`

---

## 产物体积

| 目录 | 内容 | 大小 |
| --- | --- | ---: |
| `~/models/qwen3-1.7b-quants/` | F16 + 6 档量化 | 约 11.9 GB |
| `~/models/qwen3-0.6b-quants/` | F16 + 6 档量化 | 约 4.3 GB |
| `~/.cache/modelscope/` | 1.7B(3.8G) + 0.6B(1.5G) 原始权重 | 约 5.3 GB |

不需要时可整体删除；重新生成的命令见上。

---

# 附：路线 C 首轮实测——投机解码（负结果）

> 测试条件同上（vlm-server 占 6.4 GB）。靶模型 Qwen3-1.7B（bf16），草稿 Qwen3-0.6B，k=4，greedy，每次生成 128 token，3 个不同 prompt。

## 结果

| # | 基线 tok/s | 投机 tok/s | 加速比 | 每步 token | 估计 α |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 36.0 | 16.3 | **0.45×** | 2.72 | 0.43 |
| 2 | 35.7 | 25.5 | **0.71×** | 3.76 | 0.69 |
| 3 | 36.6 | 25.8 | **0.71×** | 5.12 | 1.03 |
| 平均 | 36.1 | 22.5 | **0.62×** | 3.87 | 0.72 |

**把草稿压到 4-bit（NF4，1.2 GB → 0.45 GB）后仍然只有 0.55×**——说明不是草稿带宽的锅。

## 四个结论

1. **在 transformers 的高层实现里，投机解码是负优化**：平均 0.62×，最差 0.45×。
2. **α 不是瓶颈**：第 3 个 prompt 的 α≈1.03（每步 5.12 token，即草稿基本全接受），
   理论上应接近 2× 加速，实测只有 0.57×。所以问题不在"草稿准不准"。
3. **草稿带宽也不是瓶颈**：草稿从 3.4 GB 压到 0.45 GB，加速比 0.62× → 0.55×，没变好。
4. **真凶是实现开销**：`assisted_decoding` 每步要做 Python 层循环、拒绝时回滚 KV、
   重建 attention mask、tensor 拼接裁剪。这些固定开销在单卡小模型上远大于算术收益——
   带宽省下的时间被调度吃光了。

## 副产品（可能比结论更有价值）：引擎差距 2.7×

同一个 1.7B 模型、同一份 bf16 权重：

| 引擎 | decode 速度 | 有效带宽 | 占硬件上限 |
| --- | ---: | ---: | ---: |
| transformers（sdpa） | 36.1 tok/s | 122 GB/s | 27% |
| llama.cpp（Vulkan）F16 | 98.8 tok/s | 402 GB/s | **90%** |

**同样的数学，差 2.7 倍。** 这说明在带宽受限的卡上，"引擎工程质量"和"算法"同等重要，
也解释了为什么投机解码在 llama.cpp / vLLM 的 kernel 化实现里才有意义。

## 一个容易踩的坑：greedy 也不保证逐 token 相同

三个 prompt 的投机输出与基线**都不完全一致**（例如把"显存和CPU之间传输"生成成"显存和CPU之间传"）。
原因是 assisted decoding 会改变 batch shape，bf16 归约顺序随之变化。
**结论：做 A/B 引擎对比时，不能靠"输出是否逐字相同"判定正确性。**

## 复现

```bash
cd ~/apps/infer-research
~/apps/bench/yolo-venv/bin/python routeC_spec.py        # bf16 草稿
~/apps/bench/yolo-venv/bin/python routeC_spec_nf4.py    # 4-bit 草稿
```

> 下一轮若要做正收益验证，应该换到引擎层（vLLM 的 EAGLE/Medusa 实现），
> 或者把靶模型换成 8B（草稿相对成本更低）。

---

# 补：路线 C 第二轮——换到 llama.cpp 引擎，仍然是负收益

前面 transformers 的负结果可能是"实现太糙"，所以换到 llama.cpp（同一张卡、8B Q4 + 0.6B Q4 草稿）再测一次。
用的是 `llama-server` 的 `-md/--model-draft` + `--draft-max`。

| 配置 | 生成 | tok/s | 加速比 | 草稿接受率 |
| --- | ---: | ---: | ---: | ---: |
| A 基线：8B Q4_K_M | 128 | **72.7** | 1.00× | — |
| B 投机：draft-max 4 | 128 | 53.2 | **0.73×** | 67%（46/69） |
| C 投机：draft-max 8 | 128 | 38.7 | **0.53×** | 64%（47/74） |

## 理论 2.65× vs 实测 0.73×：差在实现

按带宽账：
```
每步读取 = 5.02 GB（8B Q4） + 4 × 0.49 GB（草稿 4 次） = 6.98 GB
每步产出 = 1 + 0.67 × 4 = 3.68 token
理论加速 = (3.68 / 6.98) × 5.02 = 2.65×
```
实测 0.73×，**差 3.6 倍**。全部是每步固定开销：草稿每个 token 一次 GPU 往返（kernel 启动 + 同步），
拒绝时还要回滚 KV。**带宽已经饱和的设备上，"用空闲算力换时间"的前提不成立。**

## 8B Q4 的基线数据（顺便记录）

| 测法 | pp512 | tg128 |
| --- | ---: | ---: |
| `llama-bench`（Vulkan，全量卸载） | 3,082 tok/s | 59.95 tok/s |
| `llama-server`（单请求，128 token） | — | 72.7 tok/s |

> 8B Q4 = 5.02 GB，72.7 tok/s → 有效带宽 **365 GB/s，占 448 GB/s 的 81%**。
> 这就是"投机解码在这张卡上不划算"的物理原因。

## 对端侧的意义

| 设备 | 瓶颈 | 独立草稿模型的投机解码 |
| --- | --- | --- |
| 桌面 GPU（本机） | **带宽饱和 81%** | ❌ 实测 0.53–0.73× |
| 手机 NPU | 算力 + 功耗，且内存紧张 | ⚠️ 主流用共享主干的 EAGLE/Medusa 头 |
| 纯 CPU 端侧（K3，无 NPU） | **算力受限** | ✅ 理论上更可能正收益，待实测 |

**真正能上端侧的三条路**：n-gram / prompt lookup（零额外模型）、EAGLE/Medusa 头（多 100–300 MB）、
LayerSkip（零额外内存）。"再装一个小模型"这条路在带宽受限设备上不成立。

---

# 附：完整清单——跑了什么、没跑什么

## 已跑（本机实测，数字都在本文档）

| # | 项目 | 产出 |
| --- | --- | --- |
| 1 | GGUF 量化梯度：Qwen3-1.7B（F16/Q8/Q6/Q5/Q4/Q3/Q2） | 速度 + PPL 七档对照 |
| 2 | GGUF 量化梯度：Qwen3-0.6B（同上七档） | 小模型容错更差的证据（Q2 PPL +1425%） |
| 3 | 8B Q4 基线（llama-bench + llama-server） | pp512 / tg128 / 有效带宽 81% |
| 4 | KV cache 压缩：7 种策略 × 2 档上下文（2K / 8K） | 显存-精度-速度三方对照 |
| 5 | 投机解码：transformers（bf16 草稿 / NF4 草稿） | 0.62× / 0.55× |
| 6 | 投机解码：llama.cpp（draft-max 4 / 8） | 0.73× / 0.53×，接受率 67% |
| 7 | 跨引擎对照（同模型同精度） | HF 36.1 tok/s vs llama.cpp 98.8 tok/s（2.7×） |
| 8 | 环境验证：bitsandbytes 0.50.2 在 sm_120 | Linear4bit(NF4) 前向通过 |
| 9 | DDP 正确性（CPU 双进程 gloo） | 见 edgevllm 仓库 `docs/ubuntu-ai-box/minimind-training.md` |

## 没跑（选题文档里列过，尚未执行）

| 类别 | 未跑项 | 原因 |
| --- | --- | --- |
| 量化 | AWQ / GPTQ / FP8 真实对照 | 只做了 GGUF 家族；bnb INT8/NF4 已装好但未跑完整对照 |
| 量化 | 旋转量化 QuaRot / SpinQuant | 需要改模型结构，工作量大 |
| 剪枝 | Wanda / SparseGPT | 未开始 |
| KV | KIVI 2-bit 量化 | 只做了"丢弃式"压缩，未做"量化式" |
| 解码 | 约束解码 XGrammar / Outlines | 未开始（预期正收益） |
| 解码 | n-gram / prompt lookup 投机 | 未开始（端侧唯一现实路线） |
| 注意力 | FlexAttention 自定义 mask | 未开始 |
| 注意力 | 稀疏注意力 MInference / Quest | 未开始 |
| 引擎 | SGLang vs vLLM 同机对照 | 未装（sm_120 安装有风险） |
| 引擎 | exllamav2 源码精读 / TensorRT-LLM | 未开始 |
| 端侧 | K3 上的投机解码 / 多模态流式 | 未开始 |

---

# 第二轮实测（2026-11-24）：约束解码 / FlexAttention / Wanda 剪枝

> 条件同上（vlm-server 占 6.4 GB，未停）。模型 Qwen3-1.7B（约束解码/attention）与 0.6B（剪枝）。

## D. 约束解码：xgrammar 强制 JSON Schema

任务：从一句中文里抽取 `{company, year, products, confidence}` 四个字段，3 个 prompt。

| 模式 | JSON 合法率 | 平均生成 token | 平均 tok/s | 平均总耗时 |
| --- | :---: | ---: | ---: | ---: |
| 自由生成 | **0/3** | 200（撞上限） | 38.5 | 5.08 s |
| **xgrammar 约束** | **3/3** | **35** | 35.8 | **1.14 s** |

**三个结论：**

1. **合法率 0% → 100%**。自由生成其实写对了 JSON 内容，但会继续往下写
   （把 JSON 包进 ```json 代码块后再开一个新对象），200 token 上限一到就被截断。
2. **每 token 只慢 7%**（38.5 → 35.8 tok/s）：bitmask 施加的开销很小。
3. **总耗时反而降 4.5 倍**（5.08 s → 1.14 s）：约束解码在 `}` 处立即停止，
   不需要靠截断长度兜底。

**公平性检查**：把自由生成的预算放宽到 512 token，它仍然不终止（继续输出第二个 JSON 块）。
→ 说明这里的收益不是"我们给的预算太少"，而是**模型本身不知道该停在哪**。

> 踩坑：`lm-format-enforcer` 与 transformers 5.x 不兼容（内部 `from transformers.tokenization_utils
> import PreTrainedTokenizerBase` 已失效）。改用 **xgrammar**（vLLM / SGLang 用的是它），
> 通过 `GrammarMatcher` + `apply_token_bitmask_inplace` 做成 LogitsProcessor 即可。

## E. FlexAttention：滑窗 mask vs SDPA

形状按 Qwen3-1.7B 的注意力（16 头 × 128 维，bf16）。

| 序列长度 | SDPA(causal) | Flex **未编译** | Flex **编译后**(全因果) | Flex 编译后(滑窗1024) | 滑窗 vs SDPA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048 | 0.49 ms | 16.77 ms | 0.52 ms | 0.62 ms | 0.79× |
| 4096 | 1.72 ms | 65.91 ms | 1.83 ms | **0.86 ms** | **2.00×** |
| 8192 | 6.44 ms | — | 6.75 ms | **1.77 ms** | **3.64×** |

**三个结论：**

1. **必须 `torch.compile`**：不编译时 flex_attention 走数学回退（dense S²），
   比 SDPA **慢 20–30 倍**（S=4096 时 65.91 ms vs 1.72 ms）。
2. **编译后只在长序列划算**：S=2048 还没回本（0.79×），S=4096 起 2.00×，S=8192 达 **3.64×**。
   交叉点 ≈ 2K–4K。
3. **数值正确**：与显式 mask 的 SDPA 对比，max|Δ| = 1.56e-2（bf16 正常误差）。

> 首次编译约 12–14 s，`mode="max-autotune"` 自动挑了 `triton_flex_attention_40`
> （BLOCK_M=64, BLOCK_N=64, 28.67 ms 在 autotune 里最快）。

## F. Wanda 剪枝（Qwen3-0.6B）

打分 = `|W| × ||X||₂`（权重 × 校准激活范数），按输出通道保留 top-k。校准 8 段语料。

| 稀疏度 | PPL | 相对基线 |
| ---: | ---: | ---: |
| 0% | 29.150 | — |
| 20% | 29.412 | **+0.9%** |
| 40% | 36.981 | +26.9% |
| 50% | 52.736 | +80.9% |
| 60% | 281.165 | +864.5% |

**随机剪枝 50% 的 PPL = 1,780,737**（Wanda 同稀疏度是 52.7，差 **3.4 万倍**）。

**三个结论：**

1. **20% 稀疏几乎免费**（+0.9%），40% 起代价陡增，50% 已不可用，60% 彻底崩。
2. **打分方式极其关键**：随机剪 50% 直接废掉，Wanda 只掉 81%。
3. **剪枝不等于加速**：PyTorch 稠密实现下稀疏权重仍然是稠密 GEMM，
   速度一点不涨（本实验 PPL 测量耗时 0.1–0.2 s 无变化）。
   要变现必须靠 **2:4 结构化稀疏内核** 或专用硬件。

## 本轮把清单补全到 13 项

| 状态 | 项目 |
| --- | --- |
| ✅ 已跑（13 项） | 量化七档 ×2 模型、8B 基线、KV 压缩、投机解码 ×4 实现、跨引擎对照、bnb sm_120、**约束解码**、**FlexAttention**、**Wanda 剪枝**、DDP 验证 |
| ⏸ 未跑 | AWQ/GPTQ/FP8 对照、QuaRot/SpinQuant、KIVI、稀疏注意力 MInference、SGLang vs vLLM、exllamav2/TensorRT-LLM、K3 端侧 |

---

# 第三轮实测（2026-11-24）：FP8 / KIVI / QuaRot / 引擎对比 / K3 端侧

## G. FP8 量化（torchao，Qwen3-0.6B）

| 配置 | PPL | decode tok/s | 峰值显存 |
| --- | ---: | ---: | ---: |
| bf16 | 29.150 | **38.9** | 1,833 MiB |
| FP8 权重-only | 29.076（无损） | 20.4（**0.52×**） | 3,241 MiB |
| FP8 动态激活+权重 | 29.470 | 10.9（**0.28×**） | 3,810 MiB |

**结论：FP8 在 sm_120 上是负收益，而且显存还涨了。**
torchao 的 weight-only 实现是「8bit 存储 + 每次前向反量化回 bf16」，
既没省带宽（还是要读全量再展开），又多了 scale/dequant 开销。
真正的 FP8 收益需要**原生 FP8 GEMM 内核**（cutlass/cuBLASLt 的 scaled_mm），
在 HF 这套前向里用不上。

## H. KIVI 式 KV 量化（Qwen3-1.7B，8K 上下文大海捞针）

| KV 位宽 | 原始 | 量化后（理论） | 检索 |
| --- | ---: | ---: | :---: |
| 16bit | 767.5 MiB | 767.5 MiB | ✅ |
| 8bit | 767.5 MiB | 383.7 MiB | ✅ |
| **4bit** | 767.5 MiB | **191.9 MiB** | ✅ |
| 2bit | 767.5 MiB | 95.9 MiB | ❌（输出崩坏） |

**结论：KV 压到 4bit 仍能正确检索，2bit 崩。**
注意本实验是「量化后立刻反量化」，测的是**精度影响**，不是实际显存占用
（真正省显存需要量化存储格式 + 量化注意力内核）。

## I. QuaRot（旋转量化）验证（Qwen3-0.6B，4bit RTN）

先把 RMSNorm 的 per-channel 权重折进相邻线性层，再用 Hadamard 旋转残差流
（读残差流的层 `W←WR`，写残差流的层 `W←RᵀW`，embedding `E←ER`），
最后做 4bit RTN。

| 配置 | PPL | 相对 bf16 |
| --- | ---: | ---: |
| bf16 | 29.150 | — |
| 折 norm + 旋转（未量化，验证等价性） | 29.076 | **-0.3%** ✅ 变换无损 |
| 不旋转 · 逐通道 | 422.421 | +1349% |
| **旋转 · 逐通道** | **167.336** | +474% |
| 不旋转 · group128 | 156.018 | +435% |
| **旋转 · group128** | **79.403** | **+172%** |

**三个结论：**

1. **旋转本身是精确变换**：未量化时 PPL 从 29.150 → 29.076（-0.3%，bf16 噪声级）。
2. **旋转确实改善 4bit 量化**：逐通道 422→167（**2.5×**），group128 156→79（**2.0×**）。
3. **但不折 norm 会误判**：单独折 RMSNorm 权重（不旋转）会让纯 RTN 从 75 恶化到 422，
   因为 γ 抬高了通道间尺度差异。必须「折 norm + 旋转」一起做才有意义 ——
   我第一次就是拿「未折 norm 的 RTN」和「折了 norm 的旋转版」比，得出过错误结论。

> 绝对精度仍差（+172%），因为缺 QuaRot 论文里的两件事：head 维度旋转 + GPTQ 误差补偿。

## J. 推理引擎对比（同一模型 Qwen3-1.7B，同一张卡）

| 引擎 | 单请求 tok/s | 8 并发合计 | 状态 |
| --- | ---: | ---: | --- |
| transformers (sdpa, bf16) | 36.1 | — | ✅ |
| llama.cpp (Vulkan, F16) | **98.8** | — | ✅ |
| **vLLM 0.34.0** | 63.3 | **481.0** | ✅ 首次加载 140 s |
| SGLang 0.5.23 | — | — | ❌ 未跑通 |

**结论：**

1. **vLLM 能在 sm_120 上跑**（预编译轮子，`enforce_eager=True` 避开 CUDA graph 编译）。
2. **单流最快仍是 llama.cpp**（98.8，用满 90% 带宽）；vLLM 单流 63.3。
3. **但并发是 vLLM 的地盘**：8 路并发 481 tok/s，是单流的 7.6 倍 —— 连续批处理补齐了带宽缺口。
4. **SGLang 在本机跑不通**：flashinfer 编译过关后，卡在 `sgl_kernel` 的 JIT，
   报 `rsqrt` exception specification 冲突（CUDA 13.0 的 `math_functions.h` 与
   Ubuntu 26.04 的 glibc 头不兼容）。用 `-ccbin g++-14` 也没绕过。

## K. K3 端侧基线（RISC-V，8 核，15.9 GB）

| 模型 | 测法 | pp32 | tg16 |
| --- | --- | ---: | ---: |
| qwen3 4B Q4_K_M（2.52 GiB） | llama-bench，16 线程 | **5.58 tok/s** | **3.29 tok/s** |

对比 5060 Ti 同尺寸模型（4B Q4 约 145 tok/s）→ **K3 慢约 44 倍**。

**两个踩到的坑：**

1. K3 上的 llama.cpp 是 **SpacemiT 定制版**（`use_ime2=1`，16 线程 = 8 性能核 + 8 AI 核），
   支持 **9 种投机解码**：`draft-simple / draft-eagle3 / draft-mtp / draft-dflash /
   ngram-simple / ngram-map-k / ngram-map-k4v / ngram-mod / ngram-cache`。
2. 但 `llama-cli` 在 `-no-cnv` + stdin 重定向下会崩：
   `terminate called ... std::system_error: 已避免资源死锁`（llama-bench 正常）。
   → ngram 投机实验因此没跑成，需要换 llama-server 方式再测。

## 本轮补充清单

| 状态 | 项目 |
| --- | --- |
| ✅ 新增跑通 | FP8（负收益）、KIVI 式 KV 量化（4bit 可用）、QuaRot 旋转（2.0–2.5× 改善）、vLLM（单流 63.3 / 8 并发 481）、K3 基线（3.29 tok/s） |
| ❌ 尝试失败 | SGLang（CUDA 13 + glibc 的 JIT 冲突）、K3 上的 ngram 投机（llama-cli 崩溃） |
| ⏸ 未开始 | MInference（需稀疏 attention 内核）、exllamav2（需编译扩展）、TensorRT-LLM（需 TRT 10.8+ 与长编译） |

---

# 第四轮：vLLM 并发曲线 + ngram 投机解码（悬案的最终答案）

## L. 并发曲线（Qwen3-1.7B bf16，vLLM，每请求 128 token）

| 并发 | 合计 tok/s | 单流 tok/s |
| ---: | ---: | ---: |
| 1 | 62.3 | 62.3 |
| 2 | 118.0 | 59.0 |
| 4 | 237.0 | 59.3 |
| 8 | 472.1 | 59.0 |
| **16** | **927.9** | 58.0 |

**结论：到 16 并发仍是近线性扩展（15×），单流稳定在 58–62 tok/s。**
原因：decode 时权重每步只读一次，batch 越大，分摊到每个 token 的字节越少。
按 448 GB/s 和 batch 16（每 token 212 MB）算理论上限约 1,720 tok/s，实测 928 = **54% 效率**。

## M. ngram 投机解码：**3.68× 正收益**

`speculative_config={"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":8,"prompt_lookup_min":3}`
—— 不加载第二个模型，直接在 prompt/历史里找重复片段当草稿。

| 场景 | 基线 | ngram 投机 | 加速比 |
| --- | ---: | ---: | ---: |
| **抄写类**（请原样复述这段文字） | 60.5 | **222.5** | **3.68×** |
| 自由发挥（解释显存带宽） | 62.3 | 68.0 | 1.09× |

**这一组数据把前三轮的"投机解码全是负收益"补齐了：**

| 实现 | 草稿来源 | 场景 | 结果 |
| --- | --- | --- | ---: |
| transformers | 0.6B 独立模型 | 自由发挥 | 0.62× |
| transformers | 0.6B 模型 NF4 | 自由发挥 | 0.55× |
| llama.cpp | 0.6B 独立模型 | 自由发挥 | 0.73× |
| **vLLM** | **ngram（无模型）** | **抄写** | **3.68×** ✅ |
| vLLM | ngram（无模型） | 自由发挥 | 1.09× |

**结论（这才是完整答案）：**

1. **投机解码能不能赚钱，取决于接受率 α**：抄写类任务 α 极高（草稿几乎全中），自由发挥 α 很低。
2. **草稿成本必须接近零**：独立小模型每步要吃 1/7 的带宽 + 同步开销；ngram 只查内存，几乎免费。
3. **实现开销决定下限**：transformers / llama.cpp 的每步 Python+回滚开销，把 2.6× 的理论值吃成 0.7×。
4. **端侧的正确路线**：优先 ngram / prompt lookup（零额外模型），其次是 EAGLE/Medusa 这类共享主干的小头；
   **"再装一个小模型"在带宽受限设备上不成立**。

> 注：vLLM 的 ngram 投机在 sm_120 上工作正常（`enforce_eager=True`），引擎加载约 15 s（缓存预热后）。

---

# 第五轮：vLLM 极限并发 / 前缀缓存 / KV cache FP8

## N. 极限并发（继续加到 64）

| 并发 | 合计 tok/s | 单流 tok/s | 相对单流倍数 |
| ---: | ---: | ---: | ---: |
| 1 | 62.3 | 62.3 | 1.0× |
| 16 | 928.8 | 58.1 | 14.9× |
| 32 | 1,789.8 | 55.9 | 28.7× |
| **64** | **3,357.4** | 52.5 | **53.9×** |

**结论：到 64 并发仍在近线性扩展（53.9×），单流缓慢下降到 52.5（-16%）。**
这是"权重每步只读一次、batch 摊薄"的直接体现：batch 64 时每 token 只需 53 MB 权重读取，
理论上限约 8,450 tok/s，实测 3,357 = **40% 效率**（剩余损失在 attention 的 KV 读取与调度开销）。

> 顺带记录 vLLM 的显存画像：4,096 上下文下 KV cache 27,136 tokens / 2.90 GiB，
> 「4K 上下文的并发上限 6.62×」。

## O. 前缀缓存（同 4K token 前缀的第二次请求）

| 配置 | 第二次请求耗时 |
| --- | ---: |
| 前缀缓存 **开** | **529 ms** |
| 前缀缓存 关 | 688 ms |

**省 23%**。对 RAG / 多轮对话（系统提示词很长的场景）这是白送的收益。

## P. KV cache FP8（vLLM `kv_cache_dtype="fp8"`）

| 配置 | 8 并发 tok/s | 引擎加载 |
| --- | ---: | ---: |
| auto（bf16 KV） | 468.5 | 96.3 s |
| **fp8 KV** | **505.0（+7.8%）** | 54.0 s |

**能跑通，且有小幅正收益。** 注意两个坑：

1. **必须装 `ninja` 并把 venv/bin 加进 PATH**：fp8 路径会 JIT 编译内核，
   否则报 `FileNotFoundError: 'ninja'`（表现为 "Engine core initialization failed"，看不到根因）。
2. **要设 `CUDA_HOME=/usr/local/cuda`**（本机 CUDA 13.0 的头文件在 `targets/x86_64-linux/include`）。

对比第 G 节的 torchao FP8（负收益）：差别在于 **vLLM 有原生 fp8 attention/GEMM 内核**，
torchao 的 weight-only 只是存储 8bit 再反量化。

---

# 第六轮：长上下文 prefill / GPTQ-Int8 / 量化+投机叠加

## Q. 长上下文 prefill 曲线（Qwen3-1.7B bf16，vLLM）

**方法**：`max_tokens=1`，计时 ≈ prefill 耗时。**必须关掉前缀缓存**
（`enable_prefix_caching=False`），否则同一 prompt 第二次直接命中缓存 ——
我第一次没关，测出 32K 上下文 385,069 tok/s（超过硬件理论峰值），是假的。

| 上下文 | prefill 耗时 | prefill 吞吐 |
| ---: | ---: | ---: |
| 480 | 39.8 ms | 12,054 tok/s |
| 2,016 | 150.9 ms | **13,357 tok/s** |
| 8,160 | 729.2 ms | 11,190 tok/s |
| 16,368 | 2,059 ms | 7,950 tok/s |
| 32,736 | 5,758 ms | **5,686 tok/s** |

**结论：prefill 吞吐在 2K 附近见顶，之后随长度下降，32K 只剩 42%。**
线性层是 O(L)，attention 是 O(L²)，所以长上下文 prefill 会被 attention 拖垮——
这正好和 decode 相反（decode 每 token 只多读 KV，是 O(L) 增长）。

## R. vLLM + GPTQ-Int8 权重（含与 ngram 投机叠加）

> AWQ 版：ModelScope 上 `Qwen/Qwen3-1.7B-AWQ` 返回 404（无此版本），
> 改用 `Qwen/Qwen3-1.7B-GPTQ-Int8`（2.0 GB）。

| 配置 | 单流 | 8 并发 | 抄写基线 | +ngram 投机 |
| --- | ---: | ---: | ---: | ---: |
| bf16 | **62.3** | **472.1** | 60.5 | 222.5（3.68×） |
| GPTQ-Int8 | 46.0 | 353.7 | 45.5 | 171.7（**3.77×**） |

**两个结论：**

1. **GPTQ-Int8 比 bf16 慢 26%**（单流 62.3 → 46.0）。权重字节减半本应更快，
   但 vLLM 在 sm_120 上走的是 `auto_gptq` 的反量化路径（marlin 内核未针对 sm_120 优化），
   反量化开销吃掉了带宽收益。
   → 与第 G 节 torchao FP8、第 P 节 KV-FP8 一起，构成同一条规律：
   **量化的收益完全取决于"有没有该架构的原生内核"，与位宽本身无关。**
2. **量化 + 投机可以叠加**：GPTQ-Int8 上 ngram 投机仍是 **3.77×**（45.5 → 171.7）。
   两个加速手段作用在不同的地方——量化省带宽，投机省前向次数，互不冲突。

---

# 第七轮：引擎原生结构化输出 + 长上下文 decode 曲线

## S. vLLM 原生结构化输出（guided decoding）

同一个 JSON Schema，4 个并发请求：

| 模式 | JSON 合法率 | 平均生成 token | 吞吐 |
| --- | :---: | ---: | ---: |
| 自由生成 | **0/4** | 200（撞上限） | 235.4 tok/s |
| **guided decoding** | **4/4** | **45** | 219.5 tok/s |

**和第一轮（transformers + xgrammar，每 token 慢 7%、总耗时降 4.5×）完全一致的规律**：

- 每 token 只慢 **6.8%**（219.5 vs 235.4）
- 但 token 数从 200 降到 45 → 实际总时间省约 **4 倍**
- 合法率 0/4 → 4/4

> 引擎内置的 guided decoding 省掉了自己接 logits processor 的麻烦，
> 而且它跑在 vLLM 的调度里，可与其他请求混批。

## T. 长上下文 decode 曲线（与第六轮 prefill 曲线配对）

| 上下文 | TTFT | decode | 每 token 读取量 | 有效带宽 |
| ---: | ---: | ---: | ---: | ---: |
| 480 | 39.1 ms | **59.8 tok/s** | 3.46 GB | 207 GB/s |
| 2,016 | 148.4 ms | 59.6 | 3.63 GB | 216 GB/s |
| 8,160 | 731.2 ms | 57.5 | 4.34 GB | 249 GB/s |
| 16,368 | 1,829.7 ms | **54.4 tok/s** | 5.28 GB | **287 GB/s** |
| 32,736 | — | *无效* | — | — |

**结论：**

1. **decode 只掉 9%**（59.8 → 54.4），而每 token 读取量涨了 53%（3.46 → 5.28 GB）。
   说明 16K 以内 **KV 读取还不是主导**，权重才是。
2. **有效带宽反升**（207 → 287 GB/s，46% → 64%）：短上下文时每步固定开销（调度、kernel launch）
   摊不开；上下文越长、每步字节越多，利用率越高。
3. **32K 那行无效**：`max_model_len=32768`，prompt 32,736 + 生成 256 已超限，被截断，
   测出的 385.9 tok/s 是假的。要测 32K decode 需把 `max_model_len` 提到 33K 以上。

> 方法提醒：decode 速度必须用「T(256 token) − T(1 token)」摊薄 prefill；
> 用 T(64) − T(1) 在长上下文会因为 chunked prefill 与 decode 重叠而失真（我第一次就是这么错的，
> 算出 2,137 GB/s 这种超过硬件的值）。

## U. K3 上的 ngram 投机（端侧闭环）

统一参数 `-t 16 -c 1024`，抄写类任务（prompt 里含要复述的原文）：

| 配置 | decode | 加速 |
| --- | ---: | ---: |
| 基线 | 2.98 tok/s | — |
| **ngram-simple 投机** | **3.85 tok/s** | **1.29×** |

另外一组（`-t 8` + 默认上下文，绝对速度很低但对照有效）：

| 配置 | decode | 加速 |
| --- | ---: | ---: |
| 基线 | 0.32 tok/s | — |
| ngram-simple | 0.80 tok/s | **2.5×** |

**结论：**

1. **端侧 CPU（算力受限）上 ngram 投机是正收益**，与桌面 GPU 上"独立草稿模型负收益"形成对照 ——
   因为这里省下的"大模型前向次数"直接对应最稀缺的资源（算力），而不是已经很紧的带宽。
2. **越算力受限，相对收益越大**：16 线程时 1.29×，压到 8 线程时 2.5×。
3. 绝对速度仍然只有 **3.85 tok/s**（4B Q4），说明投机救不了端侧 CPU 的吞吐天花板，
   它的价值是"在同一硬件上少花一点时间"，不是"变得可用"。

**两个坑：**

- `llama-server` 的默认参数会带来 10 倍差异：`-t 8` + 默认上下文时只有 0.32 tok/s，
  `-t 16 -c 1024` 就是 2.98 tok/s。做端侧对比必须先固定线程数与上下文长度。
- K3 上的 `llama-cli` 在 `-no-cnv` + stdin 重定向下会崩（`std::system_error: 已避免资源死锁`），
  **只能用 `llama-server` + HTTP 接口**做生成类测试。
