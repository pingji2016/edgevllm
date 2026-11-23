# 推理加速三轮实测：量化梯度（A）/ KV 压缩（B）/ 投机解码（C）

> 机器 `pingji-MS-7B89` · RTX 5060 Ti 16 GB（sm_120 / 36 SM / **448 GB/s**）· Ubuntu 26.04
> 实测时间：2026-11-23 · 全部数字为本机实跑
> 脚本：[`scripts/routeA_quant.py`](scripts/routeA_quant.py) · [`scripts/routeB_kv.py`](scripts/routeB_kv.py) · [`scripts/routeC_spec.py`](scripts/routeC_spec.py) · [`scripts/routeC_spec_nf4.py`](scripts/routeC_spec_nf4.py)
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
