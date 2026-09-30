# K3 上的 ASR 实测：Whisper vs sherpa-onnx 流式 Zipformer

> K3 · riscv64 · Bianbu 4.0.1 · 2026-09-30 · 测试音频：中英混说（sherpa 官方 test_wavs）

## 摘要

| 模型 | 10s 音频耗时 | RTF | 特点 |
| --- | --- | --- | --- |
| whisper.cpp **base** q5_1 | 18.6 s | **1.85** | 慢（0.5× 实时） |
| whisper.cpp **small** q5_1 | 59.9 s | **6.0** | 更慢（0.17× 实时） |
| **sherpa-onnx 流式双语 zipformer**（int8） | **5.0 s** | **0.49** | **≈2× 实时，流式 + 词级时间戳** |

**结论：K3 上要"好用/实时"就用 sherpa-onnx 流式 zipformer（你提到的那个模型）；whisper 只适合离线。**

---

## 1. 名词澄清（RTF / 延迟 / ONNX / CPU / NPU / VPU）

- **RTF（Real-Time Factor）** = 处理耗时 ÷ 音频时长。**<1 表示比实时快**。
  - 例：10 秒音频处理用 5 秒 → RTF = 0.49。
- **"5.0s、RTF 0.49" ≠ "要等 5 秒才出结果"**：
  - 这是**把整段 10s 音频处理完**的总耗时（吞吐），不是首字延迟。
  - 它是**流式**的：接麦克风时**边说边出中间结果**，最终结果在句尾（一次 endpoint）即出，**延迟 ≈ 一个 chunk（~0.3s 级）**，不是 5 秒。
- **ONNX**：模型格式 + 运行时（onnxruntime）。onnxruntime 能跑 **CPU**，也能通过 **Execution Provider(EP)** 跑加速器（CUDA/OpenVINO/QNN/RKNN…）。
  - 这里 `--provider=cpu` → **跑在 CPU（RVV）**。
  - 这个 spacemit 预编译版的 provider 只有 `cpu/cuda/coreml/rknn/qnn`，**没有 K3 的 NPU** → **ASR 未使用 NPU**。
- **VPU**：视频编解码器（H.264/H.265），只做视频编码/解码。**之前那个 Mage-ViT 没有用 VPU**——它是 Transformer，跑在 **CPU（RVV/IME）**；Mage-VL 整条链路里 VPU 一次都没用到。

> 加速器分工小结：
> **VPU = 视频编解码**（不是 AI）、**NPU = AI 加速**（走 SpacemiT 的 ONNX EP）、**CPU(RVV/IME) = 实际跑 Whisper / Zipformer / Mage-ViT / LLM 的地方**。

---

## 2. 结果与转写样例

同一段中英混说音频：

- whisper base：`昨天是 Monday / today is tomorrow / today after tomorrow 是星期三`
- sherpa 流式 zipformer：`昨天是 MONDAY TODAY IS LIBR ... 是星期三`（带 **token 时间戳**）

whisper 计时明细（10s 音频）：
```
base : total 18.6s（encode 15.5s / 2 runs，decode 0.17s）
small: total 59.9s（encode 53.3s / 2 runs，decode 0.12s）   <- 编码器是瓶颈
```
sherpa 计时：
```
0.wav: 5.0s / 10s  -> RTF 0.49
1.wav: 3.4s / 5.1s -> RTF 0.67
```

---

## 3. 为什么差这么多？（结构差异）

（配图 `whisper-vs-zipformer.png`）

- **Whisper（编码-解码 Transformer，非流式）**
  - 每次把音频**垫成固定 30 秒窗口**（1500 帧），跑**全长自注意力**编码器（O(n²)），再**自回归**解码。
  - 就算只有 10 秒，也要算满 30 秒窗口；**每段音频都要重算整窗** → 重、且不能流式。
- **Zipformer 流式 RNN-T（流式）**
  - 音频**切成小块（~0.32s/chunk）**，Zipformer 编码器用**有限的左上下文**注意力（省算力），配 **RNN-T(transducer) joiner** 逐块输出。
  - **每块只算一点、增量输出** → 轻、可流式、低延迟。
- 一句话：**whisper 是"整段重算的大模型"，zipformer 是"分块增量的流式模型"** → RTF 1.85 vs 0.49。

---

## 4. 复现命令

```bash
# ---- Whisper（apt 自带，含 RVV 后端）----
# 模型/音频在 ~/mage-vl/asr/
whisper-cli -m ~/mage-vl/asr/ggml-base-q5_1.bin  -f ~/mage-vl/asr/0.wav -t 8 -l auto
whisper-cli -m ~/mage-vl/asr/ggml-small-q5_1.bin -f ~/mage-vl/asr/0.wav -t 8 -l auto

# ---- sherpa-onnx 流式双语 zipformer（~/mage-vl/sherpa/）----
cd ~/mage-vl/sherpa
BIN=./sherpa-onnx-v1.13.8-linux-riscv64-spacemit-shared/bin/sherpa-onnx
$BIN --tokens=tokens.txt \
  --encoder=encoder-epoch-99-avg-1.int8.onnx \
  --decoder=decoder-epoch-99-avg-1.int8.onnx \
  --joiner=joiner-epoch-99-avg-1.int8.onnx \
  --provider=cpu --num-threads=4 --decoding-method=greedy_search \
  ~/mage-vl/asr/0.wav ~/mage-vl/asr/1.wav
```

---

## 5. 建议

- **要实时/流式** → 用 sherpa-onnx 流式 zipformer（可 `--provider=cpu`；还支持 `sherpa-onnx-online-websocket-server` 做服务、`sherpa-onnx-...alsa...` 接麦克风）。
- **要离线高准确率** → 可用 whisper（接受慢），或 sherpa-onnx 的 **SenseVoice / Paraformer**（本次 SenseVoice 官方包 tokens 格式不符，未跑通）。
- **关于加速**：本 build 未暴露 K3 的 NPU；ASR 现在跑在 **CPU(RVV)** 上已 2× 实时，够用。VPU 与 ASR 无关（VPU 只管视频编解码）。

### 产物（板子上）
```
~/mage-vl/asr/      # ggml-base/small-q5_1.bin + 0.wav/1.wav
~/mage-vl/sherpa/   # sherpa-onnx(spacemit 版) + 双语 zipformer(int8) + tokens/bpe
```

---

## 6. zoo 官方 ASR（SenseVoice）实测 & 对比

官方 zoo 的 ASR 是 **SenseVoice**（`archive.spacemit.com/spacemit-ai/model_zoo/asr/sensevoice.tar.gz`，416MB，**INT8**）。我们用板子上的 **sherpa-onnx** 直接加载（`--sense-voice-model`）跑通。

> 坑：zoo 的 `tokens.txt` 是 FunASR 的 `符号<TAB>分数` 格式，sherpa 要 `符号<空格>id`；转一下即可：
> `awk -F'\t' '{print $1, NR-1}' tokens.txt > tokens_sherpa.txt`

运行：
```bash
cd ~/mage-vl/sherpa
BIN=./sherpa-onnx-v1.13.8-linux-riscv64-spacemit-shared/bin/sherpa-onnx-offline
$BIN --sense-voice-model=sv/sensevoice/model_quant_optimized.onnx \
     --tokens=sv/tokens_sherpa.txt --num-threads=4 ~/mage-vl/asr/0.wav ~/mage-vl/asr/1.wav
```

结果：
```
0.wav(10s): 昨天是 monday today is礼拜二 the day after tomorrow 是星期三
            lang=<|zh|> emotion=<|NEUTRAL|> event=<|Speech|>
1.wav(5.1s):这是第一种第二种叫呃与 always s always s 什么意思啊
Elapsed 2.191s / 音频 15.153s → RTF = 0.145
```

### 对比（同一组音频）

| ASR | 模型/大小 | 语言 | RTF | 流式 | 备注 |
| --- | --- | --- | --- | --- | --- |
| **我们的 sherpa 流式 zipformer** | bilingual-zh-en int8 ~193MB | **仅中英** | 0.49 / 0.67 | ✅ 流式 | 低延迟 |
| **zoo SenseVoice** | INT8 ~238MB | **多语**（zh/en/yue/ja/ko…） | **0.145** | ❌ 离线 | 更准 + 语种/情感/事件标签 |
| （参考）whisper base / small | ggml | 99 语 | 1.85 / 6.0 | ❌ | 慢 |

**结论**：
- 我们那个 ASR（`...-bilingual-zh-en-...`）**只支持中文 + 英文**。
- **SenseVoice 更快（≈7× 实时，比 zipformer 快约 3–4 倍）、更准、还多语 + 情感/事件**；代价是**离线（非流式）**。要流式低延迟就 zipformer，要准确/多语/离线就 SenseVoice。

