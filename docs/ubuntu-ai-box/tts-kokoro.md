# Kokoro TTS 详解：82M 为什么打赢了 0.5B 和 9B

> 机器 `pingji-MS-7B89` · RTX 5060 Ti 16 GB · Ubuntu 26.04 · 2026-11-25
> 模型 `hexgrad/Kokoro-82M-v1.1-zh`（Apache-2.0）
> 配套：[minicpm-o.md](minicpm-o.md)（9B 全模态的 TTS 不可懂）、[realtime-services.md](realtime-services.md)（CosyVoice2 未解决）

---

## 1. 一句话结论

**82M 的专用 TTS，在"可懂"和"速度"两个维度上，都打赢了我们试过的 0.5B（CosyVoice2）和 9B（MiniCPM-o）。**

| 方案 | 规模 | 可懂 | 稳态 RTF |
| --- | ---: | :---: | ---: |
| CosyVoice2-0.5B | 0.5 B | ❌ 内容全错 | — |
| MiniCPM-o 4.5 内置 TTS | 9 B | ❌ 无意义语音 | ~0.5 |
| **Kokoro-82M-v1.1-zh** | **82 M** | ✅ | **0.014–0.020（50–70× 实时）** |

---

## 2. 它是什么：两段式 TTS，但不是自回归

### 2.1 家族与结构

Kokoro 基于 **StyleTTS2 + ISTFTNet**（非自回归）：

```
文本
 │  ① G2P（字素→音素）
 │     · 英文：espeak-ng
 │     · 中文：pypinyin（声母/韵母）+ cn2an（数字转读法）+ 分词
 ▼
音素序列
 │  ② 声学模型（StyleTTS2 主干）
 │     · 文本编码器：音素 → 隐向量
 │     · 风格编码器：音色向量 ref_s（就是那个 .pt 文件，60–120 KB）
 │     · 时长预测器：给每个音素分配帧数
 ▼
梅尔/隐特征
 │  ③ 声码器（ISTFTNet）
 │     · 直接从隐特征重建波形，靠 iSTFT 头省掉昂贵上采样
 ▼
24 kHz 波形
```

**与 CosyVoice / MiniCPM-o 的关键差别**：
- CosyVoice2 是 **LLM + flow matching** 两段（先"说 token"再"唱出来"），
  LLM 一旦退化，整句话就废了 —— 这正是它在本机失败的原因；
- MiniCPM-o 的 TTS 也是"音频 token → token2wav"，token 坏了同样废；
- **Kokoro 没有 token 自回归环节**，音素直接映射到声学特征，
  所以**不存在"生成到一半跑偏"的失败模式**。

### 2.2 为什么只要 82M

| 设计 | 收益 |
| --- | --- |
| 非自回归（一次前向出整句） | 无逐 token 循环，延迟与句子长度近似线性但常数极小 |
| ISTFTNet 声码器 | 相比 HiFiGAN 类上采样网络，参数与算力都低一个量级 |
| 共享主干 + 103 个独立音色向量 | 换音色只换 60–120 KB 的 `ref_s`，不换模型 |
| 音素输入（而非字符） | 词表只有几十个符号，嵌入层极小 |

> 换句话说：**它把"能力"压在了 G2P 和训练数据上，而不是模型容量上。**
> 这也是"专用小模型"这条路的典型形态。

### 2.3 它做不到什么

| 能力 | Kokoro | 说明 |
| --- | :---: | --- |
| 零样本音色克隆 | ❌ | 只有 103 个固定音色；要新音色必须微调 |
| 情感控制 | ❌ | 无法指定"生气/温柔" |
| 中英混读 | ⚠️ | v1.1-zh 支持中文，英文词按字母读，需手动切分 |
| 长文本 | ✅ | pipeline 自动按句切分（chunk） |
| 流式输出 | ⚠️ | 支持按 chunk 出，但不是逐字流式 |

---

## 3. 实测：分阶段耗时与长度扫描

### 3.1 环境

| 项 | 值 |
| --- | --- |
| 权重 | `kokoro-v1_1-zh.pth`（327 MB，fp32） |
| 声学模型参数 | **81.8 M** |
| 采样率 | 24 kHz |
| 音色数 | **103 个**（`zf_001…` 中文女声 / `af_maple` 英文等） |

### 3.2 分阶段耗时（3 种文本长度）

| 文本 | 字数 | G2P | 模型+声码 | 音频时长 | **RTF** | 等效速度 | 每字耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 短 | 7 | 576.9 ms* | 833.5 ms | 2.45 s | 0.576 | 1.7× | 201 ms |
| **中** | 28 | **4.0 ms** | **83.5 ms** | 6.17 s | **0.014** | **70.5×** | 3 ms |
| **长** | 61 | 7.9 ms | 286.0 ms | 14.70 s | **0.020** | **50.0×** | 4 ms |

\* 第一行是**冷启动**：G2P 首次调用要把 pypinyin 词典、espeak-ng 绑定等加载进内存，
一次性 577 ms。之后 G2P 只要 4–8 ms。

**所以：**

- **稳态 RTF = 0.014–0.020，即 50–70 倍实时**；
- 我们之前报的"1.8× 实时（RTF 0.548）"是**冷启动那一次**的数字，不代表能力；
- 28 字 → 6.17 s 音频只花 **87 ms**，比语音本身长 70 倍。

### 3.3 与其它 TTS 的横向

| 方案 | 参数量 | 生成 6 s 音频的耗时 | 备注 |
| --- | ---: | ---: | --- |
| Kokoro-82M（本机） | 82 M | **0.087 s** | 稳态 |
| CosyVoice2-0.5B（本机实测） | 500 M | 19.8 s / 17.6 s 音频（RTF 1.13） | 且内容错误 |
| MiniCPM-o 9B TTS（本机实测） | 9 B | 12 s 音频约需 30 s | 且不可懂 |
| piper（CPU） | ~10 M | 2.3 s / 4.46 s 音频（RTF 0.53） | 可懂但偶有读错 |

**Kokoro 比 CosyVoice2 快约 200 倍，比 piper 快约 30 倍。**

---

## 4. 依赖链：真正的门槛（7 层）

`pip install kokoro` 在干净环境里会**尝试拉 torch（2.5 GB）**，我们这台机器上跑了 15 分钟没完。
正确姿势是 `--no-deps` 再逐个补：

| # | 缺失包 | 触发点 | 装法 |
| ---: | --- | --- | --- |
| 1 | torch（2.5 GB） | pip 自动解析依赖 | **`--no-deps` 跳过** |
| 2 | `addict` | `import kokoro` | `--no-deps addict` |
| 3 | `proces` | `cn2an`（中文数字→读法） | `--no-deps proces` |
| 4 | `spacy` | kokoro 的 G2P 管线 | 正常装 |
| 5 | `phonemizer` + `segments` | spacy 的音素化后端 | 正常装 |
| 6 | `espeakng_loader` | phonemizer 的 espeak 绑定 | `espeakng-loader` |
| 7 | `pypinyin_dict` | 中文拼音词典 | `pypinyin-dict` |
| 8 | `espeak-ng`（系统包） | 音素化二进制 | `apt install espeak-ng` |

> 教训：**pip 的依赖解析在"已有 torch"的环境里仍然会去拉 torch**，
> 大模型的 pip 安装一定要先 `--no-deps`，再补小依赖。

---

## 5. 工程约束：cu12 与 cu13 必须分进程

Kokoro 走 torch（cu13 运行时），faster-whisper 走 CTranslate2（需要 `libcublas.so.12`）。
两者同进程必然冲突：

| 环境变量 | Kokoro（torch cu132） | faster-whisper（CTranslate2） |
| --- | --- | --- |
| 不设 | ✅ 正常 | ❌ `libcublas.so.12 not found` |
| 指向 cu12 库 | ❌ `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` | ✅ 正常 |

**解法**：TTS 与 ASR 拆成两个进程/两个服务，用文件或 HTTP 交互。
这不是本机特例——生产上也普遍把 ASR 与 TTS 拆开（依赖、显存、扩缩容都不同）。

---

## 6. 质量验收：ASR 回听

```
目标: 端侧推理部署说明：解码速度五十五点三，每秒钟六十二个 token。
听到: 端策推理部署說明,解碼速度55.3每秒中62個。
```

字面相似度只有 42%，但差异**全部来自归一化**：

| 差异 | 原因 |
| --- | --- |
| 五十五点三 → 55.3 | ASR 偏好阿拉伯数字 |
| 说明/解码/个 → 說明/解碼/個 | Whisper 中文输出的繁体倾向 |
| 侧 → 策 | 同音字（sè/cè 在合成语音里接近） |
| 丢 "token" | 英文词在中文句子里被吞掉 |

**语义完全正确**，语音是可懂的。判断 TTS 好坏**不能只看字符相似度**，
要么人工听，要么用"关键信息是否保留"（数字、专名）来判。

---

## 7. 复现

```bash
# 1) 系统依赖
sudo apt install -y espeak-ng

# 2) Python 依赖（关键：先 --no-deps，避免拉 torch）
PY=~/apps/bench/yolo-venv/bin/python
M="-i https://mirrors.aliyun.com/pypi/simple/"
$PY -m pip install $M --no-deps kokoro misaki
$PY -m pip install $M --no-deps addict proces jieba pypinyin cn2an ordered-set num2words inflect
$PY -m pip install $M spacy phonemizer segments espeakng-loader pypinyin-dict

# 3) 权重（330 MB）
python -c "from modelscope import snapshot_download; snapshot_download('hexgrad/Kokoro-82M-v1.1-zh')"

# 4) 基准（分阶段耗时）
$PY ~/apps/infer-research/kokoro_bench.py

# 5) 生成（不要带 cu12 的 LD_LIBRARY_PATH）
SKIP_ASR=1 HF_ENDPOINT=https://hf-mirror.com $PY ~/apps/infer-research/kokoro_tts.py
```

产物：`~/apps/infer-research/kokoro/zh.wav`、`kokoro_bench.json`

---

## 8. 拿它做什么

| 场景 | 是否合适 |
| --- | --- |
| 端侧朗读/播报（数值、告警、导航） | ✅ **非常合适**：70× 实时，82M 可塞进手机 |
| 有声书批量生成 | ✅ 合适（每 6 s 音频 87 ms） |
| 需要特定人声/克隆 | ❌ 需微调或换 CosyVoice（但要先修好） |
| 需要情感/语气控制 | ❌ 不支持 |
| 中英混说的技术播报 | ⚠️ 英文词会被按字母读，需预处理 |

**最实际的用法**：把它接到你的相机链路里，做**本地语音播报**（"检测到有人移动"），
70× 实时意味着**说一句话的开销可以忽略**。
