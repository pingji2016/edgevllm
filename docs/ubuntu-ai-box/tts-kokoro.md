# TTS 横向：Kokoro-82M 为什么反而最能用

> 机器 `pingji-MS-7B89` · RTX 5060 Ti 16 GB · 2026-11-25
> 配套：[minicpm-o.md](minicpm-o.md)（全模态的 TTS 不可懂）、[realtime-services.md](realtime-services.md)（CosyVoice 未解决）

---

## 1. 背景：我们试过的三条 TTS 路线，两条失败

| 方案 | 规模 | 结果 |
| --- | ---: | --- |
| CosyVoice2-0.5B | 0.5 B | ❌ 内容完全错误（PPL 级排查过，根因在声学栈） |
| MiniCPM-o 4.5 内置 TTS | 9 B | ❌ 有能量但不可懂（ASR 判成挪威语 0.32） |
| **Kokoro-82M-v1.1-zh** | **82 M** | ✅ **可懂** |

**结论先行：82M 的专用 TTS，比 0.5B 和 9B 两个"大而全"方案的语音输出都更可用。**

## 2. Kokoro 是什么

Kokoro-82M 是基于 **StyleTTS2 + ISTFTNet** 的轻量 TTS：
- 82M 参数，ONNX/PyTorch 双路径，CPU 可跑
- v1.1-zh 版本带 **103 个音色**（`zf_001…` 中文女声、`af_maple` 英文等）
- 音色是**独立的 .pt 张量文件**（60–120 KB），换音色不用换模型

## 3. 依赖链（这是真正的门槛）

装 Kokoro 不是 `pip install kokoro` 就完事，实际踩了 6 层：

| 缺失 | 触发点 | 解法 |
| --- | --- | --- |
| 拉 torch（2.5 GB） | `pip install kokoro` 会解析 torch 依赖 | `--no-deps` 安装 |
| `addict` | kokoro 导入 | `--no-deps addict` |
| `proces` | cn2an（中文数字转写） | `--no-deps proces` |
| `spacy` | kokoro 的 G2P | 正常安装 |
| `phonemizer` + `segments` | spacy 的 G2P 后端 | 正常安装 |
| `espeakng_loader` | phonemizer 的 espeak 绑定 | `espeakng-loader` |
| `pypinyin_dict` | 中文拼音词典 | `pypinyin-dict` |
| `espeak-ng`（系统包） | 音素化 | `apt install espeak-ng` |

**注意**：`pip install kokoro` 默认会尝试装 torch，在某些环境里会拖 15 分钟以上。
正确姿势是**先 `--no-deps`，再逐个补小依赖**。

## 4. 实测

| 项 | 值 |
| --- | ---: |
| 管线加载 | 4.0 s |
| 文本 | 端侧推理部署说明：解码速度五十五点三，每秒钟六十二个 token。 |
| 生成音频 | 6.17 s |
| 生成耗时 | **3.38 s** |
| **RTF** | **0.548（1.8× 实时）** |
| 音色 | `zf_001`（中文女声，103 个可选） |

### ASR 回听

```
目标: 端侧推理部署说明：解码速度五十五点三，每秒钟六十二个 token。
听到: 端策推理部署說明,解碼速度55.3每秒中62個。
```

**看起来 42% 相似度很低，其实是归一化差异**：
- 数字："五十五点三" → "55.3"（ASR 偏好阿拉伯数字）
- 繁体："說明/解碼/個"（Whisper 中文输出的常见行为）
- 字面：侧→策（同音）、掉了英文词 "token"

**语义完全对**，说明语音是可懂的——这与 CosyVoice2（内容全错）和 MiniCPM-o（不可懂）形成鲜明对比。

## 5. 工程注意：cu12 与 cu13 必须分进程

Kokoro（torch cu132）与 faster-whisper（CTranslate2，需 cu12）**不能在同一进程里**：

```
LD_LIBRARY_PATH 指 cu12 → Kokoro 报 CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
不指 cu12           → whisper 报 libcublas.so.12 not found
```

正确做法：**TTS 与 ASR 拆成两个进程**（或两个服务），用文件/HTTP 交互。
这也是生产上"TTS 服务 + ASR 服务分离"的原因。

## 6. 顺带记录：RWKV-7 没跑成

本来想做"SSM vs Transformer 的长上下文 decode 对比"（Transformer 实测 8K→32K 从 71.8 掉到 55.3 tok/s，
SSM 理论上应该持平）。阻塞点：

```
模型: RWKV-x070-World-1.5B-v3（ModelScope，2.9 GB，纯 state_dict）
包:   rwkv 0.8.32
错误: 'types.SimpleNamespace' object has no attribute 'n_head'
```

检查点顶层是 `emb.weight / blocks.0.att.*`（无 `args`/`model` 包装），
而包内 `model.py` 期望 `temp_z['blocks.0.att.r_k'].shape` 推导 n_head，两者版本不匹配。

**结论：RWKV-7 的 .pth 需要与 `rwkv` 包版本严格对应**（或改用 HF 上带 config 的仓库），
本轮未跑通，留作后续。

## 7. 复现

```bash
# 系统依赖
sudo apt install -y espeak-ng

# Python 依赖（关键：先 --no-deps）
PY=~/apps/bench/yolo-venv/bin/python
$PY -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --no-deps kokoro misaki
$PY -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --no-deps \
  addict proces jieba pypinyin cn2an ordered-set num2words inflect
$PY -m pip install -i https://mirrors.aliyun.com/pypi/simple/ spacy phonemizer segments espeakng-loader pypinyin-dict

# 权重
python -c "from modelscope import snapshot_download; snapshot_download('hexgrad/Kokoro-82M-v1.1-zh')"

# 生成（不要带 cu12 的 LD_LIBRARY_PATH）
SKIP_ASR=1 HF_ENDPOINT=https://hf-mirror.com $PY ~/apps/infer-research/kokoro_tts.py

# 验收（单独进程，带 cu12）
LIBS=$(ls -d ~/apps/minimind/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')
LD_LIBRARY_PATH=$LIBS $PY -c "...faster-whisper 转写 kokoro/zh.wav..."
```
