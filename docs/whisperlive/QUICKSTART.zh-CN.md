# WhisperLive 项目说明与运行指南

> 当前仓库版本：`v0.6.2`（见 [whisper_live/__version__.py](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/__version__.py)）
> 上游仓库：<https://github.com/collabora/WhisperLive>（Collabora 开源，MIT 协议）
>
> 👉 **想搞懂它内部怎么处理音频、有哪些特殊处理**，看配套的 [DESIGN.zh-CN.md](DESIGN.zh-CN.md)（含一份"从零跑通"的运行速查）。
>
> 文中 `whisper_live/server.py#L1007` 形式的引用均已改写为上游 `v0.6.2` 的永久链接。
> 本文是本地整理的学习笔记、**非上游内容**，来源与本地对照方式见 [README.md](README.md)。

---

## 1. 这是个什么项目

WhisperLive 是 OpenAI Whisper 的**近实时（nearly-live）语音转文字服务**。它把 Whisper 做成 C/S 架构：

- **服务端**（[run_server.py](https://github.com/collabora/WhisperLive/blob/v0.6.2/run_server.py) / [whisper_live/server.py](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py)）常驻运行，加载 Whisper 模型，通过 **WebSocket** 接收音频流，边收边识别，把带时间戳的文本片段推回客户端。
- **客户端**（[whisper_live/client.py](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py)）负责采集音频（麦克风 / 音频文件 / RTSP / HLS）并按 16kHz 单声道 float32 流式发给服务端，同时把收到的字幕打印到终端、落盘成 `.srt` 字幕文件。

它并不是"说完一整段再识别"，而是**每积累约 1 秒音频就切一段送去识别**，通过"最后一段不算数、反复确认后才定稿"的策略（[server.py:1045](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L1045) `update_segments`）来获得低延迟的滚动字幕效果。

典型用途：

- 会议 / 直播实时字幕
- 长音频文件批量转字幕（带 `.srt` 时间轴）
- 浏览器标签页音频实时转录（附带 Chrome / Firefox 插件）
- 摄像头 RTSP 流、网络 HLS 流的语音转写

### 组件一览

| 组件 | 路径 | 说明 |
| --- | --- | --- |
| 服务端入口 | [run_server.py](https://github.com/collabora/WhisperLive/blob/v0.6.2/run_server.py) | 命令行参数解析 + 启动 WebSocket 服务 |
| 服务端核心 | [whisper_live/server.py](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py) | 连接管理、音频缓冲、识别线程、结果推送 |
| faster-whisper 后端 | [whisper_live/transcriber.py](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/transcriber.py) | 基于 CTranslate2 的 Whisper 实现，**默认后端**，CPU/GPU 都能跑 |
| TensorRT 后端 | [whisper_live/transcriber_tensorrt.py](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/transcriber_tensorrt.py) | 基于 NVIDIA TensorRT-LLM，延迟最低，但需要手动编译 engine |
| VAD | [whisper_live/vad.py](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/vad.py) | Silero VAD（ONNX），用于剔除无人声片段 |
| 客户端 | [whisper_live/client.py](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py) | `TranscriptionClient` 等类，可直接当 Python 库调用 |
| 工具函数 | [whisper_live/utils.py](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/utils.py) | 重采样到 16kHz、生成 SRT 字幕 |
| 浏览器插件 | [Audio-Transcription-Chrome/](https://github.com/collabora/WhisperLive/tree/v0.6.2/Audio-Transcription-Chrome/)、[Audio-Transcription-Firefox/](https://github.com/collabora/WhisperLive/tree/v0.6.2/Audio-Transcription-Firefox/) | 抓取标签页音频并转发给服务端 |
| 镜像构建 | [docker/](https://github.com/collabora/WhisperLive/tree/v0.6.2/docker/) | 预置 CPU / GPU / TensorRT 三种 Dockerfile |

---

## 2. 工作原理与通信协议

```
麦克风/文件/RTSP/HLS ──► 客户端 ──WebSocket──► 服务端 ──► Whisper 模型
                          ▲                    │
                          └─── 字幕片段 JSON ───┘
```

协议非常朴素（想看细节读 [client.py:193](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L193) 和 [server.py:225](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L225)）：

1. **建连后客户端先发一条 JSON 配置**，字段包含 `uid`、`language`、`task`（`transcribe` / `translate`）、`model`（如 `small`）、`use_vad`、`max_clients`、`max_connection_time`。
2. **服务端回 `SERVER_READY`**，附带真正生效的 `backend` 名称（可能是 `faster_whisper` 或 `tensorrt`）。此时客户端开始推流。
3. **音频用二进制帧发送**：16kHz、单声道、float32 的裸 PCM 数组（客户端把 int16 除以 32768 归一化后发送，见 [client.py:646](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L646)）。
4. **服务端持续回推 JSON 字幕**，格式为 `{"uid": ..., "segments": [{"start": "0.000", "end": "2.400", "text": "...", "completed": true}, ...]}`；`completed=false` 表示这段是还没定稿的"临时结果"。服务端只回最近 10 段。
5. **文件/流处理结束时**客户端发送一个文本哨兵 `END_OF_AUDIO`，服务端收到后收尾并把剩余结果吐完。客户端再等约 15 秒没有新响应就关闭连接（[client.py:274](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L274)）。

其他要点：

- 音频缓冲上限 45 秒，超出就丢掉最旧的 30 秒（[server.py:462](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L462)），所以长时间运行内存不会无限增长。
- 默认最多 **4 个并发客户端**、**单连接最长 600 秒**，满了会回 `{"status": "WAIT"}` 并关闭连接。
- `faster_whisper` 后端的 VAD 直接用 faster-whisper 内置的 `vad_filter`；`tensorrt` 后端则用 Silero VAD + 3 帧无语音判定句尾。

---

## 3. 环境要求

- **Python 3.8 – 3.12**（CI 覆盖到 3.12）
- 系统音频库：
  - Linux 需要 `portaudio19-dev`（PyAudio 的依赖）：`bash scripts/setup.sh`
  - Windows / macOS 上 `pip install pyaudio` 直接装 wheel 即可，无需额外系统包
- 磁盘：模型首次使用时会从 Hugging Face 下载，`small` 约 0.5 GB，`large-v3` 约 3 GB
- GPU（可选）：能用 CUDA 的 NVIDIA 显卡会显著提速；纯 CPU 也能跑，`small` 模型在普通 CPU 上可用，但延迟明显更高

---

## 4. 安装

### 4.1 推荐方式：从源码装依赖

```bash
git clone https://github.com/collabora/WhisperLive.git
cd WhisperLive

# 服务端依赖
pip install -r requirements/server.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 客户端依赖（要有麦克风才需要 PyAudio，只跑服务端可跳过）
pip install -r requirements/client.txt
```

### 4.2 或者用 pip 安装（作为库使用）

```bash
pip install whisper-live
```

> 注意：pip 版本只保证把包装进 site-packages。如果你想改代码后立即生效，请在仓库根目录执行 `pip install -e .`（项目带 [setup.py](https://github.com/collabora/WhisperLive/blob/v0.6.2/setup.py)）。

### 4.3 启用 GPU

上面的命令装的是 **CPU 版 torch**。要用显卡，先按显卡/驱动换装 CUDA 版 torch，例如：

```bash
# CUDA 12.4 版本
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 40/50 系新卡（含 RTX 5060 Ti 这类 Blackwell 架构）需要 CUDA 12.8 + torch 2.7 以上
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

装完验证一下：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

服务端会自动选设备：能用 CUDA 就 `float16`，否则 CPU `int8`（[server.py:805](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L805)）。

### 4.4 TensorRT 后端（可选，进阶）

需要按 [TensorRT_whisper.md](https://github.com/collabora/WhisperLive/blob/v0.6.2/TensorRT_whisper.md) 先搭好 NVIDIA/TensorRT-LLM 并**预先编译好 whisper engine**，否则服务端起不来。官方建议直接用 TensorRT 的 Docker 镜像。

---

## 5. 启动服务端

```bash
# 最简：默认 faster_whisper 后端，监听 0.0.0.0:9090
python run_server.py --port 9090 --backend faster_whisper
```

首次连接时会下载 `small` 模型（约 0.5 GB），等待时间较长。国内网络可加镜像：

```bash
# Git Bash / Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com
# Windows CMD
set HF_ENDPOINT=https://hf-mirror.com
```

### 5.1 常用参数

| 参数 | 短写 | 默认 | 说明 |
| --- | --- | --- | --- |
| `--port` | `-p` | `9090` | WebSocket 监听端口 |
| `--backend` | `-b` | `faster_whisper` | 可选 `faster_whisper` / `tensorrt` |
| `--faster_whisper_custom_model_path` | `-fw` | 无 | 本地自训练的 faster-whisper 模型目录 |
| `--trt_model_path` | `-trt` | 无 | TensorRT engine 路径，`tensorrt` 后端必填 |
| `--trt_multilingual` | `-m` | `false` | TensorRT 模型是否多语言 |
| `--omp_num_threads` | `-omp` | `1` | OpenMP 线程数，CPU 跑多路时可调大 |
| `--no_single_model` | `-nsm` | `false` | 每个连接都单独实例化模型（见下） |

### 5.2 "单模型模式"是什么

- **不带 `-fw` / `-trt` 时**：每个客户端连接都会**按客户端请求的模型大小**新建一个 Whisper 模型。好处是客户端可以自由选 `tiny`/`small`/`large-v3`；坏处是连接时等待久、显存/内存占用成倍增长。
- **带了 `-fw` 或 `-trt` 时**：服务端只实例化一次自定义模型并复用给所有连接。加 `--no_single_model` 可以关掉这个行为。

---

## 6. 运行客户端

`TranscriptionClient` 既是命令行脚本的底座，也能直接当库用。

### 6.1 转写麦克风

```python
from whisper_live.client import TranscriptionClient

client = TranscriptionClient(
    "localhost", 9090,
    lang="zh",                 # 多语言模型可省略，服务端会自动识别
    translate=False,           # True 则翻译成英文
    model="small",
    use_vad=True,
    output_transcription_path="./output.srt",
    max_clients=4,
    max_connection_time=600,
)
client()          # 按 Ctrl+C 结束
```

### 6.2 转写音频文件

```python
client("tests/jfk.wav")
```

文件会被播放出来同时流式送去做识别。以下几点容易踩坑：

- 传进来的文件支持 flac/mp3 等多种格式 —— 客户端会先用 `av`（PyAV）转成 **16kHz 单声道 wav**，产物写在**当前工作目录**，文件名是 `<原名>_resampled.wav`（[utils.py:67](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/utils.py#L67)）。仓库里现成的样例是 `assets/jfk.flac`，所以跑起来是：

  ```python
  client("assets/jfk.flac")   # 会生成 ./jfk_resampled.wav
  ```

- 不想听到播放声音，加 `mute_audio_playback=True`（这样也会按音频时长 sleep，保持实时节奏）。
- 输出字幕默认写到 `./output.srt`，路径必须以 `.srt` 结尾，否则构造客户端时就会抛 `ValueError`。

### 6.3 转写 RTSP / HLS 流

```python
client(rtsp_url="rtsp://admin:admin@192.168.0.1/rtsp")
client(hls_url="http://example.com/live/stream.m3u8")
```

HLS 还可以 `save_file="dump.ts"` 把拉到的流顺手存下来。

### 6.4 主要构造参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `host`, `port` | 必填 | 服务端地址 |
| `lang` | `None` | 指定源语言；留空则由服务端检测 |
| `translate` | `False` | `True` 时任务变成"翻译成英文" |
| `model` | `small` | 模型尺寸，也支持 HF 仓库名如 `Systran/faster-whisper-small` |
| `use_vad` | `True` | 是否启用语音活动检测（跳过静音） |
| `save_output_recording` | `False` | 把麦克风输入另存为 wav |
| `output_recording_filename` | `./output_recording.wav` | 上面的保存路径，必须以 `.wav` 结尾 |
| `output_transcription_path` | `./output.srt` | 字幕输出路径，必须以 `.srt` 结尾 |
| `max_clients` | `4` | 传给服务端的并发上限 |
| `max_connection_time` | `600` | 单连接最长秒数 |
| `mute_audio_playback` | `False` | 转写文件时静音播放 |
| `log_transcription` | `True` | 在终端实时刷新打印字幕 |

网络不稳定或长时间跑，建议加个守护重连循环（上游 issue 里常见的做法）：

```python
from websocket import WebSocketConnectionClosedException

while True:
    try:
        client()          # 长连接断开会抛异常
    except (KeyboardInterrupt, SystemExit):
        break
    except (WebSocketConnectionClosedException, ConnectionRefusedError) as e:
        print(f"Connection error: {e}, retrying...")
```

---

## 7. 浏览器插件

以无声方式抓取**当前标签页**的音频并送去转录（服务端地址默认 `localhost:9090`，可在插件设置里改）。

Chrome：

1. 打开 `chrome://extensions`
2. 右上角开启「开发者模式」
3. 点「加载已解压的扩展程序」，选择仓库里的 [Audio-Transcription-Chrome/](https://github.com/collabora/WhisperLive/tree/v0.6.2/Audio-Transcription-Chrome/) 目录
4. 先启动服务端，再点插件图标 → **Start Capture**

Firefox 类似，加载 [Audio-Transcription-Firefox/](https://github.com/collabora/WhisperLive/tree/v0.6.2/Audio-Transcription-Firefox/) 目录即可。设置页里可以填自己的 host / port（默认 `localhost` + `9090`，见 [popup.js:73](https://github.com/collabora/WhisperLive/blob/v0.6.2/Audio-Transcription-Chrome/popup.js#L73)）。

---

## 8. 用 Docker 跑服务端

```bash
# CPU
docker run -it -p 9090:9090 ghcr.io/collabora/whisperlive-cpu:latest

# GPU + faster-whisper（需要 nvidia-container-toolkit）
docker run -it --gpus all -p 9090:9090 ghcr.io/collabora/whisperlive-gpu:latest

# TensorRT（进镜像后自己编 engine）
docker run -p 9090:9090 --runtime=nvidia --gpus all --entrypoint /bin/bash -it ghcr.io/collabora/whisperlive-tensorrt
```

也可以自己构建：`docker build -f docker/Dockerfile.cpu -t whisperlive-cpu .`

> 镜像里默认用 `small` 模型。要换成别的尺寸，改 [whisper_live/server.py](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py) 里的默认值再重新构建。

---

## 9. 跑测试

```bash
# 测试依赖（jiwer / evaluate / openai-whisper 等）在 server.txt 里已经包含
pip install -r requirements/server.txt
pip install -r requirements/client.txt

python -m unittest discover -s tests
```

代码风格检查（与 CI 一致）：

```bash
pip install flake8
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

---

## 10. 常见问题（FAQ）

**Q：`ImportError: cannot import name 'pad_or_trim' from 'faster_whisper.audio'`**
本仓库的 [transcriber.py:21](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/transcriber.py#L21) 是按 **faster-whisper 1.1.0** 的 API 写的，装成旧版（如 0.10.0）就会报这个错。请严格按 pin 的版本装：

```bash
pip install "faster-whisper==1.1.0" "tokenizers==0.20.3" soundfile
```

**Q：服务端连上就断 / 客户端一直 `Waiting for server ready`**
先看服务端日志有没有 `Running faster_whisper backend`。没有的话通常是模型下载失败（网络）或显存/内存不足。国内可设 `HF_ENDPOINT=https://hf-mirror.com`。

**Q：客户端报 `[WARN]: Unable to access microphone.`**
PyAudio 打不开录音设备。用文件转写代替实时录音做冒烟测试，或检查系统麦克风权限与输入设备。

**Q：`python run_server.py` 报找不到 `whisper_live`**
请**在仓库根目录**执行，或先 `pip install -e .`。若机器上另有旧版 `whisper-live` 包，注意它可能覆盖本仓库代码（用 `python -c "import whisper_live; print(whisper_live.__file__)"` 确认实际加载路径）。

**Q：识别延迟高 / 实时性差**
按顺序尝试：换小模型（`tiny.en`/`base`）→ 开启 `--omp_num_threads` 提升 CPU 并行 → 确认用的是 CUDA 版 torch → 换 TensorRT 后端。

**Q：中文识别不准**
用多语言模型（不要选 `.en` 结尾的），并在客户端显式指定 `lang="zh"`；必要时通过 `initial_prompt` 或客户端侧后处理改善标点。

---

## 11. 目录结构速查

```
WhisperLive/
├── run_server.py                  # 服务端入口
├── setup.py / requirements/       # 打包与依赖（区分 server / client）
├── whisper_live/                  # 核心库
│   ├── server.py                  # WebSocket 服务 + 各后端 ServeClient
│   ├── client.py                  # TranscriptionClient / Client
│   ├── transcriber.py             # faster-whisper 封装（默认后端）
│   ├── transcriber_tensorrt.py    # TensorRT-LLM 封装
│   ├── tensorrt_utils.py, vad.py  # TensorRT 辅助 / Silero VAD
│   └── utils.py                   # 重采样、SRT 生成、终端打印
├── Audio-Transcription-Chrome/    # 浏览器插件
├── Audio-Transcription-Firefox/
├── assets/jfk.flac                # 示例音频
├── tests/                         # unittest（mock 掉 websocket / 音频设备）
├── docker/                        # CPU / GPU / TensorRT 镜像
└── docs/                          # Sphinx 文档 + 本文件
```

---

## 12. 相关链接

- 上游仓库：<https://github.com/collabora/WhisperLive>
- faster-whisper：<https://github.com/SYSTRAN/faster-whisper>
- OpenAI Whisper 论文：<https://arxiv.org/abs/2212.04356>
- Silero VAD：<https://github.com/snakers4/silero-vad>
- TensorRT-LLM：<https://github.com/NVIDIA/TensorRT-LLM>
