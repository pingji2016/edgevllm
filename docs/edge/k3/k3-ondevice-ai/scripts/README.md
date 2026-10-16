# K3 端侧 AI 工具集

配合 [`../k3-ondevice-ai-report.md`](../k3-ondevice-ai-report.md) 使用。
所有脚本都在板子上实跑验证过（SpaceMiT K3 Pico ITX / Bianbu 4.0.1 / 内核 6.18.3-cf-k3-1.3.0）。

| 脚本 | 作用 | 依赖 |
|---|---|---|
| `voice-assistant.sh` | **离线语音助手**：语音→ASR→LLM→TTS→喇叭 | SpaceMiT AI Gateway (`:18790`)、ffmpeg、aplay |
| `vlm-announce.sh` | **看图说话→播报**：图片→Mage-VL→TTS→喇叭 | `~/mage-vl`（Mage-VL gguf + llama-mtmd-cli）、网关 TTS |
| `hw-encode.sh` | **VPU 硬编**：一键硬编 / 修权限 / 自检 | ffmpeg（`*_stcodec`）、video 组 |
| `a100-unlock.sh` + `k3aix.c` | **启用 A100 核组**（HMP） | 内核 `CONFIG_SPACEMIT_HMP=y`、gcc |
| `gw-yolo.sh` | **YOLO 目标检测**（int8，56× 于 CPU EP） | 网关 `/v1/vision/*` |
| `x100-vs-a100-bench.sh` | X100 vs A100 的 LLM 对照实验 | llama-bench |
| `vision-zoo.sh` | **视觉模型全家桶**（14 个：yolo11/v8、pose、seg、face、gesture、emotion、resnet50） | 网关 `/v1/vision/*` |
| `rag.py` | **本地 RAG**：embed → 余弦 → rerank → LLM 全离线 | 网关 embed/rerank/llm + python3 |
| `tts-voices.sh` | **TTS 音色/语速验证**（含 md5 比对，会暴露参数 bug） | 网关 `/v1/tts/*` |
| `asr-length.sh` | **ASR 长度-吞吐曲线**（证明长音频要 VAD 切段） | 网关 `/v1/asr/recognize`、`/v1/vad/*` |
| `long-transcribe.py` | **长音频转写 → SRT 字幕**：VAD 切段 → 合并到 ~12s → 逐块 ASR（比整段快 9×） | 网关 VAD + ASR、python3 |
| `stream-asr.py` | **流式 ASR 客户端**（WebSocket，实测是缓冲式、不出增量 partial） | 网关 `/v1/asr/stream`、python3 `websockets` |
| `stream-tts.py` | **流式 TTS 客户端**：按句 `append` 才真流式（每句 ~0.5s 出声） | 网关 `/v1/tts/stream`、python3 `websockets` |
| `stream-voice.py` | **真·流式语音助手**：LLM token 流 → 逗号切句 → TTS 逐句合成（首音频 3.0s vs 非流式 28.4s） | 网关 LLM + TTS stream、python3 `websockets` |

## 快速上手

```bash
# 语音（文件进，语音出）
./voice-assistant.sh 说话.wav
./voice-assistant.sh --text "你好"

# 看图播报
./vlm-announce.sh 图片.jpg
./vlm-announce.sh --camera 3          # 需接 MIPI-CSI/USB 摄像头

# 硬编
./hw-encode.sh fix && ./hw-encode.sh check
./hw-encode.sh in.mp4 out.h264

# A100（注意：通用 LLM 会变慢，见报告 4.2）
cc -O2 -o ~/.local/bin/k3aix k3aix.c
./a100-unlock.sh nproc                # → 8

# 目标检测
./gw-yolo.sh 图片.jpg

# 视觉全家桶（14 个模型，一次一个，别并发）
./vision-zoo.sh ~/k3-ai/out

# 本地 RAG（先建索引再问）
./rag.py --build ../k3-ondevice-ai-report.md "机器人要接什么外设？"
./rag.py --no-rerank "A100 适合跑什么？"        # rerank 太慢，日常关掉

# TTS 音色 / 语速验证（会看到参数没生效）
./tts-voices.sh

# ASR 长度-吞吐曲线
./asr-length.sh ~/k3-ai/out/a5.wav ~/k3-ai/out/a10.wav ~/k3-ai/out/a20.wav ~/k3-ai/out/a45.wav

# 长音频转 SRT 字幕（顺带三方对照：合并 / 逐段 / 整段）
python3 long-transcribe.py 长音频.wav demo --compare   # 输出 /tmp/k3tr/demo.srt

# 流式：ASR（会看到它只吐一个结果）、TTS（按句 append 逐句出声）
python3 stream-asr.py 10s.wav 1.0
python3 stream-tts.py --append
python3 stream-tts.py "一整段文本" out.wav            # 一次性 start：只回一个整块

# 真·流式语音助手（LLM 流 → 按句 TTS）
python3 stream-voice.py "K3 板子能做哪些端侧 AI？" out.wav
```

## 必读的坑

1. **硬编报段错误** → 不是硬件问题，是用户不在 `video` 组（MPP 插件要探测 `/dev/video0`）。`./hw-encode.sh fix`。
2. **`taskset -c 8` 报 EINVAL** → 不是核坏了，是 HMP 的类别限制。先写 `/proc/set_ai_thread`。
3. **A100 反而更慢**（LLM 慢 5.8×）→ 它是向量/矩阵专用核，缓存小、主频低；只用它跑 int8/IME 算子。
4. **`/dev/video0` 不是摄像头** → 它是 VPU（Linlon）的 V4L2 节点；真摄像头会枚举成 `/dev/video1+`。
5. **板上重活必须串行** → 并发下载模型 + 加载/推理会把 16 核压到 load≈100，把隧道/心跳/ssh 全饿死（不是 OOM，见报告 §12.8）。
6. **网关字段名不统一** → embed/rerank/llm 用 `model`，vad/tts/vision 用 `model_id`；`/v1/vision/inference` 的 `tasks` 必须是 JSON 数组字符串。
7. **`/v1/asr/stream` 是假流式** → 不出增量 partial：37s 音频喂完还要再等 28.6s 才出唯一结果，RTF 跟整段识别一样。要低延迟只能自己 VAD 切段（`long-transcribe.py`）。
8. **`/v1/tts/stream` 要按句 `append` 才真流式** → 一次性 `start` 整段文本只会回一个整块（等于批量）；按句 `append` 才逐句出声（每句 ~0.5s）。
