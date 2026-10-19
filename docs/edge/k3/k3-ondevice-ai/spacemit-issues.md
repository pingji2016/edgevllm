# 给 SpaceMiT 的问题清单（可直接提交的 issue 草稿）

> 环境（全部实测于 2026-10-16 ~ 10-19）：
> **K3 Pico ITX**（riscv64）· Bianbu 4.0.1 · 内核 `6.18.3-cf-k3-1.3.0`
> 16 核（8× **X100** @2.15GHz + 8× **A100** @1.8GHz）· 15GiB · **SpaceMiT AI Gateway v0.1.15**
> 详细数据见 [`k3-ondevice-ai-report.md`](k3-ondevice-ai-report.md) 对应章节。
>
> 提交建议：优先提到对应仓库（`spacemit-com/linux-6.18`、网关/MPP 的仓库）；
> 找不到仓库就发 SpaceMiT 开发者论坛，附本文件与报告。

---

## Issue 1（严重）A100 核组跑通用 LLM 反而慢 5.8×，且没有任何提示

**组件**：内核 HMP 调度（`drivers/soc/spacemit/hmp.c`、`/proc/set_ai_thread`）+ llama.cpp

**摘要**：按官方方式把线程切到 A100（HMP_AI, cpu 8–15）后，llama.cpp 的
prompt/predict 吞吐下降约 **5.8 倍**。A100 是向量/矩阵专用核，这个结果本身可以理解，
但**没有任何文档/告警**说明它不适合通用 LLM，用户很容易认为是"没提速"或"配置错了"。

**复现**：
```bash
# 1) 切到 A100
echo 0 | sudo tee /proc/set_ai_thread
nproc                     # → 8
cat /proc/self/status | grep Cpus_allowed_list   # → 8-15

# 2) 同一 gguf，对比 X100（8 线程）与 A100（8 线程）
taskset -c 0-7  llama-bench -m Qwen3.5-2B-Q4_0.gguf -t 8 -p 256 -n 16
taskset -c 8-15 llama-bench -m Qwen3.5-2B-Q4_0.gguf -t 8 -p 256 -n 16
```

**实测**：

| 核组 | pp256 (t/s) | tg16 (t/s) |
|---|---:|---:|
| X100 (0–7) | **8.91** | **5.54** |
| A100 (8–15) | 1.53 | 1.45 |

**根因（我们的分析）**：

| | X100 | A100 |
|---|---|---|
| 主频 | 2.15 GHz | **1.8 GHz** |
| L1D | 64 KB | **32 KB** |
| L2 / 4 核 | 4 MB | **1 MB** |
| VLEN | 256 | 1024 |
| IME | 无 | **有** |

→ A100 靠 IME 做 int8 矩阵算子；llama.cpp 的 Q4_0 反量化 + 标量/向量混合负载
正好踩在它的短板上（缓存小、主频低）。

**期望**：
1. 在 A100 相关文档里明确写"通用 LLM 推理请留 X100"；
2. `echo 0 > /proc/set_ai_thread` 之后能在 dmesg 或 `/proc` 里看到当前核组的用途说明；
3. 条件允许时，给 llama.cpp 的 spacemit 分支加一个"仅把 IME 算子 offload 到 A100、
   其余留在 X100"的模式（这才是 A100 的正确用法）。

---

## Issue 2（严重）VPU 硬编：用户不在 `video` 组时**段错误**，而不是权限错误

**组件**：MPP VENC (`h264_stcodec` / `libmpp` / `libvi_k3_cam_plugin.so`)

**摘要**：普通用户直接跑 `ffmpeg -c:v h264_stcodec` 会 **segfault**。把用户加进
`video` 组后立即正常。插件在打开 `/dev/video0`（VPU 的 V4L2 节点）失败后没有
检查返回值，继续用空指针。

**复现**：
```bash
id -nG                       # 不含 video
ffmpeg -f rawvideo -pix_fmt nv12 -s 1920x1080 -r 30 -i in.nv12 \
       -c:v h264_stcodec -frames:v 10 out.h264
# → Segmentation fault
sudo usermod -aG video $USER   # 重新登录
# → 正常输出
```

**期望**：返回 `EACCES` 并打印 `Permission denied: /dev/video0 (add user to 'video' group)`。

**建议**：
1. 插件里检查 `open()`/`ioctl()` 返回值；
2. 打包时带一条 udev 规则（`SUBSYSTEM=="video4linux", KERNEL=="video*", GROUP="video", MODE="0660"`），
   或在 `ffmpeg -encoders` 的帮助里注明需要 `video` 组。

**补充**：修好后 1080p150 帧硬编 **1.087s**，对比 libx264 veryfast **5.219s**（快 4.8×，
CPU 占用少约 144×）—— 这条链路值得默认打开。

---

## Issue 3（中）TTS 的 `voice_id` 与 `speed` 参数被**静默丢弃**

**组件**：AI Gateway `/v1/tts/synthesize`、`/v1/tts/voices`（Matcha / Kokoro 适配层）

**摘要**：`/v1/tts/voices` 返回 **6 个音色**，但换 `voice_id` 输出的音频**完全一样**；
`speed` 改成 0.7 / 1.5 音频时长**也完全一样**。只有换 `model` 才真的变化。

**复现**：
```bash
# 同一段"全新文本"，三个 voice_id → md5 相同
for v in $(curl -s localhost:18790/v1/tts/voices | jq -r '.[].id'); do
  curl -s -X POST localhost:18790/v1/tts/synthesize \
    -H 'Content-Type: application/json' \
    -d "{\"text\":\"这段文字是全新生成的用于测试\",\"voice_id\":\"$v\",\"model\":\"matcha_zh_en\",\"response_format\":\"wav\"}" \
    -o /tmp/t_$v.wav
  md5sum /tmp/t_$v.wav
done
```

**实测**：
- 3 个 `voice_id` → 3 个文件 md5 **全为 `29a115b7…`**
- `speed` 0.7 vs 1.5 → md5 **全为 `118beecb…`**
- matcha_zh 的 `speed=0.7/1.0/1.5` → 时长**都是 4.49s**
- 换 `model`（matcha_zh_en / kokoro）→ 输出**不同** ✅

**影响**：多音色/变语速这两个最常用的能力实际不可用，且不报错，排查成本高。

**期望**：把 `voice_id`/`speed` 透传到适配器；不支持时返回 4xx + 明确错误，而不是静默忽略。

**复现脚本**：[`scripts/tts-voices.sh`](scripts/tts-voices.sh)

---

## Issue 4（中）ASR 处理耗时随音频长度**近似二次增长**

**组件**：AI Gateway `/v1/asr/recognize`（SenseVoice）

**摘要**：同一模型，音频越长 RTF 越差，40s 时已到 **0.807**（≈1.2× 实时），
意味着 1 小时录音要跑约 48 分钟。

**实测**：

| 音频 | 处理耗时 | RTF |
|---:|---:|---:|
| 5.1s | 0.77s | 0.150 |
| 10.1s | 2.17s | 0.216 |
| 20.1s | 8.14s | 0.405 |
| 40.2s | 32.5s | **0.807** |

**期望**：接近线性；或接口层强制分块（见下面 Issue 5 的建议）。

**我们的规避**：客户端先用 `/v1/vad/segments`（34× 实时）切段、合并到 ~12s 再逐段识别，
37.3s 音频从 27.9s 降到 **2.98s（12.5× 实时，快约 9 倍）**。
脚本：[`scripts/long-transcribe.py`](scripts/long-transcribe.py)

---

## Issue 5（中）`/v1/asr/stream` 号称流式，实际**不出增量 partial**

**组件**：AI Gateway `/v1/asr/stream/session` + `/v1/asr/stream`（WebSocket）

**摘要**：按文档流程（`POST .../stream/session` → WS 带 `session_id` → 发二进制 PCM 帧 →
发 `{"type":"end"}`），`partial=true` 却**只收到 1 个 partial，且在音频喂完之后**，
RTF 与整段批量识别几乎一样。这不是"流式"，只是给批量套了个 WS 外壳。

**实测**：

| 音频 | 喂入 | 首个（也是唯一）结果 | RTF |
|---:|---|---:|---:|
| 10.1s | 实时 1× | 12.44s（结束后 +2.38s） | 0.227 |
| 37.3s | 实时 1× | **65.90s（结束后 +28.6s）** | **0.759** |

**期望**：边说话边出 partial（哪怕是每 1–2s 一个字幕行），或在 `partial_results=true`
时明确返回"不支持"。

**对照**（正向反馈）：`/v1/tts/stream` 是**真流式** —— 按句 `{"type":"append"}` 喂文本，
每句约 **0.5s** 就返回该句的 PCM。用这个能力做"边说边播"，首音频从 28.4s 降到 **3.0s**。
脚本：[`scripts/stream-voice.py`](scripts/stream-voice.py)

---

## Issue 6（低）`onnxruntime_perf_test -e spacemit` 报 "Use a '|' to separate the key and value"

**组件**：SpaceMITExecutionProvider（onnxruntime `-e spacemit`）

**摘要**：按名字指定 EP 时，provider options 解析直接报
`Use a '|' to separate the key and value`，即使没有传任何 provider option。

**复现**：
```bash
onnxruntime_perf_test -e spacemit -r 5 model.onnx
# → Use a '|' to separate the key and value
```

**期望**：不传 option 时应使用默认值，不应报错；或文档给出必需的最小 option 串
（例如 `-e spacemit|<key>=<value>`）。

---

## 附：正向结论（一并反馈）

- **VPU 硬编**（修好权限后）：1080p150 帧 1.087s，比 libx264 veryfast 快 4.8×、CPU 少 ~144×。
- **视觉模型**：int8 YOLOv8n 单帧 **14.3ms**（emotion 7.1ms），14 个模型全部可用。
- **VAD**：10s 音频 **288ms**（34× 实时）。
- **TTS 流式**：按句 append 每句 ~0.5s 出声，是整套栈里最好用的一块。
- **A100 的正确定位**：int8/IME 算子专用核；`/proc/set_ai_thread` 一行即可切换，机制本身很干净。
