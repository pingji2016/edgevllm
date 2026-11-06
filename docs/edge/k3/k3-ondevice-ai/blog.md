> 起因很简单：手上有块 SpaceMiT K3 的 Pico ITX 板子（riscv64，16 核，15G 内存），想验证一件事 —— **不联网、不挂云端 API，它能把这几年端侧 AI 的活干到哪一步**。折腾了几天，语音、视觉、硬编、大模型都跑通了，也踩出了几个值得写下来的坑。

![K3 端侧 AI 全景](/bbs/img/k3-stack.png)

## 先说结论

| 方向 | 结果 |
|---|---|
| 离线语音对话 | ASR 2.4s → LLM 9.9s → TTS 1.2s，一轮约 18 秒，全程板内 |
| 真·流式语音 | 首音从 28.4 秒降到 **3.0 秒**（LLM token 流 + 按标点切句 + TTS 逐句合成） |
| 长音频转写 | VAD 切段后整体快 **9 倍**（40 秒音频原来只能跑 1.2× 实时） |
| 视觉 | YOLOv8n int8 单帧 **14.8 ms**，比纯 CPU 快 56 倍；14 个视觉模型全测过 |
| 硬编 | 1080p 150 帧 **1.09 秒**，比 x264 快 4.8×、CPU 少 144× |
| 大模型 | 8 线程最优；**A100 反而慢 5.8×** —— 它不是“更快的 CPU” |
| 机器人接口 | CAN-FD/SPI 在设备树里**没使能**；`/dev/video0` 不是摄像头而是 VPU |
| 横向位置 | 比树莓派 5 的最快 CPU 路径快 4.5×，比 Jetson Orin Nano 慢 3.9×（不同模型，量级对比） |
| 端侧 vs 云端 | 10 秒语音：云端 ≈1.5–2.2 s、本地 2.39 s，基本打平；1080p 视频要 7.7 Mbps，实测上行只有 2.3 Mbps，传不动 |
| 工程化程度 | 单点能力已验证，**仍是「文件输入 + 手动跑脚本」**，没有实时闭环 |

板子本身：

| 项 | 值 |
|---|---|
| SoC | SpaceMiT K3（riscv64） |
| CPU | 16 核：8× X100 @2.15 GHz + 8× A100 @1.8 GHz |
| 内存 / 磁盘 | 15 GiB（Android guest 常驻 4.5 GiB）/ 117 GB |
| 系统 | Bianbu 4.0.1，内核 `6.18.3-cf-k3-1.3.0` |
| 预装 | SpacemiT AI Gateway v0.1.15，`127.0.0.1:18790`，128 个端点 |

---

## 1. 镜像里已经有一套 AI 平台

这是最省事的一点：不用自己拼 sherpa-onnx + llama.cpp，厂商把整条链路都封成了本地 HTTP 服务。

| 域 | 端点 | 后端 |
|---|---|---|
| ASR | `/v1/asr/recognize` | SenseVoice（中英日韩粤，16k） |
| TTS | `/v1/tts/synthesize`、`/v1/tts/stream/session` | Matcha-TTS、Kokoro |
| LLM | `/v1/llm/chat/completions`（OpenAI 兼容） | llama.cpp + Qwen3 系 gguf |
| VLM | `/v1/vlm/chat/completions` | Qwen3-30B-A3B-MM（17.6G，没下） |
| Vision | `/v1/vision/inference` | YOLOv8/v11、resnet50（int8） |
| VAD / Embed / Rerank | `/v1/vad/*` 等 | Silero、bge、qwen3-reranker |

模型按需从厂商 CDN（`archive.spacemit.com`）下载，300–440 KB/s，落在 `~/.cache/models/`。视觉和 ASR 走 `SpaceMITExecutionProvider`（就是 A100 的 IME 算子），LLM 用镜像自带的 `llama.cpp-tools-spacemit`。

> 两个容易踩的字段坑：`tasks` 必须是 **JSON 数组**（`-F 'tasks=["detect"]'`）；`embed/rerank/llm` 的 load 用 `model`，而 `vad/tts/vision` 用 `model_id`。模型默认 `unloaded`，直接推理会报 `model not found`。

---

## 2. 离线语音：一轮 18 秒

![语音链路](/bbs/img/k3-voice-pipeline.png)

输入是一段 10 秒的中英混说音频：

| 阶段 | 耗时 | 说明 |
|---|---:|---|
| SenseVoice ASR | 2.39 s | RTF 0.24，中英混说一次就对 |
| Qwen3.5-2B 生成 | 9.89 s | 10 token，6.3 t/s |
| Matcha-TTS | 1.20 s | 产出 3.66 s 语音 |
| 播放 | 4.51 s | 48 kHz 立体声 |

识别结果：「昨天是monday，today is礼拜2，the day after tomorrow是星期三。」回答：「昨天周一，今天周二，后天周三。」

**踩到的两个坑：**

1. **声卡只吃 48 kHz 立体声。** ES8326 的 `CHANNELS` 只有 2，TTS 输出是 16 kHz 单声道，`aplay` 直接失败。统一 `ffmpeg -ac 2 -ar 48000` 转一遍就好。
2. **板子上没有麦克风。** 录 5 秒，ASR 返回一个「？」，RTF 0.02 —— 纯本底噪声。脚本留了 `--mic` 模式，插上 USB 麦即可，不用改代码。

---

## 3. 长音频：不要整段丢给 ASR

先看一条曲线 —— 同一个 SenseVoice，喂不同长度的音频：

![ASR RTF 曲线](/bbs/img/asr-rtf.svg)

| 音频 | 处理耗时 | RTF |
|---:|---:|---:|
| 5.1 s | 0.77 s | 0.150 |
| 10.1 s | 2.17 s | 0.216 |
| 20.1 s | 8.14 s | 0.405 |
| 40.2 s | 32.5 s | **0.807** |

处理时间近似**二次增长**（attention 没有分块）。40 秒的音频已经只剩 1.2 倍实时，再长就没法用了。

正确姿势是 **VAD 切段 → 合并 → 逐块识别**。VAD（silero）是 34 倍实时，37.3 秒音频切 14 段只花 1.10 秒：

| 模式 | 块数 | ASR 耗时 | 相对实时 |
|---|---:|---:|---:|
| **合并到 ~12 s（推荐）** | 8 | **2.98 s** | **12.5×** |
| 逐段 | 14 | 2.87 s | 13.0× |
| 整段（对照） | 1 | 27.94 s | 1.34× |

快约 9 倍，而且合并模式上下文更完整 —— 逐段会把一句切成 `Day after tomorrow.` + `嗯星期三。`。实现上注意三点：过滤 250 ms 以下的杂音段（VAD 会在开头误报一个 120 ms 段，识别成「？」）、合并 600 ms 以内的间隔、单块上限 12 秒。输出直接给标准 SRT 字幕。

---

## 4. 网关的“流式”，一半是真的

![流式对比](/bbs/img/k3-streaming.png)

**ASR 的流式是假的。** `/v1/asr/stream` 协议看着很正规：拿 session → WS 连 → 喂 PCM → `end` → 回 partial / sentence_end / final。实测按 1× 实时喂 37.3 秒音频，**全程只有 1 个 partial，而且要等音频喂完 28.6 秒后才出**，RTF 0.759 与整段识别（0.807）几乎一样 —— 只是给整段识别套了层 WebSocket 外壳。

**TTS 的流式是真的，但必须按句 append。** 一次性 `start` 整段 84 字，要 3.98 秒才回一整个 526 KB 的块（等于批量）；改成每 1.5 秒 `append` 一句，第一句 0.55 秒就出声，后面 2.09 / 3.59 秒。

所以真·流式语音助手的做法是：**LLM 的 SSE 流每来一小段就按 `，、。` 切句，切出一句立刻 append 给 TTS**，PCM 块边到边拼，最后一次性转码播放。首音 3.03 秒，非流式 28.41 秒。

> 一个诚实的说明：非流式那次回答更长（258 字 vs 流式 9.41 秒音频），属同题不同长度的示意对比；但对齐长度后流式依然更早出声，因为它在 LLM 还没生成完时就开口了。

---

## 5. 视觉：单帧 14.8 ms

![视觉链路](/bbs/img/k3-vision.png)

`POST /v1/vision/inference`，模型 `yolov8n`（int8，3.37 MB）：预处理 12.6 ms + 推理 **14.8 ms** + 后处理 0.43 ms = 27.9 ms；HTTP 端到端 65–83 ms（含 multipart 上传）。

同一张 `bus.jpg`，检出 4 个人和 1 辆公交，和肉眼一致：

![YOLOv8n 在 K3 上的检出](/bbs/img/yolo-bus.jpg)

和纯 CPU 比：

| 路径 | 单帧 | 相对 |
|---|---:|---:|
| `onnxruntime_perf_test -e cpu`（1 / 2 / 4 / 8 线程） | 860 / 820 / 840 / 943 ms | 1× |
| 网关 native + SpaceMIT EP（int8） | **14.8 ms** | **≈56×** |

注意 CPU 那一行：**加线程没用，8 线程反而最慢**。这条通用 ONNX 路径在 RISC-V 上根本没并行起来 —— 不是「没调好线程数」。

一次把 14 个视觉模型全测了（单帧 `model_infer_ms`）：

| 模型 | 任务 | 推理 | 模型 | 任务 | 推理 |
|---|---|---:|---|---|---:|
| emotion | 表情 | **7.1 ms** | yolov8n-seg | 分割 | 19.6 ms |
| resnet50 | 分类 | **7.7 ms** | yolov11n | 检测 | 20.1 ms |
| yolov8n | 检测 | 14.3 ms | yolov5-gesture | 手势 | 21.4 ms |
| yolov8n-pose | 姿态 | 16.2 ms | yolov8s | 检测 | 24.8 ms |
| yolov8s-pose | 姿态 | 27.3 ms | yolov5-face | 人脸 | 28.4 ms |
| yolov11s | 检测 | 30.9 ms | yolov8s-seg | 分割 | 32.5 ms |
| yolov8m | 检测 | 48.0 ms | yolov11m | 检测 | 65.2 ms |

小目标假阳性不少（yolov8s/m 会把领带、盆栽当目标，0.50~0.54）；resnet50 把整张公交照认成 `minibus 0.61` —— 分类模型看多目标图，正常。

**VLM 部分**：Mage-VL 4B 看图 58.7 秒，其中 **26.8 秒**是 `mtmd batch encoding`（图像 → vision token），瓶颈在视觉编码而不是生成。板子没有摄像头（`/dev/video0` 是 VPU 节点），所以视频理解先用现成视频做抽帧验证：逐帧出时间线，或者拼成一张 3×N 图让模型整体概括。

---

## 6. 跟主流方案比，K3 站在哪一档

单看 14.8 ms 没什么感觉，得放进几个坐标系里。下面四组对比，可信度不一样，我分开标。

### 6.1 同一块板，换个实现（本机实测）

| 任务 | 没有加速的路径 | 加速路径 | 倍数 |
|---|---:|---:|---:|
| 检测一帧 | ONNX Runtime CPU EP **832 ms** | SpaceMIT EP int8 **14.4 ms** | **57×** |
| 1080p / 150 帧编码 | x264 veryfast 5.22 s（CPU 29.3 s） | VPU H.264 1.09 s（CPU 0.20 s） | **4.8×**，CPU **144×** |
| 37 s 音频转写 | 整段识别 27.94 s | VAD 合并切段 2.98 s | **9.4×** |
| 语音首音 | 非流式 28.41 s | 流式 3.03 s | **9.4×** |
| LLM 生成 | A100 核组 1.53 t/s | X100 核组 8.91 t/s | **5.8×** |

这张表的重点是：**同一份模型、同一台机器，只是换了一条路**。所以倍数不会有"测试条件不同"的争议 —— 有加速器和没有加速器，就是两个世界。

### 6.2 换块板子，跑同一个模型（别人家的公开数据）

![跨设备单帧对比](/bbs/img/device-cmp.svg)

| 设备 | 实现 | 单帧 | 相对 K3 |
|---|---|---:|---:|
| Jetson AGX Orin 64G | TensorRT INT8 | 2.3 ms | 0.16× |
| Jetson Orin NX 16G | TensorRT INT8 | 3.5 ms | 0.24× |
| Jetson Orin Nano Super | TensorRT INT8 | 3.8 ms | 0.26× |
| **K3** | **SpaceMIT EP int8** | **14.8 ms** | **1×** |
| 树莓派 5 | NCNN（CPU） | 67 ms | 4.5× |
| 树莓派 5 | ONNX Runtime（CPU） | 126 ms | 8.5× |
| K3 | ONNX Runtime CPU | 832 ms | 56× |

来源：K3 两行是本次实测（`model_infer_ms` 与 `onnxruntime_perf_test`），其余是 Ultralytics 官方的 YOLO26n / 640 基准（[Jetson 一栏](https://docs.ultralytics.com/guides/nvidia-jetson/)、[树莓派一栏](https://docs.ultralytics.com/guides/raspberry-pi/)），都不含前后处理。

**这不是严格跑分**：模型不同（YOLOv8n vs YOLO26n，mAP 都在 0.47 上下）、精度不同（int8 vs fp32）、机器不同。但量级可信，三个结论也够清楚：

- 有 NPU/GPU 的那一档（Jetson 2–4 ms）比 K3 快 4 倍，这是硬件代差，认；
- K3 的加速路径，比树莓派 5 **最快**的 CPU 路径还快 4.5 倍；
- 但同一块板换回通用 ONNX，K3 比树莓派 5 慢 6.6 倍 —— **差的是算子库，不是算力**。

换句话说：买 K3 买的就是它的 EP + IME。如果你打算在上面跑通用框架，它会掉到"比树莓派还差"的那一档。

### 6.3 多开几路会怎样（本机实测）

| 并发 | 端到端墙钟 | 吞吐 | 单帧推理 |
|---:|---:|---:|---:|
| 1 | 72 ms | 13.9 次/秒 | 14.4 ms |
| 2 | 101 ms | 19.7 次/秒 | 14.4 ms |
| 4 | 167 ms | 23.9 次/秒 | 14.3 ms |
| 8 | 300 ms | 26.6 次/秒 | 14.3 ms |

单帧推理耗时**完全不随并发上升**（14.4 → 14.3 ms），说明 IME 内部是并行/流水处理的；但吞吐只放大 2 倍就饱和，瓶颈不在 AI 算力，在网关的 HTTP 层和预处理（单进程）。

要跑多路摄像头，别用多开 curl 解决，得在进程内批量喂帧，或者干脆多进程 + 负载均衡。

### 6.4 模型选型：这块板能跑多大的（本机实测）

| 模型 | 文件 | 生成速度 | 适合干什么 |
|---|---:|---:|---|
| qwen3-0.6b | 365 MB | **13.9 t/s** | 意图解析、槽位提取、路由 |
| deepseek-r1-distill-1.5b | 1.0 GB | 6.1 t/s | 带思维链，数学题能算对 |
| hy-mt1.5-1.8b | 1.1 GB | 7.0 t/s | 翻译专用 |
| qwen3.5-2b | — | 6.3 t/s | 通用对话（本文基线） |
| LFM2-24B-A2B | 17.6 GB | — | 装不下（可用内存 4–9 GB） |

**2B 是这块板的分水岭**：0.6B 能到 14 t/s，适合当"前置小模型"；2B 是 6 t/s 的可用对话速度；再大就只剩"能跑，但等不起"。

### 6.5 端侧 vs 云端（网络实测 + 算账）

反过来看现在最主流的做法：调云端 API。我在板子上把网络也量了：

| 项 | 实测 |
|---|---:|
| RTT（到 223.5.5.5，WiFi） | 101 ms 平均（66–147 ms） |
| 上行带宽 | 288 KB/s ≈ 2.3 Mbps |
| 下行（从 111 拉一张 200 KB 图） | 63 KB/s（当时链路偏保守） |

拿 10 秒语音算账：16 kHz 单声道 16 bit = 320 KB，上传 1.11 秒，加一次往返 0.1 秒，再加云端推理（公开 API 通常 0.3–1 秒）—— **云端大概 1.5–2.2 秒，端侧 2.39 秒**，基本打平，而且云端识别更准。

那端侧的意义在哪？把同一个账算到视频上就露馅了：1080p H.264 一路 7.7 Mbps，实测上行只有 2.3 Mbps —— **视频根本传不上去**。再叠加断网可用、隐私不出门、不按次计费，才轮到端侧出场。

所以结论不是"端侧更快"，而是：**短语音请求云端更划算，连续视觉 / 离线 / 隐私场景端侧不可替代**。K3 这种板子的正确定位，是"把摄像头和麦克风的数据就地消化掉"。

## 7. VPU 硬编：根因是个权限 bug

一开始 `ffmpeg -c:v h264_stcodec` 直接段错误：

```
[MPP-ERROR] check_v4l2_linlonv5v7:578 Can not open '/dev/video0' (Permission denied)
[MPP-ERROR] module_init:199 can not find v4l2_linlonv5v7
[MPP-ERROR] module_get_so_path:281 module is NULL
→ 段错误
```

根因链条很清楚：ffmpeg 的 `*_stcodec` → MPP 插件 `libv4l2_linlonv5v7_codec2.so` 初始化时逐个探测 `/dev/video0..31`；而 `/dev/video0` 是 `root:video 0660`，普通用户不在 `video` 组 → 探测全失败 → 插件返回 NULL → ffmpeg 空指针崩溃。

不是硬件坏，也不是内核缺配置（`CONFIG_VIDEO_LINLON_K3=y`，dmesg 里 `MVX dev: Linlon v5276 identified. cores=4` 一切正常）。修复就一行：

```bash
sudo usermod -aG video $USER     # 重新登录生效
```

修完之后（1080p / 150 帧 / 同一份 NV12，排除解码干扰）：

![硬编对比](/bbs/img/encode.svg)

| 编码器 | 墙钟 | user CPU | 产物 |
|---|---:|---:|---:|
| **VPU H.264** @4M | **1.087 s** | 0.203 s | 4.82 MB |
| 软编 x264 veryfast @4M | 5.219 s | 29.276 s | 2.50 MB |
| **VPU HEVC** @4M | **0.792 s** | 0.166 s | 3.20 MB |
| VPU H.264 @12M | 1.057 s | 0.191 s | 4.82 MB |

**快 4.8×、CPU 少 144×**，1080p 单流约 138 fps。

但还有个尾巴：`-b:v` **没有映射到 MPP 的码率控制** —— 4M 和 12M 的产物字节数完全相同（都是 4.82 MB，实际约 7.7 Mbps）。要真调码率得走原生 `VencChnAttr` 的 `eRcMode / u32Bitrate / u32MinQp / u32MaxQp`。我按这套接口写了 `mpp_enc.c`（直连 VENC/VB、零拷贝 dma-buf、支持 CBR/VBR/CVBR/FIXQP），语法已过，板上编译和实测还在排队。

---

## 8. A100 不是“更快的 CPU”

这是整轮最有意思的发现。写 `/proc/set_ai_thread` 把线程切进 A100 组，`taskset -c 8-15` 也合法了，但跑 llama.cpp 的结果是：

| 核组 | pp256 | tg16 |
|---|---:|---:|
| X100（cpu0-7） | **8.91 t/s** | **5.54 t/s** |
| A100（cpu8-15） | 1.53 t/s | 1.45 t/s |

**慢 5.8 倍。** 原因不是故障，是定位不同：

| | X100 | A100 |
|---|---|---|
| 主频 | 2150 MHz | 1800 MHz |
| L1I / L1D | 64K / 64K | 32K / 32K |
| L2（4 核共享） | 4 MB | 1 MB |
| VLEN | 256 | 1024 |
| IME | 无 | 有 |

A100 是向量/矩阵专用核，缓存小、主频低，跑「从 DDR 流式读权重」的通用 Q4 路径必然吃亏。它的正确打开方式是 **IME 算子 + int8**（也就是视觉那 56 倍的来源）。**别拿它跑通用 LLM。**

顺带量了线程数：4 线程 pp 4.53 / tg 3.47，**8 线程 8.90 / 5.26**，16 线程 8.72 / 4.35 —— 超订反而掉 17%，所以网关默认 `--threads 8` 是合理的。

---

## 9. 外设：芯片很豪华，设备树没全开

| 接口 | 芯片能力 | 板子实际 |
|---|---|---|
| CAN-FD | 10 路 | ❌ `ip link` 里没有 `can0` |
| SPI | 有 | ❌ 无 `/dev/spidev*` |
| I2C | — | ✅ `/dev/i2c-{1,2,3,8,9,10,11,12}` |
| UART | 17 | ✅ `/dev/ttyS0..16` |
| PWM | 30 | ✅ `pwmchip0`、`pwmchip1` |
| GPIO | — | ✅ `/dev/gpiochip0..3` |
| USB | 6 条 | ✅ 2×USB3 + 4×USB2 |
| 摄像头 | MIPI-CSI | 驱动齐，但没插 |

想直连 CAN 关节电机得先改 DT/内核（要人在现场）；图省事就上 USB-CAN 适配器。另外 **K3 没有 NPU**，真正的加速器就是 A100 的 IME 和 VPU 编码器。

---

## 10. 踩坑清单

| # | 问题 | 严重度 |
|---|---|---|
| 1 | A100 跑通用 LLM 慢 5.8×，无任何提示 | 严重 |
| 2 | VPU 硬编在用户不在 `video` 组时**段错误**而非权限错误 | 严重 |
| 3 | TTS 的 `voice_id` / `speed` 被**静默丢弃** | 中 |
| 4 | ASR 耗时随音频长度近似**二次增长** | 中 |
| 5 | `/v1/asr/stream` 号称流式却**不出增量** | 中 |
| 6 | `onnxruntime_perf_test -e spacemit` 报参数解析错 | 低 |

第 3 条值得单独说：用全新文本排除缓存后，`zf_xiaobei / zm_yunxi / af_heart` 三个音色的输出 **md5 完全相同**，`speed=0.7 / 1.5` 的输出也完全相同，`matcha_zh` 两种语速时长都是 4.49 秒 —— 只有换 `model` 才变。所以音色和语速参数根本没传下去。

除了厂方的问题，自己也栽了几次：并发下模型（3 个下载 + 14 个视觉模型 + RAG）把 16 核压到 load 108，心跳停摆 28 分钟，**不是 OOM**，负载回落后自愈；Qwen3-30B-A3B 也不是调优问题 —— 17.6 GB 权重、CDN 11–16 小时、可用内存只有 9 GB，装不下。

最后附一个运维事故，因为它比任何 benchmark 都值得记：

![隧道失联](/bbs/img/k3-tunnel.png)

10/19 09:14 反向隧道断了，SSH 彻底不可达，但**门户心跳每 30 秒照常更新**。板子活着、出网正常，只有 111:22 的 SSH 在 banner 交换阶段被中间设备掐掉；systemd 自动重试了 228 次，每次卡满 10 秒 / 120 秒超时，直到 23:37 网络侧放行，下一次重试立刻连上。

教训有三条：**心跳通 ≠ 可管理**（两条独立链路）；`ConnectTimeout` 只管 TCP 连接，`ServerAliveInterval` 只在认证成功后生效；以及一定要留一条带外通道（备用端口 / 局域网直连 / 串口）。

结果今天上午又复发一次，症状还不一样：隧道进程活着、`Send-Q` 恒为 0、心跳照常，但所有新连接都卡在 banner 交换 —— 板上那个"按 Send-Q 判卡死"的看门狗永远等不到。最后靠在 111 上 kill 掉那条会话、让板子的客户端以 255 退出重连才恢复，耗时 25 秒。

所以现在在 111 上加了一层：每 2 分钟直连 `127.0.0.1:2222` 读 `SSH-` 开头那 4 个字节，连续两次读不到就杀掉持有端口的会话，让 systemd 把客户端拉起来。装上之后当场自愈了一次，25 秒。

教训补一条：**看门狗要探"能不能真的用"，而不是探"进程还在不在"。**

---

## 11. 还没做的：一张路线图

![从脚本到产品](/bbs/img/k3-roadmap.png)

写到这儿最容易被误读的是"这板子已经能用了"。其实差得还远，我按"卡在哪"分三类列清楚。

**已经跑通（但都是脚本 + 文件输入）**：离线语音整轮 18 s、图片检测 14.8 ms、VAD 长音频快 9×、VPU 硬编 138 fps、流式首音 3.03 s。

**差一步：硬件或工程**

| 事项 | 卡在哪 |
|---|---|
| 摄像头实时闭环 | 板子没插摄像头，`/dev/video0` 还是 VPU 节点而不是 sensor |
| 麦克风常驻 | 没有声卡输入；ES8326 只吃 48 kHz 立体声 |
| 唤醒词 + 打断（barge-in） | 现在是一问一答，说话得等它讲完 |
| 常驻服务 | 全是手动脚本，没做成 systemd 单元 + 失败重启 |
| 流式 ASR | 网关的 `/v1/asr/stream` 不出增量，得换 sherpa-onnx 流式模型 |
| 多路视频 | 并发只到 26.6 次/秒，且没有进程内批处理 |
| TTS 音色 / 语速 | 厂方把参数丢了（见 §10 第 3 条），要用只能绕开网关 |
| VLM 视频理解 | Mage-VL 单帧 58.7 s，抽帧都嫌慢 |
| RAG / 知识库 | rerank 11 秒/篇，比向量检索贵 8000 倍 |
| LLM 再加速 | 没有 IME 的 LLM 内核，只能吃 RVV + 老实的 8 线程 |

**还没开始：产品化**

| 事项 | 说明 |
|---|---|
| CAN / 机器人控制 | CAN-FD 在设备树里没使能，`ip link` 里连 can0 都没有 |
| ROM / OTA 集成 | AI 栈现在还装在用户目录，没进系统镜像 |
| 功耗 / 温度 / 降频 | 只读到 53 °C，没有功率计，也没做长时间满载曲线 |
| 压测 / 降级 | 没做"模型挂了怎么退"的兜底 |
| 端侧微调 | 一次没试过 |
| 语音克隆 / 方言 | 未尝试 |
| 跟踪 / OCR / 姿态接业务 | 模型都在，没接进去 |

所以准确的说法是：**单点能力验证完了，工程化才刚开始**。下一步我自己排的顺序是 —— 先插一只 USB 麦和摄像头把"实时"跑通（最便宜，也最能暴露问题），再把语音链路做成 systemd 常驻 + 支持打断，最后才谈 ROM 集成。

## 12. 怎么复现

```bash
# 网关
systemctl is-active spacemit-ai-gateway
curl -s http://127.0.0.1:18790/v1/asr/models

# 语音（文件模式；插上麦用 --mic 5）
~/k3-ai/voice-assistant.sh some_speech.wav
~/k3-ai/stream-voice.py --file ~/k3-ai/out/long.wav

# 长音频 → SRT
~/k3-ai/long-transcribe.py ~/k3-ai/out/long.wav

# 硬编（先确保在 video 组）
~/k3-ai/hw-encode.sh check
CODEC=hevc ~/k3-ai/hw-encode.sh in.mp4 out.h265

# 视觉
curl -s -X POST http://127.0.0.1:18790/v1/vision/models/load \
  -H 'content-type: application/json' -d '{"model_id":"yolov8n"}'
curl -s -X POST http://127.0.0.1:18790/v1/vision/inference \
  -F file=@bus.jpg -F model_id=yolov8n -F 'tasks=["detect"]'
```

所有脚本都在仓库的 `docs/k3-ondevice-ai/scripts/` 下，报告原文在同目录的 `k3-ondevice-ai-report.md`。

唯独 `mpp_enc.c` 的码率扫描还没跑完（验证 `eRcMode` 的 CBR 是否真的生效），这是我下一步第一个要补的实测。

---

*本文所有数字均为板内实测（不联网、不调云端 API），同一份输入、同一台机器。模型耗时取自网关返回的 `model_infer_ms`，与 HTTP 墙钟分开标注。*
