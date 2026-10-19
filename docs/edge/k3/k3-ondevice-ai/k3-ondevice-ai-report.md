# K3 端侧 AI 落地实测报告（2026-10-16）

> 板子：SpaceMiT K3 Pico ITX（riscv64）· Bianbu 4.0.1 · 内核 `6.18.3-cf-k3-1.3.0`
> 16 核（8× X100 @2.15GHz + 8× A100 @1.8GHz）· 15GiB 内存 · 117G 盘
> 全部实验**在板子上离线完成**，不依赖云端。所有数字都是本机实测，不是厂商标称值。

---

## 0. 结论速览

| # | 任务 | 状态 | 关键结果 |
|---|---|---|---|
| 1 | **离线语音助手** | ✅ 跑通 | ASR 2.4s → LLM 9.9s → TTS 1.2s → 喇叭，一轮 ≈18s，全程离线 |
| 2 | **摄像头 → VLM → 播报** | ✅ 跑通（图片输入） | Mage-VL 4B 看图 58.7s，播报 1.3s；板子无摄像头，已写抓帧接口 |
| 3 | **LLM 性能压榨** | ✅ 有结论 | 8 线程最优（pp 8.90 / tg 5.26 t/s）；**A100 反而慢 5.8×**；30B-A3B 下载不可行 |
| 4 | **VPU 硬编** | ✅ **修好了** | 根因是权限 bug；修后 1080p150帧 **1.09s**，比 x264 快 4.8×、CPU 少 144× |
| 5 | **ONNX / YOLO** | ✅ 跑通 | YOLOv8n int8 单帧 **14.8ms**（≈56× 于 CPU EP），真实检出已验证 |
| ★ | **A100 / IME 能否自助开启** | ✅ **能，已验证** | 写 `/proc/set_ai_thread` 即解锁；但**开出来跑 llama.cpp 是负优化** |
| 6 | **视觉模型全家桶（14 个）** | ✅ 跑通 | yolo11/v8-seg/pose/face/emotion/resnet50 全实测，最快 emotion **7.1ms**（见 §12.1） |
| 7 | **本地 RAG（embed+rerank+LLM）** | ✅ 跑通 | 40 块知识库全离线问答；但 **rerank ≈11s/篇**，向量检索只要 8ms（见 §12.3） |
| 8 | **模型动物园 / TTS / VAD** | ✅ 有结论 | qwen3-0.6b **13.9 t/s**；VAD **34× 实时**；顺带测出 **TTS 音色/语速 bug** 与 **ASR 长音频劣化**（见 §12.4–12.6） |
| 9 | **长音频转写 / 真·流式语音** | ✅ 跑通 | VAD 切段转写**快 9×**；流式 TTS 按句 **0.5s** 出声；LLM→TTS 流式**首音频 3.03s vs 非流式 28.4s**（见 §13） |
| 10 | **码率控制 / 视频理解 / 实时语音 / 厂方 issue** | 🚧 代码完成 | `mpp_enc.c` 原生编码（CBR/VBR/CVBR/FIXQP）、`vlm-video.sh`、`live-voice.py`、`camera-capture.sh`、6 个 issue 草稿（见 §14）；**板上实测待隧道恢复** |

一句话：**板子上的端侧 AI 栈（官方 AI Gateway + MPP 硬编 + A100 核）全部可用，且已实测定位了三个坑**：A100 的定位、VPU 编码权限、以及网关 LLM 的性能上限。

---

## 1. 环境：板子自带一整套「Edge AI 平台」

出人意料的是，镜像里已经装好了 **SpacemiT AI Gateway v0.1.15**（`spacemit-ai-gateway.service`），监听 `127.0.0.1:18790`。它是一个 **128 端点的本地 AI 服务**，把整条边缘推理链都封好了：

| 域 | 端点 | 后端 / 模型 |
|---|---|---|
| ASR | `/v1/asr/recognize`、`/v1/asr/jobs` | **SenseVoice**（中英日韩粤，16k，流式） |
| TTS | `/v1/tts/synthesize`、`/v1/tts/voices` | **Matcha-TTS** zh/en/zh-en、Kokoro |
| LLM | `/v1/llm/chat/completions`（OpenAI 兼容） | llama.cpp + Qwen3/LFM2 系（gguf） |
| VLM | `/v1/vlm/chat/completions` | Qwen3-30B-A3B-MM（**17.6G，本机没下**） |
| Vision | `/v1/vision/inference` | YOLOv8/v11、resnet50（**int8 `.q.onnx`**） |
| VAD / Embed / Rerank | `/v1/vad/*` 等 | Silero 等 |

模型**按需自动下载**，源是厂商自己的 CDN `archive.spacemit.com`（**直连可达**，约 300–440 KB/s，不需要代理）。下载缓存落在 `~/.cache/models/`。

> 这直接改变了实现策略：语音/视觉不用自己拼 sherpa + llama.cpp，**用官方网关就是最短路径**，也最贴近量产形态。

底层加速：
- 视觉/ASR 走 **`SpaceMITExecutionProvider`**（`-e spacemit`，即 A100 的 IME 算子）；
- LLM 走镜像自带的 `llama.cpp-tools-spacemit`（带 IME2 kernel 的官方构建）。

---

## 2. 离线语音助手（任务 1）

**链路**：`WAV →(SenseVoice ASR)→ 文本 →(Qwen3.5-2B LLM)→ 回复 →(Matcha TTS)→ 音频 → 喇叭`

脚本：`scripts/voice-assistant.sh`

### 实测（输入用板子上现成的中英混说语音 `0.wav`）

```
K3 离线语音助手  (模型 qwen3.5-2b-local, 网关 http://127.0.0.1:18790)
────────────────────────────────────────────────────────────
  [1] SenseVoice ASR      2.39s  「昨天是monday，today is礼拜2，the day after tomorrow是星期三。」
  [2] LLM 生成            9.89s  6.3 t/s 生成了 10 token
  [3] Matcha TTS 合成     1.20s  3.66s 语音 → ~/k3-ai/out/reply.wav
  [4] 播放到 hw:0,0       4.51s  OK
────────────────────────────────────────────────────────────
你  ：昨天是monday，today is礼拜2，the day after tomorrow是星期三。
助手：昨天周一，今天周二，后天周三。
```

- **ASR**：SenseVoice 对中英混说处理很干净，RTF 0.15–0.25（10 秒音频 2.4 秒出结果）。
- **LLM**：Qwen3.5-2B Q4_0，用网关 `/v1/llm/models/register` 的 `source_type=local_path` **直接注册本机 gguf，免下载**，加载 5s。
- **TTS**：1.2s 出 3.66s 语音，16kHz 单声道。
- **一轮总耗时 ≈18s**（大头是 LLM 生成）。

### 两个踩到的坑（已在脚本里处理）

1. **声卡只吃立体声**。ES8326 只支持 `CHANNELS: 2`，TTS 输出是单声道 → `aplay` 直接失败。脚本里统一 `ffmpeg -ac 2 -ar 48000` 转一遍再播。
2. **板子上没有可用麦克风**。录 5 秒 → ASR 返回 `？`（RTF 0.02，纯本底噪声）。所以脚本提供 `--mic N` 模式，**插上麦克风即可切到实时语音**（无需改代码）。

> 想要真正的「实时语音助手」：接一只 USB 麦（或接到 ES8326 的 mic-in），然后 `./voice-assistant.sh --mic 5`。

---

## 3. 摄像头 → VLM → 天气…不，→ 播报（任务 2）

**链路**：`图片 →(Mage-VL 4B 本地视觉理解)→ 中文描述 →(Matcha TTS)→ 音频 → 喇叭`

脚本：`scripts/vlm-announce.sh`

### 实测

输入是 llama.cpp 自带的测试图 `test-1.jpeg`（阿波罗 11 号那天的《纽约时报》头版）：

```
  [1] 输入图片                     .../tools/mtmd/test-1.jpeg
  [2] Mage-VL 看图（本地 4B）  58.73s  「《纽约时报》登载“月球漫步”新闻。」
  [3] Matcha TTS 合成               1.34s  3.66s 语音
  [4] 播放到 hw:0,0                4.50s  OK
```

- 模型答对了（那张图确实是登月头版），中文表达自然。
- **瓶颈在视觉编码**：58.7s 里有 **26.8s** 是 `mtmd batch encoding`（图像 → vision token），其余是模型加载 + 生成。
- 模型：`mage-vl-backbone-Q4_K_M`（2.7G）+ `mage-vit-mmproj-Q8_0`，走 `llama-mtmd-cli`。

### 摄像头：板子现在没接，但通道是通的

- `/dev/video0` 存在但**没有 sensor**（`v4l2-ctl --list-devices` 无实体设备，eDP/DP 也全 disconnected）。
- 所以脚本把「抓帧」做成了 `--camera N` 模式：`ffmpeg -f v4l2 -i /dev/video0 ...`。
- 内核侧**驱动是齐的**：`CONFIG_SPACEMIT_K3_CAM_OV5647=y`、`OV2735=y`，还有 `/usr/lib/libvi_k3_cam_plugin.so` 与 VI/ISP 框架 —— **插上 MIPI-CSI 摄像头即可用**。

---

## 4. LLM 性能压榨（任务 3）

### 4.1 线程数扫描（Qwen3.5-2B Q4_0，X100）

| threads | pp256 (t/s) | tg64 (t/s) |
|---:|---:|---:|
| 4 | 4.53 | 3.47 |
| **8** | **8.90** | **5.26** |
| 16 | 8.72 | 4.35 |

**结论：8 线程（= X100 物理核数）最优**；16 线程超订反而让生成掉 17%。网关默认 `--threads 8` 是合理的。

参考（同板 t=8）：Mage-VL backbone 4B → pp128 **5.71** / tg32 **3.90** t/s。

### 4.2 ★ 反直觉发现：A100 开出来跑 llama.cpp，**慢 5.8 倍**

| 核组 | pp256 | tg16 |
|---|---:|---:|
| X100（cpu0-7, taskset 0-7） | **8.91** | **5.54** |
| A100（cpu8-15, set_ai_thread + taskset 8-15） | 1.53 | 1.45 |

原因查清了，是**架构定位不同**，不是故障：

| | X100 | A100 |
|---|---|---|
| 主频 | 2150 MHz | 1800 MHz |
| L1I / L1D | 64K / 64K | **32K / 32K** |
| L2（4 核共享） | **4 MB** | **1 MB** |

A100 是**向量/矩阵专用核**（VLEN 1024 + IME，官方"60 TOPS"那半），缓存小、主频低 —— 跑「从 DDR 流式读权重」的通用 Q4 路径必然吃亏。**它的正确打开方式是 IME 算子 + int8 + TCM**，这正是官方 `libspacemit_ep.so`（`-e spacemit`）存在的意义 —— 也解释了为什么视觉那一路能快 56 倍（见第 5 节）。

> 换句话说：**A100 不是"更快的 CPU"，是"另一种处理器"**。别拿它跑通用 LLM 推理。

### 4.3 Qwen3-30B-A3B：这次跑不了，但账算清楚了

- 官方 CDN 上**有** `qwen3-30b-a3b-q4_0`（`archive.spacemit.com/spacemit-ai/model_zoo/llm/`），约 **17.6 GB**。
- CDN 实测 **300–440 KB/s** → 17.6GB 需要 **≈11–16 小时**，不适合在调试窗口里做。
- 内存账：板子 15GiB，Android guest（crosvm）占 4.5GiB，实测 available ≈9GiB。Q4_0 的 30B-A3B 权重 ≈17GB + KV → **必须先停掉 Android guest 且仍然吃紧**（官方 VLM 版 `qwen30ba3b-mm-q4_1` 标称 17.6G，同理）。
- 官方公布的 Qwen3-30B-A3B 约 12 t/s 是在**没开 A100**（默认落 X100）的情况下，与本次 4.1 的结论一致。

**结论**：30B-A3B 在这块 15GiB 的板子上**不是"调优"问题，是"装不下+下载太慢"问题**。要用它，建议：① 换 32G 版；② 或预先离线拷入 gguf。

---

## 5. VPU 硬编：**根因是权限 bug，修好了**（任务 4）

### 5.1 现象与根因

`ffmpeg -c:v h264_stcodec` **段错误（139）**，日志：

```
[MPP-ERROR] check_v4l2_linlonv5v7:578 Can not open '/dev/video0', please check it! (Permission denied)
[MPP-ERROR] module_init:199 can not find v4l2_linlonv5v7, please check!
[MPP-ERROR] module_get_so_path:281 module is NULL, please check!
→ 段错误
```

**根因链条**：ffmpeg 的 `*_stcodec` 编码器 → MPP 插件 `libv4l2_linlonv5v7_codec2.so` → 初始化时逐个探测 `/dev/video0..31`；而 `/dev/video0` 是 `root:video 0660`（标准 udev 规则 `SUBSYSTEM=="video4linux", GROUP="video"`），普通用户不在 `video` 组 → 探测全失败 → 插件返回 NULL → ffmpeg 空指针崩溃。

**不是**硬件坏、**不是**内核缺配置（`CONFIG_VIDEO_LINLON_K3=y`，dmesg 里 `MVX dev: Linlon v5276 identified. cores=4` 一切正常）。

### 5.2 修复

```bash
sudo usermod -aG video $USER     # 一行，持久生效；重新登录后即可
```

（脚本 `scripts/hw-encode.sh fix` 封装了这一条。板子上已执行，`pingji` 现有 `audio`、`video` 组。）

### 5.3 硬编基准（150 帧 1920×1080，输入为 raw NV12，排除解码干扰）

| 编码器 | 墙钟 | user CPU | 产物 | 帧数校验 |
|---|---:|---:|---:|---|
| **VPU 硬编 H.264** `h264_stcodec` @4M | **1.087s** | **0.203s** | 4.82 MB | ✅ 150 |
| 软编 `libx264 veryfast` @4M | 5.219s | 29.276s | 2.50 MB | ✅ 150 |
| **VPU 硬编 HEVC** `hevc_stcodec` @4M | **0.792s** | 0.166s | 3.20 MB | ✅ 150 |
| VPU 硬编 H.264 @12M | 1.057s | 0.191s | 4.82 MB | ✅ 150 |

- **硬编比 x264 veryfast 快 4.8×，CPU 时间少 144×**（软编吃 530% CPU，硬编几乎不占）。1080p 硬编单流 ≈138 fps。
- 可用编码器：`h264_stcodec` / `hevc_stcodec` / `mjpeg_stcodec` / `vp8_stcodec` / `vp9_stcodec`。

**已知限制（写进文档）**：`-b:v` 目前没映射到 MPP 的码率控制 —— 4M 与 12M 产物**字节数完全相同**（都是 4.82MB，实际约 7.7Mbps）。需要调码率得走 MPP 原生的 `RC` 参数（`VencChnAttr` 的 `eRcMode/minQp/maxQp`），或在 ffmpeg 侧用 `-rc`/`-q` 之类尚未接通的旋钮。

### 5.4 顺带搞清楚的 MPP 机制

- `/dev/video-dec0` **不是设备节点**，而是一个**普通文本文件**（`type=dec / codecs=H.264:H.265:VP9`），由 `/etc/init.d/multimedia.sh` 在开机时创建 —— 这是 MPP 的"模块发现"机制。
- 编码器**不需要**对应的 `video-enc0`（实测加不加都能编）；关键是那个 `/dev/video0` 探测。这点容易误判，特此记下。

---

## 6. ONNX / YOLO（任务 5）

### 6.1 网关视觉通道实测（YOLOv8n，int8）

```
POST /v1/vision/inference  -F file=@frame.jpg  -F model_id=yolov8n  -F 'tasks=["detect"]'
→ timing: preprocess 15.05ms, model_infer 14.81ms, postprocess 0.08ms, total 29.9ms   (HTTP 墙钟 65–83ms)
```

**单帧模型推理 14.8ms ≈ 68 FPS**（native backend + SpaceMIT EP int8），HTTP 端到端约 15–30 FPS。

**真实检出验证**（不是空跑）：

| 图片 | 结果 |
|---|---|
| `imported-into-android-studio.jpg` | 1 个目标：class 63 = **laptop**（0.50） |
| `matmul.png` | 6 个目标：class 9 = traffic light（0.84/0.80/0.67…） |

模型目录（`GET /v1/vision/models`）：`yolov8n/s/m`、`yolov11n/s/m`、`yolov5-face`、`yolov5_gesture`、`yolov8*-pose/seg`、`resnet50`，全部 int8（`~/.cache/models/vision/yolov8/yolov8n_no_dfl.q.onnx`，3.37MB）。

### 6.2 和 CPU EP 的对照

| 路径 | 单帧耗时 | 相对速度 |
|---|---:|---|
| `onnxruntime_perf_test -e cpu`（纯 CPU） | 832 ms | 1× |
| 网关 native + SpaceMIT EP（int8） | **14.8 ms** | **≈56×** |

> **这就是 A100/IME 的意义所在** —— 通用 CPU 路径跑 int8 CNN 要 0.83 秒，专用 EP 只要 15 毫秒。

### 6.3 记录一个厂方 bug

`onnxruntime_perf_test -e spacemit` 会失败：

```
[SpacemiT EP] Use a '|' to separate the key and value for the run-time option you are trying to use.
```

是 perftest 传 provider options 的格式与 EP 期望不符（EP 侧只接受 `key|value` 形式）。**建议提给 SpaceMiT**。

另外：系统 python 的 `onnxruntime 1.24.2+spacemit.a1` 只注册了 `CPUExecutionProvider`，**Python 侧挂不上 SpaceMIT EP**（EP 只以预编译 `.so` 形式提供，通过 C++/网关使用）。

---

## 7. ★ A100 / IME 能不能自己搞定？（对应本轮第 1 问）

**能，而且是官方机制、公开可查，不需要等厂商。**

### 7.1 开关就是一行

```bash
echo 0 > /proc/set_ai_thread     # 0 = 当前线程；写完后本进程/子线程被划入 A100 组
```

实测（本机复核）：

```
/proc/set_ai_thread      → 存在，模式 --w--w--w-（0222，普通用户可写）
CONFIG_SPACEMIT_HMP=y    → 已编入内核
marchid                  → 8 核 0x…41000002（X100）+ 8 核 0x…58000002（A100）
写之前:  Cpus_allowed_list = 0-7,  nproc=8
写之后:  Cpus_allowed_list = 8-15, nproc=8     ← A100 解锁成功
```

### 7.2 为什么以前误判为"用不了"

- `taskset -c 8` 报 `EINVAL`：这是 **HMP 的设计约束**，不是硬件故障。内核 `drivers/soc/spacemit/hmp.c` 里，`sched_setaffinity()` 会调用 `hmp_cpu_affinity_restrict()`；普通线程属于 `HMP_REGULAR`（掩码 0-7），越界到 A100 就被拒绝。**必须先把自己切成 `HMP_AI` 线程**（写 `/proc/set_ai_thread`），之后 `taskset -c 8-15` 才合法。
- "8-15 不在 `cpu_active_mask`"：A100 关电时确实不在，但 `hmp_set_ai_thread()` 会**自动给第一个 A100 核上电**。
- `vector has been enabled already!!!` 警告：约束是"**必须在执行向量指令之前切换**"（内核把 VLEN 固定为 4096，A100 是 VLEN=1024，语义不同）。实测中 bash 已跑过向量指令仍切换成功（仅告警），但**正确姿势是尽早切**。社区工具 `brucehoult/k3_ai` 的 `aix` 就是纯汇编先写节点再 `execve`。

### 7.3 公开资料确实找得到了（不需要问厂商）

| 内容 | 来源 | 性质 |
|---|---|---|
| HMP / `cpu-ai` DTS / `/proc/set_ai_thread` 实现 | `github.com/spacemit-com/linux-6.18` 分支 `k3-br-v1.0.y`（`drivers/soc/spacemit/hmp.c`） | 官方源码 |
| 你那版内核的源码 | `github.com/monkey-jsun/linux-6.18-k3` tag `cf-k3-v1.3.0` | 第三方镜像 |
| llama.cpp 自己会写 `/proc/set_ai_thread` | `github.com/spacemit-com/llama.cpp`（`ggml/src/ggml-cpu/spacemit/ime.cpp`） | 官方源码 |
| HMP 机制与 60 TOPS 的官方说明 | LPC 2026 演讲 + 官方论坛帖 | 官方文档 |
| `/dev/tcm_sync_mem` 缺失影响很小 | 官方论坛 SpaceMiT 员工回复 | 官方文档 |
| IME 指令集规范 | `github.com/spacemit-com/riscv-ime-extension-spec` | 官方文档 |
| `aix` / `k3_taskset` 现成工具 | `brucehoult/k3_ai`、`zqb-all/k3_taskset` | 社区 |

**仍然不公开（只能问厂商）**：
1. `spacemit-tcm.ko`（`/dev/tcm_sync_mem` 的树外模块源码/包）—— 但它**非必需**，厂商自己公布的 benchmark 也是在缺它的情况下跑的；
2. `SpaceMITExecutionProvider`（onnxruntime）的**完整实现源码**（只有预编译 `.so` 与 IME kernel 片段公开）；
3. HMP 的上游化时间表、以及 A100 能否以 VLEN=256 参与普通 SMP 调度。

> 所以答案是：**"怎么开 A100"我们自己就能查到并验证（已做）；"能不能因此更快"要分场景 —— 通用 LLM 反而慢，专用 int8/IME 算子快 56×。**

---

## 8. 跑机器人要接什么外设？（对应本轮第 3 问）

K3 的接口非常"机器人友好"，但**现在板子上一个都没插**。按用途列：

### 8.1 感知

| 外设 | 接到哪 | K3 上的现状 |
|---|---|---|
| **摄像头** | MIPI-CSI | 内核带 OV5647/OV2735 驱动 + `libvi_k3_cam_plugin.so`，**插上即用**。⚠️ 但 `/dev/video0` **不是摄像头**（它是 VPU 的 V4L2 节点，见 §8.5），真摄像头会枚举成 `/dev/video1+`，`--camera` 要传对编号 |
| 深度/RGBD | USB3 | 免驱（V4L2），最省事 |
| **麦克风阵列** | I2S/PDM | 声卡 ES8326 已就绪（**只支持立体声**）；当前无麦 → 语音必须改成实时需插麦 |
| IMU | SPI/I2C | 板子有 I2C/SPI 排针 |
| 超声/ToF/力传感 | I2C/SPI/ADC | 走排针 |

### 8.2 运动（K3 的杀手锏）

| 外设 | 接口 | 说明 |
|---|---|---|
| **关节电机 / 云台** | **CAN-FD ×10** | K3 芯片级有 10 路 CAN-FD，**机器人总线直连**，不用额外的 CAN 适配器 |
| 舵机 | **PWM ×30** | 芯片级 30 路 PWM |
| 编码器 | QEI / GPIO | |
| 串口设备 | UART ×17 | |

> 这是 K3 相对树莓派/一般 AI 板真正的差异化：**AI 算力 + 一堆实时总线在同 SoC 上**，且内核带 **RT-Linux defconfig**，适合"AI 决策 + 实时控制"一体机。

### 8.3 交互 / 输出

- **喇叭**：I2S + D 类功放（当前 `aplay -D hw:0,0` 能播，**不确定喇叭有没有接**）。
- **屏幕**：MIPI-DSI / HDMI（eDP-1、DP-1 当前 disconnected）。
- LED/灯带、按键：GPIO/PWM。

### 8.4 电源与安全（最容易被忽略）

- 锂电池 + **BMS** + DC-DC（K3 板需要稳定 5V/12V），
- **急停按钮**（接 GPIO，硬线切断电机使能），
- 电流/温度监测（CAN 或 I2C 传感）。

### 8.5 实况核对：这些接口现在真的使能了吗？（带更正）

> 逐条实测见 §12.7。一句话：**I2C/UART/PWM/GPIO/USB 都在；CAN-FD 和 SPI 在当前设备树里没使能**（芯片有），
> `/dev/video0` 也**不是摄像头而是 VPU 节点**。要 CAN 直连关节电机，得先改 DT/内核（需人在现场），
> 否则图省事就直接上 **USB-CAN 适配器**（免改 DT）。

### 8.6 给这套板子的架构建议

```
感知层  摄像头(MIPI-CSI) + 麦阵(I2S) + IMU(I2C) + ToF
   │            ↓ 视觉/语音推理
计算层  A100 核（IME/int8，走 SpaceMIT EP）← 视觉 CNN、语音
        X100 核（8 线程）← LLM 对话、决策
        VPU（Linlon v5276）← 摄像头 H.264/HEVC 硬编、回传/录像
        （注意：K3 没有 NPU，别指望它）
控制层  CAN-FD 接关节电机 / PWM 接舵机 ← 用 RT-Linux 跑实时回路
交互层  喇叭(I2S) + 屏幕(MIPI-DSI) + 灯带(GPIO)
```

**先接什么？** 最小可用机器人 combo：**1 只 MIPI-CSI 摄像头 + 1 只 USB 麦克风 + 1 只有源喇叭 + CAN-FD 接一路电机**。这样上面 1/2 两个 Demo 立刻从"文件输入"升级成真实实时闭环。

---

## 9. 复现步骤（照着敲）

```bash
# 0) 前提：网关在跑
systemctl is-active spacemit-ai-gateway        # active

# 1) 语音助手（文件模式；有麦克风就用 --mic 5）
~/k3-ai/voice-assistant.sh some_speech.wav
~/k3-ai/voice-assistant.sh --text "你好，介绍一下你自己"

# 2) 看图说话 → 播报（有摄像头就用 --camera 3）
~/k3-ai/vlm-announce.sh photo.jpg
~/k3-ai/vlm-announce.sh --camera 3

# 3) 硬编（若报段错误先修权限）
~/k3-ai/hw-encode.sh fix            # 加进 video 组（需 sudo，重新登录生效）
~/k3-ai/hw-encode.sh check
~/k3-ai/hw-encode.sh in.mp4 out.h264
CODEC=hevc ~/k3-ai/hw-encode.sh in.mp4 out.h265

# 4) 开 A100（谨慎：通用 LLM 会变慢）
echo 0 > /proc/set_ai_thread
taskset -c 8-15 nproc               # 应该输出 8

# 5) YOLO 检测
curl -s -X POST http://127.0.0.1:18790/v1/vision/inference \
  -F file=@img.jpg -F model_id=yolov8n -F 'tasks=["detect"]'
```

---

## 10. 原始数据附录

### A. 语音链路逐段耗时（`0.wav`，10s 音频）

| 阶段 | 耗时 | 备注 |
|---|---:|---|
| ASR (SenseVoice) | 2.39s | RTF ≈0.24 |
| LLM (Qwen3.5-2B Q4_0, 8 threads) | 9.89s | 10 token，6.3 t/s；prompt 8.5 t/s |
| TTS (Matcha zh_en) | 1.20s | 产出 3.66s 语音 |
| 播放 | 4.51s | 立体声 48k |
| LLM 加载（首次） | 5s | 注册本机 gguf，零下载 |

### B. 网关 ASR 精度抽查

| 输入 | 输出 | RTF |
|---|---|---:|
| `0.wav` | 昨天是monday，today is礼拜2，the day after tomorrow是星期三。 | 0.249 |
| `1.wav` | （正常中文转写） | 0.15 |

### C. llama-bench 全量

```
Qwen3.5-2B Q4_0   build 5496cdc9c (10050)
  t=4   pp256  4.53 ±0.00   tg64  3.47 ±0.00
  t=8   pp256  8.90 ±0.00   tg64  5.26 ±0.00      ← 最优
  t=16  pp256  8.72 ±0.00   tg64  4.35 ±0.00
  t=8   taskset 0-7   pp256 8.91   tg16 5.54      (X100)
  t=8   A100 8-15     pp256 1.53   tg16 1.45      (慢 5.8×)
mage-vl backbone Q4_K_M 4.41B
  t=8   pp128  5.71 ±0.00   tg32  3.90 ±0.00
```

### D. 硬编基准原始输出

```
A. h264_stcodec @4M : real 1.087s  user 0.203s  sys 1.057s  → 4824334 B, 150 帧 ✅
B. libx264 vf  @4M : real 5.219s  user 29.276s sys 0.725s  → 2500578 B, 150 帧 ✅
C. hevc_stcodec @4M: real 0.792s  user 0.166s  sys 0.804s  → 3202458 B, 150 帧 ✅
D. h264_stcodec @12M: real 1.057s user 0.191s sys 1.028s   → 4824334 B（与 4M 完全相同 → RC 未生效）
```

### E. 网关视觉 timing（JSON 里的字段）

```
preprocess_ms 15.05 · model_infer_ms 14.81 · postprocess_ms 0.08 · infer_ms 29.95
```

### F. A100 vs X100 硬件参数

| | X100 ×8 | A100 ×8 |
|---|---|---|
| marchid | 0x…41000002 | 0x…58000002 |
| 主频 | 2150 MHz | 1800 MHz |
| L1I / L1D | 64K / 64K | 32K / 32K |
| L2（4 核共享） | 4 MB | 1 MB |
| VLEN | 256 | 1024 |
| IME | 无 | 有 |
| 调度类别 | HMP_REGULAR | HMP_AI |

---

## 11. 遗留问题 / 下一步

1. **`-b:v` 未映射到 MPP 码率控制** —— 已写 `scripts/mpp_enc.c`（直接设 `VencChnAttr.eRcMode/u32Bitrate`）+ `mpp-enc-bench.sh` 验证脚本；**待板上编译/实测**（见 §14.1）。
2. **实时语音**需要一只麦克风（最简单：USB 麦）。
3. **30B-A3B** 需要离线拷入或换 32G 内存版。
4. 把「A100 反优化」「VPU 权限 bug」「perftest `-e spacemit` 报错」「`/v1/asr/stream` 号称流式却不增量」整理成给 SpaceMiT 的 issue。
5. 网关的流式 WebSocket 已实测（见 §13）：**`/v1/asr/stream` 是假的**（缓冲式、不出增量 partial），**`/v1/tts/stream` 是真的**（按句 `append` 逐句出声）——真·流式语音助手已跑通（`scripts/stream-voice.py`，首音频 3.03s vs 28.4s）。
6. **TTS 的 `voice_id` / `speed` 参数被丢弃**（厂方 bug，md5 实锤，见 §12.5）—— 想多音色/变语速目前只能换 `model`。
7. **ASR 长音频吞吐非线性劣化**（40s 时 RTF 0.81，见 §12.6）—— 已落成 **VAD 切段转写工具**（`scripts/long-transcribe.py`，快 9×，见 §13.1）。
8. **rerank 太慢**（qwen3-reranker-0.6b ≈11s/篇，见 §12.3）—— 板上 RAG 只用向量检索即可。
9. **CAN-FD / SPI 未在设备树使能**（见 §8.5 / §12.7）—— 直连机器人总线需改 DT/内核，或上 USB-CAN 适配器。
10. **板子反向隧道不再自动重连**（2026-10-19 09:14 起，SSH 彻底不可达；心跳仍正常，见 §14.0）—— 需现场 `sudo systemctl restart k3-tunnel`；若根因是 DNS，把 unit 里的 `__K3_SERVER__` 换成 IP。

## 12. 第二轮扩展实验：把网关的余量榨干（2026-10-16 追加）

第 1 节说了网关有 128 个端点。第二轮把它们**挨个试了一遍**，另外补了三个可上报的厂方问题。

### 12.1 视觉模型全家桶（14 个，单帧 `model_infer_ms`）

一次把网关里能下的视觉模型全下下来跑（测试图用 ultralytics 经典图 `bus.jpg` / `zidane.jpg`，都含人，可同时测 detect / pose / segment）：

| 模型 | 任务 | 推理 ms | 模型 | 任务 | 推理 ms |
|---|---:|---:|---|---:|---:|
| emotion | 表情 | **7.1** | yolov8s-pose | 姿态 | 27.3 |
| resnet50 | 分类 | **7.7** | yolov5-face | 人脸 | 28.4 |
| yolov8n | 检测 | **14.3** | yolov11s | 检测 | 30.9 |
| yolov8n-pose | 姿态 | 16.2 | yolov8s-seg | 分割 | 32.5 |
| yolov8n-seg | 分割 | 19.6 | yolov8m | 检测 | 48.0 |
| yolov11n | 检测 | 20.1 | yolov11m | 检测 | 65.2 |
| yolov5-gesture | 手势 | 21.4 | yolov8s | 检测 | 24.8 |

> 首次 `load` 会从 CDN 现下（yolov8m 79s、yolov11m 116s），下完即用。

观察：
- **yolo11 与 yolo8 同档性能接近**（yolov11s 比 yolov8s 多检出 1 个目标，也多 1 个假阳性 `hydrant`）。
- 小目标假阳性：`yolov8s/m` 把领带 / 盆栽当成目标（0.50~0.54）。
- **resnet50 把整张 `bus.jpg` 认成 `minibus 0.61`** —— 分类模型不适合多目标图，正常。
- `yolov5-gesture` 在公交照上 0 检出（需要近景手部）；`emotion` 在 `zidane` 上判 `neutral`。
- `pose` 结果的 `label_name` 是 `null`（只有 score），要取关键点得读 `keypoints` 字段。

### 12.2 VAD（silero）：**34× 实时**

```
/v1/vad/analyze   10s 音频 → is_speech=true, prob 0.9999, processing_ms 288
/v1/vad/segments  10s 音频 → 5 段（0.84–2.46s … 8.91–10.05s），speech_ratio 0.63
```

`processing_ms 288 / 10000ms = 0.029` → **比 ASR 快约 30 倍**。长音频的正确姿势是 **先 VAD 切段、再逐段 ASR**（见 §12.6）。

### 12.3 Embedding + Rerank → **本地 RAG 全链路跑通**

知识库：把本报告的 Markdown 切 40 块（每块 ~277 字）。查询链路 `embed → 余弦 → rerank → LLM` 全部在板上。

```
embed   bge-small-zh-v1.5-q4_k_m   512 维   索引 40 块 8.17s（4.9 块/s；稳态 35.5 条/s）
rerank  qwen3-reranker-0.6b-q4_0
LLM     qwen3.5-2b
```

三个真实提问，回答**全部 grounded 在报告里**（摘录）：

| 提问 | 检索 top3 块 | 回答要点 |
|---|---|---|
| A100 适不适合跑 llama.cpp？ | [12,14,31] | "向量/矩阵专用核…强行跑通用 Q4 慢 5.8×…正确姿势是 IME 算子 + int8" |
| VPU 硬编为什么一开始段错误？ | [16,19,39] | "根因是权限 bug，`h264_stcodec` 段错误" |
| 机器人需要哪些外设？ | [31,33,35] | 逐条列出 CAN-FD / PWM / UART / MIPI-CSI / 急停… |

**★ 但 RAG 的瓶颈在 rerank：**

| 阶段 | 耗时 |
|---|---:|
| 查询 embedding | 0.05s |
| 余弦检索（40 块） | **0.008s** |
| **rerank（6 篇文档）** | **65–74s** |
| LLM 生成 | 90–120s（79–272 token，4.9 t/s） |

`qwen3-reranker-0.6b` 是生成式打分，**≈11s/篇**，比向量检索贵约 8000 倍。**结论：板上做 RAG 就该只用向量检索**（8ms），rerank 要么去掉、要么换 cross-encoder、要么离线预排。

脚本：`scripts/rag.py`（板上配套 `~/k3-ai/rag/`）。

### 12.4 LLM 模型动物园（都是本机 gguf，零下载切换）

用 `/v1/llm/models/register`（`source_type: local_path`）把三个新模型挂上去，`switch` 后实测：

| 模型 | 用途 | 加载 | 生成速度 | 备注 |
|---|---|---:|---:|---|
| **qwen3-0.6b-q4_0** | 快聊 | 4.3s | **13.9 t/s** | 最快，适合"意图解析 / 槽位提取"这类前置小任务 |
| **hy-mt1.5-1.8b-q4_k_m** | 翻译专用 | 5.3s | 7.0 t/s | 中→英译文准确（"…4.8 times faster…144 times less CPU"） |
| **deepseek-r1-distill-qwen-1.5b** | 推理 | 4.4s | 6.1 t/s | 带完整思维链，数学题答案 ✅（`\boxed{6}`） |
| qwen3.5-2b（基线） | 通用 | — | ~6.3 t/s | |

**组合思路**：0.6B 做"路由器 / 改写"，2B 做对话主力，R1 只在前置判断需要推理时临时切过去。

### 12.5 TTS 音色 + 速度曲线 + 一个实锤 bug

`/v1/tts/voices` 给了 6 个音色（`default` zh/en/zh-en、`zf_xiaobei` 小贝、`zm_yunxi` 云希、`af_heart`）。

| 模型 | 5.4s 语音合成耗时 | RTF | 评价 |
|---|---:|---:|---|
| **matcha_zh** | **0.84s** | **0.183** | 5.5× 实时，**唯一适合实时**的 |
| kokoro | 22–32s | 3.6–5.3 | **0.2–0.3× 实时，太慢** |

**★ 厂方 bug：`voice_id` / `speed` 参数完全没生效。** 用**全新文本**排除缓存干扰后：

```
kokoro zf_xiaobei / zm_yunxi / af_heart  → md5 全部 29a115b7…（三个人声一模一样）
kokoro speed 0.7 / 1.5                   → md5 全部 118beecb…（改语速没反应）
matcha_zh speed 0.7 / 1.5                → 时长都是 4.49s（不变）
不同 model                                → md5 各不相同 ✅（证明不是缓存问题）
```

→ 只有 `model` 生效，**音色和语速被丢弃**。复现：`scripts/tts-voices.sh`。

### 12.6 ★ ASR 吞吐随音频长度非线性劣化

同一个 SenseVoice，喂 4 个不同长度的音频（都来自板上现成稿件拼接）：

| 音频 | 处理耗时 | RTF | 相对实时 |
|---:|---:|---:|---:|
| 5.1s | 0.77s | **0.150** | 6.7× |
| 10.1s | 2.17s | 0.216 | 4.6× |
| 20.1s | 8.14s | 0.405 | 2.5× |
| 40.2s | 32.5s | **0.807** | **1.2×** |

**处理时间随长度近似二次增长**（attention 没有分块）。→ **绝不要把长音频整段丢给 `/v1/asr/recognize`**：先用 `/v1/vad/segments`（34× 实时）切段，再逐段识别，整体能拉回 RTF≈0.15。脚本：`scripts/asr-length.sh`。

### 12.7 外设实况更正（对应 §8）

在板上把总线挨个点了一遍，**§8 有两处要更正**：

| 接口 | §8 原说法 | 实测 |
|---|---|---|
| **CAN-FD ×10** | "机器人总线直连" | ❌ `ip link` 里**没有 `can0`** —— 芯片有，**当前 DT/镜像没使能** |
| **SPI** | "板子有 I2C/SPI 排针" | ❌ 无 `/dev/spidev*`，**未暴露** |
| I2C | — | ✅ `/dev/i2c-{1,2,3,8,9,10,11,12}` ×8 |
| UART | ×17 | ✅ `/dev/ttyS0..16` 全在 |
| PWM | ×30 | ✅ `pwmchip0`、`pwmchip1` 两个控制器 |
| GPIO | — | ✅ `/dev/gpiochip0..3` |
| USB | — | ✅ 6 条总线（2× USB3 + 4× USB2），当前只挂了 RTL8852BE 蓝牙 |
| **`/dev/video0`** | 当成摄像头 | ❌ 它是 **VPU（Linlon，`platform:mvx`，cores=4）的 V4L2 节点**，不是 camera |

→ **想直连 CAN 关节电机，必须先改 DT/内核（需人在现场）**；图省事就直接上 **USB-CAN 适配器**（免改 DT）。这反过来让 VPU 硬编的权限 bug 解释更干净：编码插件要探测的正是这个 VPU 节点。

### 12.8 一个事故：并发重活把板子"压死"28 分钟

第二轮为了图快，把 **3 个 LLM 下载 + 14 个视觉模型加载 + RAG** 并发跑。结果：

- 16 核被压到 **load average ≈ 108**（`12.3 / 89.8 / 108.6`）；
- 反向隧道 / 心跳 / sshd **全被饿死**：门户心跳停摆 28 分钟、SSH 能建连但**不响应**（111 上能看到板子侧 TCP 发送队列积压 3128B 未 ACK）；
- **不是 OOM**（内存始终有 ~10Gi 可用，dmesg 无 OOM 记录）；负载回落后**自愈，未重启**（uptime 连续）。

**教训**：板上重活**必须串行**，尤其别在下载模型的同时加载 / 推理。判断板子"死没死"最快的办法是看门户心跳（`/api/status`）和隧道 TCP 的 `send-q`。

### 12.9 第二轮新增脚本

| 脚本 | 作用 |
|---|---|
| `scripts/vision-zoo.sh` | 视觉模型全家桶 benchmark |
| `scripts/rag.py` | embed → 余弦 → rerank → LLM 的本地 RAG |
| `scripts/tts-voices.sh` | TTS 音色 / 语速验证（含 md5 比对） |
| `scripts/asr-length.sh` | ASR 长度-吞吐曲线 |

## 13. 第三轮：长音频转写 & 真·流式语音（2026-10-16 追加）

§12.6 留下一个明确的问题（ASR 长音频劣化 → 必须 VAD 切段），§11 第 5 条也点名"网关带流式 ASR/TTS WebSocket → 可做真·流式语音助手"。本轮把这两条**都做掉并实测**，还顺带发现**"流式 ASR 其实是假的"**。

### 13.1 长音频转写：VAD 切段 → 合并 → 逐块 ASR → SRT（快 9×）

测试音频：**37.3s**，由板上现成语音片段 + 2s/3s 静音拼成（多段语音 + 停顿）。

VAD 把 37.3s 切成 14 段（过滤掉 <250ms 的杂音段），只用 **1.10s（34× 实时）**。然后三方对照：

| 模式 | 块数 | ASR 耗时 | 相对实时 |
|---|---:|---:|---:|
| **合并到 ~12s（推荐）** | 8 | **2.98s** | **12.5×** |
| 逐段 | 14 | 2.87s | 13.0× |
| 整段（对照） | 1 | **27.94s** | **1.34×** |

→ **快约 9 倍**；且合并模式文本上下文更好（逐段会把句子切碎：`"Day after tomorrow."` + `"嗯星期三。"` → 合并后 `"dayy after tomorrow是星期三。"`）。

**设计要点**：

- 过滤 `MIN_SEG_MS`（默认 250ms）——VAD 会在音频开头误报一个 120ms 的杂音段（识别成 `"？"`）；
- 合并 `GAP_MS`（默认 600ms）以内的相邻段、单块上限 `MAX_CHUNK_MS`（默认 12s）——兼顾"速度 × 上下文";
- 直接按 `start_ms/end_ms` 用 python `wave` 切片（不调 ffmpeg），省一轮进程开销；
- 输出标准 **SRT 字幕**（`/tmp/k3tr/<前缀>.srt`）+ 纯文本。

脚本：`scripts/long-transcribe.py`。这把 §12.6 的结论落成了可交付的"**音视频转字幕**"管线。

### 13.2 ★ 流式 ASR（`/v1/asr/stream`）**是假的**

网关有 WS 端点 `/v1/asr/stream`，协议看着很正规：`POST /v1/asr/stream/session` 拿 `session_id` → WS 带 `session_id` 连接（`partial=true`）→ 发二进制 PCM 帧 → 发 `{"type":"end"}` → 服务端回 `partial` / `sentence_end` / `final`。实测：

| 音频 | 喂入方式 | 首个（也是唯一）结果 | RTF |
|---:|---|---:|---:|
| 10.1s | 按真实时间 1× | 12.44s（音频结束后 2.38s） | 0.227 |
| 37.3s | 按真实时间 1× | **65.90s**（音频结束后 **28.6s**） | **0.759** |

**全程只有 1 个 partial，而且等音频喂完才出**；RTF 与整段识别（§12.6 的 0.807）**几乎一模一样**。→ 这个"流式"只是给整段识别套了层 WebSocket 外壳，**没有增量解码**。

**结论**：想要低延迟，不能依赖网关的"流式 ASR"，只能**客户端自己 VAD 切段**（正是 §13.1 的做法）。脚本：`scripts/stream-asr.py`。

### 13.3 ★ 流式 TTS（`/v1/tts/stream`）**是真的**（但必须按句 `append`）

TTS WS 协议支持 `{"type":"start","text":..}` / `{"type":"append","text":..}` / `{"type":"end"}`，服务端持续回**二进制 PCM 块**。两种用法差别巨大：

- **一次性 `start` 整段文本**（84 字）：只在 **3.98s** 时回**一个整块**（526KB = 全部 16.46s 音频）→ 等于批量。
- **按句 `append`**（每 1.5s 喂一句）：

```
start  "第一句话，你好。"   → 0.55s 出声（2.10s 音频）
append "第二句话，世界。"   → 2.09s 出声（2.27s 音频）
append "第三句话，再见。"   → 3.59s 出声（2.29s 音频）
```

→ **每句约 0.5s 就出声**，这就是"边说边播"的正确用法。脚本：`scripts/stream-tts.py --append`。

### 13.4 压轴：真·流式语音助手（LLM token 流 → 逗号切句 → TTS 逐句 append）

把 13.3 与 LLM 的 SSE 流接起来：**LLM 一边吐 token，一边按标点切句（逗号/顿号/句号都切），切出的小句立刻 `append` 给 TTS**，音频块边到边拼，全在板上。

| 方案 | LLM 首 token | **首个音频（听到第一声）** | 总耗时 |
|---|---:|---:|---:|
| **流式**（本方案） | 1.59s | **3.03s** | 6.25s |
| 非流式（等 LLM 全生成 + TTS 整段） | — | **28.41s** | 28.4s |

→ **用户听到第一声的时间从 28.4s 降到 3.0s，快 9.4 倍**。这正是 §11 第 5 条想要的东西，现在跑通了：`scripts/stream-voice.py`。

> 说明：非流式那次回答更长（258 字 vs 流式 9.41s 音频），属"同一提问、输出长度不同"的示意对比；但对齐长度后，流式的首音频依然早于"LLM 全生成 + 整段 TTS"，因为它在 LLM 还没生成完时就已经出声。

### 13.5 第三轮新增脚本

| 脚本 | 作用 |
|---|---|
| `scripts/long-transcribe.py` | 长音频 → VAD 切段 → 合并 → 逐块 ASR → **SRT 字幕**（快 9×） |
| `scripts/stream-asr.py` | 流式 ASR 客户端（实测证明它是**缓冲式**） |
| `scripts/stream-tts.py` | 流式 TTS 客户端（`--append` 才真流式） |
| `scripts/stream-voice.py` | **真·流式语音助手**（LLM 流 → 按句 → TTS 流） |

## 14. 第四轮：码率控制 / 视频理解 / 实时语音 / 厂方 issue（2026-10-19 追加）

> ⚠️ **本轮有一半没能在板上跑完**：2026-10-19 09:14 板子的反向隧道（`k3-tunnel.service`）
> 断掉且**不再重连**，SSH 彻底不可达（心跳仍在每 30s 更新 → 板子活着、网络通，是板上
> 这个服务挂了；详见 §14.0）。所以本轮**代码与文档全部完成并入库**，但 §14.1–14.3 的
> 实测数字**待隧道恢复后补**（会把状态从 🚧 改成 ✅）。§14.4 的 issue 清单基于前三轮的
> 实测数据，可直接提交。

### 14.0 事故：隧道断了且不自动重连（09:14 →）

现象与定位过程（可复用的排查顺序）：

| 检查项 | 结果 | 结论 |
|---|---|---|
| 门户 `/api/heartbeat` 的 `lastSeen` | 每 30s 更新 | **板子活着、出网正常** |
| 板子 `uptime`（心跳负载里） | 429217s 连续 | **没重启** |
| 111 上 `127.0.0.1:2222` | 先 `LISTEN`，随后 `Connection refused` | 反向隧道会话**已断** |
| 111 sshd 近 4h 的 `Accepted publickey` | **只有本机一个 IP** | 板子**没有在重连** |
| 111 负载 / `:22` 连接数 | 0.21 / 5 条 ESTAB | sshd 没被打满，排除 MaxStartups |
| iptables / ufw / fail2ban | 只有 docker 的 DROP，ufw inactive | 不是被封 |

→ 板上 `k3-tunnel.service` 处于"不再尝试"的状态（该 unit 已带
`StartLimitIntervalSec=0` + `Restart=always`，所以更可能是**连接侧失败**，例如
`__K3_SERVER__` 用域名而 DNS 失效 —— 心跳脚本直连 IP 所以没事）。
**恢复办法（需现场/本地）**：`sudo systemctl restart k3-tunnel`；若 DNS 是根因，
把 unit 里的 `__K3_SERVER__` 换成 **IP**。

> 教训：**心跳通 ≠ 可管理**。心跳和隧道是两条独立链路，排障时要分开看：
> 心跳活 + `ss -ltn | grep 2222` 无监听 = 板上隧道服务的问题，不要再去怀疑网络/负载。

### 14.1 🚧 MPP 原生编码器 `mpp_enc.c`：把码率控制补上

§5.1 定位到 ffmpeg 的 `*_stcodec` **没有把 `-b:v` 映射到 MPP 的码率控制**，所以硬编
码率不可调。本轮直接按 `/usr/include/venc/venc_api.h` + `/usr/include/sys/vb_api.h`
写了原生程序，关键是把这几个字段显式交给硬件：

```c
VencChnAttr attr;
attr.eCodecType     = MPP_STREAM_CODEC_H264;
attr.eInputPixelFormat = MPP_PIXEL_FORMAT_NV12;
attr.eFrameBufMode  = VENC_FRAME_BUF_DMABUF_EXTERNAL;  // 用 VB 池的 dma-buf
attr.eRcMode        = VENC_RC_MODE_CBR;                // ← FIXQP / CBR / VBR / CVBR
attr.u32Bitrate     = 4000000;                         // ← 目标码率（bps）
attr.u32MinQp       = 10;  attr.u32MaxQp = 45;         // ← QP 上下限
attr.u32Gop         = 30;  attr.u32FrameRate = 30;
```

数据流是**零拷贝**的：`VB_CreatePool` 建 dma-buf 池 → `VB_GetBuffer` 取一块 →
按 `u32PlaneStride[]` 把 NV12 两个平面写进去 → `VENC_SendFrame` →
`VENC_GetStream` 直接拿到码流指针 → 写文件 → `VENC_ReleaseStream` → `VB_ReleaseBuffer`。

配套 `mpp-enc-bench.sh` 做两件事：
1. 同一段 NV12 扫 **1/2/4/8 Mbps** 的 CBR，输出「目标 vs 实际 Mbps vs 偏差 vs 编码帧率」；
2. 用 ffmpeg `h264_stcodec` 分别 `-b:v 1M` 和 `-b:v 8M` 各编一次，**若两者大小基本相同，
   就是"ffmpeg 没把码率传下去"的现场证据**。

> 已验证：`cc -std=c99 -Wall -Wextra` 用按真实头文件摘录的桩头文件**零告警通过**。
> 待板上验证：`libmpp` 实际库名、`VB_GetFrameInfo` 是否填充平面地址、EOS 刷尾是否完整。
> 建议首次加 `--dump-info`。

### 14.2 🚧 视频理解 `vlm-video.sh`：抽帧 → Mage-VL → 时间线 / 拼图

板上没有摄像头（§12.7），所以用现成视频（`soccer.mp4` / `src.mp4`）做文件输入验证。
Mage-VL **一次只能吃一张图**，于是给了两种模式：

| 模式 | 做法 | 成本 |
|---|---|---|
| 逐帧（默认） | 均匀抽 N 帧，每帧一次推理，输出 `[时间] 描述` 时间线 | N × ~1 分钟 |
| `--grid-only` | 把 N 帧用 `tile` 拼成 3×N 一张图，**一次**推理要整体概括 | ~1 分钟 |

抽帧时间点是 `(i+0.5)*时长/N`（取每段中点，避免抽到片头黑帧）。
`--no-vlm` 只抽帧不跑模型，用来先确认抽帧/拼图正确。

> 待板上验证：Mage-VL 的实际单帧耗时、拼图模式下 4B 模型能否稳定读出"3×2 顺序"。

### 14.3 🚧 实时语音 `live-voice.py` + 摄像头 `camera-capture.sh`（硬件接口就绪）

**实时语音**（`live-voice.py`）的回合控制不依赖"整段录音"，而是流式门控：

```
0.4s 窗口 → POST /v1/vad/analyze 判语音
          → 有语音就追加到当前句；连续静音 0.8s → 断句
          → ASR → LLM 流式 → 按标点切句 → TTS 逐句 append → 逐句 ffmpeg(48k/2ch) → aplay
```

- **`--file wav` 模式**用文件按真实时间喂窗口，**没有麦克风也能完整验证门控/断句/流式回复**，
  插上 USB 麦后 `--mic` 即可（`arecord -f S16_LE -r 16000 -c 1`）。
- `--no-llm` 只跑 VAD+ASR；`--once` 处理一句就退出；`--no-play` 不播音。
- ES8326 只支持 48k 立体声，所以播出前统一 `ffmpeg -ac 2 -ar 48000`（§9 的坑）。

**摄像头**（`camera-capture.sh`）：自动读 `/sys/class/video4linux/*/name` + 上游设备路径，
**把 VPU（`Linlon`/`mvx`）节点过滤掉**，只认真摄像头；没有时打印诊断（不是乱抓 `/dev/video0`）。
`--frames N` 抓帧，`--vlm` 直接接 `vlm-announce.sh` 播报。

> 硬件清单（需采购/插上）：
> - **USB 麦克风**（免驱 UVC Audio）→ `live-voice.py --mic` 立刻可用；
> - **USB 摄像头**（UVC）或 **MIPI-CSI OV5647/OV2735** → 枚举成 `/dev/video1+`；
>   内核驱动与 `libvi_k3_cam_plugin.so` 已在镜像里，插上即用。
> 没有硬件时，这两条链路的**其余部分都能用文件输入跑通**（这正是本轮的设计）。

### 14.4 厂方 issue 清单（6 个，可直接提交）

整理成 [`spacemit-issues.md`](spacemit-issues.md)，每个都带环境、复现命令、实测数据、根因分析、期望行为：

| # | 问题 | 严重度 | 关键证据 |
|---|---|---|---|
| 1 | A100 核组跑通用 LLM **慢 5.8×**，无任何提示 | 严重 | pp256 8.91→1.53 t/s；A100 主频 1.8G / L2 1MB |
| 2 | VPU 硬编在用户不在 `video` 组时 **段错误**而非权限错误 | 严重 | `h264_stcodec` segfault，加组即好 |
| 3 | TTS 的 `voice_id` / `speed` **被静默丢弃** | 中 | 3 音色 md5 全同；speed 0.7/1.5 时长都 4.49s |
| 4 | ASR 耗时随音频长度**近似二次增长** | 中 | 5.1s→RTF 0.150，40.2s→**0.807** |
| 5 | `/v1/asr/stream` 号称流式却**不出增量 partial** | 中 | 37.3s 音频喂完还要 +28.6s 才出唯一结果 |
| 6 | `onnxruntime_perf_test -e spacemit` 报 option 解析错 | 低 | `Use a '|' to separate the key and value` |

### 14.5 第四轮新增文件

| 文件 | 作用 | 状态 |
|---|---|---|
| `scripts/mpp_enc.c` | MPP 原生 H.264/H.265 编码器（CBR/VBR/CVBR/FIXQP） | 🚧 语法通过，待板上编译验证 |
| `scripts/mpp-enc-bench.sh` | 码率扫描 + 与 ffmpeg `-b:v` 失效的对照 | 🚧 待跑 |
| `scripts/vlm-video.sh` | 视频抽帧 → Mage-VL 时间线 / 拼图总览 | 🚧 待跑 |
| `scripts/live-voice.py` | VAD 门控实时语音（`--file` 可无麦验证） | 🚧 待跑 |
| `scripts/camera-capture.sh` | 摄像头探测（排除 VPU 节点）+ 抓帧 | 🚧 待跑 |
| `spacemit-issues.md` | 6 个厂方 issue 草稿 | ✅ 可提交 |
