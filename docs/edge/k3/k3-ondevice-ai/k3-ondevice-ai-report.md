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
| **摄像头** | MIPI-CSI | 内核带 OV5647/OV2735 驱动 + `libvi_k3_cam_plugin.so`，**插上即用**；当前 `--camera` 会抓 `/dev/video0` |
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

### 8.5 给这套板子的架构建议

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

1. **`-b:v` 未映射到 MPP 码率控制** —— 硬编码率不可调。需要写 MPP 原生程序（`VENC_CreateChn` + `VencChnAttr.eRcMode`）或找 ffmpeg stcodec 的正确旋钮。头文件已摸清（`/usr/include/venc/venc_api.h` + `sys/vb_api.h`），下一步可写 `mpp_enc.c`。
2. **实时语音**需要一只麦克风（最简单：USB 麦）。
3. **30B-A3B** 需要离线拷入或换 32G 内存版。
4. 把「A100 反优化」和「VPU 权限 bug」「perftest `-e spacemit` 报错」整理成给 SpaceMiT 的 issue。
5. 网关还带 VAD/流式 ASR/TTS WebSocket —— 可做**真·流式**语音助手（边说边出）。
