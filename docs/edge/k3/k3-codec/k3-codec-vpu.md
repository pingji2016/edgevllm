# K3 视频编解码 · VPU 硬解绕行 · 与 Mage-VL 的关系（全记录）

> K3 · riscv64 · Bianbu 4.0.1 · ffmpeg 8.0.1 · 2026-09-30

## 结论（TL;DR）

- **软件编解码**在 K3 上可用；**硬解**默认走 ffmpeg 会**挂死/段错误**。
- 我们**绕开 ffmpeg，直接用 SpacemiT 的 MPP C API 驱动 VPU** → **硬解跑通**：150 帧 / ~0.5s / **297fps / CPU 仅 0.11s**（软解要 2.87s）。
- **硬解对 Mage-ViT 没有加速意义**：Mage-VL 的瓶颈是 **Mage-ViT 编码（占 96%+）**，是纯矩阵/注意力算力，不是在解码视频；硬解只作用于链路里 ~26ms 的那一步。

---

## 1. 整体流程

### 1.1 视频理解链路（Mage-VL）

```
mp4 --> [H.264 解码] --> [codec-aware 抽帧/拼 canvas] --> [Mage-ViT 编码] --> [StreamMind EPFE] --> [分类器] --> silent/speak
         26ms             (cv-preinfer)                  68690ms            177ms               2230ms
```

### 1.2 编解码测试链路

```
                              ┌──────────── 软件：libx264/libx265（CPU 跑算法）
源视频(1080p30,150帧) ──编码──>┤
                              └──────────── 硬件：VPU（Linlon/mvx）硬流水线

encoded.mp4 ──解码──> ┌ 软件：native 解码器（CPU）
                      └ 硬件：VPU（MPP API，绕开 ffmpeg）
```

---

## 2. 原理：软编解码 vs 硬编解码

- **软件编解码**：CPU 跑编解码算法（x264/x265…）。**压缩率高、参数全可调**（CRF/preset/GOP/码控），但**吃 CPU、耗电**。
- **硬件编解码**：专用 **VPU** 硬流水线（K3 上是 VeriSilicon **Linlon(mvx)**）。**CPU≈0、可实时、省电**，适合 NVR/直播/转码/播放；代价是**压缩率略低、参数少、格式/分辨率受限**。
- 关键指标：软解吃满若干 CPU 核；硬解把活交给 VPU，CPU 基本空出来（能并发多路）。

---

## 3. 问题现象（默认用 ffmpeg）

| 操作 | 结果 |
| --- | --- |
| 硬解 `h264_v4l2m2m` | **进程卡死在内核 D 态**（`kill -9` 无效，只能重启/驱动返回） |
| 硬编 `h264_stcodec` / `hevc_stcodec` | **段错误**，`[MPP-ERROR] module_get_so_path: module is NULL` |
| 硬编 `h264_v4l2m2m` | `Task finished with error code: -22 (Invalid argument)`，输出 0 字节 |

**原因（dmesg 实证）**：
```
MVX session: Unsupported V4L2 pixel format. format=0x32315559 (YU12) / 0x47504a4d (MJPG)
```
→ ffmpeg 的 `v4l2m2m` 封装与 Linlon 驱动**协商到了不支持的像素格式**，ioctl 不返回 → D 态。
→ `stcodec` 则是 **MPP 编解码模块未成功 init**。

**D 态说明**：`TASK_UNINTERRUPTIBLE`，进程发起 ioctl 后**内核驱动卡住不返回**，不响应任何信号。用户态无法强制杀死；驱动返回才退出，否则**只能重启**。（驱动线程 `vq0/vq1` 常驻 D 态属正常。）

---

## 4. 解决措施：绕开 ffmpeg，直连 MPP

SpacemiT 的 **MPP 用户态 API** 是正路，板子上现成（`libmpp.so` + `/usr/include/vdec|venc|sys`）。

关键点：
1. **初始化顺序**：`SYS_Init()` → `VB_Init()` → `VDEC_Init()`（漏了 VB_Init 就会 `VB_CreatePool failed`/NOMEM）。
2. **必须 root**：`cma_discover_base` 要读 debugfs 拿 CMA 物理基址，非 root 会 `Permission denied` → 解码不完整。
3. **按「slice」分帧喂流**：`mp4toannexb` 不保证有 AUD，所以用 **H.264 slice NAL(type 1/5)** 作为访问单元边界（不能按固定字节切，也不能依赖 AUD）。
4. **发送遇忙要边取帧边重试**：输入队列满会返回非 0，需 `GetFrame` 排空再重发。
5. **结尾发一个空的 EOS 包**触发 flush，再连续收帧。

核心 API（`vdec_api.h`）：`VDEC_CreateChn → VDEC_EnableChn → VDEC_SendStream → VDEC_GetFrame → VDEC_ReleaseFrame`。

程序：`~/mage-vl/mpp/mpp_dec.c`（一键编译运行见 §6）。

---

## 5. 解决成果

编译运行（root）：
```bash
sudo ~/mage-vl/mpp/mpp_dec ~/mage-vl/codec/sw_h264_aud.h264
# packets=150 decoded=149 frames in 0.501s -> 297.2 fps (last_get=-2011=EOS)
# real 0.828s | user 0.111s | sys 0.035s
```

### 软/硬解码 & 软编码 对比（1080p30，150 帧）

| 任务 | 墙钟 | 帧率 | CPU(user) | 备注 |
| --- | --- | --- | --- | --- |
| 软解 H264 | 0.575 s | 260 fps | **2.87 s** | 占 CPU 核 |
| 软解 H265 | 0.738 s | 203 fps | 4.18 s | |
| **硬解 H264（VPU/MPP）** | 0.501 s | **297 fps** | **0.11 s** | 活交给 VPU |
| 软编 H264 `libx264 -preset veryfast` | 5.41 s | 27.7 fps | 32.4 s | 6–7 核跑满 |
| 软编 H265 `libx265 -preset ultrafast` | 10.92 s | 13.7 fps | 80.2 s | 8 核跑满 |
| 硬编 H264/H265 | 失败 | — | — | stcodec 缺模块 / v4l2m2m -22（本次未打通） |

**关键差距**：硬解 wall 与软解接近，但 **CPU 用量差 ~26 倍**（0.11s vs 2.87s）。这正是硬件编解码的价值：**省 CPU、能并发多路、可实时**。

（配图：`k3-codec-bench.png` 软硬对比；`vpu-bypass.png` 问题→绕行。）

---

## 6. 复现命令

```bash
# 素材 + 软编解
cd ~/mage-vl/codec
ffmpeg -y -f lavfi -i testsrc2=size=1920x1080:rate=30 -t 5 -c:v libx264 -preset veryfast -crf 18 src.mp4
ffmpeg -y -i src.mp4 -c:v libx264 -preset veryfast -b:v 6M -x264-params aud=1 -an sw_h264_aud.mp4
ffmpeg -y -i sw_h264_aud.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 sw_h264_aud.h264
# 硬解（绕开 ffmpeg）
cd ~/mage-vl/mpp && gcc -O2 -I/usr/include -o mpp_dec mpp_dec.c -lmpp
sudo ~/mage-vl/mpp/mpp_dec ~/mage-vl/codec/sw_h264_aud.h264
```

---

## 7. 对我们那个 Mage-ViT 模型有积极意义吗？能加速它吗？

**没有积极意义，也不能加速。**（配图 `mage-vl-bottleneck.png`）

- Mage-VL 的耗时分布（3 秒片段 / 16 canvas）：
  `视频解码 ~26ms` → **`Mage-ViT 编码 ~68690ms`（96%+，瓶颈）** → `EPFE ~177ms` → `分类器 ~2230ms`。
- **硬解只作用于最左边那 ~26ms 的解码**；Mage-ViT 是**纯矩阵/注意力算力**（Transformer 前向），跟视频编解码是两码事——VPU 不参与、也帮不上。
- 而且 Mage-VL 是 **codec-native**：它要的是压缩域的 **bitcost/运动矢量** 信息（cv-preinfer 的补丁 FFmpeg 抽的），**V4L2/VPU 接口根本不暴露这些** → 硬解连"替代预处理"都做不到。
- 结论：**要加速 Mage-ViT，得靠更好的算子/后端（NPU、更好的 RVV kernel、量化），而不是视频硬编解码。**

> 那硬解什么时候有用？——**视频转码 / NVR 多路 / 直播推流 / 播放**这类"编解码本身是主体"的场景。对 LLM/VLM 推理没有帮助。

---

## 8. 产物

```
~/mage-vl/
├── codec/            # src.mp4, sw_h264_aud.mp4/.h264, sw_h264/h265.mp4, hw_*（失败）
├── mpp/mpp_dec.c     # VPU 硬解程序（绕开 ffmpeg，MPP API）
└── logs/
```
