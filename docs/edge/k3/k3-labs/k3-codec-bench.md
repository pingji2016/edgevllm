# K3 软/硬编解码对比测试报告

> K3 · riscv64 · Bianfu 4.0.1 · ffmpeg 8.0.1 · 2026-09-30

## 摘要

- **软件编解码**（CPU：libx264/libx265）在 1080p 上**可用且解码很快**（解码 200+ fps）。
- **硬件编解码（VPU）在本镜像上开箱不可用**：`stcodec` 找不到 MPP 编解码模块（段错误），`v4l2m2m` 编码 `-22`、解码直接**卡死在内核态**。
- 结论：当前 Bianbu 镜像 + ffmpeg 8.0.1 下，**实际只能用软件编解码**；要启用 VPU 需修 MPP 模块/驱动（SpacemiT 侧）。

## 测试环境

- 板：SpacemiT K3（riscv64），可用 8 核；ffmpeg 8.0.1-3ubuntu2bb1。
- VPU：`/dev/video0` = **Linlon Video device (platform:mvx)**，支持 H264 / HEVC / VP9 / VP8 / JPEG 解码（V4L2 multiplanar）。
- 素材：`testsrc2` **1920×1080@30，5 秒（150 帧）**。

## 软件编解码结果（可用）

| 任务 | 墙钟 | 帧率 | CPU(user) | 说明 |
| --- | --- | --- | --- | --- |
| 软编 H264 `libx264 -preset veryfast` | 5.41 s | **27.7 fps** | 32.4 s | 6–7 核跑满 |
| 软编 H265 `libx265 -preset ultrafast` | 10.92 s | **13.7 fps** | 80.2 s | 8 核跑满 |
| 软解 H264（native） | 0.58 s | **260 fps** | 2.87 s | 解码很轻松 |
| 软解 H265（native） | 0.74 s | **203 fps** | 4.18 s | |

> 编码 CPU 时间 ≈ 墙钟的 6–8 倍（多核并行的体现）；解码几乎不吃力。

## 硬件编解码尝试（失败）

| 任务 | 结果 |
| --- | --- |
| HW 编码 `h264_stcodec` / `hevc_stcodec` | **段错误**；`[MPP-ERROR] module_get_so_path: module is NULL`（MPP 编解码模块缺失） |
| HW 编码 `h264_v4l2m2m` | `Task finished with error code: -22 (Invalid argument)`，输出 0 字节 |
| HW 解码 `h264_v4l2m2m` | **进程卡死在内核 D 态**（`timeout` 都杀不掉，只能 `pkill -9`） |
| HW 解码 `hevc_stcodec` | 段错误 |

附：`/dev/video0` 默认 `root:video`（需 `video` 组或 ACL）；已装 `mpp2`（提供 `libmpp.so`/`libspacemit_mpp.so`），但 ffmpeg 的 `stcodec` 封装仍加载不到模块。

## 概念：软编解码 vs 硬编解码

（见配图 `k3-codec-bench.png`）

- **软件编解码**：CPU 上跑编码/解码算法（x264/x265…）。**压缩率高、参数全可调**（CRF/preset/GOP/码控），生态成熟；代价是**吃 CPU、耗电**，实时性差。
- **硬件编解码**：专用 **VPU** 硬流水线。**CPU≈0、可实时、省电**，适合 NVR/直播/转码/媒体播放；代价是**压缩率略低、参数少、格式/分辨率受限**。
- **本板实际情况**：VPU（Linlon）链路在本镜像上与 ffmpeg 不兼容，所以"硬解的红利"现在拿不到。

## 复现命令

```bash
cd ~/mage-vl/codec
# 素材
ffmpeg -y -f lavfi -i testsrc2=size=1920x1080:rate=30 -t 5 -c:v libx264 -preset veryfast -crf 18 src.mp4
# 软编解
ffmpeg -y -i src.mp4 -c:v libx264 -preset veryfast -b:v 6M -an sw_h264.mp4
ffmpeg -y -i src.mp4 -c:v libx265 -preset ultrafast -b:v 6M -an sw_h265.mp4
ffmpeg -y -i sw_h264.mp4 -f rawvideo -pix_fmt yuv420p /dev/null
# 硬编（会失败，供对照）
ffmpeg -y -i src.mp4 -pix_fmt nv12 -c:v h264_stcodec -b:v 6M hw_h264.mp4
ffmpeg -y -i src.mp4 -pix_fmt nv12 -c:v h264_v4l2m2m -b:v 6M hw_h264v.mp4
```

## 结论

1. **软件编解码才是这块板子当前可用的路线**：1080p 软解 200+ fps 完全够用；软编 1080p 约 14–28 fps。
2. **VPU 硬编解在本镜像不可用**（缺 MPP 模块 + v4l2m2m 驱动挂死）；想吃到"硬解红利"需要 SpacemiT 修模块/驱动。
3. 如果你的用途是**转码/监控/直播**（需要省 CPU、实时），硬编解是关键；但在 K3 现状下暂时只能软编解，CPU 会被占满。
