# 配图与生成脚本

博客 [`../blog.md`](../blog.md) 用到的图都在这里，`src/` 是可重新生成的源。

## 图

| 文件 | 用在 | 怎么来的 |
|---|---|---|
| `k3-stack.png` | 开篇全景（感知/计算/控制/交互四层） | `src/k3-stack.json` |
| `k3-voice-pipeline.png` | §2 离线语音链路 | `src/k3-voice.json` |
| `k3-streaming.png` | §4 流式 vs 非流式 | `src/k3-streaming.json` |
| `k3-vision.png` | §5 单帧 28 ms 的路径 | `src/k3-vision.json` |
| `k3-tunnel.png` | §10 那次 14 小时失联 | `src/k3-tunnel.json` |
| `k3-roadmap.png` | §11 从脚本到产品的三栏路线图 | `src/k3-roadmap.json` |
| `device-cmp.svg` | §6.2 跨设备单帧对比（对数刻度） | `src/device-cmp.py` |
| `asr-rtf.svg` | §3 ASR 的 RTF 随长度二次增长 | `src/charts.py` |
| `encode.svg` | §7 硬编 vs x264 墙钟 | `src/charts.py` |
| `yolo-bus.jpg` | §5 实测检出（4 人 + 1 公交） | 板子上跑 yolov8n 后把框画回原图，裁掉上半部分 |

## 怎么重新生成

手绘风格的 6 张图用 excalidraw 渲染器（roughjs + resvg）：

```bash
# 需要 excalidraw skill 的 scripts/render.mjs
node <skill>/scripts/render.mjs src/k3-stack.json .
```

数据图是纯手写 SVG（没有 matplotlib）：

```bash
python3 src/device-cmp.py     # -> device-cmp.svg
python3 src/charts.py         # -> asr-rtf.svg / encode.svg / first-audio.svg
```

## 两个约定

1. **图里的数字必须是实测的**。所有耗时都来自 `k3-ondevice-ai-report.md`，没有估算值。
2. **跨设备那张图是唯一有"别人的数据"的**：Jetson / 树莓派 的数字来自 Ultralytics 官方
   benchmark（YOLO26n / 640 / 不含前后处理），图上和正文里都标了来源与不可比之处。
