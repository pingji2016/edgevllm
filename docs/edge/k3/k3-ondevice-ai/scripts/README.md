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
```

## 三个必读的坑

1. **硬编报段错误** → 不是硬件问题，是用户不在 `video` 组（MPP 插件要探测 `/dev/video0`）。`./hw-encode.sh fix`。
2. **`taskset -c 8` 报 EINVAL** → 不是核坏了，是 HMP 的类别限制。先写 `/proc/set_ai_thread`。
3. **A100 反而更慢**（LLM 慢 5.8×）→ 它是向量/矩阵专用核，缓存小、主频低；只用它跑 int8/IME 算子。
