# K3 实机实测集（SpacemiT K3 · RISC-V）

2026-09-29 ~ 09-30 在一块 **SpacemiT K3**（`riscv64`，Bianbu 4.0.1，16 GB 内存）
上跑出来的一批记录，从板子桌面目录整体搬入，图与 Excalidraw 源文件一并保留。

怎么先摸到这块板子（网线 / ARP / SSH），见上一级的
[01-k3-riscv-board.md](../01-k3-riscv-board.md)；想在边缘设备上做优化，看
[kernel-opt/07-edge.md](../../kernel-opt/07-edge.md)。这里只记**这块板上实际跑出了什么**。

## 目录

| 目录 | 内容 |
| --- | --- |
| [k3-report/](k3-report/) | **汇总报告**（`report.html`，含 4 张图与复现命令）：内存带宽实测、各模型速度对比、加速器分工 |
| [k3-npu/](k3-npu/) | 「K3 到底有没有 NPU」—— 没有，官方定位是 RISC-V AI CPU（RVV + IME），AI SDK 全走 CPU/ONNX |
| [k3-asr/](k3-asr/) | ASR 实测：whisper.cpp 与 sherpa-onnx 流式 zipformer 的 RTF 对比，以及 RTF / 延迟 / NPU / VPU 的名词澄清 |
| [k3-asr-service/](k3-asr-service/) | 把流式 zipformer 封装成 WebSocket ASR 服务：启动脚本、systemd 单元、Python 流式客户端 |
| [k3-codec/](k3-codec/) | 视频编解码：ffmpeg 硬解挂死的原因与**绕开 ffmpeg 直接用 SpacemiT MPP C API 驱动 VPU** 的做法 |
| [k3-labs/](k3-labs/) | 两份早期实验记录：软/硬编解码对比（其「VPU 不可用」的结论已被 k3-codec 推翻）、Mage-VL 视频链路跑通 |
| [mage-vl-on-k3/](mage-vl-on-k3/) | Mage-VL 4B（Mage-ViT + Qwen3-4B）在板上做图片问答的实跑记录 |

## 一句话结论

| 问题 | 结论 |
| --- | --- |
| 能跑 8B 大模型吗 | 能，但只有 ~2.9 tok/s —— 卡在内存带宽（CPU 实测读 ~20 GB/s，远低于标称 52 GB/s） |
| 那实际主力是什么 | 官方调过的小模型与 MoE：Qwen3.5-2B ~13 t/s、0.8B ~29 t/s、Qwen3-30B-A3B（激活 3B）~12 t/s |
| 有 NPU 吗 | 没有独立 NPU。AI 算力在 CPU 内（RVV 向量 + IME 矩阵扩展） |
| 有 VPU 吗 | 有（H.264/H.265），但 ffmpeg 路径会挂死；绕开 ffmpeg 直接用 MPP 可跑通硬解 |
| 语音怎么落 | sherpa-onnx 流式 zipformer（int8）RTF 0.49 ≈ 2× 实时；whisper 只适合离线 |

各条结论的原始数据、复现命令和失败路径都在对应目录里，报告表格以
[k3-report/report.html](k3-report/report.html) 为准。

## 关于图

- `*.png` 是已经渲染好的图。
- `d*.json` 是 **Excalidraw 源文件**（`k3-report/d1~d4`、`mage-vl-on-k3/d5`、`k3-codec/d6~d8`、
  `k3-asr/d9`、`k3-npu/d10`），用 [excalidraw.com](https://excalidraw.com) 打开即可改。
- `k3-labs/d6.json` 与 `k3-codec/d6.json` 是同一份，两个目录各自引用。

## 说明

- 目录名沿用板子上的原名，`k3-labs/` 与 `k3-codec/` 有部分内容重叠（同名图与 Excalidraw 源相同），
  差异在正文：`k3-labs/k3-codec-bench.md` 是**早期**结论，`k3-codec/k3-codec-vpu.md` 是**后续**的突破记录。
  两份都保留，按时间顺序读。
- `k3-asr-service/asr-server.service` 里的路径写死了板上的 `/home/pingji/mage-vl/sherpa`，
  换机器要改。
