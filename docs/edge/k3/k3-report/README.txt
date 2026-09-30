SpacemiT K3 (RISC-V) 大模型推理实测报告
板子: pingji-k3 / Bianbu 4.0.1 / 16GB   —   2026-09-29

打开报告: 双击 report.html（浏览器查看）

== TL;DR ==
- K3 能跑 8B，但生成只有 ~2.9 tok/s —— 卡在内存带宽(CPU 实测 ~20GB/s，不是标称 52)。
- 官方调优小模型才是主力: Qwen3.5-2B ~13 t/s, 0.8B ~29 t/s。
- 官方"大模型"招牌 = MoE Qwen3-30B-A3B(只激活3B) ~12 t/s；官方清单里没有 8B 稠密。

== 目录 ==
report.html   完整报告(含图)
fig1.png      访问链路
fig2.png      内存墙原理
fig3.png      各模型速度对比
fig4.png      加速器分工 & 模型动物园
d1~d4.json    Excalidraw 源文件(可改，用 excalidraw.com 打开)

== 关键数据 ==
内存带宽(CPU实测)    读 ~20GB/s   拷贝 ~23GB/s   (8/16线程同 → 共享端口瓶颈)
标称峰值             52 GB/s (LPDDR5-6400×64bit, 颗粒峰值, CPU拿不到)
8B  Q4_K_M           ollama 2.45 t/s | spacemit llama.cpp 2.86 t/s
2B  Q4_0             pp128 115.7 | tg 13.29 t/s
硬件编解码           H.264/H.265 4K (ffmpeg h264_stcodec / v4l2m2m)

== 复现 ==
见 report.html 末尾"复现命令"。
