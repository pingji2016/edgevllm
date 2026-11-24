# 实时链路实测：流式 ASR / 流式视觉 / 分割 / 语音合成诊断

> 机器 `pingji-MS-7B89` · RTX 5060 Ti 16 GB（sm_120）· Ubuntu 26.04
> 实测日期：2026-11-24 · 配套 [capability-internals.md](capability-internals.md)（原理）与
> [inference-acceleration.md](inference-acceleration.md)（LLM 推理）

---

## 0. 本轮结论速览

| 项目 | 结果 | 关键数字 |
| --- | --- | --- |
| **流式 ASR**（faster-whisper-small） | ✅ 可用 | 每 0.5–2 s 更新一次，**更新延迟 110–130 ms**，RTF 0.06–0.26 |
| **流式视觉**（Qwen3-VL-8B Q4） | ✅ 可用 | 稳态 **0.41 s/帧**（≈2.4 FPS），冷启动首帧 36 s |
| **SAM2 分割**（hiera-tiny） | ✅ 可用 | **18 ms/帧（54.9 FPS）**，273 MiB |
| **CosyVoice2 语音克隆** | ❌ 仍未解决 | 已定位两个 bug，根因指向 transformers 5.x |
| **MInference** | ❌ 装不上 | 源码构建需 C++20 + 匹配工具链，本轮失败 |
| **TensorRT-LLM** | ⏭ 未尝试 | 需 TRT 10.8+ 与 GB 级安装 + 30 min engine 编译 |

---

## 1. 流式 ASR（滚动窗口）

模型 faster-whisper-small（CTranslate2，CUDA fp16），音频 13.75 s 中文。
做法仿 whisperlive：**滚动缓冲 6 s**，每 W 秒触发一次转写。

| 分块 | 块数 | 更新延迟 mean | p95 | 总耗时 | RTF |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 s | 28 | 128 ms | 227 ms | 3.60 s | 0.26 |
| 1.0 s | 14 | **110 ms** | 148 ms | 1.55 s | 0.11 |
| **2.0 s** | 7 | 116 ms | 140 ms | **0.81 s** | **0.06** |
| 离线（一次整段） | 1 | — | — | 0.83 s | 0.06 |

**结论：**

1. **2 s 分块是甜点**：总耗时与离线持平（0.81 vs 0.83 s），但能每 2 s 吐一次字。
2. **单次更新延迟稳定在 110–130 ms**，与分块大小关系不大（因为都只转写 6 s 缓冲）。
3. **0.5 s 分块反而更慢**（3.60 s）：调用次数翻倍，固定开销摊不开。

**必须记的坑：朴素增量合并会重复**
按「已输出文本的后缀 vs 新文本前缀」求最长重叠再拼接，输出会出现重复段落
（如“保持管理层的一致性那家公司因此保持管理层的一致性…”）。这正是 whisperlive 需要
**双游标 + 重复检测**的原因。本轮只验证了延迟指标，文本合并仍需完善。

---

## 2. 流式视觉（Qwen3-VL-8B）

用的是本机常驻的 `vlm-server.service`（llama-server + Vulkan，配置：
`-c 4096 -ngl 99 -b 256 -ub 256 --image-max-tokens 256 --jinja`）。
逐帧 POST `/v1/chat/completions`，每帧要求 48 token 描述。

| 帧 | 冷启动 | 稳态 |
| --- | ---: | ---: |
| 第 1 帧 | **36.15 s** | 0.45 s |
| 第 2 帧 | 0.83 s | 0.38 s |
| 第 3 帧 | 0.63 s | 0.41 s |

**稳态 0.41 s/帧（≈2.4 FPS）**，输出 20–26 token。

**结论：**

1. **冷启动 36 s** 是图片编码首次触发 kernel 编译/缓存构建，必须预热后再测。
2. 与相机链路对比：`camfilter` 的 detect 是 6.7 ms/帧、整链 16.6 FPS →
   **VLM 描述慢约 60 倍**，只能"抽帧送检"（每 6 帧取 1 帧）而不是逐帧。
3. 8B Q4 的显存占用约 6.2 GB，与 16 GB 卡上其他模型共存需要规划。

---

## 3. SAM2 逐帧分割（hiera-tiny）

transformers 5.23 自带 `Sam2Model`，点提示 = 画面中心。

| 项 | 实测 |
| --- | ---: |
| 参数量 | 31.4 M |
| 单帧耗时 | **18 ms（54.9 FPS）** |
| 显存 | 273 MiB |
| 预测 mask IoU | 0.922 |
| mask 覆盖率 | 0.2%（点是鼻尖附近的小区域，属正常） |

**结论：SAM2-tiny 可以进实时链路**——18 ms 在 60 ms 的 16.6 FPS 预算里只占 30%，
和 Depth-Anything（9.4 ms）、YOLO 检测（6.7 ms）三者叠加仍在预算内。

> 踩坑：v5 的 `post_process_masks` 需要 `reshaped_input_sizes`，而处理器输出里没有；
> 且返回的 `pred_masks` 是 5 维 `(1,1,3,256,256)`。索性绕开处理器，
> 按「最后两维是空间」自己 `F.interpolate`（见脚本）。

---

## 4. CosyVoice2 语音克隆：仍未解决，但定位到两层

原始症状（文档记录）：能跑，但合成内容与目标句完全不符。

### 4.1 第一层：混合精度导致崩溃（已修）

```
llm   {torch.bfloat16, torch.float32}   ← 同一个模块里混着两种精度
flow  {torch.float32}
hift  {torch.float32}
```

`load_jit=False, load_trt=False, fp16=False` 时，LLM 仍是 bf16 → 与 fp32 的 prompt 特征
相乘报 `mat1 and mat2 must have the same dtype`；随后 HiFiGAN 收到过短输入报
`Kernel size can't be greater than actual input size`。

把三个模块统一 `.float()` 后**能跑通**，但——

### 4.2 第二层：内容仍然错（未解）

| 项 | 结果 |
| --- | --- |
| 目标句 | 你好，今天天气不错。（8 字） |
| 生成音频 | **8.00 s**（正常应 2–3 s） |
| ASR 回听 | 其實我只要過一段時間我就還能夠過一段時間跑在那邊為啥我都做不了還要過一段時間 |
| 字符相似度 | **4%** |

即：LLM 阶段产生了退化/重复的 speech token。

### 4.3 根因判断：transformers 5.x 与 CosyVoice2 的接口不兼容

CosyVoice2 的 LLM 包装层用的是**旧版 `past_key_values` 协议**：

```python
outs = self.model(..., use_cache=True, past_key_values=cache)
new_cache = outs.past_key_values          # 期望是 tuple
```

transformers 5.x 已换成 `Cache` 对象体系。官方要求 `torch 2.3.1 / torchaudio 2.3.1 /
diffusers 0.29.0`，本机是 2.15 / 2.11 / 0.42。

**验证进展**：已建 `~/apps/cosy-env`（transformers 4.51.3 + diffusers 0.29.0），
但 torch 轮子仍在下载（2.5 GB，本机到 PyPI 镜像约 1 MB/s）。装完即可做 A/B 验证。

**结论**：这个能力项要可用，必须**独立 venv + 严格锁版本**，不要在通用环境里跑。

---

## 5. MInference / TensorRT-LLM

### MInference：构建失败

| 阶段 | 现象 | 处理 |
| --- | --- | --- |
| 默认安装 | 构建隔离环境里 `ModuleNotFoundError: torch`（setup.py 读 torch 版本） | 改 `--no-build-isolation` |
| 绕过隔离 | `#error C++20 or later compatible compiler is required` | 加 `CXXFLAGS=-std=c++20` |
| 加 C++20 | 仍然构建失败（CUDA 扩展与 CUDA 13.0 / torch cu132 组合） | 放弃本轮 |

结论：MInference 0.1.6 需要与其 CUDA 扩展匹配的工具链，**本机 CUDA 13 + gcc 15 组合过新**。
可行的替代路线：用其 Triton-only 路径（若有），或在更老的环境（CUDA 12.x + gcc 12）里装。

### TensorRT-LLM：未尝试（工程量明确超标）

| 需求 | 说明 |
| --- | --- |
| 包体积 | `tensorrt-llm 1.3.0` + TensorRT + CUDA 组件，GB 级 |
| 版本要求 | TRT ≥ 10.8 才支持 Blackwell（sm_120） |
| 编译时间 | 每个模型要 build engine，30 min 起 |
| 磁盘 | 当前仅剩 27 GB |

判断：投入产出比低，**本轮跳过**。要做的话建议先在临时目录装、只编 1 个小模型验证。

---

## 6. 本轮对"实时链路"的整体判断

| 环节 | 单帧/单块延迟 | 能否进 16.6 FPS（60 ms）预算 |
| --- | ---: | --- |
| YOLO11n 检测 | 6.7 ms | ✅（已有） |
| Depth-Anything-V2-S | 9.4 ms | ✅ |
| SAM2 hiera-tiny | 18 ms | ✅（占 30%） |
| Qwen3-VL-8B 描述 | 410 ms | ❌ 只能抽帧 |
| faster-whisper-small（ASR） | 110–130 ms/次 | ✅（音频异步并行） |

**结论：**视觉侧"检测+深度+分割"三者可并行进实时链路；
**VLM 描述与 ASR 必须异步/抽帧**，不能串在逐帧回调里。

---

## 7. 复现

```bash
# 流式 ASR（需 cu12 库路径）
LIBS=$(ls -d ~/apps/minimind/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')
LD_LIBRARY_PATH=$LIBS ~/apps/bench/yolo-venv/bin/python ~/apps/infer-research/stream_asr.py

# 流式视觉（打常驻服务）
~/apps/bench/yolo-venv/bin/python ~/apps/infer-research/vl_direct.py 8092 /tmp/vlframes

# SAM2
~/apps/bench/yolo-venv/bin/python ~/apps/infer-research/sam2_test.py

# CosyVoice2 诊断
~/apps/bench/yolo-venv/bin/python ~/apps/infer-research/cosy_diag2.py
```
