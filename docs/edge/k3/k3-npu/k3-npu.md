# K3 到底有没有 NPU？——没有（"RISC-V AI CPU"）+ 官方 AI 路径

> 2026-09-30 · K3 riscv64 · Bianbu 4.0.1

## 0. 首先：K3 没有独立 NPU

- 官方定位是 **"RISC-V AI CPU"、"RISC-V 同构融合计算"**（`spacemit.com/products/keystone/k3`）——AI 算力在 **CPU 内**（**RVV 向量 + IME 矩阵扩展**），不是 NPU ASIC。
- 官方 **AI SDK** 仓库（`github.com/spacemit-com/ai-sdk`）里 ASR / VLM / Vision 等示例**全程没有 NPU**，都是 CPU(+ONNX) 跑。
- 硬件证据：**设备树无 NPU 节点**；`/dev/npu*` 不存在；dmesg 无 NPU 驱动；`/sys/class/misc` 只有 `v2d_dev`/`tcm`/`ddr_perf` 等，无 npu。
- 所以：**之前的"NPU"说法是错的**——`libspacemit_ep.so` / `provider: spacemit` 是 **CPU 侧的 ONNX Runtime 加速后端（RVV/IME）**，不是 NPU。

> 结论：**用不上"NPU"不是通路问题，是因为根本没有 NPU。** 官方的"AI 加速"就是 **CPU 的 RVV + IME**。

## 官方 AI 路径（可直接用）

`github.com/spacemit-com/ai-sdk`（子模块：`asr/ tts/ vad/ llm/ vlm/ vision/ gateway/`），一键编译 `source build/envsetup.sh && m`：
- **ASR**：`asr_file_demo <wav>` / `asr_stream_demo -i 0`（SenseVoice，CPU）
- **TTS**：`tts_file_demo -p "你好" -l matcha:zh`
- **VLM**：`fastvlm_demo --config ... --image ...`（FastVLM，58 tok/s）
- **LLM**：通过 `llama-server`（llama.cpp-tools-spacemit）
- 模型下载：`archive.spacemit.com/spacemit-ai/model_zoo/...`

---

## TL;DR

- **没有 NPU**；AI 在 **CPU（RVV+IME）** 上跑。
- 官方**有**适配路径：**ONNX + `spacemit-ort`(CPU/RVV/IME 后端) + 官方 AI SDK/ModelZoo**。
- EP 能注册（`['SpaceMITExecutionProvider','CPUExecutionProvider']`），但它作用于 **CPU**。
- 板子上的 AI 工作（我们的 ASR/LLM/VLM）本来就在 CPU(RVV) 上，**已经是在用这块芯片的"AI 算力"了**（只是它不是 NPU）。

## 1) 官方适配途径（模型怎么跑）

1. 模型导出/转成 **ONNX**（静态形状 + 支持的算子）。
2. 用 **`spacemit-ai-sdk`（x86_64）做量化/转换**（INT8，需校准数据）。
3. 部署到板子，用 **`spacemit-ort` + `SpaceMITExecutionProvider`** 运行（CPU/RVV/IME）。
4. 或直接用 **ModelZoo / AI SDK** 官方适配好的模型（SenseVoice、FastVLM、YOLO…）。


## 1) 官方适配途径（模型怎么上 NPU）

1. 把模型导出/转成 **ONNX**（静态形状 + 支持的算子）。
2. 用 **`spacemit-ai-sdk`（x86_64）做量化/转换**（INT8，需校准数据）。
3. 部署到板子，用 **`spacemit-ort`（riscv）+ `SpaceMITExecutionProvider`(EP)** 运行 → 支持的子图自动落 NPU。
4. 或直接用 **ModelZoo** 官方适配好的模型。

（配图 `npu-path.png`）

## 2) 公开的地址（这些是能拿到的）

下载站：**`https://archive.spacemit.com/spacemit-ai/`**
- `/spacemit-ai-sdk/` — **转换+量化工具链**（x86_64；`quick_start.tar.gz` 含 ImageNet 校准数据）= 官方适配入口
- `/onnxruntime/spacemit-ort.riscv64.*.tar.gz` — riscv 运行时
- `/ModelZoo/` — 适配好的模型：`asr/sensevoice.tar.gz`、`vision/`、`llm/`、`ggml/` …
- `/llama.cpp/`、`/ollama/`、`/localai/`、`/dify/`、`/openwebui/` 等

apt（板子源）：
- `resolute-porting/main`：`spacemit-onnxruntime`、`python3-spacemit-ort`、`spacemit-runtime`
- `resolute-customization/main`：`spacemit-ai-gateway`

> 找过但**没有**公开的 pip 源：`pypi.org` 上无 `spacemit-asr`/`spacemit-ai-gateway`；apt 里也没有对应 python 包。→ gateway 的 py 依赖属**内部源**。

## 3) 实测卡点

| 项 | 结果 |
| --- | --- |
| EP 安装/注册 | ✅ `['SpaceMITExecutionProvider','CPUExecutionProvider']` |
| gateway 安装 | ✅ `/opt/spacemit-ai-gateway`（configs/schema/venv） |
| gateway 启动 | ❌ venv 缺 `spacemit-ai-gateway` 可执行；`pip install -r requirements-runtime.txt` → **no matching distribution**（公网无） |
| 官方 SenseVoice 跑 NPU | ❌ 受上面两条阻塞 |
| NPU 设备 | ❌ `/dev/npu0`、`/dev/npuctl` 不存在；dmesg 无 NPU 驱动 |
| 普通 ONNX 丢给 EP | ❌ **直接崩**（native，无异常） |

## 4) 结论与下一步

- **想真正用上 NPU**，需要：
  1. SpacemiT 的**内部 pip 源 / 离线 wheel**（装 `spacemit-asr`、`spacemit-ai-gateway` 等）；
  2. 确认内核 **NPU 驱动/固件已启用**（出现 `/dev/npu*`）。
- 或者自己走 **SDK 量化 + 自建 ORT/EP** 路线（仍需 NPU 设备可用）。
- **对我们的 ASR 不是必须**：sherpa 流式 zipformer 在 **CPU(RVV)** 已经 **≈2× 实时**，够用。

## 5) 参考命令

```bash
# 装 + 验证 NPU EP
sudo apt-get install -y spacemit-onnxruntime spacemit-runtime python3-spacemit-ort
python3 -c "import spacemit_ort,onnxruntime as ort; print(ort.get_available_providers())"
# 期望：['SpaceMITExecutionProvider', 'CPUExecutionProvider']

# 安装官方 gateway（注意：其 py 依赖需内部源）
sudo apt-get install -y spacemit-ai-gateway
sudo systemctl start spacemit-ai-gateway   # 当前会因缺 venv 可执行而失败

# 官方资产
# https://archive.spacemit.com/spacemit-ai/spacemit-ai-sdk/
# https://archive.spacemit.com/spacemit-ai/onnxruntime/
# https://archive.spacemit.com/spacemit-ai/ModelZoo/asr/sensevoice.tar.gz
```
