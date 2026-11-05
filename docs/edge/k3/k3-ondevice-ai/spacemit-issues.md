# 给 SpaceMiT 的问题清单（可直接提交的 issue 草稿）

> 环境（实测于 2026-10-16 ~ 11-05）：
> **K3 Pico ITX**（riscv64）· Bianbu 4.0.1 → **4.0.8** ·
> 内核 `6.18.3-cf-k3-1.3.0` / `6.18.3-generic` **1.0.1.4 → 1.0.10.4**
> 16 核（8× **X100** @2.2GHz + 8× **A100** @1.8GHz）· 15GiB · **SpaceMiT AI Gateway v0.1.15**
> 详细数据见 [`k3-ondevice-ai-report.md`](k3-ondevice-ai-report.md) 对应章节。
>
> 提交建议：优先提到对应仓库（`spacemit-com/linux-6.18`、网关/MPP 的仓库）；
> 找不到仓库就发 SpaceMiT 开发者论坛，附本文件与报告。

---

## Issue 1（严重）A100 核组跑通用 LLM 反而慢 5.8×，且没有任何提示

> 2026-11-02 补：开关的准确用法（`echo $$ > /proc/set_ai_thread` / `echo 0`、
> 子进程继承但跨会话不继承）、X100 vs A100 的完整对照，见
> [`../k3-a100-cores.md`](../k3-a100-cores.md)。

**组件**：内核 HMP 调度（`drivers/soc/spacemit/hmp.c`、`/proc/set_ai_thread`）+ llama.cpp

**摘要**：按官方方式把线程切到 A100（HMP_AI, cpu 8–15）后，llama.cpp 的
prompt/predict 吞吐下降约 **5.8 倍**。A100 是向量/矩阵专用核，这个结果本身可以理解，
但**没有任何文档/告警**说明它不适合通用 LLM，用户很容易认为是"没提速"或"配置错了"。

**复现**：
```bash
# 1) 切到 A100
echo 0 | sudo tee /proc/set_ai_thread
nproc                     # → 8
cat /proc/self/status | grep Cpus_allowed_list   # → 8-15

# 2) 同一 gguf，对比 X100（8 线程）与 A100（8 线程）
taskset -c 0-7  llama-bench -m Qwen3.5-2B-Q4_0.gguf -t 8 -p 256 -n 16
taskset -c 8-15 llama-bench -m Qwen3.5-2B-Q4_0.gguf -t 8 -p 256 -n 16
```

**实测**：

| 核组 | pp256 (t/s) | tg16 (t/s) |
|---|---:|---:|
| X100 (0–7) | **8.91** | **5.54** |
| A100 (8–15) | 1.53 | 1.45 |

> **2026-11-05 修正**：A100 慢的是标量/通用计算，**不是内存带宽**。
> 同一块板子跑 STREAM（`N=2e7` doubles，`gcc -O3 -march=rv64gcv -fopenmp`，单位 GB/s）：
>
> | 核组 / 线程 | Copy | Scale | Add | Triad |
> |---|---:|---:|---:|---:|
> | X100 1T | 6.52 | 6.45 | 6.82 | 10.26 |
> | X100 8T | 9.48 | 9.63 | 12.53 | 18.74 |
> | A100 1T | 10.74 | 8.89 | 14.35 | 19.11 |
> | A100 8T | **10.97** | **10.97** | **14.33** | **22.13** |
>
> A100 单线程 Copy 就比 X100 高 65%，8 线程 Triad 高 18%，与 VLEN 1024 的宽向量访问一致。
> 所以"别用 A100"要限定在**标量/分支密集、对小缓存敏感**的负载——
> CoreMark 4972 vs 1539 才是它的真实短板，带宽密集任务反而该用它。

**根因（我们的分析）**：

| | X100 | A100 |
|---|---|---|
| 主频 | 2.15 GHz | **1.8 GHz** |
| L1D | 64 KB | **32 KB** |
| L2 / 4 核 | 4 MB | **1 MB** |
| VLEN | 256 | 1024 |
| IME | 无 | **有** |

→ A100 靠 IME 做 int8 矩阵算子；llama.cpp 的 Q4_0 反量化 + 标量/向量混合负载
正好踩在它的短板上（缓存小、主频低）。

**期望**：
1. 在 A100 相关文档里明确写"通用 LLM 推理请留 X100"；
2. `echo 0 > /proc/set_ai_thread` 之后能在 dmesg 或 `/proc` 里看到当前核组的用途说明；
3. 条件允许时，给 llama.cpp 的 spacemit 分支加一个"仅把 IME 算子 offload 到 A100、
   其余留在 X100"的模式（这才是 A100 的正确用法）。

---

## Issue 2（严重）VPU 硬编：用户不在 `video` 组时**段错误**，而不是权限错误

**组件**：MPP VENC (`h264_stcodec` / `libmpp` / `libvi_k3_cam_plugin.so`)

**摘要**：普通用户直接跑 `ffmpeg -c:v h264_stcodec` 会 **segfault**。把用户加进
`video` 组后立即正常。插件在打开 `/dev/video0`（VPU 的 V4L2 节点）失败后没有
检查返回值，继续用空指针。

**复现**：
```bash
id -nG                       # 不含 video
ffmpeg -f rawvideo -pix_fmt nv12 -s 1920x1080 -r 30 -i in.nv12 \
       -c:v h264_stcodec -frames:v 10 out.h264
# → Segmentation fault
sudo usermod -aG video $USER   # 重新登录
# → 正常输出
```

**期望**：返回 `EACCES` 并打印 `Permission denied: /dev/video0 (add user to 'video' group)`。

**建议**：
1. 插件里检查 `open()`/`ioctl()` 返回值；
2. 打包时带一条 udev 规则（`SUBSYSTEM=="video4linux", KERNEL=="video*", GROUP="video", MODE="0660"`），
   或在 `ffmpeg -encoders` 的帮助里注明需要 `video` 组。

**补充**：修好后 1080p150 帧硬编 **1.087s**，对比 libx264 veryfast **5.219s**（快 4.8×，
CPU 占用少约 144×）—— 这条链路值得默认打开。

---

## Issue 3（中）TTS 的 `voice_id` 与 `speed` 参数被**静默丢弃**

**组件**：AI Gateway `/v1/tts/synthesize`、`/v1/tts/voices`（Matcha / Kokoro 适配层）

**摘要**：`/v1/tts/voices` 返回 **6 个音色**，但换 `voice_id` 输出的音频**完全一样**；
`speed` 改成 0.7 / 1.5 音频时长**也完全一样**。只有换 `model` 才真的变化。

**复现**：
```bash
# 同一段"全新文本"，三个 voice_id → md5 相同
for v in $(curl -s localhost:18790/v1/tts/voices | jq -r '.[].id'); do
  curl -s -X POST localhost:18790/v1/tts/synthesize \
    -H 'Content-Type: application/json' \
    -d "{\"text\":\"这段文字是全新生成的用于测试\",\"voice_id\":\"$v\",\"model\":\"matcha_zh_en\",\"response_format\":\"wav\"}" \
    -o /tmp/t_$v.wav
  md5sum /tmp/t_$v.wav
done
```

**实测**：
- 3 个 `voice_id` → 3 个文件 md5 **全为 `29a115b7…`**
- `speed` 0.7 vs 1.5 → md5 **全为 `118beecb…`**
- matcha_zh 的 `speed=0.7/1.0/1.5` → 时长**都是 4.49s**
- 换 `model`（matcha_zh_en / kokoro）→ 输出**不同** ✅

**影响**：多音色/变语速这两个最常用的能力实际不可用，且不报错，排查成本高。

**期望**：把 `voice_id`/`speed` 透传到适配器；不支持时返回 4xx + 明确错误，而不是静默忽略。

**复现脚本**：[`scripts/tts-voices.sh`](scripts/tts-voices.sh)

---

## Issue 4（中）ASR 处理耗时随音频长度**近似二次增长**

**组件**：AI Gateway `/v1/asr/recognize`（SenseVoice）

**摘要**：同一模型，音频越长 RTF 越差，40s 时已到 **0.807**（≈1.2× 实时），
意味着 1 小时录音要跑约 48 分钟。

**实测**：

| 音频 | 处理耗时 | RTF |
|---:|---:|---:|
| 5.1s | 0.77s | 0.150 |
| 10.1s | 2.17s | 0.216 |
| 20.1s | 8.14s | 0.405 |
| 40.2s | 32.5s | **0.807** |

**期望**：接近线性；或接口层强制分块（见下面 Issue 5 的建议）。

**我们的规避**：客户端先用 `/v1/vad/segments`（34× 实时）切段、合并到 ~12s 再逐段识别，
37.3s 音频从 27.9s 降到 **2.98s（12.5× 实时，快约 9 倍）**。
脚本：[`scripts/long-transcribe.py`](scripts/long-transcribe.py)

---

## Issue 5（中）`/v1/asr/stream` 号称流式，实际**不出增量 partial**

**组件**：AI Gateway `/v1/asr/stream/session` + `/v1/asr/stream`（WebSocket）

**摘要**：按文档流程（`POST .../stream/session` → WS 带 `session_id` → 发二进制 PCM 帧 →
发 `{"type":"end"}`），`partial=true` 却**只收到 1 个 partial，且在音频喂完之后**，
RTF 与整段批量识别几乎一样。这不是"流式"，只是给批量套了个 WS 外壳。

**实测**：

| 音频 | 喂入 | 首个（也是唯一）结果 | RTF |
|---:|---|---:|---:|
| 10.1s | 实时 1× | 12.44s（结束后 +2.38s） | 0.227 |
| 37.3s | 实时 1× | **65.90s（结束后 +28.6s）** | **0.759** |

**期望**：边说话边出 partial（哪怕是每 1–2s 一个字幕行），或在 `partial_results=true`
时明确返回"不支持"。

**对照**（正向反馈）：`/v1/tts/stream` 是**真流式** —— 按句 `{"type":"append"}` 喂文本，
每句约 **0.5s** 就返回该句的 PCM。用这个能力做"边说边播"，首音频从 28.4s 降到 **3.0s**。
脚本：[`scripts/stream-voice.py`](scripts/stream-voice.py)

---

## Issue 6（低）`onnxruntime_perf_test -e spacemit` 报 "Use a '|' to separate the key and value"

**组件**：SpaceMITExecutionProvider（onnxruntime `-e spacemit`）

**摘要**：按名字指定 EP 时，provider options 解析直接报
`Use a '|' to separate the key and value`，即使没有传任何 provider option。

**复现**：
```bash
onnxruntime_perf_test -e spacemit -r 5 model.onnx
# → Use a '|' to separate the key and value
```

**期望**：不传 option 时应使用默认值，不应报错；或文档给出必需的最小 option 串
（例如 `-e spacemit|<key>=<value>`）。

---
## Issue 7（严重）IME2 路径的 flash-attn 在「逐 token 解码」时精度错误（一行修复，上游未合）

**组件**：`llama.cpp-tools-spacemit` 的 IME2 后端（`ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp`）

**来源**：SpaceMiT 论坛已有人完整定位并给出一行修复 ——
[`t/1733`](https://forum.spacemit.com/t/1733)。这里补一条**版本 / 行号 / 本地对照**的证据链，
方便厂商直接并进下一版。

**受影响范围**（我们这边核实过）：

| 项 | 值 |
| --- | --- |
| 本机安装版本 | `llama.cpp-tools-spacemit` **0.1.9**（与论坛点名的 0.0.8 / 0.1.1 / 0.1.9 一致） |
| 本机源码树 | `~/llama-src/llama.cpp-master`，第 **669** 行仍是 `float M[QLEN] = { -INFINITY };` |
| 实际会走的内核 | 只实例化了 `..._mrow<4>`（1262 行）和 `..._mrow<2>`（1283 行） |
| 触发条件 | `n_ubatch = 1`（逐 token 解码）+ flash-attn 开 + 每个 KV 组 2 或 4 个 query head |

**机制**：聚合初始化只把 `M[0]` 设成 `-∞`，`M[1..QLEN-1]` 是 0。
第 2/3/4 个 head 的 online-softmax 初始最大值变成 0，注意力分数全为负时 `exp(s-0)`
在转成 `_Float16`（V 累加器）时下溢 → 输出被缩小，生成出现 "the the"、JSON 抽错。

**本地对照（2026-11-02 / 11-03，`pingji-k3`，512 ctx，`-b 512 -ub 1 -t 8`）**：
`-fa 0` 作基准，`-fa 1` 与它比 KLD（同一份文本、同一颗芯片，只改 flash-attn）：

| 模型 | q / kv（每 KV 组 query 数） | 核簇 | KLD 均值 | top-p 一致 |
| --- | --- | --- | ---: | ---: |
| Qwen3-0.6B Q4_0 | 16 / 8（2，`_mrow<2>`） | X100 | 0.0046 | 96.5% |
| Qwen3-0.6B Q4_0 | 同上 | A100 | 0.000000 | 100% |
| Qwen2.5-7B Q4_0 | 28 / 4（**7，无 `_mrow<7>`**） | X100 | **0.001464**（中位 0.000953 / 最大 0.0106） | 98.43% |
| Qwen2.5-7B Q4_0 | 同上 | A100 | 0.000000 | 100% |
| **Qwen3-4B Q4_K_M** | 32 / 8（**4，正好命中 `_mrow<4>`**） | X100 | **0.002029 ± 0.000248**（中位 0.000652 / 最大 0.041432） | 97.255 ± 1.025% |
| Qwen3-4B Q4_K_M | 同上 | A100 | 0.000000 | 100% |

4B 的基准 PPL = **14.0710 ± 3.14562**（X100，`-fa 0`，173.38 s/pass）。

**结论**：三种 GQA 组合都试过了，包括**真正会实例化 `_mrow<4>` 的 4:1**
（Qwen3-4B，X100 上 KLD 也只有 0.0020），仍然**没有复现论坛那 0.2148 量级的偏差**。
所以问题要么不在我们这版 0.1.9 的二进制里（编译期走了别的分支），
要么论坛那组数据来自**更早的构建 / 不同的 `-ub` 与 `-fa` 组合**。
建议厂商给出那组 0.2148 的**完整复现命令与版本号**；无论如何，
把 §修复 那一行并进主线仍然是必要的（代价为零）。

附带一个反直觉的观察：**Qwen3-4B 在 A100 上反而快**（`-fa 0` 61.73 s/pass vs
X100 173.38 s/pass，2.8×），而 Qwen2.5-7B 在 A100 上慢 5.8×（1688 s vs 289 s）。
说明 IME2 路径是否命中与模型形状强相关，不能一概而论"别用 A100"。

**修复（论坛给出，我们照抄一行）**：
```diff
-    float M[QLEN] = { -INFINITY };  // maximum KQ value
+    float M[QLEN];                  // maximum KQ value
+    std::fill(M, M + QLEN, -INFINITY);
```
临时规避：`-fa off`（pp512 约慢 11%）。

**期望**：把这一行并进 `llama.cpp-tools-spacemit` 下一版，同时提上游
`ggml-org/llama.cpp`（`-DGGML_CPU_RISCV64_SPACEMIT=ON` 也有同样问题）。

## Issue 8（严重）SpaceMIT EP 在「QDQ 量化模型」上 **SIGSEGV**，崩在 `SpineGraph::Build`

**组件**：`spacemit_ort` EP（`libspacemit_ep.so.2.0.6`）+ ORT `1.24.2+spacemit.a1`

**来源**：论坛 [t/1744](https://forum.spacemit.com/t/1744) 有人报 Depth-Anything-3 int8
在 EP 上段错误（ORT `1.24.0+spacemit.a3`）。我们用板上**现成的** int8 模型复现，
说明它不是某个模型的特例。

**复现**（2026-11-03，`pingji-k3`，Bianbu 4.0.1）：

```python
import spacemit_ort, onnxruntime as ort
s = ort.InferenceSession("yolo11n.q.onnx",
        providers=['SpaceMITExecutionProvider'])   # ← 这里就崩
```

**结果**（三个不同来源的 QDQ 模型，全部在 `InferenceSession` 构造阶段崩溃）：

| 模型 | 大小 | 节点数 | EP | CPU |
| --- | ---: | ---: | --- | --- |
| `yolo11n.q.onnx` | 2.9 MB | 975（Q/DQ 673） | **SIGSEGV (139)** | ✅ 正常输出 `(1,64,80,80)`、`(1,80,80,80)` |
| `resnet50.q.onnx` | 25 MB | 273（Q/DQ 200） | **SIGSEGV (139)** | ✅ |
| `yolov5n-face_cut.q.onnx` | 2.0 MB | — | **SIGSEGV (139)** | ✅ |

崩溃栈（`gdb -batch -ex run -ex bt`，符号齐全）：

```
SIGSEGV in onnxruntime::spacemit::SpineNode::SpineNode(onnxruntime::Node const*)
  #1 SpineGraph::Build(GraphViewer const&)
  #2 SpineGraph::SpineGraph(GraphViewer const&)
  #3 SpaceMITExecutionProvider::Compile(...)
  #4 PartitionOnnxFormatModelImpl(...)
  #5 GraphPartitioner::Partition(...)
  #6 InferenceSession::TransformGraph(...)
  #7 InferenceSession::Initialize()
所有帧都在 libspacemit_ep.so.2.0.6 / libonnxruntime.so.1.24.2+spacemit.a1
```

**影响**：**所有 int8（QDQ）ONNX 模型都无法用 AI Core 加速**，只能退回 CPU
（`providers=['CPUExecutionProvider']` 同一文件跑得好好的）。
这也解释了论坛 t/1744 的现象——不是 DA3 的问题，是 EP 的 QDQ 图构建。

**期望**：
1. `SpineNode` 构造里对缺失的 optional attribute / 子图做判空，**返回 Status 而不是段错误**；
2. 修好后请给一个最小的 QDQ conv 单测（本仓库这份 `yolo11n.q.onnx` 2.9 MB 就够）。
3. 临时规避：int8 模型用 CPU EP；或先 `quantize` 成 `MatMulInteger`/FP16 再试。

---

## Issue 9（中）`onnxruntime.quantization` 模块无法导入：protobuf 描述符冲突直接 abort

**组件**：`python3-onnx` 1.20.0 + ORT `1.24.2+spacemit.a1`（同一进程）

```bash
$ python3 -c "import onnxruntime.quantization"
terminate called after throwing an instance of 'google::protobuf::FatalException'
  what():  CHECK failed: GeneratedDatabase()->Add(encoded_file_descriptor, size):
Aborted (core dumped)
```

- `import onnx; import onnxruntime` ✅（顺序反过来就行）
- `import onnxruntime.quantization` **单独 import 也崩** ❌
- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` **无效**（崩在 ORT 自带的 C++ protobuf）

**影响**：板上**无法做离线量化**（`quantize_static` / `quantize_dynamic` 全部不可用），
而 K3 的卖点之一就是 int8；用户只能去别的机器量化好再拷过来。

**期望**：统一 protobuf 版本 / 让 ORT 的 python 包用 `protobuf` 的 dynamic message，
或至少把 `quantization` 子模块单独打包，别让它把描述符池搞崩。

## Issue 10（严重，已在 1.0.10.4 修复）`vsetvli x0,x0`（keep-vl 形式）之后的向量指令必然 SIGILL

**组件**：内核 `6.18.3-generic` **#1.0.1.4**（Bianbu 4.0.1）的向量状态管理

**摘要**：只要用 keep-vl 写法 `vsetvli x0, x0, e8, m1, ta, ma` 设一次 vl，
**紧随其后的第一条向量指令必定 SIGILL（signal 4 / exit 132）**，
两条核簇都 100% 复现。换成 `rd ≠ x0` 的任意写法立即正常。

**复现**（`gcc -O1 -march=rv64imafdcv`）：

```c
static unsigned char buf[64] __attribute__((aligned(64)));
int main(void){
  /* rd = x0, rs1 = x0：keep-vl 形式 */
  __asm__ volatile("vsetvli zero,%0,e8,m1,ta,ma"::"r"(0UL));
  __asm__ volatile("vle8.v v0,(%0)"::"r"(buf));   /* ← 这里 SIGILL */
  return 0;
}
```

```bash
taskset -c 0 ./v                      # X100：132 (SIGILL)
echo $$ > /proc/set_ai_thread; ./v    # A100：132 (SIGILL)
```

**实测（1.0.1.4）**：

| 写法 | X100 (0–7) | A100 (8–15) |
| --- | --- | --- |
| `vsetvli zero, zero, e8, m1, ta, ma`（keep-vl） | **SIGILL 15/15** | **SIGILL 15/15** |
| `vsetvli a2, zero, e8, m1, ta, ma`（rd≠x0） | ✅ 正常 | ✅ 正常 |
| `vsetvli zero, 1, e8, m1, ta, ma`（显式 vl） | ✅ 正常 | ✅ 正常 |

**机制（我们的推断）**：keep-vl 形式本应"只改配置、保留当前 vl"，
但这一版内核在 HMP 的向量上下文切换里没有区分该形式，
把 `vill`（vector illegal）无条件置位了，下一条向量指令就触发非法指令异常。
由于这属于内核态行为，`/proc/cpuinfo` 与 ISA 串看不出任何异常，排查成本很高。

**修复验证**：升级到 **`6.18.3-generic #1.0.10.4`**（Bianbu 4.0.8）后，
同一份二进制在两簇各连跑 6 次，**全部正常**（`vl=0`、`vle8.v` 通过）：

| 内核 | X100 | A100 |
| --- | --- | --- |
| 1.0.1.4 | SIGILL 15/15 | SIGILL 15/15 |
| **1.0.10.4** | ✅ 6/6 | ✅ 6/6 |

**结论**：这是**内核 bug，不是硬件问题**，1.0.10.4 已修复。
在 1.0.1.4 上编译 RVV 代码时，临时规避是**避免手写 keep-vl 形式的
`vsetvli`**（GCC/LLVM 正常生成的代码很少用这种形式，所以影响面有限，
但手写 intrinsics / 汇编的库会踩到）。

---

## Issue 11（中）两簇 VLEN 不同（256 vs 1024），但 `/proc/cpuinfo` 的 ISA 串完全一样且缺 `supm`

**组件**：内核 `/proc/cpuinfo`、`ext_status`（Bianbu 4.0.1 / 4.0.8 均如此）

**摘要**：X100 与 A100 的**向量长度差 4 倍**（256 bit vs 1024 bit），
但两簇在 `/proc/cpuinfo` 里暴露的 `isa` 字符串**逐字节相同**，
并且都没有声明 `supm`（RVA23 的指针掩码扩展）。
仅凭 `/proc/cpuinfo` 做能力探测的软件会得出错误结论。

**实测（`csrr vlenb`，两版内核都一致）**：

| 核簇 | CPU | `vlenb` | VLEN |
| --- | --- | ---: | ---: |
| X100 | 0–7 | 32 | **256 bit** |
| A100 | 8–15 | 128 | **1024 bit** |

```bash
taskset -c 0  ./vlenb     # X100 → 32
echo $$ > /proc/set_ai_thread; ./vlenb   # A100 → 128
grep -m1 isa /proc/cpuinfo   # 两簇字符串相同，且不含 supm
```

**矛盾点**：板上 `/bin/ls` 的 `Tag_RISCV_arch` 已包含
`supm`、`zic64b`、`zvl128b`，说明 **Bianbu 的用户态二进制已经是按
RVA23U64 基线编译的**；但内核的 `/proc/cpuinfo` 没有把这些能力报出来。

**同一个问题的第二个例子（H 扩展）**：`/proc/cpuinfo` 的 isa 串明确写着
`rv64imafdcv`**`h`**`_...`，但用户态执行 `csrr vsstatus` 在
**X100 和 A100 上都触发 SIGILL**（两簇一致，1.0.8/1.0.10.4 都一样）：

| 核簇 | isa 串含 `h` | `csrr vsstatus` |
| --- | --- | --- |
| X100 (0–7) | 是 | **SIGILL** |
| A100 (8–15) | 是 | **SIGILL** |

也就是说 isa 串声明了 H，但 H 并没有真正对用户态可用（该固件的
OpenSBI 没有打开，或按 K3 的规则只对 X100 开、而实际未生效）。
这条和 `supm` 一样，都说明 **`/proc/cpuinfo` 的 isa 串只能当参考，
不能当作能力探测依据**——要么用 `csrr vlenb` / `riscv_hwprobe`，
要么直接探测指令是否可用（`SIGILL` 兜底）。

**RVA23U64 强制项实测**（板上 GCC 15.2.0，用展开式 `-march=` 逐条探测）：

| 类别 | 扩展 | 结果 |
| --- | --- | --- |
| 基础 | M / A / F / D / C / B | ✅ |
| 向量 | V（RVV 1.0） | ✅ |
| 缓存块 | Zicbom / Zicbop / Zicboz | ✅ |
| 条件运算 | Zicond | ✅ |
| 浮点 | Zfa / Zfhmin / Zfh | ✅ |
| 压缩 | Zcb / Zcmop | ✅ |
| 向量位操作 | Zvbb / Zvkt | ✅ |
| 其他 | Zawrs / Zimop / Zihintntl | ✅ |
| **可选**（非 RVA23 强制） | Zacas / Zabha | **SIGILL**（符合预期） |

**工具链问题**：板上 GCC 15.2.0 **不认 profile 名**，
`-march=rva23u64` / `-march=rva23s64` 一律报
`ISA string must begin with rv32 or rv64`（升级到 `15.2.0-16ubuntu1bb5` 后仍如此），
必须写成展开的扩展列表才能编译。

**期望**：
1. `/proc/cpuinfo` 的 `isa` 按**每簇**真实能力输出（至少 VLEN 要能区分）；
2. 补上 `supm` / `zic64b` / `zvl128b` 等已在实际使用的扩展声明；
3. GCC 支持 `-march=rva23u64` profile 名（这是上游 GCC 14+ 就有的能力）。

**规避**：能力探测用 `csrr vlenb` / `riscv_hwprobe`，不要相信 `/proc/cpuinfo` 的字符串；
跨簇代码必须**运行时**读 `vlenb`，不能假设两簇一致。

---

## Issue 12（严重，1.0.10.4 仍未修）进程内切 A100：向量状态不匹配，栈被写坏 → SIGSEGV

**组件**：内核 HMP `/proc/set_ai_thread` + 向量上下文切换

**来源**：论坛 [t/1482](https://forum.spacemit.com/t/1482)（GCC 在 A100 上"栈溢出"）、
[t/1684](https://forum.spacemit.com/t/1684)（OpenBLAS 作者也提到这条迁移路径会 SIGSEGV）。
我们补一份**最小复现 + 对照实验**，并确认 **1.0.10.4 没有修**。

**复现**（RV 向量内核，和论坛同构）：

```c
static int32_t work(const int16_t *p, int len, int shift){
  int64_t acc = 0; int nc = len & ~3;
  for (int n = 0; n < nc;) {
    size_t vl = __riscv_vsetvl_e16m1((size_t)(nc - n));
    vint16m1x2_t v = __riscv_vlseg2e16_v_i16m1x2(p + 2*n, vl);
    vint16m1_t cr = __riscv_vget_v_i16m1x2_i16m1(v, 0);
    vint16m1_t ci = __riscv_vget_v_i16m1x2_i16m1(v, 1);
    vint32m2_t pw = __riscv_vwmacc_vv_i32m2(
        __riscv_vwmul_vv_i32m2(cr, cr, vl), ci, ci, vl);
    pw = __riscv_vsra_vx_i32m2(pw, shift, vl);
    acc += __riscv_vmv_x_s_i64m1_i64(
        __riscv_vwredsum_vs_i32m2_i64m1(pw, __riscv_vmv_s_x_i64m1(0,1), vl));
    n += (int)vl;
  }
  return (int32_t)acc;
}

work(a, len, 2);                    /* 先在 X100 上跑向量，VLEN=256 */
write(open("/proc/set_ai_thread", O_WRONLY), "self_pid", ...);
work(a, len, 4);                    /* 已在 A100，VLEN=1024 → 栈被写坏 */
```

**实测（`pingji-k3`，Bianbu 4.0.8，内核 `6.18.3-generic #1.0.10.4`）**：

| 场景 | 结果 |
| --- | --- |
| 纯 X100，跑两遍向量（不切换） | ✅ `rc=0`，结果正确 |
| **exec 前**切 A100（`echo $$ > /proc/set_ai_thread; exec prog`） | ✅ `rc=0`，结果正确 |
| 进程内跑过向量再切 A100（`-fstack-protector-all`） | ❌ **`rc=134`** 栈保护触发（计算值仍正确，canary 被写坏） |
| 同上，`-fno-stack-protector` | ❌ **`rc=139` SIGSEGV**（3/3 稳定） |

**关键点**：不是"你显式写过向量指令"才会中招 —— **glibc 启动时 `memcpy` 就用 RVV**，
所以任何正常启动的进程在切换前都已经持有 VLEN=256 的向量状态。
论坛里 `aix` / `k3_taskset` 之所以有效，就是它们在 **exec 之前**完成切换。

**期望**：
1. 切换核簇时按目标 hart 的 `vlenb` 完整保存/恢复向量状态（或直接丢弃并重置 `vill`），
   不要把旧 VLEN 的寄存器内容按新 VLEN 解释；
2. 或者检测到线程已有向量上下文时**拒绝切换**并返回 `-EBUSY`，而不是静默写坏用户栈；
3. 文档里把"必须在 exec 前切换"写成硬性前提（现在是社区自己踩出来的）。

---

## Issue 13（中）PMU `stalled-cycles-frontend` / `stalled-cycles-backend` 语义疑似对调

**组件**：K3 X100 的 PMU 通用事件映射（perf `PERF_COUNT_HW_STALLED_CYCLES_FRONTEND` = 0x7、`..._BACKEND` = 0x8）

**来源**：论坛 [t/1500](https://forum.spacemit.com/t/1500)（厂商已回复"会尽快修复"）。

**我们的验证**（`perf stat`，X100 单核，各跑 1.5s）：

| 负载 | `stalled-cycles-frontend` | `stalled-cycles-backend` |
| --- | ---: | ---: |
| 随机内存访问（pointer chase，L1D/LLC miss 为主） | **27.28%** | 3.85% |
| 间接跳转 / 不可预测分支（函数指针表） | 8.64% | **91.44%** |

按标准语义应当是"内存 miss → backend 高、控制流 → frontend 高"，实测**正好相反**：

| 负载 | 期望表现 | 实测表现 |
| --- | --- | --- |
| 随机内存访问 | backend 高 | **frontend 高** |
| 间接跳转 | frontend 高 | **backend 高** |

**影响**：任何按标准语义调优的 perf 分析都会得出相反结论（例如把内存瓶颈误判为取指瓶颈）。

**期望**：修正两个通用事件到硬件事件的映射，或提供 K3 的 PMU 事件对照表
（`mhpmevent3..31` 的 event id / 过滤域），并在 `perf list` 里给出 K3 专属别名。

---

## Issue 14（中）VPU 硬编：`h264_stcodec` 静默忽略码率参数，`h264_v4l2m2m` 直接失败

**组件**：MPP VENC 的 ffmpeg 封装（`h264_stcodec`、`h264_v4l2m2m`）

**来源**：论坛 [t/1405](https://forum.spacemit.com/t/1405)、[t/1299](https://forum.spacemit.com/t/1299)。

**现象 A：`-b:v` 完全无效**

```bash
for br in 1M 8M 20M; do
  ffmpeg -nostdin -f lavfi -i testsrc2=size=1920x1080:rate=30 -t 3 \
         -pix_fmt yuv420p -c:v h264_stcodec -b:v $br /tmp/o_$br.h264 -y
done
ls -l /tmp/o_*.h264
```

| `-b:v` | 输出大小 | 实际码率 |
| ---: | ---: | ---: |
| 1M | 3,263,994 B | 8.70 Mbps |
| 8M | 3,263,994 B | 8.70 Mbps |
| 20M | 3,263,994 B | 8.70 Mbps |

三档输出**字节数完全相同**——参数被静默丢弃，编码器固定在一个约 8.7 Mbps 的档位。
`ffmpeg -h encoder=h264_stcodec` 里也确实没有任何码率相关选项，但 ffmpeg 侧
没有报 "option not found"，所以用户完全无从发现。

**现象 B：`h264_v4l2m2m` 直接失败**

```bash
ffmpeg -nostdin -f lavfi -i testsrc2=size=1280x720:rate=30 -t 2 \
       -pix_fmt yuv420p -c:v h264_v4l2m2m -b:v 2M /tmp/o.h264 -y
# → Task finished with error code: -22 (Invalid argument)
# → Nothing was written into output file
```

输出 0 字节，退出即报 `-22`，没有任何更具体的提示。

**期望**：
1. `h264_stcodec` 要么实现 `-b:v`/`-maxrate`/`-bufsize`，要么在未实现时明确报错；
2. 把当前实际使用的码率/量化参数打到日志里，便于确认；
3. `h264_v4l2m2m` 区分"格式不支持"和"驱动缺陷"（`-22` 太笼统）。

**影响**：用 ffmpeg 硬编做录制/推流的用户会得到"码率调了没用"或"编码器不可用"，
两种情况都没有明确错误信息。

---

## 附：正向结论（一并反馈）

- **VPU 硬编**（修好权限后）：1080p150 帧 1.087s，比 libx264 veryfast 快 4.8×、CPU 少 ~144×。
- **视觉模型**：int8 YOLOv8n 单帧 **14.3ms**（emotion 7.1ms），14 个模型全部可用。
- **VAD**：10s 音频 **288ms**（34× 实时）。
- **TTS 流式**：按句 append 每句 ~0.5s 出声，是整套栈里最好用的一块。
- **A100 的正确定位**：int8/IME 算子专用核；`/proc/set_ai_thread` 一行即可切换，机制本身很干净。
- **EP 的 Clip / 5 维输入**：Bianbu 4.0.8 + `libspacemit_ep.so.2.0.6` 上，4D/5D、
  opset 6/13、标量/`[1]` 张量 min/max 全部正常；论坛 t/1762 的
  OpenHarmony 崩溃在我们这套上不复现（疑似 OH 侧 EP 是另一个构建）。
- **TCM**：`/dev/tcm` 存在（3 MB，8×384 KB 块），`CONFIG_SPACEMIT_TCM=y`；
  `/dev/tcm_sync_mem` 缺失但厂商已确认不影响性能。补充实测：直接 `mmap` 后
  `memcpy` 只有 **~0.01 GB/s**（同尺寸 heap 15.8 GB/s、64 MB DDR 流式 5.4 GB/s），
  说明这块 TCM 不能当 CPU scratchpad 用，而应走 AI 核/EP 的专用通路。
- **x86_64 qemu-user + binfmt**：正常工作。`qemu-x86_64` 与 binfmt 直跑
  （`/proc/sys/fs/binfmt_misc/qemu-x86_64` 已注册）都能跑 x86_64 static busybox，
  `uname -m` 返回 `x86_64`。论坛 [t/865](https://forum.spacemit.com/t/865) 的
  SIGILL 在我们这套 Bianbu 4.0.8 上不复现。
- **Vulkan**：默认 ICD 只枚举到 **llvmpipe**（软件渲染，`apiVersion 1.4.335`）；
  强制 `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/powervr_mesa_icd.json` 报
  `Failed to detect any valid GPUs in the current config`。SSH/headless 场景下
  PowerVR 设备不可见，需有真实图形会话再验证
  [t/1340](https://forum.spacemit.com/t/1340) 的 `VK_EXT_external_memory_dma_buf`。
