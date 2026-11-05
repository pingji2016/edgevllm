# SpaceMiT 论坛问题实测验证记录（K3）

> 环境：K3 Pico-ITX · Bianbu **4.0.8**（resolute）· 内核 `6.18.3-generic #1.0.10.4` ·
> Python **3.14.4** · `python3-spacemit-ort 2.0.7+2` · `spacemit-onnxruntime 2.0.7+2` ·
> `spacemit-tcm 3.0.1` · ONNX Runtime `1.24.2+spacemit.a1`
>
> 目的：把官方论坛里「能验证但没人给数据」的帖子逐条实测一遍。
> 结论分三类：**✅ 复现**、**❌ 不复现**、**⚠️ 与官方规格不符**。
> 上一轮（STREAM / stcodec 码率 / qemu / Vulkan / TCM 带宽）见
> [`spacemit-issues.md`](spacemit-issues.md) 的 Issue 1、14 与附录。

## 汇总

| 论坛帖 | 问题 | 结论 | 关键数据 |
| --- | --- | --- | --- |
| [t/1465](https://forum.spacemit.com/t/1465) | K3 JPEG 编码支持情况 | ⚠️ **规格不符** | 4K@**85 fps**（规格 60），但编码上限 **4096×4096**，非 8192²；4097 宽会**静默裁成 4096** |
| [t/1439](https://forum.spacemit.com/t/1439) | bianbu chromium 是否强制 `--no-sandbox` | ❌ **不复现** | 默认带沙盒 rc=0；setuid helper 在位；启动项无 `--no-sandbox` |
| [t/1716](https://forum.spacemit.com/t/1716) | `python3-spacemit-ort` 用法 / import 顺序 | ✅ **规则确认** | 只 import onnxruntime → 仅 CPU EP；`import spacemit_ort` 后 → EP 出现 |
| [t/1308](https://forum.spacemit.com/t/1308) | A100 核 TCM 状态不一致 | ❌ **不复现** | X100 写入 → A100 读 → X100 读，校验和完全一致 |
| [t/1311](https://forum.spacemit.com/t/1311) | A100 可否跑非 AI 任务 | ✅ **可以但慢** | sha256 **3.6×** 慢、gzip **2.3×** 慢 |
| [t/1503](https://forum.spacemit.com/t/1503) | 知了向量化后黑屏 | ❌ **不是软件问题** | 原帖作者后来确认是 TYPE-C 供电不足 |
| [t/935](https://forum.spacemit.com/t/935) | 60 TOPS 论文指标 | ⚠️ **当前栈复现不了** | EP fp32 MatMul 上限 **86.5 GFLOPS**（=0.087 TOPS）；A100 上无提升；int8 模型全部无法加载 |
| [t/1127](https://forum.spacemit.com/t/1127) | EP 上 reshape 维度报错 | ❌ **2.0.7 不复现** | 静态 / `-1` / 动态 batch 三种 reshape 全部通过 |
| [t/1342](https://forum.spacemit.com/t/1342) | Spine-Triton `make_block_ptr` | ⏸ **无法验证** | 板子上没有 triton（私有源），`import triton` 失败 |

---

## 1. t/1465：JPEG 硬编能跑 4K@85fps，但上限只有 4096×4096

**官方回复**：JPEG（Baseline sequential）最大分辨率 **8192×8192**，单路最高 4K@60fps。

**实测编码吞吐**（`mjpeg_stcodec`，4K 120 帧，`-q:v 5`）：

| 轮次 | 耗时 | 帧率 |
| ---: | ---: | ---: |
| 1 | 1.41 s | **85.1 fps** |
| 2 | 1.43 s | 83.9 fps |
| 3 | 1.41 s | 85.1 fps |

输出 34,325,389 B（平均 286,044 B/帧，约 68.6 Mbps）。**吞吐超过规格的 4K@60**。

**分辨率上限**（单帧，`testsrc2`）：

| 请求尺寸 | rc | 实际输出尺寸 | 说明 |
| --- | ---: | --- | --- |
| 1920×1080 | 0 | 1920×1080 | 正常 |
| 3840×2160 | 0 | 3840×2160 | 正常 |
| 4096×4096 | 0 | 4096×4096 | **上限** |
| 4097×4096 | 0 | **4096×4096** | ⚠️ **静默裁掉 1 列**，无任何警告 |
| 4100×1000 | 171 | — | `do not support the size` |
| 5120×2880 | 171 | — | 同上 |
| 7680×4320 | 171 | — | 同上 |
| 8192×8192 | 171 | — | 同上 |

**解码侧同样到不了 8192**：

| 输入 | 解码器 | 结果 |
| --- | --- | --- |
| 4096×4096 JPEG | `mjpeg_stcodec` | ✅ rc=0（日志 `Resolution changed.`） |
| 8192×8192 JPEG（软件编码 1,580,789 B） | `mjpeg_stcodec` | ❌ rc=171 `Stcodec decoder do not support the size` |
| 同上 | 软件 `mjpeg` | ✅ 输出 100,663,296 B（8192×8192×1.5） |

**复现**：

```bash
# 4097 宽：返回成功，但产物是 4096 宽
ffmpeg -nostdin -f lavfi -i testsrc2=size=4097x4096:rate=1 -frames:v 1 \
       -pix_fmt yuv420p -c:v mjpeg_stcodec -q:v 5 out.jpg
ffprobe -v error -show_entries stream=width,height -of csv=p=0 out.jpg
# → 4096,4096

# 8192：直接拒绝
ffmpeg -nostdin -f lavfi -i testsrc2=size=8192x8192:rate=1 -frames:v 1 \
       -pix_fmt yuv420p -c:v mjpeg_stcodec -q:v 5 out.jpg
# → Stcodec encoder do not support the size(8192 x 8192), too big or too small.
```

**影响**：需要 8K 静帧的场景（文档扫描、拼接监控、印刷预览）用不了硬编；
更危险的是 4097 这种「差一点」的尺寸会**静默产出错误尺寸的图**，调用方无法察觉。

**期望**：① 文档把上限改成 4096×4096（或说明硬编/硬解各自上限）；
② 超限时返回明确错误，而不是静默裁剪。

---

## 2. t/1716：`import spacemit_ort` 的真实规则

论坛答主说「注意 import 顺序，需要先 `import spacemit_ort`，再 `import onnxruntime`」。
实测这条**结论对，但表述过严**：

| 场景 | `ort.get_available_providers()` | 能否用 EP |
| --- | --- | --- |
| 只 `import onnxruntime` | `['CPUExecutionProvider']` | ❌ |
| `import onnxruntime` → 再 `import spacemit_ort` | `['SpaceMITExecutionProvider', 'CPUExecutionProvider']` | ✅ |
| `import spacemit_ort` → `import onnxruntime` | 同上 | ✅ |

**真正的规则是：`import spacemit_ort` 必须发生在创建 `InferenceSession` 之前。**

| 场景 | session 的 providers |
| --- | --- |
| 先建 session（只 import onnxruntime），之后才 import spacemit_ort | `['CPUExecutionProvider']`（已建好的 session 不会变） |
| import 之后再建 session | `['SpaceMITExecutionProvider', 'CPUExecutionProvider']` |

也就是说：**已经创建的 session 不会被"补救"**，但 import 顺序本身不限。
论坛上常见的报错 `SpaceMIT EP does not support Compile`（t/1716 原帖）和
「没加速」多半是 session 建早了。

**顺带**：`import onnxruntime` 静默只给 CPU EP，不报错、不提示，这是最容易踩的坑。
建议 EP 包在 import 时打印一行 `SpaceMIT EP registered`。

---

## 3. t/1439：Bianbu 4.0.8 的 chromium **没有**强制 `--no-sandbox`

原帖称「bianbu 源的 chromium 似乎被强制以 `--no-sandbox` 启动」。实测：

| 检查项 | 结果 |
| --- | --- |
| `chromium --headless=new --dump-dom about:blank`（默认，带沙盒） | ✅ rc=0，1.4 s，输出 40 B |
| 同上加 `--no-sandbox` | ✅ rc=0，0.9 s |
| `/usr/lib/chromium/chrome-sandbox` | `-rwsr-xr-x root root`（setuid 在位） |
| `unshare -Ur true` | ✅ 通过 |
| `max_user_namespaces` | 62,093 |
| `/usr/share/applications/chromium.desktop` 的 `Exec` | `/usr/bin/chromium %U`（无 `--no-sandbox`） |
| `/etc/chromium.d/*` 中含 `no-sandbox` 的文件 | 无 |

**结论**：4.0.8 上沙盒默认可用，原帖描述的现象不复现（可能是 4.0.1/旧版镜像）。
代价是启动慢约 0.5 s（1.4 s vs 0.9 s）。

> 局限：这是 headless 测试；真实桌面会话下是否有额外 flag 由 `.desktop` 决定，
> 上面已确认桌面启动项没有加 `--no-sandbox`。

---

## 4. t/1308：A100 核 TCM 状态不一致 —— 未复现

原帖称 ORT 的 SpaceMiT EP 与 Spine-Triton 用了不同的 TCM 管理策略，导致状态不一致。
我们按「写 → 切核簇 → 读」的方式直接验证内存内容：

```c
/* 打开 /dev/tcm，mmap 一个 384 KB block，写模式填充 (i*37+11)，读模式算滚动校验和 */
```

| 步骤 | 校验和 | 前 4 字节 |
| --- | ---: | --- |
| X100（cpu0）写入 | 3,003,218,137 | `26 e2 79 db` |
| X100（cpu0）读回 | 3,003,218,137 | `26 e2 79 db` |
| **A100（cpu8，exec 前切核）读回** | 3,003,218,137 | `26 e2 79 db` |
| 再切回 X100 读回 | 3,003,218,137 | `26 e2 79 db` |

**结论**：跨核簇访问同一 block，数据完全一致，没有观察到"状态不一致"。
原帖的问题可能只出现在「EP + Spine-Triton 同时初始化 TCM 管理器」的场景，
纯 `mmap` 路径下不可复现。

> 附带结论（上一轮）：`/dev/tcm` 直接 mmap 的 `memcpy` 只有 ~0.01 GB/s，
> 它是给 AI 核/EP 的通路，不适合 CPU 当 scratchpad。

---

## 5. t/1311：A100 可以跑非 AI 任务，但慢 2–3.6×

原帖问「K3 中的 A100 核心可以调度非 AI 任务吗」。答案是可以（`/proc/set_ai_thread`
之后就是普通 hart），但性能是这样：

| 负载 | X100（cpu0） | A100（cpu8） | 倍率 |
| --- | ---: | ---: | ---: |
| `openssl speed -evp sha256` | **1,438,294 kB/s**（@16k） | 402,403 kB/s | **3.6× 慢** |
| `gzip -9`（40 MB 随机数据） | **2.23 s** | 5.03 s | **2.3× 慢** |
| CoreMark（单核，上轮） | 4,972 | 1,539 | 3.2× 慢 |

原因是标量性能与缓存（A100 1.8 GHz / L1D 32 KB / L2 1 MB，X100 2.15 GHz /
64 KB / 4 MB）。**能用，但没有理由把通用任务放上去**；A100 的价值在 IME 矩阵算子。

---

## 6. t/1503：知了向量化黑屏 —— 已由原帖作者定性为供电问题

原帖描述了「Bianbu 4.0.4 下 `apt install zenow`，向量化后立刻黑屏、桌面重新登录」。
该帖 #5 楼主自己更新：

> 后来发现是 TYPE-C 的供电不足，导致 NPU 功耗拖垮了供电，不是系统的问题。

我们不再重复测试。**结论：不是软件 bug**，用带 PD 的足额电源即可。
这一类"跑 AI 负载就黑屏/重启"的现象，建议统一先查供电（K3 + NPU 峰值电流较高）。

---

## 7. t/935：60 TOPS 在当前软件栈上复现不了

论文标题是《SpacemiT K3: A RVA23 RISC-V AI CPU with 60 TOPS AI Compute》。
我们用 ONNX Runtime + SpaceMiT EP 2.0.7 做端到端测量。

**fp32 MatMul（EP vs CPU EP，8 线程）**：

| 模型 | 每次算力 | EP | CPU EP | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| MatMul 64×4096×4096 | 2.15 GFLOP | 81.5 GFLOPS | 40.7 GFLOPS | 2.0× |
| MatMul 512×4096×4096 | 17.2 GFLOP | **87.0 GFLOPS** | — | — |
| MatMul 1024×4096×4096 | 34.4 GFLOP | **86.8 GFLOPS** | — | — |

N 从 64 加到 1024，GFLOPS 基本不变（81.5 → 86.8），说明已经算力受限、不是带宽受限。

**X100 vs A100**（同一个 1024×4096×4096 MatMul）：

| 核组 | 耗时 | GFLOPS |
| --- | ---: | ---: |
| X100（cpu0-7） | 397.4 ms | **86.5** |
| A100（cpu8-15，exec 前切核） | 402.7 ms | 85.3 |

**A100 没有任何提升**，说明 fp32 MatMul 走的是向量路径，没有用上 IME。
只有 int8 才会走 IME2。

**int8 路径：三个真实模型全部无法加载**

| 模型 | 类型 | 结果 |
| --- | --- | --- |
| `dino-vits16/onnx/model_quantized.onnx`（22 MB） | QDQ int8 | ❌ EP 初始化失败 |
| `von13_int8.onnx`（379 MB） | int8 | ❌ EP 初始化失败 |
| `sherpa/encoder-...int8.onnx`（174 MB） | int8 | ❌ EP 初始化失败 |
| 合成 `QLinearConv` | QLinear int8 | ❌ EP 初始化失败 |

统一报错：

```
Exception during initialization: compiler_depend.ts:371
static const onnxruntime::logging::Logger&
onnxruntime::logging::LoggingManager::DefaultLogger()
Attempt to use DefaultLogger but none has been registered.
```

这和 Issue 8（EP 在 QDQ 模型上崩溃）是同一类问题：现在不崩了，但换成了初始化异常，**int8 依然不可用**。

**结论**：

- 当前 `python3-spacemit-ort 2.0.7+2` + 通用 ONNX 模型，**可用算力上限约 87 GFLOPS（fp32）**；
- 60 TOPS 是 IME2 的峰值指标，需要 int8 才能触发，而 int8 路径在我们这版上**完全跑不起来**，
  所以无法通过 ORT 复现，只能标注为「未能验证」；
- 即使按 int8 ≈ 4× fp32 粗估（约 0.35 TOPS），距离 60 TOPS 仍有约 170× 的差距——
  这个差距来自「峰值 vs 可达」，不是芯片虚标，但**官方文档需要说明怎么才能接近峰值**
  （IME intrinsics？Spine-Triton？专用量化工具？）。

> 附：单节点 `Conv 256ch 3×3 56×56`（3.7 GFLOP）在 EP 下仍由 CPU 执行
> （40–43 GFLOPS），说明 EP 并不会把任意单个 Conv 节点 offload；
> 真实模型的加速来自整图融合，不能靠单算子微基准衡量。

---

## 8. t/1127：reshape 维度问题在 EP 2.0.7 上不再复现

原帖（2.0.2+beta1）在跑 `mixformer_v2.onnx` 时报 reshape 维度错误。我们无法拿到该模型，
改用三种常见写法构造最小模型（`Reshape → MatMul`，输入 `[1,3,8,8]`）：

| 写法 | EP 结果 | CPU 结果 |
| --- | --- | --- |
| 静态目标 `[1,192]` | ✅ rc=0，输出 `(1,192)` | ✅ |
| `-1` 推导 `[1,-1]` | ✅ rc=0 | ✅ |
| 动态 batch（`Shape→Slice→Concat→Reshape`） | ✅ rc=0 | ✅ |

动态写法连跑 3 次全部 rc=0，没有崩溃。

**结论**：2.0.7 上这三类 reshape 都正常。原帖的失败很可能是 2.0.2 的旧问题，
官方建议的「升级 `spacemit-ort` + 用 `xslim` 精简模型」方向正确；但因为没有
`mixformer_v2` 本体，**不能 100% 断言原模型已修**。

> `xslim` 在板子上未安装（`pip install xslim` 可从公网装），本次未验证。

---

## 9. t/1342：Spine-Triton 无法验证

`import triton` 直接失败，`pip list` 里也没有 `triton` / `spine-triton`；
板子上只有 `onnx`、`onnxruntime`、`spacemit-ort`、`llama.cpp-tools-spacemit` 等。
Spine-Triton 的安装包在私有源，无法从公网获取，因此 `make_block_ptr` 相关问题**未验证**。

---

## 附：本轮未跑但已排除的帖子

| 帖子 | 原因 |
| --- | --- |
| t/1342 / t/1298 / t/1320 Spine-Triton 相关 | 需要私有源里的 triton，板子上没装（见第 9 节） |
| t/1341 `libspine_gemm_bridge.so` | 楼主已在 #3 自述解决；该库不在 `spacemit-tcm` 包内 |
| t/1127 reshape 维度（20 回复） | 构造的三个 reshape 最小例都正常，但无 `mixformer_v2` 本体（见第 8 节） |
| t/1612 / t/1748 / t/1727 | 分别需要 K1 场景 / amd64·arm64 / RX550，K3 上不可验证 |
