# MNN 在 K3 上跑 LLM：标准 RVV 5 tok/s vs IME2 226 tok/s

> 2026-10-21 · `pingji-k3`（SpacemiT X100）· Bianbu · gcc 15.2.0 · cmake 4.2.3
> MNN master `d662ce36` · 模型 `Qwen3-0.6B-MNN`

把 MNN（阿里开源推理引擎）拉到板子上编了两版——**标准 RVV** 和 **SpaceMiT K3 IME2**——跑同一套
`llm_bench`，顺手把 2026 年 4 月以来上游那一批 RISC-V PR 复盘了一遍。

**主要结论**：在 K3 上跑量化 LLM，**必须开 `MNN_RVV_SPACEMIT_IME2`**。只开 `MNN_USE_RVV` 不会报错、
测试也全过，但主算子会静默回退到标量实现，速度差 **40 倍**。

---

## TL;DR

| 构建 | pp512 | tg128 | 相对标准 RVV |
| --- | ---: | ---: | ---: |
| 标准 RVV（`-DMNN_USE_RVV=ON`） | 5.62 ± 0.01 | 4.97 ± 0.02 | 1× |
| **+ IME2（`-DMNN_RVV_SPACEMIT_IME2=ON`）** | **225.88 ± 0.40** | **30.40 ± 0.98** | **40.2× / 6.1×** |

模型 `Qwen3-0.6B-MNN`（W4 块量化），8 线程，pp512 / tg128。

| 问题 | 结论 |
| --- | --- |
| 标准 RVV 为什么这么慢 | **RVV 目录里只有 FP32 的 GEMM**；LLM 低内存模式用的"W4 块量化 + 边算边反量化"GEMV/GEMM **没有 RVV 实现**，回退标量 C++ |
| 是不是 runtime gate 把 RVV 关了 | 不是。`riscv_hwprobe(2)` 实测 `key=4 → 0x38dbff9ffff00ff`，bit2(V)=1 |
| IME2 能直接编吗 | **不能**，CMake 给的 march 少了 `zfh` / `zihintpause`，需手动补（见 §3.2） |
| 该开 KV 量化吗（`-qa`） | **不要**。`-qa 1` pp 222→142，`-qa 2` 降到 111，纯负优化 |
| 能对上官方 381 / 54.5 吗 | 拿到的是 0.59× / 0.56×，差距原因见 §4.4（**假设，未验证**） |

---

## 0. 环境【实测】

| 项 | 值 |
| --- | --- |
| 芯片 | `SpacemiT(R) X100`（`/proc/cpuinfo model name`） |
| 核 | `/proc/cpuinfo` 见 16 个处理器，但**进程可用 `nproc=8`**（8×X100 + 8×A100，HMP 分组，见 [k3-hardware](../k3-hardware/k3-hardware.md)） |
| 内核 | `Linux 6.18.3-generic riscv64` |
| 内存 / 磁盘 | 15971 MB / 117 GB（用 32 GB） |
| 工具链 | gcc 15.2.0 (Bianbu)、cmake 4.2.3、git 2.53 |
| **实测 VLEN** | **256 bit**（e8m1 一次 32 字节） |
| ISA | `rv64imafdcvh` + `zve32f/zve64d/zvfh/zvbb/zvk*` + `zihintpause` + `zfh` |

VLEN 实测方法（比查文档可靠）：

```c
unsigned long vl;
__asm__ volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(vl));  // vl*8 = VLEN
```

---

## 1. 上游 RVV 支持的时间线（2026-04 ~ 10）

MNN 的 RISC-V 支持其实很新，一年内从"能编"走到"能跑 LLM"：

| 时间 | PR | 性质 |
| --- | --- | --- |
| 2026-04-02 | #4331 | **首次引入 RVV**（+ 修 tokenizer 头文件） |
| 2026-05 ~ 07 | #4359 #4433 #4426 #4484 #4508 #4531 #4588 #4590 #4690 | Int8 函数、common opt、SiLu/Exp、Norm/Gelu、pack 修复、`riscv_hwprobe` 能力探测 |
| 2026-07-31 | **#4702** | SpacemiT **K3 IME2** 专项：38 文件 **+11020/-61**，阿里内部仓库同步 |
| 2026-09-11 ~ 09-16 | **15 个 PR** | 标准 RVV 内核"补全潮"，合计约 **+3785/-236**，15 个里 12 个作者是 XJJ |
| 2026-10-15 | #4869 | INT8 求和 tail 修复（最后一个） |
| 未合入 | #4907 | 4 字节 raster bitcopy 广播优化（2026-09-23 开） |

### 1.1 9 月那批到底改了什么

| 主题 | PR | 关键内容 | 实测（kernel 级） |
| --- | --- | --- | --- |
| **LLM 算子** | #4843 | KV cache 量化（`MNNQuantAttentionKey/Value.cpp`，661 行） | key 1.79× / value 3.23× |
| | #4841 | attention mask（显式 + 三角，`MNNAttentionMaskQK.cpp`） | 1.27× / 4.16× |
| | #4842 | linear attention（rank-one update、gated delta） | 5.81× |
| **卷积 / 图像** | #4874 | C4 bilinear sampler（`vl` 固定 4，`rtz+vmfge` 代替 `+0.5`） | 38.9 → 15.2 ns/像素（2.55×） |
| | #4821 | AbsMax 改沿 channel 向量化；修 depthwise kernel **死代码** | — |
| | #4839 | 注册 matrix kernels + depthwise 反卷积；修 `MNNMatrixProd` 可达性 | — |
| | #4879 | depthwise 输入 padding 的 memcpy → `MNNMemcpyBytes_RVV` | — |
| | #4868 | 稀疏 im2col blit（放弃 gather/scatter，改循环方向） | 4.9× |
| **Int8** | #4867 | Int8 Min/Max（并主动删掉自己写的 ScaleAndAddBias） | 2.31× / 2.26× |
| | #4838 | count/min/max + Int8 ReLU | 2.89× / **17.78×** |
| | #4547 | QuantizedAdd（SG2044 上测） | 4.30× |
| | #4869 | INT8 求和 tail：tail-agnostic 指令覆盖非活跃 lane | 17 通道时 `[-13,-11]` vs 正确 `[17,34]` |
| **布局** | #4840 | C4 pack/unpack，坚持 pack=4、**不改 tensor ABI** | 2.59× / 2.08× |
| **调度正确性** | #4871 | 区分编译期 `MNN_USE_RVV` 与运行期 `supportRVV`；6 个函数在无 V 扩展的 C906/C908 上会崩 | — |

### 1.2 两个值得记住的教训

1. **"内核存在但从未被调用"反复出现三次**（#4821 / #4839 / #4846）：C++ linkage 与 C linkage 同名、
   或缺少 `CoreFunctions` 槽位 → 链接期覆盖后调用点仍解析到通用实现，优化是空转。
   **→ 看老版本 MNN 的 RVV 加速别只看有没有那个 .cpp，要确认 dispatch 真接上了。**
2. **验证标准极高**：每个 PR 都要求 native + QEMU **VLEN 128/256/512/1024** 四档、
   Qwen logits **逐字节一致**、独立 oracle、AB/BA 多进程测速。原因是 FMA 收缩和 round 顺序一改就漂。

---

## 2. 构建：标准 RVV

```bash
git clone --filter=blob:none https://github.com/alibaba/MNN.git   # 98 MB，40 秒
cmake -S MNN -B mnn-build/rvv \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_USE_RVV=ON \
  -DMNN_BUILD_TEST=ON \
  -DMNN_BUILD_LLM=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DMNN_BUILD_TOOLS=ON
cmake --build mnn-build/rvv -j8
```

- 编译 **10 分 21 秒**（CPU 时间 72 分），0 错误
- 产物：`libMNN.so` / `libllm.so` / `run_test.out` / `llm_bench` / `llm_demo`
- 核实编进去了：`libMNN.so` 含 **67 个 RVV 符号、3935 条 `vsetvli`**；`MNNCPU` 的 `flags.make` 里带 `-DMNN_USE_RVV`

---

## 3. IME2：快，但默认编不过

### 3.1 直接开的后果

```
-DMNN_RVV_SPACEMIT_IME2=ON   →  编译失败（EXIT=2）
```

CMake 给 IME2 目标算的 march 是 `${MNN_RVV_BASE_MARCH}_xsmtvdotii`，默认 `MNN_RVV_BASE_MARCH=rv64gcv`。
但几个 kernel 里用了半精度标量指令和 `pause`，于是连挂三次：

| 轮次 | 报错 | 缺的扩展 |
| --- | --- | --- |
| 1 | `unrecognized opcode 'flh'` / `'fcvt.s.h'` | `zfhmin` |
| 2 | `unrecognized opcode 'fmul.h'` | `zfh`（完整半精度） |
| 3 | `unrecognized opcode 'pause'` | `zihintpause` |

### 3.2 可用的构建命令

```bash
cmake -S MNN -B mnn-build/ime2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_USE_RVV=ON -DMNN_RVV_SPACEMIT_IME2=ON \
  -DMNN_BUILD_TEST=ON -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_TOOLS=ON \
  -DMNN_RVV_MARCH="rv64gcv_zfh_zihintpause_zba_zbb_zbs"
cmake --build mnn-build/ime2 -j8
```

- 编译 **6 分 30 秒**，0 错误；`libMNN.so` 里 152 个 spacemit/ime2 符号
- 板子这几条扩展都支持（`zfh` / `zihintpause` 在 ISA 里），**是 CMake 少写，不是硬件不支持**

> 这一条值得给上游提 issue：`source/backend/cpu/riscv/CMakeLists.txt` 的
> `MNN_SPACEMIT_IME2_MARCH` 应至少包含 `zfh` 与 `zihintpause`，否则默认配置编不过。

---

## 4. 基准数据

### 4.1 主结果（同一台板、8 线程、同一模型）

| 构建 | pp512 | tg128 | bench 总耗时 |
| --- | ---: | ---: | ---: |
| 标准 RVV | 5.62 ± 0.01 | 4.97 ± 0.02 | 482 s |
| **IME2** | **225.88 ± 0.40** | **30.40 ± 0.98** | **36 s** |

### 4.2 算子级拆解（`llm_bench --profile`，p32 / n8）

| 阶段 | 算子 | 标准 RVV | IME2 | 变化 |
| --- | --- | ---: | ---: | ---: |
| prefill | **Convolution**（= Linear/MatMul） | **5,540,067 µs** | **66,414 µs** | **83×** |
| prefill | Attention | 8,768 µs | 8,499 µs | 1.03× |
| decode | **Convolution** | **1,580,981 µs** | **209,094 µs** | **7.6×** |
| decode | Attention | 10,749 µs | 8,828 µs | 1.2× |

Convolution 占 prefill 的比例：**98.6% → 22%**。IME2 之后 decode 的瓶颈转到 **Attention**
（224 次调用共 8.8 ms，约 39 µs/次）。

### 4.3 标准 RVV 只有 5 tok/s 的根因链

1. `--profile`：**Convolution 占 prefill 98.6%** → 问题在主算子，不在框架调度
2. 采样整机 CPU：**只有 42%（约 3.4/8 核）** → 不是线程没起来，是单核算力被浪费
3. 直接调 `riscv_hwprobe(2)`：`key=4 → value=0x38dbff9ffff00ff`，**bit2(V)=1**
   → `supportRVV` 为真，**不是 runtime gate 把 RVV 关了**
4. 回源码：`source/backend/cpu/riscv/rvv/` 只有 **FP32** 的
   `MNNPackedMatMulFP32 / MNNPackedMatMulRemainFP32` 和一个 `MNNGemmInt8AddBiasScale_16x4_Unit_RVV`；
   而 `Int8FunctionsOpt.cpp` 里**没有** LLM 低内存模式所用的"W4 块量化、边算边反量化"
   那个 `MNNPackedMatMul` 的 RVV 版本

> 结论：**5 tok/s 不是编译错误，而是标准 RVV 路径缺少量化 GEMM 内核，主算子静默回退标量 C++。**
> 这种情况下测试照样全过、日志一句警告都没有 —— 只看单测结果会被骗过去。

### 4.4 和官方 381 / 54.5 的差距：三条已查证，一条仍未验证

我们拿到 221~227 / 30~31，约为官方文档（`docs/perf/riscv_k3_ime2_asymmetric_w4b64.md`）的 0.58~0.60×。
原来列的三个假设，2026-10-21 又上板验证了一轮：

| 假设 | 结论 | 实测依据 |
| --- | --- | --- |
| 向量宽度不同（官方写"1024-bit"） | **不适用**。本板确实有 1024-bit 的核，但那是 A100，而 MNN 在 A100 上比 X100 **慢约 4 倍** | A100 实测 `VLEN=1024`；p8/tg2 → **10.75 / 7.42**（X100 同负载 42.17 / 32.64）；p128 那轮 7 分钟没出结果 |
| 统计口径不同 | 不足以解释 1.7×，重复次数只影响小数位 | `-rep 2` 与 `-rep 3` 差 < 2% |
| 核组不同 | 同上，A100 是反向的 | 见上一行 |
| **模型规格不同** | **最可能的主因（待验证）** | 官方口径 4bit / block 64 / `sym:false` / fp16 scale ≈ **4.25 bit ≈ 302 MiB**；ModelScope 上这份权重 **429.9 MiB / 596 M 参数 = 6.05 bit/参数** |

为什么模型规格能解释一大半：decode 是纯带宽活。按 429.9 MiB/token 算，
31.23 tok/s × 429.9 MiB ≈ **13.2 GB/s**，已经是本板实测读带宽（~20 GB/s）的 66%；
若换成 302 MiB 的 W4B64 模型、维持同样效率，应到 **~44 tok/s**，与官方 54.49 同一量级。
旁证：llama.cpp 在同一块板上跑 0.8B 也只有 ~29 t/s，**大家都撞在同一堵带宽墙上**。

prefill 那 1.7 倍另有来源（block-scale IME2 指令路径、官方 `-n 0` 的纯 prefill 口径），**这部分仍未验证**。

A100 核组**不是**提 decode 的路子：仓库里 [k3-ondevice-ai](../k3-ondevice-ai/k3-ondevice-ai.md) 已实测
它在通用 LLM（llama.cpp Q4）上比 X100 **慢 5.8 倍**（L2 1 MB vs 4 MB、主频 1.8 G vs 2.15 G），
只适合 int8 CNN / IME 算子。

### 4.5 调优试探（IME2 构建，`-rep 2`）

| 配置 | pp512 | tg128 | 判断 |
| --- | ---: | ---: | --- |
| 默认 | 222.94 | 31.44 | 基准 |
| `-qa 1`（KV QK-int8，对应 #4843） | 142.28 | 31.14 | **负优化** |
| `-qa 2`（KV QKV-int8） | 111.00 | 23.91 | **更差** |
| `-dyo 8` | **237.12** | 31.08 | prefill +6%，可用 |
| `-dyo 4` | 224.90 | 31.01 | 无收益 |
| `-dyo 16` | 231.18 | 30.79 | 与 `-dyo 8` 同级（±1%） |
| `-t 6` | 193.19 | 27.61 | 线程不够，掉 |
| `-t 4` | 141.98 | 22.28 | 掉更多 |

`-qa` 用的正是 9 月那批 PR 新加的 KV 量化内核，但在 0.6B + VLEN=256 这个组合下不划算：量化/反量化的
开销超过了节省的带宽。**结论：保持关闭。**

线程数与 `-dyo` 都扫过了：`-t 8` 已经是最优（`-t 6` 掉 10%、`-t 4` 掉 35%），
`-dyo 8~16` 只给 prefill 多 1~6%，decode 不动。**这套组合的天花板就在 ~230 / 31**，
再往上只能靠换模型规格（§4.4）或新内核（见[上游贡献路线](上游贡献路线.md)）。

---

## 5. 正确性测试

| 测试组 | 结果 |
| --- | --- |
| `backend/cpu/rvv`（attention_mask / linear_attention / image_process / sum_by_axis_tail） | **4/4 通过**（两种构建都是） |
| `op/unary` 42、`op/prelu` 9、`op/conv` 14、`expr/MatMul` 19、`op/scale` 2、`op/binary` 2、`op/attention` 1 | 共 **89 个用例，失败 0** |
| `cv/image_process` | 跑完无失败 |

> 注意：`run_test.out` 的汇总里 `TEST_CASE_AMOUNT_UNIT` 与 `TEST_CASE` 会各报一次，直接 `grep passed`
> 会把数字翻倍。

---

## 6. 模型准备

用的是官方 benchmark 同款 `Qwen3-0.6B-MNN`。

| 文件 | 大小 | 说明 |
| --- | ---: | --- |
| `llm.mnn.weight` | 450,810,338 | sha256 `953afb7e0165818a…`（与 ModelScope 文件清单一致） |
| `llm.mnn` | 461,520 | sha256 `d426c65a5159c938…` |
| `tokenizer.txt` | 3,193,569 | sha256 `77009625db0b18e7…` |
| `config.json` / `llm_config.json` | 403 / 4,880 | `precision: low`、`memory: low` |

下载踩坑记录：ModelScope 的 `api/v1/models/.../repo?FilePath=` 对 LFS 文件返回的是配置文件内容而不是文件本身；
**`hf-mirror.com/taobao-mnn/Qwen3-0.6B-MNN/resolve/main/<file>` 可直连**，450 MB 约 18 秒下完。

---

## 7. 复现

一条命令（板上执行，会编两版并跑完整对比）：

```bash
bash docs/edge/k3/k3-mnn-rvv/scripts/build-and-bench.sh all
```

脚本里就是上文 §2 / §3.2 的命令，日志写 `~/mnn-build/log/`。本次跑出来的日志：

| 文件 | 内容 |
| --- | --- |
| `0-env.log` | 环境采集 |
| `1-cmake.log` / `2-build.log` | 标准 RVV 配置 / 编译（10m21s） |
| `3-test.log` / `4-optests.log` | RVV 专项 / 89 项算子测试 |
| `5-download.log` | 模型下载 |
| `6-llmbench-baseline.log` / `7-profile.log` | 标准 RVV 基线 / 算子级 profile |
| `8~11-ime2-build*.log` | IME2 四次编译（含三次失败原因） |
| `12-llmbench-ime2.log` / `13-profile-ime2.log` | IME2 bench / profile |
| `14-tuning.log` / `15-bench-*.log` | `-qa` / `-dyo` 调优明细 |

两个构建目录都保留着，后续复测不需要重编。

---

## 8. 结论与建议

1. **在 K3 上跑 MNN 的量化 LLM，必须开 `MNN_RVV_SPACEMIT_IME2=ON`**，否则主算子回退标量，
   而且**不会报错**——只能靠 benchmark 发现。
2. **IME2 构建要补 march**（`zfh` + `zihintpause`），见 §3.2；这是上游 CMake 的问题，建议提 issue。
3. **标准 RVV 并非没用**：FP32 路径（`MNNPackedMatMulFP32`）和 CNN/图像类算子（C4 sampler、
   depthwise、resize、颜色转换）都有 RVV 实现；本次 RVV/图像测试全过。它缺的是**量化 GEMM**。
4. **KV 量化（`-qa`）先别开**，等更宽向量或更小 batch 再试。
5. 想看 MNN 在这块板上的上限，下一步值得试：`MNN_RVV_FAST_MATH=ON`、更大的模型（1.7B）看
   prefill/decode 的 scale 表现、以及把 `nproc` 之外的 X100 核调度策略再调一调。

---

## 关联文档

- 硬件边界（8×X100 + 8×A100、内存带宽、VLEN）→ [k3-hardware](../k3-hardware/k3-hardware.md)
- "K3 没有 NPU，AI 算力在 CPU 的 RVV+IME" → [k3-npu](../k3-npu/k3-npu.md)
- 端侧推理其它实测（ASR / VLM / ONNX）→ [k3-ondevice-ai](../k3-ondevice-ai/k3-ondevice-ai.md)
- 上游 K3 IME2 官方说明（仓内）→ `MNN/docs/perf/riscv_k3_ime2_asymmetric_w4b64.md`
