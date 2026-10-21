# MNN 在 K3 上跑 LLM：标准 RVV 从 5 补到 40 tok/s，以及和官方数字的对账

> 2026-10-21 · `pingji-k3`（SpacemiT X100）· Bianbu · gcc 15.2.0 · cmake 4.2.3
> MNN master `d662ce36` · 模型 `Qwen3-0.6B-MNN` / `Qwen3-0.6B-MNN-W4B64`

把 MNN（阿里开源推理引擎）拉到板子上编了两版——**标准 RVV** 和 **SpaceMiT K3 IME2**——跑同一套
`llm_bench`，顺手把 2026 年 4 月以来上游那一批 RISC-V PR 复盘了一遍；最后把标准 RVV 缺的
**W4 量化 GEMM** 自己写了，并把官方 381 / 54.5 的口径复现出来。

**三条结论**：

1. 在 K3 上开箱跑 MNN 的量化 LLM，**必须开 `MNN_RVV_SPACEMIT_IME2`**。只开 `MNN_USE_RVV` 不会报错、
   测试也全过，但主算子静默回退标量，速度差 40 倍。
2. 标准 RVV 缺的 W4 GEMM 补上后，同一块板从 **5.65 / 4.07** 提升到 **40.07 / 25.89**
   （7.1× / 6.4×），实现与单测见 [rvv-w4-gemm.md](rvv-w4-gemm.md)。
3. 官方 **381.24 / 54.49** 的对账：decode 复现到 **94%**，主因是**权重规格不同**（手上的
   429.9 MiB vs 官方口径 320.16 MiB）；prefill 只到 76%，剩余差距见 §4.4。

---

## TL;DR

| 构建 / 模型 | pp512 | tg128 | 口径 |
| --- | ---: | ---: | --- |
| 标准 RVV（原样），ModelScope 429.9 MiB | 5.62 ± 0.01 | 4.97 ± 0.02 | 早期 |
| IME2，ModelScope 429.9 MiB | 225.88 ± 0.40 | 30.40 ± 0.98 | 早期 |
| RVV W4 GEMM，ModelScope 429.9 MiB | 38.70 | 21.87 | 官方 |
| **RVV W4 GEMM，官方规格 320.16 MiB** | **40.07** | **25.89** | 官方 |
| IME2，官方规格 320.16 MiB | 289.62 | **51.21** | 官方 |
| 官方文档 `docs/perf/riscv_k3_ime2_asymmetric_w4b64.md` | 381.24 | 54.49 | 官方 |

- 早期口径：`-p 512 -n 128 -rep 2`，单进程。
- 官方口径：pp512 `-p 512 -n 0 -rep 5 -t 8 -load false -fa 1 -kv false`；tg128
  `-p 1 -n 128 -rep 5 -t 8 -load false -fa 0 -kv true`；3 进程 × 5 次取均值。
- 跨口径不比绝对值。

| 问题 | 结论 |
| --- | --- |
| 标准 RVV 为什么这么慢 | **RVV 目录里只有 FP32 的 GEMM**；LLM 低内存模式用的「W4 块量化 + 边算边反量化」GEMM **没有 RVV 实现**，回退标量 C++（本次已补，见 [rvv-w4-gemm.md](rvv-w4-gemm.md)） |
| 是不是 runtime gate 把 RVV 关了 | 不是。`riscv_hwprobe(2)` 实测 `key=4 → 0x38dbff9ffff00ff`，bit2(V)=1 |
| IME2 能直接编吗 | **不能**，CMake 给的 march 少了 `zfh` / `zihintpause`，需手动补（见 §3.2） |
| 该开 KV 量化吗（`-qa`） | **不要**。`-qa 1` pp 222→142，`-qa 2` 降到 111，纯负优化 |
| 能对上官方 381 / 54.5 吗 | decode 复现 **94%**（51.21），prefill 只有 **76%**（289.62）；主因是权重规格，剩余差距见 §4.4 |

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

### 4.4 官方 381 / 54.5 的对账：模型规格是主因（已定稿）

上一版这里留了一条「假设，未验证」：官方数字可能来自 4bit / block 64 / `sym:false` 的权重，
而模型库上那份 `Qwen3-0.6B-MNN` 是 5.77 bit/参数。现已按官方规格自己导出并上板复测，结论定稿。

原先查证过的三条仍然成立，不再展开：**统计口径**（`-rep 2` vs `-rep 3` 差 < 2%）、
**核组**（A100 在通用 LLM 上比 X100 慢 4~5.8 倍，p8/tg2 → 10.75 / 7.42，不可能是 381 的来源）、
**调参**（`-t 8`、`-dyo 8~16` 的天花板就在 ~230 / 31，见 §4.5）。

#### 4.4.1 权重规格差异

| 模型 | 权重文件 | 位宽折算 | 来源 |
| --- | ---: | ---: | --- |
| `Qwen3-0.6B-MNN`（ModelScope） | 450,810,338 B = 429.93 MiB | 5.77 bit/参数 | 来源不明，无导出参数记录 |
| `Qwen3-0.6B-MNN-W4B64`（自建） | 335,712,498 B = 320.16 MiB | 4.30 bit/参数 | 4bit + block64 + 非对称 + FP16 scale + HQQ |

导出配方（见 §6.1）：`--quant_bit 4 --quant_block 64 --hqq`，`--sym` 默认 `false`、
`--scale_bit` 默认 16，与官方文档 §2.1 的「4 bit / block 64 / `sym: false` / FP16 scale」一致。

#### 4.4.2 实测矩阵（官方口径，3 进程 × 5 次）

| 后端 | 模型 | pp512 | tg128 |
| --- | --- | ---: | ---: |
| 官方文档 | W4B64 | **381.24** | **54.49** |
| IME2 | 320.16 MiB（官方规格） | **289.62**（76%） | **51.21**（**94%**） |
| IME2 | 429.93 MiB | 214.97 | 30.82 |
| RVV W4 GEMM | 320.16 MiB | 40.07 | 25.89 |
| RVV W4 GEMM | 429.93 MiB | 38.70 | 21.87 |
| 标量 C++ | 320.16 MiB | 5.65 | 4.07 |

**decode：94% 复现。** 只换模型，IME2 从 30.82 到 51.21（+66%）。带宽核算也对得上：
51.21 tok/s × 320.16 MiB ≈ **16.8 GB/s**，是板上实测读带宽（~20 GB/s）的 84%
——decode 到这一步是真的带宽墙，不是 kernel 的问题。

剩下那 6% 与硬件状态吻合：板上 governor 是 `userspace`，A100 组锁在 **1.8 GHz**（上限 2.0 GHz，
= 90%），且没有免密 sudo，改不了：

```bash
$ cat /sys/devices/system/cpu/cpu8/cpufreq/scaling_governor   # userspace
$ cat /sys/devices/system/cpu/cpu8/cpufreq/scaling_cur_freq   # 1800000
$ cat /sys/devices/system/cpu/cpu8/cpufreq/cpuinfo_max_freq   # 2000000
```

**prefill：只有 76%，这条没闭合。** prefill 计算受限，不能用模型规格解释；频率修正
（×1.11 ≈ 321 tok/s）后仍差约 16%。可能是板级热/内存配置差异，或文档数字来自更新的 kernel
revision——手上没有证据，不猜。

#### 4.4.3 结论对照

| 早先的判断 | 现在 |
| --- | --- |
| 官方数字来自不同的量化规格 | **成立**（decode 侧），误差量级与带宽核算都对得上 |
| 模型规格能解释 prefill 差距 | **不成立**，prefill 仍差 24%（频率修正后 ~16%），来源待查 |
| 标准 RVV 慢是 kernel 缺失 | **成立**：W4 GEMM 确实没有，补上后 7.1× / 6.4×；补完仍只有 IME2 的 50%（decode）/ 14%（prefill），这是 ISA 决定的 |

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
再往上只能靠换模型规格（§4.4，已实测 decode +66%）或补新内核（RVV 的 W4 GEMM 已补，见 [rvv-w4-gemm.md](rvv-w4-gemm.md)）。

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

### 6.1 官方规格模型的导出（要复现官方数字必须自己导）

模型库上那份 `Qwen3-0.6B-MNN`（429.93 MiB）不是官方 benchmark 的规格。4bit / block 64 / 非对称
要自己导，Mac 上跑（需要 torch + MNN python 包）：

```bash
python transformers/llm/export/llmexport.py \
  --path Qwen3-0.6B --export mnn \
  --quant_bit 4 --quant_block 64 --hqq \
  --dst_path Qwen3-0.6B-MNN-W4B64
# llm.mnn.weight = 335,712,498 B = 320.16 MiB（4.30 bit/参数）
```

同一后端只换这个模型：IME2 decode 30.82 → **51.21**；RVV W4 GEMM 21.87 → **25.89**（§4.4）。

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

1. **在 K3 上跑 MNN 的量化 LLM，开箱只有两条路**：开 `MNN_RVV_SPACEMIT_IME2`（厂商矩阵单元，
   decode 到官方 94%），或者自己补 RVV 的量化 GEMM（本次 40.07 / 25.89，7.1× / 6.4×，
   见 [rvv-w4-gemm.md](rvv-w4-gemm.md)）。只开 `MNN_USE_RVV` 会静默回退标量，且不报错。
2. **复现官方数字之前先对模型规格**：429.93 MiB 的权重只能到官方 decode 的 56%，换成
   4bit/block64/HQQ 自导出（320.16 MiB）就是 94%。这一步比任何 kernel 优化都便宜。
3. **IME2 构建要补 march**（`zfh` + `zihintpause`），见 §3.2；这是上游 CMake 的问题，建议提 issue。
4. **标准 RVV 并非没用**：FP32 路径（`MNNPackedMatMulFP32`）和 CNN/图像类算子（C4 sampler、
   depthwise、resize、颜色转换）都有 RVV 实现，本次 `backend/cpu` 回归 5 passed / 0 failed。
   它缺的是量化 GEMM，现在补上了。
5. **KV 量化（`-qa`）先别开**，量化/反量化开销超过省下的带宽。
6. **不要为了 decode 上 A100 核组**：通用 LLM 上比 X100 慢 4~5.8 倍，只适合 int8 CNN / IME 算子。
7. 还没闭合的：官方 prefill 381 只复现到 76%（289.62），频率修正后仍差 ~16%，
   需要更新的 kernel revision 或不同的板级配置才能解释。

---

## 关联文档

- 硬件边界（8×X100 + 8×A100、内存带宽、VLEN）→ [k3-hardware](../k3-hardware/k3-hardware.md)
- "K3 没有 NPU，AI 算力在 CPU 的 RVV+IME" → [k3-npu](../k3-npu/k3-npu.md)
- 端侧推理其它实测（ASR / VLM / ONNX）→ [k3-ondevice-ai](../k3-ondevice-ai/k3-ondevice-ai.md)
- 上游 K3 IME2 官方说明（仓内）→ `MNN/docs/perf/riscv_k3_ime2_asymmetric_w4b64.md`
