# 给 MNN 补上 RVV 的 W4 量化 GEMM：标准 RVV 从 5.6 到 40 tok/s

> 2026-10-21 · `pingji-k3`（SpacemiT X100，VLEN=256）· MNN master `d662ce36` · gcc 15.2.0
> 模型 `Qwen3-0.6B-MNN-W4B64`（4bit / block 64 / 非对称 / HQQ，320.16 MiB）· 8 线程
> 本地分支 `feature/rvv-w4-gemm`，提交 `a3bbe3e1`，3 文件 +485 行（未推送上游）

MNN 在 RVV 上缺一个函数：`Int8GemmKernel_W4`。ARM、x86、SpacemiT IME2 都有实现，
只有通用 RISC-V 是空的——于是「4bit 权重 + 动态 int8 激活」的 LLM 主算子回退到标量 C++。
本文记录这个 GEMM 的定位、实现、验证和实测，以及为什么它只能补到 IME2 的 50%（decode）。

---

## TL;DR

| 实现 | pp512 | tg128 | 相对标量 |
| --- | ---: | ---: | ---: |
| 标量 C++（上游现状） | 5.65 ± 0.01 | 4.07 | 1× |
| **本次 RVV W4 GEMM** | **40.07** | **25.89** | **7.1× / 6.4×** |
| 参考：SpacemiT IME2（厂商矩阵单元） | 289.62 | 51.21 | 51.3× / 12.6× |

同一块板、同一个官方规格模型、官方 benchmark 口径（`-p 512 -n 0 -fa 1 -kv false` /
`-p 1 -n 128 -fa 0 -kv true`，3 进程 × 5 次，取均值）。

- 缺口不是"少写了一个 kernel"，而是**一个没被赋值的函数指针**（§1）。
- 单测 72 组形状与标量参考实现**逐位一致**，走的是生产用的 dispatch 表（§3）。
- 剩下与 IME2 的差距是 ISA 决定的：IME2 有 int4/int8 矩阵乘（`smat.vmadot*`），
  通用 RVV 没有 int8 点积（板上 `/proc/cpuinfo` 确认无 `zvdot*`），只能 int16 `vwmacc`（§5）。

---

## 1. 缺口在哪：一个没人赋值的函数指针

LLM 的 Linear 层在 MNN 里被降成 1×1 `Convolution`，低内存量化权重走：

```
CPUBackend::int8GemmFunctions()  →  MatmulRelatedFunctions::Int8GemmKernel_W4
```

```c
// source/backend/cpu/compute/Int8FunctionsOpt.h
void(*Int8GemmKernel_W4)(int8_t* dst, const int8_t* src, const int8_t* weight,
                         size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                         const QuanPostTreatParameters* post, size_t realCount);
```

各平台赋值情况：

| 平台 | 赋值位置 |
| --- | --- |
| 通用默认 | `Int8FunctionsOpt.cpp:2688` → `MNNGemmInt8AddBiasScale_16x4_w4_Unit`（标量） |
| ARM82 / ARM86 / SME2 | `CommonOptFunction.cpp`（汇编/NEON 版本） |
| x86 AVX2 / AVX512-VNNI | `x86_x64/*/GemmInt8.cpp` |
| SpacemiT IME2 | `MNNSpacemitIme2FastPathRegistration.cpp:52` |
| **RVV** | **无** —— `Int8FunctionsOpt.cpp` 的 RVV 注册块只设了 `Int8GemmKernel`（W8），W4 保持默认 |

所以 `MNN_USE_RVV=ON` 能编译、能过测试、不报任何警告，只是主算子静默走标量。
这就是 [mnn-rvv-on-k3.md](mnn-rvv-on-k3.md) §4.3 那条根因链的落点。

### 1.1 权重布局

低内存模式按 **block 64** 量化，`GEMM_INT8_UNIT = 4`（每 unit 4 个输出通道）、
`GEMM_INT8_SRC_UNIT = 16`（每步 16 个输入 lane）。每个（输出 unit，量化 block，深度步）：

```
32 字节 = 64 个 4bit 权重
byte k 的高 4 位 = 第 k 个值
byte k 的低 4 位 = 第 k+32 个值
值下标 m = 输出通道(0..3) * 16 + lane(0..15)
之后紧跟：4 个 float scale + 4 个 float zero-point 修正（非对称量化的 offset 折算在这里）
```

注意 4bit 权重是**无符号** 0..15，零点是靠 `weightBias`（配 `srcKernelSum` 行和）事后修正的，
所以 kernel 里是零扩展而不是符号扩展。

---

## 2. 实现

文件：`source/backend/cpu/riscv/rvv/MNNGemmInt8AddBiasScale_16x4_w4_Unit_RVV.cpp`

三个关键决定：

1. **一个 tile 只解包一次，供两个激活行复用。**
   prefill 每一步会把 8 个 token 走过同一份权重，如果按行循环，每个行都要重新
   `vle8 + vsrl + vand + 2×vslidedown + 4×vzext`。改成 tile 在外、行在内之后，
   prefill 直接快了 1.4×（38.5 vs 26.85）。
2. **累加器用 `int32m2`（16 lane × 4 通道），跨 lane 的归约每个 tile 只做一次。**
   原先每个深度步都要横归约，现在只在 tile 结束后 `vredsum` 四次。权重用
   `vzext_vf2`（u8→u16）+ `vlmul_trunc` 压到 `int16m1`，正好配 `vwmacc_vv_i32m2`；
   这样 4 通道 × 2 行的 8 个累加器只占 16 个寄存器，不会溢出。
3. **epilogue 按编译期模板特化。**
   `inputScale`（动态量化）与 `inputBias`（残差项）是否存在的 4 种组合各实例化一份，
   内层 4 个通道里不再有分支；跨 quant block 的部分和在 `bk > 0` 时加回。

一个约束：`32 字节 tile + 两个 16 lane 半区`要放进一个 LMUL=1 寄存器，因此要求 **VLEN ≥ 256**。
注册处按 `csrr vlenb` 门控，窄向量（VLEN=128 的 VisionFive 2 等）继续用标量实现——不会错，只是不加速。
这个取舍写在提交信息里，后续可以把 VLEN=128 的半区拆成两次 m1 处理。

---

## 3. 正确性

`test/backend/cpu/RVVW4GemmTest.cpp`，两个设计点：

- **走生产 dispatch 表**而不是直接调符号（RVV 目标文件在共享库里没有导出符号），
  这样测的就是"运行时真正会被调用的那个函数"；
- **参考实现是标量 kernel 的镜像**。整数点积精确、浮点后处理顺序一致，所以两边应当**逐位相等**，
  用 `memcmp` 比较，不是容差比较。

覆盖 72 组形状：深度步 1/2/5/16 × 量化 block 1/2/4 × 行数 1/3/8 × 输出 unit 1/3 ×
`inputScale`/`inputBias` 三种后处理组合。

```
W4 GEMM dispatch: RVV=1, VLEN=256 bit
W4 GEMM: 72 cases matched the scalar reference exactly
√√√ all <backend/cpu/rvv/w4gemm> tests passed.
```

`run_test.out backend/cpu` 全量回归：`5 passed / 0 failed`。

实现过程中这套测试抓到两个真 bug：行步长把"字节数"当成"float 个数"（越界写），
以及重构 epilogue 时漏掉了跨 quant block 的累加。两者在端到端 demo 里都只表现为"输出变差"，
不看数值对比根本发现不了——这也是为什么这个 kernel 一定要带测试提交。

---

## 4. 性能

### 4.1 官方规格模型（320.16 MiB，官方口径，3 进程 × 5 次）

| 实现 | pp512 | tg128 |
| --- | ---: | ---: |
| 标量 C++ | 5.65 ± 0.01 | 4.07 |
| **RVV W4 GEMM** | **40.07**（39.76 / 40.11 / 40.33） | **25.89**（25.86 / 26.01 / 25.81） |
| IME2 | 289.62 | 51.21 |

### 4.2 优化过程（ModelScope 模型 429.9 MiB，`-p 512 -n 128 -rep 2`，8 线程）

| 版本 | pp512 | tg128 | 改了什么 |
| --- | ---: | ---: | --- |
| 标量 C++ | 5.61 | 5.00 | 上游现状 |
| v1 向量化 | 26.85 | 19.06 | 解 nibble + `vwmacc`，每 tile 归约一次 |
| v2（提交版） | 38.5 ± 0.4 | 21.7 ± 0.2 | tile 复用 2 行 + epilogue 特化 |

> 4.2 与 4.1 的模型、口径不同，不要跨表比绝对值；4.1 才是可对标官方文档的数字。

### 4.3 试过但没收益的（已回退）

| 试验 | 结果 | 判断 |
| --- | --- | --- |
| `__builtin_prefetch` 预取 6 个 tile | 37.98 / 20.81 | 噪声内；顺序访问已被硬件预取覆盖 |
| `#pragma GCC unroll 4` 展开深度步 | 36.59 / 21.97 | 无提升，寄存器压力反而变大 |

### 4.4 为什么到不了 IME2

线程扩展近乎线性（同一模型，t=2/4/8）：

| 线程 | pp512 | tg128 |
| ---: | ---: | ---: |
| 2 | 13.12 | 6.72 |
| 4 | 23.42 | 12.79 |
| 8 | 38.31 | 21.58 |

线性扩展说明**不是带宽饱和、也不是共享缓存瓶颈**，而是每核的流式加载速率受限：
每核约 1.06 GB/s（`25.89 tok/s × 320 MiB / 8`）。IME2 在同板上也只有约 1.7 GB/s/核，
说明这是这颗 SoC 的单核流式上限，不是我们 kernel 的访存写法问题。

prefill 的差距则来自 ISA：

| | 通用 RVV（本次） | IME2 |
| --- | --- | --- |
| 点积 | int16 `vwmacc`（16 lane） | `smat.vmadot` INT4 `8×32×8` / INT8 `8×16×8`，带 FP16 block-scale 变体 |
| int8 点积扩展 | **无**（板上无 `zvdot*`） | 硬件单元 |
| 实测 pp512 | 40.07 | 289.62 |

一句话：decode 靠带宽（我们到 IME2 的 50%），prefill 靠矩阵算力（我们到 14%）。
在没有 `vdot` 的通用 RVV 上，这个比例已经是"用向量指令模拟点积"的天花板附近。

---

## 5. 复现

```bash
# 1) 构建（标准 RVV，不开 IME2）
cmake -S MNN -B build-rvv -DCMAKE_BUILD_TYPE=Release \
  -DMNN_USE_RVV=ON -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_TEST=ON -DMNN_BUILD_TOOLS=ON
cmake --build build-rvv -j8

# 2) 单测
./build-rvv/run_test.out backend/cpu/rvv/w4gemm

# 3) 测速（官方口径）
./build-rvv/llm_bench -m <MODEL>/config.json -p 512 -n 0 -rep 5 -t 8 -load false -fa 1 -kv false
./build-rvv/llm_bench -m <MODEL>/config.json -p 1 -n 128 -rep 5 -t 8 -load false -fa 0 -kv true
```

模型按官方规格自己导（见 [mnn-rvv-on-k3.md](mnn-rvv-on-k3.md) §4.4），权重 335,712,498 B。

---

## 关联文档

- 上游缺口与贡献优先级 → [上游贡献路线.md](上游贡献路线.md)
- 官方 381 / 54.5 的对账（模型规格是主因）→ [mnn-rvv-on-k3.md](mnn-rvv-on-k3.md) §4.4
- 硬件边界（X100/A100、带宽、VLEN）→ [../k3-hardware/k3-hardware.md](../k3-hardware/k3-hardware.md)
