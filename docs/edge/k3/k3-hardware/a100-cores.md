# A100 核组到底能不能用：`/proc/set_ai_thread` 与「切过去反而更慢」

> **本文数据全部是 2026-11-02 在 `pingji-k3` 上实测**，环境：Bianbu 4.0.1 · 内核
> **`6.18.3-generic`**（原厂 Bianbu 内核，**不是**跑 Cuttlefish 用的 `6.18.3-cf-k3-1.3.0`）·
> 15 GiB 内存 · 无 swap · `llama.cpp-tools-spacemit` `b10050 (5496cdc9c)`。
>
> 这块板子的硬件清单见 [`k3-hardware.md`](k3-hardware.md)；
> 模型推理的横向对比见 [`../k3-ondevice-ai/k3-ondevice-ai.md`](../k3-ondevice-ai/k3-ondevice-ai.md)；
> 给厂商的问题清单见 [`../k3-ondevice-ai/spacemit-issues.md`](../k3-ondevice-ai/spacemit-issues.md)。

---

## 0. 一句话结论

| 问题 | 结论 |
| --- | --- |
| A100（CPU 8–15）真的"摸不到"吗 | **不是。** 有一个开关 `/proc/set_ai_thread`，写一下就能把当前进程（及其子进程）迁到 8–15 |
| 为什么以前 `taskset -c 8` 报 EINVAL | 那只发生在 **`6.18.3-cf-k3-1.3.0`**（Cuttlefish 定制内核）上：8–15 online 但不在 `cpu_active_mask` |
| 换上 A100 会更快吗 | **通用 LLM 反而慢 4~6 倍**（实测 0.6B：tg 18.6 → 4.4 t/s） |
| 什么情况下值得用 | 只有在**按 VLEN=1024 编译的内核**或 **IME2 矩阵路径**上才划算（见 §4） |

**一句话：A100 不是"禁用"，是"另一个物种"；能上去，但通用代码上去只会更慢。**

---

## 1. 开关：`/proc/set_ai_thread`

```bash
$ ls -l /proc/set_ai_thread
--w--w--w- 1 root root 0 Nov  2 13:22 /proc/set_ai_thread   # 只写，谁都能写
```

两种写法都有效（实测 T1–T4，都是同一条命令）：

```bash
# 写法 A：登记当前进程
echo $$ > /proc/set_ai_thread

# 写法 B：写 0（效果相同，0 = 当前线程）
echo 0 > /proc/set_ai_thread

# 然后绑定（顺序反过来会 EINVAL）
taskset -c 8-15 你的程序
```

### 1.1 三个容易踩的点（逐条实测）

| 现象 | 实测 | 说明 |
| --- | --- | --- |
| 登记后**能不能 fork** | ✅ 能，子进程直接继承 | T1/T4：`echo 0` 之后另起 `taskset` 子进程也成功 |
| 登记后**能不能 exec** | ✅ 能，PID 不变沿用 | T3：`echo $$; exec taskset …` 成功 |
| **跨会话**还生效吗 | ❌ 不生效 | T5：新开一个 shell 直接 `taskset -c 8-15` → `EINVAL` |
| 谁登记谁有效 | 每个**进程组/线程组**自己登 | 不是全局开关，也没有持久化 |

```bash
# 失败的样子（没登记就绑）
$ taskset -c 8-15 true
taskset: 设置 pid 111914 的亲和力失败: 无效的参数
```

> 论坛上有人提到「必须在 `exec` 之前登记，否则从 X100 迁到 A100 会 SIGSEGV」——
> 那是另一个问题：动态链接器 / OpenBLAS 在 X100 上已经按 **VLEN=256** 缓存了决策，
> 迁到 VLEN=1024 的核上那些假设就不成立了。**先登记、再启动**最省事。

---

## 2. 实测：0.6B 在 X100 vs A100

模型 `Qwen3-0.6B-Q4_0.gguf`（358 MiB），`llama-bench -t 8`：

```bash
# X100（默认会话亲和性 0-7）
llama-bench -m Qwen3-0.6B-Q4_0.gguf -p 256 -n 32 -t 8 -r 1

# A100
bash -c 'echo $$ > /proc/set_ai_thread; exec taskset -c 8-15 \
  llama-bench -m Qwen3-0.6B-Q4_0.gguf -p 256 -n 32 -t 8 -r 1'
```

| 核簇 | pp256 | tg32 |
| --- | ---: | ---: |
| **X100 ×8**（0–7） | **27.3 t/s** | **18.6 t/s** |
| **A100 ×8**（8–15） | 4.78 t/s | 4.43 t/s |
| 倍率 | **5.7× 慢** | **4.2× 慢** |

`llama-perplexity`（同一段 14.7 KB 文本，`-c 512 --chunks 1 -b 512 -ub 1 -t 8`）也印证了同一件事：

| 核簇 | 每 pass 耗时 |
| --- | ---: |
| X100 | **39.6 s** |
| A100 | **192.3 s**（4.85×） |

> 注意：`llama.cpp-tools-spacemit` 启动横幅里 `cpu_mask: ff00`（= 想用 8–15），
> 但会话默认 `Cpus_allowed_list: 0-7` 把它挡回去了 —— 所以它**实际上一直跑在 X100 上**，
> 那些 IME2 优化很多时候根本没被用上。这条对"为什么官方标称和实测差那么多"是个解释方向。

---

## 3. 为什么慢：VLEN=1024 需要匹配的内核

A100 的 RVV 长度是 **1024 bit**（X100 是 256 bit），而绝大多数 RISC-V 二进制里的向量内核
是按 VLEN=256 写死的（tile 形状、`vsetvl` 假设、`vget` 用法）。把同一份代码丢到 1024 上：

- 每次 `vsetvli` 能处理的元素数变了 → tile 形状不匹配，寄存器利用率掉下来；
- 访存/广播模式按 256 设计 → 宽核的优势全浪费在搬运上。

第三方（opensolvers）在 Banana Pi BPI-SM10 上用 HPL 量过这件事（见 §6 参考 1）：

| 配置 | OpenBLAS 目标 | GFLOP/s |
| --- | --- | ---: |
| X100 ×8 | `ZVL256B` | **52–53** |
| A100 ×8 | `ZVL256B`（能跑，但没对齐） | 15 |
| A100 ×8 | 新增 `RISCV64_ZVL1024B`（DGEMM 16×8） | **35.9** |
| 8 X100 + 8 A100 混合 | 各自匹配 | **57.5** |

结论很清楚：**A100 把内核按 VLEN=1024 重写后能追回 2.4×，但仍不如 X100；只有混合跑才小胜。**
而且 `RISCV64_ZVL1024B` 的二进制**不能**在 X100 上跑（会挂），必须 `DYNAMIC_ARCH` 或分后端。

---

## 4. 那 A100 到底该拿来干什么

| 场景 | 结论 |
| --- | --- |
| 通用 LLM 推理（llama.cpp CPU 后端） | ❌ 别切，慢 4~6× |
| 手写 RVV 内核（VLEN=1024 tile） | ⚠️ 能追到 X100 的 2/3，除非混合跑 |
| **IME2 矩阵扩展**（A100 独有的 2x2x4[x2] MAC） | ✅ 这才是它的主业，但要走 TCM + `ai_dma` + 厂商运行时 |
| 已见成效的例子 | MNN 的 **IME2** 路径（本仓库 `k3-mnn-rvv/`）：pp512 226 / tg128 30 tok/s，是标准 RVV 的 5 倍量级 |

所以「A100 能不能用」的正确答案不是能/不能，而是：
**能用，但请让厂商的 IME2 运行时去用它；自己把通用代码 `taskset` 上去，只会更慢。**

---

## 5. 为什么以前的文档说"不可用"

同一块板子、不同内核，结论相反，这不是矛盾：

| 内核 | `taskset -c 8` | 原因 |
| --- | --- | --- |
| `6.18.3-cf-k3-1.3.0`（Cuttlefish 定制） | **EINVAL** | 8–15 online 但不在 `cpu_active_mask`，`__sched_setaffinity()` 直接拒 |
| **`6.18.3-generic`（原厂 Bianbu）** | **成功** | 有 `/proc/set_ai_thread`（HMP）这条路，登记后就能绑 |

也就是说：**先看 `uname -r`，再决定要不要试这个开关。**
跑 Cuttlefish / Android 的那套内核为了给 KVM 让路，把 A100 摘出了调度域；
原厂内核则把它当作 HMP 的"大核"暴露出来。

---

## 6. 参考

1. 论坛：**SpacemiT K3 has 8× VLEN=1024 "AI" cores — we taught OpenBLAS about them and ran HPL**
   → <https://forum.spacemit.com/t/1684>（HPL 数据、`set_ai_thread` 用法、OpenBLAS PR）
2. OpenBLAS 草稿 PR：`Add RISCV64_ZVL1024B (wide VLEN / SpacemiT K3 A100)` → <https://github.com/OpenMathLib/OpenBLAS/pull/6066>
3. 论坛：`AI 核(A100) 可以分配给 KVM 虚拟机使用吗？` → <https://forum.spacemit.com/t/1310>
   （A100 没有 H 扩展，KVM guest 用不上）
4. 论坛：`Bug with CONFIG_SPACEMIT_HMP=y in SpacemiT Linux-6.18` → <https://forum.spacemit.com/t/1701>
   （HMP 打开后 `signal.c` 里 `vstate.vlenb` 用户可控，属于稳定性隐患）
5. 本仓库 [`../k3-ondevice-ai/spacemit-issues.md`](../k3-ondevice-ai/spacemit-issues.md) Issue 1：
   同一现象（A100 跑通用 LLM 慢 5.8×）当时已提交给厂商，缺的是"能上去"的开关说明。
