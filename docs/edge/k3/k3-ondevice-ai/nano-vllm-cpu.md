# 在 K3 上跑 nano-vllm（纯 CPU 移植）

> 环境：K3 Pico-ITX · Bianbu **4.0.8** · 内核 `6.18.3-generic #1.0.10.4` ·
> Python **3.14.4** · `torch 2.9.1+debian`（16 线程）· RAM 16 GB
>
> 模型：`Qwen/Qwen3-0.6B`（bf16 权重 1.5 GB，加载后转 fp32）
>
> 补丁：[`patches/nano-vllm-cpu.patch`](patches/nano-vllm-cpu.patch)（446 行，8 个文件）

## 结论

**能跑，结果正确，但慢**：

| 指标 | 数值（实测） |
| --- | ---: |
| 模型加载 | 12.0 s（RSS 4105 MB，含 1.4 GB fp32 KV cache） |
| Prefill（354 token prompt） | **29 tok/s** |
| Decode，bs=1 | **1.31 tok/s**（762 ms/token） |
| Decode，bs=2（聚合） | 2.39 tok/s（418 ms/token/条） |
| Decode，bs=4（聚合） | **3.93 tok/s**（254 ms/token/条） |
| 正确性 | 与 HF `transformers` greedy **逐 token 一致** |

（数字为第二轮优化后的结果；第一版是 1.27 / 2.23 / 3.43，见下文「优化」一节。）

和三方对比：

| 方案 | Prefill | Decode | 权重 |
| --- | ---: | ---: | ---: |
| nano-vllm（本项目移植，fp32） | 29 tok/s | 1.31 tok/s | 2.4 GB |
| llama.cpp SpaceMIT 版（Q8_0，8 线程） | **28.7 tok/s** | **18.3 tok/s** | 0.60 GB |
| ORT + SpaceMiT EP（真实形状微基准外推） | ~107 tok/s | ~9.7 tok/s | 2.4 GB |

**prefill 三方基本一致，差距全在 decode**：nano-vllm 每 token 只有 3.1 GB/s 的
有效权重带宽，llama.cpp 是 11.0 GB/s，SpaceMiT EP 单算子能到 24 GB/s。
换句话说，**K3 硬件没问题，是 torch 的 fp32 GEMV 内核没吃满带宽**。

放在一起看会更清楚：

| 子项 | 实测 | 说明 |
| --- | ---: | --- |
| Prefill 有效算力 | ~35 GFLOPS | `2 × 0.6e9 × 354 / 12.37 s` |
| Decode 有效算力 | ~1.6 GFLOPS | 每 token 只搬一次权重，纯延迟受限 |
| Decode 权重带宽 | ~3.1 GB/s | 2.4 GB fp32 / 0.762 s，远低于 STREAM 22 GB/s |

也就是说 **瓶颈不是内存带宽，是逐 token 的 GEMV 效率**（单 batch 矩阵向量乘，
16 线程也吃不满）。所以并发是唯一有效的杠杆：bs=4 时聚合吞吐是 bs=1 的 3.0×。
想真正提高单流速度只能换内核——见下文「和 llama.cpp 对比」「ONNX Runtime + SpaceMiT EP」两节。

## 为什么需要补丁

nano-vllm 原版只能在 CUDA 上跑（triton + flash-attn + cudagraph + nccl）。
板子上这些全都没有，改动分四类：

| # | 文件 | 改动 | 原因 |
| --- | --- | --- | --- |
| 1 | `layers/attention.py` | 整体重写：triton `store_kvcache` + `flash_attn_*` → 纯 torch eager SDPA + 手工 paged KV gather | 没有 triton / flash-attn |
| 2 | `engine/model_runner.py` | `nccl`→`gloo`、设备 `cuda`→`cpu`、强制 eager、KV cache 按 `/proc/meminfo` 预算、去掉 `.cuda()`/`cudaGraph` | 没有 CUDA |
| 3 | `layers/layernorm.py` | 重写为无 in-place 版本 | **原版在 fp32 下算错**（见下） |
| 4 | `layers/sampler.py`、`engine/block_manager.py`、`@torch.compile` | 去 in-place、`xxhash`→`hashlib` 回退、去掉 4 处 `@torch.compile` | 板上没装 xxhash；CPU 上 inductor 编译无收益 |

### 顺手发现的一个真 bug（fp32 下静默算错）

原版融合 RMSNorm：

```python
x = x.float().add_(residual.float())   # x.float() 在 fp32 时返回自身
residual = x.to(orig_dtype)            # 同一个 tensor 的别名！
x.mul_(torch.rsqrt(var + self.eps))    # 原地归一化，residual 一起被改
```

`Tensor.float()` 和 `Tensor.to(same_dtype)` 都是**返回自身**，所以当激活是 fp32 时
`residual` 和归一化结果是同一块内存，残差被就地覆写。

- 在 GPU + bf16 下：`x.float()` 会产生副本，所以从来没暴露；
- 在 CPU + fp32 下：结果直接错，而且**不报错**，只是生成的文本莫名其妙。

我们第一次跑就撞上了：和 HF greedy 的 token 完全对不上（`[369,98,461,...]` vs
`[491,150,150,...]`）。把 layernorm 改成无 in-place 之后逐 token 一致。

> 这一点对所有把 nano-vllm 移植到 CPU / fp32 的人都是必踩的坑，
> 已在上面的 patch 里修掉。

## 正确性验证方法

拿不到真模型时先用小模型对齐，比"看起来能生成"可靠得多：

1. 用 `transformers` 造一个随机初始化的小 Qwen3（4 层、hidden 128、词表 512），
   连同 tokenizer 一起 `save_pretrained`；
2. 同一个 prompt 分别跑 nano-vllm（`temperature=1e-6` 近似贪心）和 HF greedy；
3. 比 token 序列。

结果：

```
nano-vllm: [491, 150, 150, 150, 150, 150]
HF greedy: [491, 150, 150, 150, 150, 150]
MATCH
```

中途还对比过 embedding、逐层输出和 logits，最终误差为 0。

## 和 llama.cpp 对比（同一块板子、同一个模型）

模型 `Qwen3-0.6B`；llama.cpp 用官方 `llama.cpp-tools-spacemit 0.2.2`
（`ggml-spacemit`，`use_ime2=1`，TCM 启用），GGUF 只发布了 Q8_0。

```bash
llama-bench -m Qwen3-0.6B-Q8_0.gguf -p 354 -n 32 -r 2 -t 8
```

| 线程 | pp354（prefill） | tg32（decode） |
| ---: | ---: | ---: |
| 4 | 14.80 tok/s | 11.45 tok/s |
| **8** | **28.72 tok/s** | **18.29 tok/s** |
| 16 | 26.73 tok/s | 9.46 tok/s |

8 线程是最优的（和 `ggml-spacemit` 自己报的 `num_cores=8` 一致）；
16 线程 decode 掉到一半——**跨核簇调度是负收益**，和我们之前在 A100 上看到的一致。

换算成有效带宽，差距一目了然：

| 方案 | 每 token 权重流量 | Decode | 有效带宽 |
| --- | ---: | ---: | ---: |
| nano-vllm fp32 | 2.4 GB | 1.31 tok/s | **3.1 GB/s** |
| llama.cpp Q8_0 | 0.60 GB | 18.3 tok/s | **11.0 GB/s** |
| SpaceMiT EP（单算子） | 2.4 GB | ~9.7 tok/s | ~24 GB/s |

所以 14× 的差距里，约 4× 来自量化（Q8 对 fp32），**剩下 3.5× 纯粹是内核效率**。

> 注：GGUF 仓库只有 Q8_0 和 Q4_K_M，没有 F16/BF16，所以没法做「同精度」对比。
> 这一点在报告里属于**已知偏差**，不能把 14× 全部算作框架差距。

## 第二轮优化：做了什么、值不值

第一版 decode 是逐序列 gather + 带 mask 的 SDPA。改动两处：

1. decode 时**一次 gather 所有序列**，用一次批量 SDPA 代替 `bs` 次小调用；
2. 用 `enable_gqa=True` 省掉 `repeat_interleave`（Qwen3-0.6B 是 16 头 / 8 KV 头，原本要把 KV 复制 2 倍）；
3. `bs=1` 走无 mask 快路径（带 mask 会让 CPU 上的 SDPA 走慢分支）。

效果：

| 场景 | 优化前 | 优化后 | 提升 |
| --- | ---: | ---: | ---: |
| bs=1 | 1.27 tok/s | **1.31 tok/s** | +3% |
| bs=2 | 2.23 tok/s | **2.39 tok/s** | +7% |
| bs=4 | 3.43 tok/s | **3.93 tok/s** | +15% |

提升有限，因为**瓶颈不在 attention**。做了个拆解实验（把 attention 直接短路、
或只读 1 个 block）来量化：

| 变体 | bs=1 每 token 耗时 | 说明 |
| --- | ---: | --- |
| 完整 | 762 ms | |
| 跳过 attention（`ATT_MODE=noattn`） | 688 ms | attention 占 **74 ms（9.7%）** |
| 只读 1 个 block（`ATT_MODE=short`） | 758 ms | KV gather 只占 ~4 ms |

结论：**attention 总共只占 10%，剩下 690 ms 全是线性层**，
折算 2.4 GB / 0.69 s = 3.5 GB/s——这就是 torch fp32 GEMV 的真实效率。
继续优化 attention 没有意义，要么换量化内核，要么换运行时（见下）。

顺带做了线程扫描（bs=1）：

| torch 线程 | 16 | 8 | 4 |
| --- | ---: | ---: | ---: |
| decode | **1.31 tok/s** | 0.97 | 1.12 |
| prefill | **29 tok/s** | 23 | 29 |

和 llama.cpp 相反，torch 是 16 线程最好——因为它的 GEMV 单线程效率太低，
需要更多线程凑带宽。

## ONNX Runtime + SpaceMiT EP：比 torch 快 4–5×

本来想把 Qwen3-0.6B 导成 ONNX 直接跑 EP，**结果卡在导出上**：

```python
torch.onnx.export(torch.nn.Linear(3,3), (torch.randn(2,3),), 't.onnx', opset_version=17)
# [libprotobuf FATAL] File already exists in database: onnx/onnx-ml.proto
# CHECK failed: GeneratedDatabase()->Add(encoded_file_descriptor, size)
```

连 `x*2+1` 这种模型都会让进程 abort。这是 Debian 打包的 `torch 2.9.1+debian`
和 `onnx 1.20.0` 的 protobuf 描述符冲突（不是 2 GB 限制——8 层切片 1.1 GB 一样崩）。
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 也救不了（冲突在 C++ 层）。

**影响**：所有 PyTorch→ONNX 的流程（optimum 导出 LLM、YOLO 导出、自定义算子）
在 Bianbu 4.0.8 自带的 torch 上全部不可用。绕法是改用 dynamo 导出（需 `onnxscript`）
或直接用 onnx Python API 手工构图（这条路是通的，我们所有手工模型都能 `onnx.save`）。

于是改成**用真实 LLM 权重形状做单算子微基准**（Qwen3-0.6B：hidden 1024、
q 2048 / kv 1024、intermediate 3072、28 层）：

**M=128（prefill 形状）**

| 算子 | 形状 | EP | CPU | 加速 |
| --- | --- | ---: | ---: | ---: |
| qkv_proj | 128×1024×4096 | 8.66 ms（124.0 GF） | 62.01 ms（17.3 GF） | **7.16×** |
| o_proj | 128×2048×1024 | 12.51 ms（42.9 GF） | 46.11 ms（11.6 GF） | 3.68× |
| gate_up | 128×1024×3072 | 7.35 ms（109.5 GF） | 42.00 ms（19.2 GF） | 5.71× |
| down_proj | 128×3072×1024 | 13.99 ms（57.6 GF） | 36.54 ms（22.0 GF） | 2.61× |

→ 28 层纯 matmul：EP 1.19 s vs CPU 5.23 s，**加速 4.39×**；
折算 prefill 24.5 → **107.5 tok/s**。

**M=1（decode / GEMV 形状）**

| 算子 | EP | CPU | 加速 |
| --- | ---: | ---: | ---: |
| qkv_proj | 1.06 ms | 8.88 ms | **8.35×** |
| o_proj | 0.61 ms | 0.80 ms | 1.30× |
| gate_up | 0.91 ms | 2.69 ms | 2.94× |
| down_proj | 1.08 ms | 6.74 ms | 6.26× |

→ 28 层：EP 0.10 s vs CPU 0.54 s，**加速 5.21×**；折算 decode 1.87 → **9.74 tok/s**。

对照我们的 nano-vllm 实测（1.31 tok/s）：**EP 理论上能快 5–7×**，
和 llama.cpp Q8_0 的 18.3 tok/s 在一个量级（考虑 Q8 是 1/4 权重流量）。

**必须说清楚的两个局限**：

1. 这是**单算子微基准**，同一个权重反复跑会命中缓存，所以 24 GB/s 是乐观值；
   真实整网 decode 受 DRAM 冷读限制，估计落在 5–8 tok/s，而不是 9.7。
2. 整网 EP 没能实测，因为 `torch.onnx.export` 挂了。

## bf16 试过了，不值得

权重是 bf16，直觉上把激活也改成 bf16 能把带宽砍半。实测：

| dtype | Decode（bs=1） | 模型加载 |
| --- | ---: | ---: |
| fp32 | **1.27 tok/s** | 11.5 s |
| bf16 | 1.55 tok/s（+22%） | 171 s（15× 慢） |

只快 22%，说明 decode 并不是单纯卡带宽；而加载慢 15 倍（权重转换走的是标量路径）。
**结论：RISC-V 上老老实实用 fp32。**

## 复现步骤

```bash
# 1. 拉代码（板上已有 git 和代理）
git clone --depth 1 https://github.com/GeeeekExplorer/nano-vllm /tmp/nano-vllm
cd /tmp/nano-vllm

# 2. 打补丁（本目录的 patches/nano-vllm-cpu.patch）
git apply /path/to/nano-vllm-cpu.patch

# 3. 下模型（约 1.5 GB）
export https_proxy=http://127.0.0.1:10808 http_proxy=http://127.0.0.1:10808
python3 -c "from huggingface_hub import snapshot_download as d; \
  d('Qwen/Qwen3-0.6B', local_dir='/home/pingji/huggingface/Qwen3-0.6B')"

# 4. 跑
python3 - <<'PY'
import sys; sys.path.insert(0, '/tmp/nano-vllm')
from nanovllm import LLM, SamplingParams
llm = LLM('/home/pingji/huggingface/Qwen3-0.6B', enforce_eager=True,
          max_model_len=1024, max_num_batched_tokens=512, num_kvcache_blocks=24)
out = llm.generate(["用两句话介绍一下 RISC-V 架构。"],
                   SamplingParams(temperature=0.7, max_tokens=32))
print(out[0]["text"])
llm.exit()
PY
```

llama.cpp 对照（官方 `llama.cpp-tools-spacemit`，GGUF 只有 Q8_0）：

```bash
python3 -c "from huggingface_hub import hf_hub_download as d; \
  d('Qwen/Qwen3-0.6B-GGUF', 'Qwen3-0.6B-Q8_0.gguf', local_dir='/home/pingji/gguf')"
llama-bench -m /home/pingji/gguf/Qwen3-0.6B-Q8_0.gguf -p 354 -n 32 -r 2 -t 8
```

EP 单算子微基准（`scripts/ep_shapes.py`，用 onnx Python API 手工构图，
绕开坏掉的 `torch.onnx.export`）：

```bash
python3 scripts/ep_shapes.py    # 打印 EP vs CPU 的 GFLOPS 与加速比
```

## 关于 LiteRT / vLLM

用户原本还想试 LiteRT 和 vLLM，实测这条路在当前环境走不通：

| 包 | 结果 |
| --- | --- |
| `ai-edge-litert` | ❌ PyPI 无 riscv64 / py3.14 轮子 |
| `tflite-runtime` | ❌ 同上 |
| `triton` | ❌ 无 riscv64 轮子（vLLM / SGLang 的硬依赖） |
| `vllm` | ⚠️ PyPI 有版本列表，但依赖 triton，装不上 |
| `sglang` | ⚠️ 同上，还依赖 CUDA kernel 包 |

想在 K3 上跑 LiteRT，只能从源码编译（bazel / CMake），成本远高于收益。
**真要在这块板子上做推理服务，现实选择是 `llama.cpp`（官方已带 SpaceMIT 优化）
或 ONNX Runtime + SpaceMiT EP。**

## 局限

- 只测了 Qwen3-0.6B；更大的模型（7B fp32 = 28 GB）装不下，需要量化。
- CPU 版 attention 是逐序列 gather + eager SDPA，**没有** flash-attention 的
  分块优化；本报告只测到 354 token 的 prefill。
- batching 只测到 4 条并发，更大 batch 会先撞到 KV cache 和内存。
- 和 llama.cpp 的对比**不同精度**：GGUF 仓库只发布了 Q8_0 / Q4_K_M，
  没有 F16/BF16。所以 14× 的差距里只有约 3.5× 是内核效率，另外约 4× 是量化红利。
- SpaceMiT EP 的 107 / 9.7 tok/s 是**单算子微基准外推**，不是整网实测；
  整网没法测是因为 `torch.onnx.export` 在这套环境里直接 abort（见
  [`spacemit-issues.md`](spacemit-issues.md) Issue 9）。
