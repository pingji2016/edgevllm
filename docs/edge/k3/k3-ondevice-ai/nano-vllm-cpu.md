# 在 K3 上跑 nano-vllm（纯 CPU 移植）

> 环境：K3 Pico-ITX · Bianbu **4.0.8** · 内核 `6.18.3-generic #1.0.10.4` ·
> Python **3.14.4** · `torch 2.9.1+debian`（16 线程）· RAM 16 GB
>
> 模型：`Qwen/Qwen3-0.6B`（bf16 权重 1.5 GB，加载后转 fp32）
>
> 补丁：[`patches/nano-vllm-cpu.patch`](patches/nano-vllm-cpu.patch)（427 行，8 个文件）

## 结论

**能跑，结果正确，但慢**：

| 指标 | 数值（实测） |
| --- | ---: |
| 模型加载 | 11.5 s（RSS 2761 MB） |
| Prefill（354 token prompt） | **29 tok/s** |
| Decode，bs=1 | **1.27 tok/s**（786 ms/token） |
| Decode，bs=2（聚合） | 2.23 tok/s（449 ms/token/条） |
| Decode，bs=4（聚合） | **3.43 tok/s**（292 ms/token/条） |
| 正确性 | 与 HF `transformers` greedy **逐 token 一致** |

放在一起看会更清楚：

| 子项 | 实测 | 说明 |
| --- | ---: | --- |
| Prefill 有效算力 | ~35 GFLOPS | `2 × 0.6e9 × 354 / 12.14 s` |
| Decode 有效算力 | ~1.5 GFLOPS | 每 token 只搬一次权重，纯延迟受限 |
| Decode 权重带宽 | ~3.0 GB/s | 2.4 GB fp32 / 0.786 s，远低于 STREAM 22 GB/s |

也就是说 **瓶颈不是内存带宽，是逐 token 的 GEMV 效率**（单 batch 矩阵向量乘，
16 线程也吃不满）。所以并发是唯一有效的杠杆：bs=4 时聚合吞吐是 bs=1 的 2.7×。

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
  分块优化，长上下文会明显退化；本报告只测到 354 token 的 prefill。
- batching 只测到 4 条并发，更大 batch 会先撞到 KV cache 和内存。
- 没有和 `llama.cpp` 做同模型对比（llama.cpp 需要 GGUF，Qwen3-0.6B 未量化转换）。
