# vLLM 学习笔记

从"数学公式 → 代码路径 → 硬件约束"这条线拆解 vLLM 的算子实现。

配套的仓库总览见上一级的 [`vLLM-项目介绍.md`](../vLLM-项目介绍.md)。

## 目录

| 文档 | 内容 |
| --- | --- |
| [01-矩阵运算-GEMM.md](01-矩阵运算-GEMM.md) | Transformer 里最典型的线性层：数学、TP 分片、代码路径，以及"为什么最典型的矩阵运算反而不是定制 kernel 的战场" |
| [02-RMSNorm-归一化.md](02-RMSNorm-归一化.md) | RMSNorm 的数学、CUDA kernel 逐行注释、三个关键优化点，以及 decode 场景的真实瓶颈 |

## 阅读顺序建议

1. 先看 **01** 建立"算子 ≠ 一个公式，而是一堆工程决策"的认识
2. 再看 **02** 逐行读一个真实的 CUDA kernel，理解向量化 / fp32 累加 / 跨线程规约
3. 然后回到仓库看 [docs/design/vllm_ir.md](../docs/design/vllm_ir.md)，理解为什么同一个算子要注册多个实现

## 贯穿两份文档的三个结论

**1. 模型适配 ≠ 读论文。** 大部分适配是架构复用 + 胶水代码。注册表里 `LlamaForCausalLM` 一份实现被 TeleChat3 / IQuestCoder / Cwm 等几十个架构复用。

**2. 优化空间的来源不是"公式相同"，而是"同一公式在不同硬件约束下最优实现不同"。**
`hidden_size=4096` 放得进 shared memory 而 `8192` 放不进；fp16 能一次搬 128 bit；累加必须切 fp32。公式的每一行都对应一个可以优化或写错的工程决策。

**3. 最典型的算子恰恰不是 vLLM 自研 kernel 的战场。** 未量化的 GEMM 直接调 `torch.nn.functional.linear` 走 cuBLAS。自研集中在量化、MoE、融合算子 —— 也就是"别人做不好的地方"。

## 常用路径速查

```
数学公式        vllm/ir/ops/                      算子语义声明（native 参考实现）
算子实现        vllm/kernels/                     register_impl 注册各 provider
               vllm/model_executor/kernels/       量化 linear 家族（一 dtype 一目录）
               csrc/                              C++/CUDA 源 → torch.ops._C
模型骨架        vllm/model_executor/models/       一模型族一文件 + registry.py 注册
可复用层        vllm/model_executor/layers/       Linear / RMSNorm / MoE / RoPE
算子测试        tests/kernels/                     按领域分（attention/quantization/moe/core/ir）
               tests/models/                      对 HF Transformers 的数值比对
性能基准        benchmarks/kernels/                性能只放这里，不放 tests
```

## 环境（AGENTS.md 硬规定）

禁止使用系统 `python3` 或裸 `pip`，一律走 `uv` + `.venv/bin/python`：

```bash
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# 跑测试
.venv/bin/python -m pytest tests/kernels/core/test_activation.py -v

# lint
pre-commit run
```
