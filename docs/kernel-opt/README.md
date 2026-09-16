# vLLM 算子优化笔记

面向 LLM 推理（尤其是边缘 / 端侧部署）的算子与矩阵运算优化技巧整理。
不追求覆盖所有 kernel，只挑**实际会决定吞吐和延迟的那几个**。

## 目录

| 文件 | 内容 |
| --- | --- |
| [01-roofline.md](01-roofline.md) | Roofline 模型、算术强度、prefill 与 decode 的瓶颈差异 |
| [02-cuda-basics.md](02-cuda-basics.md) | 合并访存、共享内存、寄存器压力、occupancy、warp shuffle |
| [03-gemm.md](03-gemm.md) | GEMM tiling、双缓冲、split-K、Tensor Core、GEMV 与 batched GEMM |
| [04-attention.md](04-attention.md) | FlashAttention、online softmax、PagedAttention、GQA/MQA |
| [05-quantization.md](05-quantization.md) | INT8/INT4/FP8、W4A16 反量化 GEMM、group-wise scale |
| [06-fusion-layout.md](06-fusion-layout.md) | 算子融合、RMSNorm+Quant、RoPE、内存布局与 KV cache 写入 |
| [07-edge.md](07-edge.md) | 边缘设备实践：带宽受限、KV cache 压缩、静态图、功耗墙 |
| [08-profiling.md](08-profiling.md) | 性能分析方法：Nsight、roofline 画图、常见反模式 |

## 建议阅读顺序

1. **先看 01**，把「这个 kernel 到底受限于什么」想清楚。大多数优化失败的原因不是代码写得不漂亮，而是优化错了瓶颈。
2. 02 / 03 是通用基本功，任何 GPU 算子都用得上。
3. 04 / 05 / 06 是 LLM 特有的部分，也是 vLLM 里改动最频繁的地方。
4. 07 / 08 讲怎么在边缘设备上落地、怎么验证优化真的有效。

## 三条贯穿全文的经验

- **先测量，再优化。** 没有 profile 数据的优化都是猜测。一个 kernel 慢，90% 的情况是访存没合并 / 没向量化 / 有 bank conflict，而不是算术不够快。
- **decode 阶段几乎永远是带宽瓶颈。** batch=1 时 GEMM 退化成 GEMV，算术强度约等于 1 FLOP/Byte，距离 A100 的拐点（约 78）差了两个数量级。这时候唯一有用的手段是：减少要搬运的字节（量化）、或提高每次搬运的复用率（增大 batch、KV cache 复用）。
- **能用库就别手写。** cuBLAS / CUTLASS / FlashAttention / Marlin 已经调得很好了。手写的价值在于「融合」——把多个小算子合成一个，省掉中间张量的读写。

## 约定

- 代码片段以 CUDA C++ 和 Triton 为主，伪代码会明确标注。
- 提到具体数字时默认是 A100-SXM-80G（2 TB/s HBM，FP16 Tensor Core 312 TFLOP/s）这一档；边缘设备的数字单独在 [07-edge.md](07-edge.md) 里给。
- 引用 vLLM 源码时按 `vllm/` 下的路径写，版本以 v0.6.x 为参考。
