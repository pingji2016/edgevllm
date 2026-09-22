# 01 · 最典型的矩阵运算：Linear / GEMM

> 本文回答三个问题：
> 1. Transformer 里的线性层数学上是什么？
> 2. vLLM 怎么把它切成多卡并行？
> 3. 这个算子到底在代码的哪里，以及**为什么它反而不是 vLLM 自研 kernel 的战场**。

## 0. 为什么从这里开始

一个 8B 模型推理时，绝大部分 FLOPs 和绝大部分显存带宽都花在矩阵乘法上。所以"vLLM 怎么算矩阵"决定了它的性能上限。

但结论可能会让你意外：**最典型的那部分，vLLM 自己一行 kernel 都没写。**

---

## 1. 数学

Transformer 里所有线性层都是同一个公式：

$$y = xW^T + b$$

| 符号 | 形状 | 含义 |
| --- | --- | --- |
| `x` | `[num_tokens, in_features]` | 输入激活值 |
| `W` | `[out_features, in_features]` | 权重（HF 的约定是 `[out, in]`，所以公式里要转置） |
| `y` | `[num_tokens, out_features]` | 输出 |
| 计算量 | `2 × num_tokens × in_features × out_features` | FLOPs |

拆开看就是每个输出元素做一次点积：

$$y_{i,o} = \sum_{j=1}^{\text{in}} x_{i,j} \cdot W_{o,j} + b_o$$

**这个公式简单到不需要读任何论文。** 优化空间全部来自工程。

### 补一句：为什么是 `W^T` 而不是 `W`

这是历史遗留。PyTorch 的 `nn.Linear` 存的是 `[out, in]`，因为这样 `y = x @ W.T` 可以映射到 cuBLAS 的列主序布局不用拷贝。如果你自己写 kernel，反而不一定要转置 —— 这正是"公式相同、实现不同"的第一个例子。

---

## 2. 具体例子：Llama-3-8B 的 QKV 投影

配置：`hidden=4096, num_heads=32, num_kv_heads=8, head_dim=128`

```
q_proj: [4096, 4096]     # 32 个 head × 128 dim
k_proj: [1024, 4096]     # 只有 8 个 KV head（GQA）
v_proj: [1024, 4096]
```

如果分别算，就是 3 次 GEMM + 3 次 kernel launch + 3 份 `x` 的内存往返。

vLLM 把三个**合并成一次**：

```python
# LlamaAttention 里
self.qkv_proj = QKVParallelLinear(
    hidden_size=4096,
    head_size=128,
    total_num_heads=32,
    total_num_kv_heads=8,
    ...
)
```

底层 weight 是 `[6144, 4096]`（= 4096 + 1024 + 1024），**一次 GEMM 算完 QKV**。

这就是归一化之外的第一个通用优化手法：**算子融合**。代价是 `load_weights()` 变复杂 —— 得把 HF checkpoint 里三个独立的 `q_proj.weight` / `k_proj.weight` / `v_proj.weight` 按正确的偏移量塞进一个张量。这就是 `QKVParallelLinear.weight_loader()` 那 100 多行在干的事（[vllm/model_executor/layers/linear.py:1198](../vllm/model_executor/layers/linear.py#L1198)）。

---

## 3. TP 分片数学（vLLM 的精髓）

模型太大一张卡放不下，权重得切到多卡。切法有两种，名字对应切哪一维。

### 3.1 ColumnParallelLinear —— 按输出维切

```
W = [W_0 | W_1 | W_2 | W_3]        (tp_size=4，沿 out 维切)
每卡算: y_i = x @ W_i^T            x 完整复制到 4 张卡
结果:   y = [y_0 | y_1 | y_2 | y_3]   天然就是分片好的
通信:   0 次
```

```python
# vllm/model_executor/layers/linear.py:603
def forward(self, input_):
    bias = self.bias if not self.skip_bias_add else None
    output_parallel = self.quant_method.apply(self, input_, bias)   # 各卡独立算
    if self.gather_output and self.tp_size > 1:
        output = tensor_model_parallel_all_gather(output_parallel)  # 默认不 gather
    else:
        output = output_parallel
```

**关键点：`gather_output` 默认为 False。** 因为下一层如果是 RowParallelLinear，它要的刚好就是分片形式的输入 —— 不用 gather 再切一遍。

### 3.2 RowParallelLinear —— 按输入维切

```
W = [W_0 / W_1 / W_2 / W_3]        (沿 in 维切)
每卡算: y_i = x_i @ W_i^T           x 也切成 4 份
结果:   y = Σ y_i                  ← 部分和，必须相加
通信:   1 次 all-reduce
```

```python
# vllm/model_executor/layers/linear.py:1762
if self.input_is_parallel:
    input_parallel = input_          # 上一层的输出已经是切好的，不用再切
else:
    split_input = split_tensor_along_last_dim(input_, self.tp_size)
    input_parallel = split_input[self.tp_rank].contiguous()

# 只有 rank 0 把 bias 融进 GEMM，否则 TP>1 时 bias 会被加 tp_size 次
bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
output_parallel = self.quant_method.apply(self, input_parallel, bias_)

if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
```

**`bias_` 那一行是个隐藏的坑。** 每张卡算出的是部分和，all-reduce 后会相加。如果每张卡都把自己的 bias 加进去，结果就是 `y + tp_size × b`。所以只在 rank 0 加。这种 bug 在小 tp_size 下不容易发现，tp=8 时误差会大 8 倍。

### 3.3 为什么这个组合是经典设计

MLP 的结构是 `ColumnParallel → activation → RowParallel`：

```
x ──[ColParallel: gate_up_proj]──> 分片好的激活前值
                                      ↓ 激活（逐元素，无通信）
                                      ↓ 分片好的激活后值
   ──[RowParallel: down_proj]────> 部分和
                                      ↓ all-reduce（唯一一次通信）
                                   y
```

**两个 GEMM 之间零通信。** 因为 ColumnParallel 的输出分片形状，刚好就是 RowParallel 需要的输入分片。这是 Megatron-LM 的核心设计（对应论文 *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*），vLLM 直接沿用。

一次 MLP 只有 1 次 all-reduce 通信。这是 TP 能高效的原因。

### 3.4 三种 Linear 的对照

| 类 | 切哪一维 | 通信 | 典型用途 |
| --- | --- | --- | --- |
| `ReplicatedLinear` | 不切 | 0 | 小权重（如 MoE router） |
| `ColumnParallelLinear` | out | 0（或 all-gather） | QKV、gate_up_proj |
| `RowParallelLinear` | in | all-reduce | o_proj、down_proj |
| `QKVParallelLinear` | out（且处理 GQA 的 head 数不等） | 0 | attention 的 QKV |
| `MergedColumnParallelLinear` | out（多权重拼一起） | 0 | gate + up 合并 |

---

## 4. 代码路径 —— 以及那个反直觉的结论

调用链：

```
LlamaMLP.gate_up_proj  (MergedColumnParallelLinear)
  └─ quant_method.apply()                      # UnquantizedLinearMethod.apply
      └─ dispatch_unquantized_gemm()            # vllm/model_executor/layers/utils.py:584
          └─ default_unquantized_gemm()
              └─ torch.nn.functional.linear     # ← 就是它
```

```python
# vllm/model_executor/layers/linear.py:229  (UnquantizedLinearMethod.apply)
def apply(self, layer, x, bias=None):
    if envs.VLLM_BATCH_INVARIANT and (...):
        return linear_batch_invariant(x, layer.weight, bias)
    return self._gemm_impl(layer, x, layer.weight, bias)

# vllm/model_executor/layers/utils.py:83
def default_unquantized_gemm(layer, x, weight, bias=None):
    return torch.nn.functional.linear(x, weight, bias)
```

### 反直觉的地方

**未量化的 GEMM，vLLM 自己一行 kernel 都没写，全部交给 cuBLAS。**

原因很直接：NVIDIA 在这个赛道上投了几十年，手写 GEMM 打不过 cuBLAS / cuBLASLt。vLLM 的选择是"不重复造轮子"。

```python
# vllm/model_executor/layers/utils.py:584
def dispatch_unquantized_gemm(linear_backend="auto"):
    if current_platform.is_rocm():
        return rocm_unquantized_gemm       # AMD 走 hipBLASLt
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    elif not current_platform.is_cuda():
        return default_unquantized_gemm
    backend_spec = _FLASHINFER_BF16_BACKENDS.get(linear_backend)
    ...
```

注意连**平台分派本身都是在选"调谁的库"**，不是在选"用哪个自研 kernel"。

### vLLM 自研的 GEMM 在哪

自研全部集中在"别人做不好的地方"：

| 场景 | 实现 | 为什么必须自研 |
| --- | --- | --- |
| fp8 / int8 量化 | `ops.cutlass_scaled_mm`（[scaled_mm/cutlass.py:152](../vllm/model_executor/kernels/linear/scaled_mm/cutlass.py#L152)） | cuBLAS 对 block-wise scale 支持差 |
| int4（Marlin / Machete） | [mixed_precision/marlin.py](../vllm/model_executor/kernels/linear/mixed_precision/marlin.py) | dequant 必须融进 GEMM 才不亏 |
| nvfp4 / mxfp4 | [nvfp4/](../vllm/model_executor/kernels/linear/nvfp4/)、[mxfp4/](../vllm/model_executor/kernels/linear/mxfp4/) | 新硬件格式，生态没跟上 |
| MoE grouped GEMM | [csrc/moe/](../csrc/moe/) | 每个 expert 一个变长 batch，cuBLAS 无此接口 |

### 组织范式：一个 dtype 一个目录

[vllm/model_executor/kernels/linear/](../vllm/model_executor/kernels/linear/) 下每个量化格式一个目录，目录内**一个 base 接口 + 各后端一份实现**：

```
nvfp4/
  base.py        ← 接口
  cutlass.py     ← CUTLASS 实现
  marlin.py      ← Marlin 实现
  flashinfer.py  ← FlashInfer
  b12x.py / humming.py / fbgemm.py   ← 其他第三方
  emulation.py   ← 回退（用 bf16 模拟）
  pytorch.py     ← 纯 PyTorch 参考
```

同一个算子 8 种实现，运行期按平台 + 硬件能力挑。**想给某个量化格式加一个新后端，就是复制一个文件 + 实现同样的接口。**

---

## 5. 优化空间在哪

看完上面的结论，方向就清楚了：

| 层次 | 有没有空间 | 说明 |
| --- | --- | --- |
| 未量化 GEMM 本身 | ❌ | cuBLAS 已经吃满了 |
| 算子融合 | ✅ | 合并 QKV、gate+up、norm+quant+RoPE |
| 量化 GEMM | ✅✅ | vLLM 的主战场，见上表 |
| MoE grouped GEMM | ✅✅ | 变长 batch 调度是难题 |
| 通信与计算重叠 | ✅✅ | all-reduce 期间 GPU 在等，可以拆成 chunk 流水 |
| 融合并减少显存往返 | ✅✅ | 每个融合都是省一次 `[num_tokens, hidden]` 的读写 |

**判断一个优化值不值得做**：数一数它省了几次 global memory 往返。`[num_tokens, hidden]` 在 bf16 下是 `num_tokens × 8192` 字节 —— decode 阶段 `num_tokens` 小，prefill 阶段可以有几十万。省一次读写的收益通常远大于把 FLOPS 抠快 10%。

---

## 附：术语表

| 术语 | 含义 |
| --- | --- |
| **GEMM** | General Matrix Multiply，`C = αAB + βC`。所有线性层的底层 |
| **TP** | Tensor Parallelism，把权重切到多卡 |
| **Column Parallel** | 按输出维切权重，输出天然分片，无通信 |
| **Row Parallel** | 按输入维切权重，输出是部分和，需 all-reduce |
| **all-reduce** | 所有卡的张量求和后广播回所有卡 |
| **all-gather** | 所有卡的张量拼接后广播回所有卡 |
| **GQA** | Grouped Query Attention，KV head 数少于 Q head 数（Llama-3-8B 是 8 vs 32） |
| **grouped GEMM** | 一次 kernel launch 里算多个不同形状的 GEMM，MoE 专用 |
