# 02 · 归一化：RMSNorm 逐行注释

> 本文用 RMSNorm 当样本，完整走一遍"**数学公式 → 接口设计 → CUDA kernel 逐行 → 优化权衡**"。
> 读完你应该能：看懂任何一个 elementwise / reduction 类 kernel，并判断它的瓶颈在哪。

## 1. 数学

论文：Zhang & Sennrich 2019, *Root Mean Square Layer Normalization*
（就是代码 docstring 里引的那篇 —— [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)）

$$y_i = w_i \cdot \frac{x_i}{\sqrt{\frac{1}{n}\sum_{j=1}^{n}x_j^2 + \epsilon}}$$

对比 LayerNorm：

$$\text{LN: } y_i = w_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + b_i \qquad \text{RMS: } y_i = w_i \cdot \frac{x_i}{\sqrt{\text{mean}(x^2) + \epsilon}}$$

**RMSNorm 只差"不减均值"这一项。**

省掉的是什么？层归一化里 $\mu$ 和 $\sigma^2$ 各需要一次跨线程规约。RMSNorm 把两次合成一次（因为 $\sigma^2 = E[x^2] - \mu^2$，不减均值就只剩 $E[x^2]$）。少一次规约、少一次全张量减法、少一个 bias 参数。

这是它被 Llama 系全面采用的原因 —— **不是精度更好，是更快**。

### 为什么它能不减均值还不掉点

$\frac{x_i}{\text{RMS}(x)}$ 已经是单位尺度了，减去均值带来的"中心化"收益，在 RMSNorm 的设定下不值那一次规约的开销。实验上（原论文）效果相当。

---

## 2. 接口里一个不起眼但重要的参数

```python
# vllm/model_executor/layers/layernorm.py:37
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(self, hidden_size, eps=1e-6, var_hidden_size=None, has_weight=True, ...):
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
```

`variance_size` 的语义是：**只对最后一维的前 `variance_size` 个元素算方差，但归一化作用在全部元素上。**

```python
# vllm/ir/ops/layernorm.py:16
x_var = x if variance_size is None else x[..., :variance_size]
variance = x_var.pow(2).mean(dim=-1, keepdim=True)
x = x * torch.rsqrt(variance + epsilon)      # ← 归一化用的是完整的 x
if weight is not None:
    x = x.to(weight.dtype) * weight
```

### 谁在用这个参数

[vllm/model_executor/models/interns1_vit.py:202](../vllm/model_executor/models/interns1_vit.py#L202)：

```python
self.dummy_dim = (num_dummy_heads + self.num_heads) * self.head_dim
...
self.q_norm = RMSNorm(self.dummy_dim, var_hidden_size=self.embed_dim)
```

Q / K 被**填充了额外的 dummy head 来做对齐**。这些 dummy head 是纯 padding，不能参与统计量计算，否则会污染方差。所以：

- 归一化的范围 = `dummy_dim`（含 padding，因为 padding 位也要有值）
- 方差的计算范围 = `embed_dim`（只有真实 head）

**这类需求不会出现在任何论文里**，它来自具体实现的工程约束。这也印证了那个判断：模型适配 ≠ 读论文。

---

## 3. 这个参数暴露了 IR 派发机制的真实作用 ⚠️

一个重要的事实：**CUDA 版的 `rms_norm` kernel 根本不接受 `variance_size`。**

看 C++ 的算子签名（[csrc/libtorch_stable/torch_bindings.cpp:367](../csrc/libtorch_stable/torch_bindings.cpp#L367)）：

```cpp
"rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon) "
```

只有四个参数，没有 `variance_size`。所以 `vllm_c` 这个 provider 遇到 `variance_size != None` 时必须**拒绝服务**。拒绝的方式是 IR 的 `supports_args` 谓词：

```python
# vllm/kernels/vllm_c.py:20
rms_no_var_size = lambda x, weight, epsilon, variance_size=None: (
    variance_size is None and (weight is None or weight.dtype == x.dtype)
)
"""vLLM kernel requires no variance_size override and matching input/weight dtype."""

@ir.ops.rms_norm.register_impl(
    "vllm_c", supports_args=rms_no_var_size, supported=GPGPU_DEVICE
)
def rms_norm(x, weight, epsilon, variance_size=None):
    assert variance_size is None
    ...
```

**这就是 vLLM IR 存在的意义。** 一个算子声明一次语义（native 参考实现），多个 provider 各自声明"我在什么条件下可用"：

| provider | `supported` 条件 |
| --- | --- |
| `native` | 永远可用（PyTorch 参考实现） |
| `vllm_c` | GPGPU 设备 **且** `variance_size is None` 且 dtype 匹配 |
| `aiter` | ROCm 平台 |
| `oink` | `device_capability >= 100` 且装了 oink 插件 |

InternVL 的 qk-norm 因为带了 `variance_size`，**自动回退到 native 的 PyTorch 实现** —— 慢但正确。不需要调用方写任何 `if` 判断。

顺带回答一个常见疑问：为什么 `layernorm.py` 里只有 2 个 op、整个 `vllm/ir/ops/` 只有 3 个？因为 IR 还在从旧 `CustomOp` 迁移中（git log 里 PR 标题带序号 `[vLLM IR] 2/N`），目前 **3 / 78**。

---

## 4. CUDA kernel 逐行注释

来源：[csrc/libtorch_stable/layernorm_kernels.cu:13](../csrc/libtorch_stable/layernorm_kernels.cu#L13)

开头就有一行注释，很有信息量：

```cuda
// TODO(woosuk): Further optimize this kernel.
```

### 4.1 签名 —— 模板参数是核心手法

```cuda
template <typename scalar_t, int VEC_SIZE, int NUM_DIMS, bool HasWeight>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const int64_t input_stride_d2,        // input.stride(-2)
    const int64_t input_stride_d3,        // input.stride(-3)
    const int64_t input_stride_d4,        // input.stride(-4)
    const int64_t input_shape_d2,         // input.size(-2)
    const int64_t input_shape_d3,         // input.size(-3)
    const scalar_t* __restrict__ weight,  // [hidden_size] 或 [num_groups, hidden_size]；!HasWeight 时为 null
    const int64_t weight_stride,          // 0 或 weight.stride(0)
    const float epsilon, const int num_tokens, const int hidden_size) {
```

四个模板参数全是**编译期常量**：

| 参数 | 作用 |
| --- | --- |
| `scalar_t` | fp16 / bf16 / fp32 |
| `VEC_SIZE` | 一次搬几个元素（向量化宽度） |
| `NUM_DIMS` | 2D / 3D / 4D，编译期分派掉 stride 计算 |
| `HasWeight` | 有无可学习权重，编译期消除分支 |

**`__restrict__` 是给编译器的承诺**：这些指针不重叠。编译器因此可以放心重排访存指令、用非阻塞加载。

### 4.2 预备 —— 两级累加器

```cuda
  __shared__ float s_variance;      // 一个 block 共享一个标量，用于广播
  float variance = 0.0f;            // 每线程的私有累加器（在寄存器里）
  const scalar_t* input_row;
  const scalar_t* weight_row;
```

**注意区分这两个**：`variance` 是 per-thread 的寄存器变量，每个线程只累加自己负责的那部分元素；最后才跨线程求和。这个"先局部、后全局"的模式是所有 reduction kernel 的骨架。

### 4.3 定位本 block 负责的行

```cuda
  if constexpr (NUM_DIMS == 2) {
    // 2D for layernorm normal case [batch_size, hidden]
    input_row = input + blockIdx.x * input_stride_d2;
    weight_row = weight + blockIdx.x * weight_stride;
  } else if constexpr (NUM_DIMS == 3) {
    // 3D for q/k norm [batch_size, num_heads, head_size]
    int batch_idx = blockIdx.x / input_shape_d2;
    int head_idx = blockIdx.x % input_shape_d2;
    input_row = input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
    weight_row = weight + batch_idx * weight_stride;
  } else if constexpr (NUM_DIMS == 4) {
    // 4D for transformers model_impl qk norm [batch, seq, head, head_dim]
    int batch_idx = blockIdx.x / (input_shape_d3 * input_shape_d2);
    int remaining = blockIdx.x % (input_shape_d3 * input_shape_d2);
    int seq_idx = remaining / input_shape_d2;
    int head_idx = remaining % input_shape_d2;
    input_row = input + batch_idx * input_stride_d4 +
                seq_idx * input_stride_d3 + head_idx * input_stride_d2;
    weight_row = weight + batch_idx * weight_stride;
  }
```

**核心设计：一个 block 负责一行。** `blockIdx.x` 就是行号。

`if constexpr`（C++17）让编译器**只保留当前维度对应的那份代码**，其余在编译期丢弃。运行期一个分支都没有。

三种维度的用途：

- **2D** `[batch, hidden]` —— 标准情况（每个 token 归一化一次）
- **3D** `[batch, num_heads, head_size]` —— q/k norm（每个 head 单独归一化）
- **4D** `[batch, seq, head, head_dim]` —— 对齐 HF 原生实现，兼容性用

`weight_stride` 的用法很巧：weight 是所有行共享的，3D/4D 时 `weight_stride = 0`，让 `weight + blockIdx.x * 0` 恒等于 `weight`，一套代码适配所有情况。

### 4.4 第一遍：累加平方和

```cuda
  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(vec.val[i]);   // ← 转 fp32 再平方
      variance += x * x;
    }
  };
  auto scalar_op = [&variance](const scalar_t& val) {
    float x = static_cast<float>(val);
    variance += x * x;
  };
  vllm::vectorize_read_with_alignment<VEC_SIZE>(
      input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);
```

#### 三个关键优化

**① 向量化访存（`vec_n_t<scalar_t, VEC_SIZE>`）**

一次搬 `VEC_SIZE` 个元素。fp16 下 `VEC_SIZE=8` → **128 bit = 16 字节**，一条指令搬完。

为什么这是第一优化？因为 **RMSNorm 是 memory-bound 的**。每个元素只做 2 次乘加（1 次乘法 + 1 次加法），算术强度极低。此时吞吐**完全由访存决定**，向量化直接决定性能上限。

```
不向量化：每个元素一条 global load    → 指令发射成为瓶颈
向量化：  8 个元素一条 global load    → 逼近显存带宽上限
```

`vectorize_read_with_alignment` 还处理了**对齐**问题：hidden_size 不一定是 `VEC_SIZE` 的整数倍，前段用向量化读、尾部用 `scalar_op` 标量收尾。它还会检查指针是否真的对齐，未对齐就整体退化为标量。

**② fp32 累加（`static_cast<float>`）**

这行**不是可有可无的类型转换**。

fp16 的尾数只有 10 位（约 3 位十进制精度）。累加 8192 个平方和，每个数量级相近，用 fp16 累加会快速丢失有效位 —— 后面的加数直接"消失"（swamping）。最后方差算小或算大，整个归一化的缩放因子就是错的。

**规律：reduction 必须在高精度下做。** 这也是为什么后面 `BlockReduce` 用的模板参数是 `float` 而不是 `scalar_t`。

**③ 每线程负责交错的一段**

`threadIdx.x, blockDim.x` 作为起点和步长 → 线程 `t` 处理元素 `t, t+blockDim, t+2*blockDim, ...`。这叫 **grid-stride 交错访问**，保证同一 warp 内 32 个线程访问相邻地址 → 合并成一次内存事务（coalesced access）。如果改成"线程 t 处理前 1/blockDim 段"，每次访存就散开了。

### 4.5 跨线程规约

```cuda
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);   // ← 公式里的 1/sqrt(...)
  }
  __syncthreads();
```

每个线程只累加了自己那部分，必须求和。这里用 `cub::BlockReduce`。

**为什么不用手写？** 手写 block 内规约要自己处理：warp 内 shuffle 树形规约（5 步）、warp 间通过 shared memory 传递、避免 bank conflict、以及最经典的漏写 `__syncthreads()` 导致的竞态。`cub` 是 NVIDIA 官方的原语库，正确性和性能都有保证。

**`rsqrtf` 只算一次。** 反平方根是特殊函数单元（SFU）指令，比普通乘法贵得多。把它提到循环外、只由一个线程算、存进 shared memory，是一次除法换 n 次。

`__syncthreads()` 是必须的：不然其他线程可能读到还没写好的 `s_variance`。这是 GPU 编程里出错最多的一行。

**顺序注意**：先规约求和 → 再除以 `hidden_size` 求均值 → 再加 `epsilon` → 最后 `rsqrtf`。公式 `1/sqrt(mean(x²) + ε)` 里的 ε 在根号**内**，位置写错会得到不同的数值行为（ε 在根号外就变成了可加常数，数值稳定性完全不同）。

### 4.6 第二遍：读输入、乘权重、写出

```cuda
  scalar_t* out_row = out + blockIdx.x * hidden_size;
  auto* v_in  = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
  auto* v_w   = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(weight_row);
  auto* v_out = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE>*>(out_row);
  for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
    vec_n_t<scalar_t, VEC_SIZE> dst;
    vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[i];      // ← 第二次读 input！
    vec_n_t<scalar_t, VEC_SIZE> src2;
    if constexpr (HasWeight) {
      src2 = v_w[i];
    }
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float x = static_cast<float>(src1.val[j]);
      if constexpr (HasWeight) {
        float w = static_cast<float>(src2.val[j]);
        dst.val[j] = static_cast<scalar_t>(x * s_variance * w);
      } else {
        dst.val[j] = static_cast<scalar_t>(x * s_variance);
      }
    }
    v_out[i] = dst;
  }
```

#### 最重要的一个设计决策：为什么第二次读 input？

这里**又读了一遍 `input`**，而不是把数据缓存到 shared memory 里。看着很浪费 —— 每个元素被读了两遍。

这是一个**权衡**，算的是这笔账：

| 方案 | 收益 | 代价 |
| --- | --- | --- |
| 缓存到 shared memory | 第二遍省一次 global read | 每 block 占 `hidden × 2 byte` = 8 KB（hidden=4096） |
| 读两遍（当前做法） | zero shared memory 占用，occupancy 高 | 多一次 global read，赌**命中 L2** |

H100 每个 SM 的 shared memory 上限约 228 KB。看着 8 KB 不多，但**shared memory 是限制同一 SM 上能并存多少 block 的关键资源之一**。占满之后 occupancy 掉下来，隐藏访存延迟的能力就没了 —— 这个损失通常**大于**多读一遍数据的损失。

而第二遍读的数据刚被读过，几乎必然还在 L2 里。**L2 命中比 global 快一个数量级**，所以这笔账是划算的。

> 这是 kernel 优化里最典型的模式：**用带宽换并行度（occupancy）**。
> 判断依据永远是"这个 kernel 是 latency-bound 还是 bandwidth-bound"。

#### `HasWeight` 编译期模板

```cuda
    if constexpr (HasWeight) {
      float w = static_cast<float>(src2.val[j]);
      dst.val[j] = static_cast<scalar_t>(x * s_variance * w);
    } else {
      dst.val[j] = static_cast<scalar_t>(x * s_variance);
    }
```

某些模型的 RMSNorm **没有可学习权重**（例如 Gemma 的某些变体）。编译期去掉这个分支后，weightless 路径就只做一次乘法，而且完全不读 `weight`。

这对应 Python 侧的：

```python
# vllm/model_executor/layers/layernorm.py:67
# When has_weight=False, pass weight=None so implementations that
# support a weightless path can skip the per-channel multiply.
# Implementations that require weight (e.g. oink) fall back via IR
# op priority when weight=None is unsupported.
self.pass_weight = self.has_weight
```

一条链走下来：`has_weight=False` → `pass_weight=False` → 传 `weight=None` → `HasWeight=false` → 编译期消除乘法。**从 Python 的一个布尔值直达 GPU 指令数。**

---

## 5. 这个 kernel 的瓶颈与优化空间

### 5.1 什么时候它很快，什么时候很慢

**一个 block 处理一行** 意味着：**并行度 = 行数 = num_tokens**。

| 阶段 | num_tokens | GPU 占用 | 表现 |
| --- | --- | --- | --- |
| **prefill** | 几百到几十万 | 打满（H100 约 132 个 SM 都有活干） | 很好 |
| **decode** | 可能只有 1（单序列） | 1 个 block / 132 个 SM | **约 1% 利用率** |

**这才是真瓶颈。** 不是公式慢，是 decode 时整个 GPU 只有一行数据要处理。

这正好印证了那份笔记里的判断：

> 优化空间的来源不是"公式相同"，而是"同一公式在不同硬件约束下最优实现不同"。

同一个 kernel，prefill 时是带宽瓶颈，decode 时是并行度瓶颈。**公式一个字没变。**

### 5.2 可以怎么优化

**① Split-K 式的方差规约**

decode 时把一行切成几段，多个 block 并行累加方差，再用 global memory 做二次规约。

- 收益：并行度从 1 提升到几十，直接吃掉那 100 倍差距
- 代价：多一次 kernel launch（或 atomic），多一次 global memory 往返
- 权衡点：只有 num_tokens 小的时候才划算，所以需要**运行期判断** —— 这正是为什么同一个 `rms_norm` 要注册多个 provider 并按优先级选

**② 算子融合（最有效）**

把 RMSNorm 和它后面的操作融成一个 kernel，省掉中间那个 `[num_tokens, hidden]` 的 global write + read。

vLLM 已经在做了，看这些文件：

| 文件 | 融合了什么 |
| --- | --- |
| [csrc/libtorch_stable/layernorm_quant_kernels.cu](../csrc/libtorch_stable/layernorm_quant_kernels.cu) | RMSNorm + 量化（int8/fp8），直接输出量化结果 |
| [csrc/libtorch_stable/fused_qknorm_rope_kernel.cu](../csrc/libtorch_stable/fused_qknorm_rope_kernel.cu) | QK 归一化 + RoPE 位置编码 |
| [csrc/libtorch_stable/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu](../csrc/libtorch_stable/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu) | 归一化 + per-token 动态量化 |
| [csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu](../csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu) | 归一化 + RoPE + **写入 KV cache** |

最后一个最能说明问题：它把 4 个操作融成 1 个 kernel。省下的是 3 次 `[num_tokens, hidden]` 级别的 global memory 往返 —— 在 prefill 阶段这是实打实的大头。

**③ 用 IR 加一个 provider**

如果你不想改 C++（要重编译），可以只加一个 Triton 实现：

```python
# vllm/kernels/triton/layernorm.py (示意)
import triton
import triton.language as tl
from vllm import ir

@ir.ops.rms_norm.register_impl("triton_decode", supported=...)
def rms_norm_triton(x, weight, epsilon, variance_size=None):
    ...
```

模式已经定死了，参考 [vllm/kernels/triton/activation.py](../vllm/kernels/triton/activation.py)。

---

## 6. 通用套路总结（怎么读任何 GPU kernel）

把上面这个 kernel 抽象出来，就是 GPU 编程的固定骨架：

```
1. 模板参数化    把 VEC_SIZE / 维度 / 有无权重 变成编译期常量
2. 划分线程      grid-stride 交错访问，保证 coalesced
3. 局部累加      每个线程在寄存器里累加自己那段
4. 跨线程规约    warp shuffle + shared memory 树形规约（用 cub 别手写）
5. 广播 + 同步   算一次标量 → shared memory → __syncthreads()
6. 第二遍遍历    应用结果 + 写出（向量化写）
```

**判断瓶颈的问法**：

| 问题 | 如果答案是…… |
| --- | --- |
| 每个元素做几次算术？ | 少（< 10）→ **memory-bound**，优化向量化和访存合并 |
| 还是多（> 100，如 GEMM）？ | → **compute-bound**，优化分块、寄存器复用、指令级并行 |
| 并行度够吗？ | 行数 < SM 数 → **并行度不足**，考虑 split-K |
| 中间数据能不能不落回显存？ | 能 → **融合**，这是收益最大的方向 |

**RMSNorm 属于第一类**（memory-bound）+ decode 时第三类（并行度不足）。

## 附：术语表

| 术语 | 含义 |
| --- | --- |
| **occupancy** | 每个 SM 上实际并存的 warp 数 / 硬件上限。决定隐藏延迟的能力 |
| **coalesced access** | 同一 warp 内 32 线程访问相邻地址，合并成一次内存事务 |
| **向量化访存** | 一条指令搬 128 bit（`float4` / `__half8`），减少指令数 |
| **shared memory** | 每个 SM 上的高速暂存，容量有限（H100 约 228 KB/SM），是 occupancy 的约束之一 |
| **bank conflict** | 多线程访问 shared memory 同 bank 不同地址，被迫串行 |
| **`__syncthreads()`** | block 内屏障。漏写 = 竞态 |
| **`rsqrtf`** | 反平方根，走 SFU（特殊函数单元），比乘法贵 |
| **swamping** | 低精度累加时大数吞掉小数，reduction 必须切 fp32 的原因 |
| **memory-bound** | 算术强度低，吞吐由访存决定（RMSNorm、elementwise） |
| **compute-bound** | 算术强度高，吞吐由算力决定（GEMM） |
