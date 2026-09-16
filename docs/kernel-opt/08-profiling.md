# 08 · 性能测量与分析方法

> 「我们没有时间做性能分析，但一直有时间重写代码。」—— 大多数优化工作的真实写照。

这一篇讲怎么**用数据定位瓶颈**，而不是靠直觉猜。

## 1. 第一步：先量端到端，再量 kernel

**不要一上来就打开 ncu。** 先回答：这个 kernel 占端到端多少时间？

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    for _ in range(5):
        model.generate(...)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

得到的表里看 **`cuda_time_total` 的百分比**。按 Amdahl 定律：

```text
优化一个占 5% 的算子，哪怕快 10 倍，端到端也只快 4.5%
优化一个占 40% 的算子，快 2 倍，端到端快 20%
```

**先动占比最大的那个。** 这条规则能挡掉 80% 的无用功。

### LLM 推理的典型耗时分部

| 阶段 | 主要耗时 | 优化方向 |
| --- | --- | --- |
| Prefill | GEMM（60–80%）、Attention（10–20%） | Tensor Core、FlashAttention |
| Decode | GEMM/GEMV（50–70%）、Attention（20–40%）、elementwise（5–15%） | **量化**、融合、CUDA Graph |

序列越长，attention 占比越高；batch 越大，GEMM 占比越高。

## 2. 正确的计时方法

### 用 CUDA Event，不要用 `time.time()`

```python
# 错误：GPU 是异步的，Python 计时测的是 CPU 提交时间
import time
t0 = time.time()
out = model(x)
print(time.time() - t0)          # 测出来接近 0，完全是错的

# 正确
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

start.record()
out = model(x)
end.record()
torch.cuda.synchronize()          # 必须在读 elapsed_time 前同步
print(start.elapsed_time(end), "ms")
```

### 一定要预热

```python
for _ in range(10):               # 预热：JIT 编译、cuBLAS 算法选择、cache 预热
    model(x)
torch.cuda.synchronize()

# 然后才计时，跑多次取中位数（别取平均，会被偶发停顿污染）
times = []
for _ in range(50):
    start.record(); model(x); end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))
print(f"median {sorted(times)[len(times)//2]:.3f} ms")
```

**为什么会需要预热：**

- cuBLAS 首次调用会做 heuristic 选择算法（可能几十毫秒）
- CUDA Graph 捕获第一次很慢
- JIT 编译的 kernel（Triton）第一次编译要几秒
- 权重从 page cache 冷读到显存

**实践：跑 benchmark 时永远报告「预热后 + 中位数」。**

### 端到端指标：TTFT / TPOT / 吞吐

```text
TTFT (Time To First Token)  —— prefill 延迟，影响「首字响应」
TPOT (Time Per Output Token) —— decode 单步延迟，影响「打字速度」
ITL  (Inter-Token Latency)  —— 连续 token 间隔，看抖动
吞吐 (tokens/s)             —— 单位时间总输出量
```

**这四个指标会互相矛盾。** 加大 batch 提升吞吐但恶化 TPOT；量化改善 TPOT 但可能影响质量。
所以先明确优化目标，再选指标。

vLLM 自带工具：

```bash
vllm bench throughput --model <model> --num-prompts 100
vllm bench latency   --model <model> --batch-size 1
```

## 3. 把 kernel 放回 Roofline 上

拿到一个 kernel 的时间和字节数，算出实际带宽利用率：

```python
def bandwidth_utilization(kernel_ms, bytes_moved, peak_bw_gbps):
    achieved = bytes_moved / (kernel_ms * 1e-3) / 1e9   # GB/s
    return achieved / peak_bw_gbps

# 例：RMSNorm, hidden=4096, batch=1, BF16
# 读 8 KB + 写 8 KB = 16 KB，假设耗时 2 微秒
print(bandwidth_utilization(0.002, 16 * 1024, 2039))   # → 0.004 ?!
```

上面这个数字小得离谱 —— 因为 **batch=1 的 RMSNorm 太小了，根本打不满带宽**（16 KB 对一个 2 微秒的 kernel 来说，说明瓶颈根本不是带宽，是 launch 开销）。

**这就是「把算子放进 roofline 框架」的价值**：它会告诉你这个算子优化空间本来就不大 —— 该做的不是优化它，而是**融合掉它**（见 [06-fusion-layout.md](06-fusion-layout.md)）。

### 判断规则

| 带宽利用率 | 结论 | 下一步 |
| --- | --- | --- |
| > 85% | 已经打满 | 只能减少字节数：量化 / 融合 |
| 60–85% | 还可优化 | 查向量化（float4）、合并访存、occupancy |
| 30–60% | 明显有问题 | 查 bank conflict、访存模式、block 配置 |
| < 30% | 瓶颈不是带宽 | 可能太小（融合掉）、或 launch 受限、或 CPU 受限 |

## 4. 用 Nsight Compute 深挖单个 kernel

```bash
# 快速看「这个 kernel 受限于什么」
ncu --set speedoflight ./your_program

# 完整分析（很慢，只跑一次）
ncu --set full -k "layernorm_kernel" -c 1 ./your_program

# 只看某几个指标
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    -k "gemm" ./your_program
```

### 值得看的指标（按重要性）

| 指标 | 含义 | 看什么 |
| --- | --- | --- |
| `dram__throughput.avg.pct_of_peak...` | DRAM 带宽利用率 | > 80% 说明是带宽瓶颈 |
| `sm__throughput.avg.pct_of_peak...` | 计算单元利用率 | 和上面一起看，谁高谁是瓶颈 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 实际 occupancy | 低于 30% 要查寄存器和 SMEM |
| `launch__registers_per_thread` | 寄存器占用 | > 128 要警惕，> 200 基本会 spill |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | SMEM bank conflict | 非 0 就要加 padding |
| `smsp__inst_executed.avg.per_cycle_active` | 指令发射效率 | 太低说明有 stall |

**`speedoflight` 视图最有用的地方**：它直接告诉你「Compute 90% / Memory 20%」或反过来，省掉自己算 roofline 的功夫。

### 常见的 stall 原因

```text
stall_long_scoreboard      → 等全局内存，说明延迟没藏住（或访存没合并）
stall_short_scoreboard     → 等 SMEM / MIO，可能是 bank conflict
stall_wait                 → 等固定延迟指令
stall_barrier              → __syncthreads() 等太久，负载不均衡
stall_not_selected         → 有太多 warp 可跑（其实是好事）
```

## 5. 常见反模式（先查这些，能省一半时间）

### 反模式 1：在循环里触发 CPU-GPU 同步

```python
# 每个 .item() / .cpu() / .tolist() / print(tensor) 都会强制同步
for i in range(n_steps):
    if loss.item() > threshold:        # ← 每次同步，GPU 流水线断裂
        ...
    print(f"step {i}, loss {loss}")    # ← 更糟
```

**表现**：GPU 利用率曲线是锯齿状的。**修法**：累积到 CPU 一侧的 list，最后一起转换。

### 反模式 2：关键路径上动态分配张量

```python
# cudaMalloc 可能同步，且在边缘设备上很慢
for _ in range(100):
    x = torch.empty(batch, hidden, device="cuda")
```

**修法**：预分配 buffer 池，或让框架的 caching allocator 复用（PyTorch 默认会，但**跨 shape 复用不了**）。

### 反模式 3：Python / 框架层开销

每个 `torch` 算子的 dispatch 开销约 **5~20 微秒**。decode 阶段 300+ 个算子：

```text
300 × 10 微秒 = 3 毫秒        ← 可能比 GPU 计算本身还慢
```

**边缘设备 CPU 更弱，这个问题更严重。** 修法：

- CUDA Graph（最有效）
- 算子融合，减少算子数量
- 用 `torch.compile` / 图模式

**检查方法**：在 profiler 里看 `cpu_time_total`，如果它显著大于 `cuda_time_total`，就是 CPU 受限。

### 反模式 4：动态 shape 破坏 CUDA Graph

```python
# 每次 batch size 都不同 → 每步都要重新捕获 CUDA Graph → 更慢
batch = len(requests)      # 1, 3, 5, 7, ...
```

**修法**：分桶到 `{1, 2, 4, 8, 16}`，向上取整到最近的桶，不足的用 padding（或让 attention kernel 处理 mask）。

### 反模式 5：用平均值而不是分位数

```python
print(f"平均延迟 {np.mean(latencies):.1f} ms")     # 掩盖了长尾
print(f"P99 延迟 {np.percentile(latencies, 99):.1f} ms")   # ← 用户真正感受到的
```

LLM 服务的 SLA 一般看 **P95/P99**。平均值好看但 P99 有 2 秒尖刺的服务是不可用的。

### 反模式 6：改了配置没重新测量

CUDA Graph 的开启、block size 的变化、`max-num-seqs` 的调整，**都会改变最优的其他参数**。
改一个参数就要重跑一次 benchmark。**别凭感觉连着改三个。**

## 6. 一个可复用的 benchmark 脚本骨架

```python
import torch, statistics

def benchmark(fn, warmup=10, iters=50, device="cuda"):
    """返回中位数、P99（毫秒）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return {
        "median": times[len(times) // 2],
        "p99":    times[int(len(times) * 0.99)],
        "min":    times[0],
    }

# 用的时候，把「关掉优化」和「打开优化」两版都跑一遍
base  = benchmark(lambda: model_naive(x))
fused = benchmark(lambda: model_fused(x))
print(f"median: {base['median']:.3f} → {fused['median']:.3f} ms"
      f"  ({base['median'] / fused['median']:.2f}×)")
```

**每次优化都要有这样一个对照。** 没有对照的数字就是猜测。

## 检查清单

- [ ] 用 profiler 确认了目标 kernel 占端到端的比例吗？（< 5% 就别碰了）
- [ ] 计时用 CUDA Event + 预热 + 中位数吗？
- [ ] 算了带宽利用率吗？知道瓶颈是带宽还是算力吗？
- [ ] 循环里有 `.item()` / `.cpu()` / 动态分配吗？
- [ ] CPU 时间和 GPU 时间哪个大？（决定要不要上 CUDA Graph）
- [ ] 报告的是 P99 还是平均值？
- [ ] 改一个变量测一次，还是一次改三个？（后者等于没测）
