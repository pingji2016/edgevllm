# LeetGPU 01 · Vector Addition

> 题源：`third_party/leetgpu-challenges/challenges/easy/1_vector_add/`
> （该目录为本地拉取、不进版本控制，原文见本地 `challenge.html`）
>
> 本文是对题目的**重新表述 + 代码注释 + 性能分析**，不是原文的转载。
> 涉及上游题面/模板的完整内容，请以本地 clone 为准。

---

## 1. 题目在问什么

把两个等长的 `float32` 向量逐元素相加，结果写进第三个向量：

```
C[i] = A[i] + B[i],  i = 0 .. N-1
```

约束（按题面整理）：

| 项 | 值 |
| --- | --- |
| 数据类型 | `float32` |
| `N` 范围 | 1 ~ 100,000,000 |
| 性能测试用的 `N` | **25,000,000** |
| 允许用外部库 | 否（cuBLAS / CUB 都不行） |
| `solve` 函数签名 | 不可改 |
| 结果落点 | 必须写进 `C`（原地输出，不是返回新张量） |

一眼看上去这是最平凡的 kernel——确实也是。但它作为第一题是**刻意**的：它把所有干扰项都拿掉，
只剩下「访存」这一件事。**这题唯一要证明的能力，是你的 kernel 能跑满显存带宽。**

---

## 2. 评测框架怎么运作

### 2.1 `challenge.py`（题目定义）

题目的 CPU 侧逻辑全部在 `challenge.py` 的 `Challenge` 类里，它继承 `core/challenge_base.py`
的 `ChallengeBase`。四个关键成员：

```python
class Challenge(ChallengeBase):
    name = "Vector Addition"
    atol = 1e-05      # 判定用的绝对容差
    rtol = 1e-05      # 判定用的相对容差
    num_gpus = 1      # 只用一张卡
    access_tier = "free"
```

**`reference_impl`** —— 标准答案，就是一行 `torch.add`：

```python
def reference_impl(self, A, B, C, N):
    assert A.shape == B.shape == C.shape
    assert A.dtype == B.dtype == C.dtype
    assert A.device == B.device == C.device
    torch.add(A, B, out=C)          # out= 表示原地写入 C
```

注意 `out=C`：参考实现和你的 `solve` 必须**写同一个缓冲区**，语义才对齐。

**`get_solve_signature`** —— 这是整个框架里最需要看懂的部分。它把参数列表翻译成
`ctypes` 的类型描述，供评测器通过 `ctypes.CDLL(...).solve` 去调用你的 `.so`：

```python
def get_solve_signature(self):
    return {
        "A": (ctypes.POINTER(ctypes.c_float), "in"),    # 只读输入
        "B": (ctypes.POINTER(ctypes.c_float), "in"),    # 只读输入
        "C": (ctypes.POINTER(ctypes.c_float), "out"),   # 只写输出
        "N": (ctypes.c_size_t,             "in"),    # ← 注意这里
    }
```

三个细节：

1. **参数顺序即字典顺序**，`solve` 必须按 `A, B, C, N` 这个顺序接收。
2. **`"in"` / `"out"` 决定评测器的搬运方向**：先把 `A`、`B` 拷到显存，`C` 是评测器在显存上
   分配好的空缓冲区；跑完再拷回 CPU 与 `reference_impl` 的结果比对。
3. **`N` 声明成 `ctypes.c_size_t`（64 位无符号），而 starter 里写的是 `int`（32 位）。**
   在 x86-64 System V 调用约定下，参数都走 64 位寄存器，被调方只读低 32 位——小端序下低 32 位
   正好是对的，所以 `N ≤ 2^31-1` 时**能跑通**。但这是个隐式依赖，真要出问题得等 `N` 上到 21 亿。
   本题 `N` 最大 1 亿，安全。

### 2.2 三类测试

`generate_example_test` / `generate_functional_test` / `generate_performance_test`
分别产出三个用途不同的用例集。

**example（1 例）**：`A=[1,2,3,4]`，`B=[5,6,7,8]`。`C` 用 `torch.empty` 分配——**不初始化**。

**functional（13 例）**：全部在 CPU 上用小样例把张量建好再搬上显存。按用途分四组：

| 分组 | 用例 | 想抓的 bug |
| --- | --- | --- |
| 极短向量 | `scalar_tail_1/2/3`（N=1,2,3） | 边界检查写错、一个 warp 都不满时崩掉 |
| 基本正确性 | `basic_small`(4)、`all_zeros`(16)、`non_power_of_two`(30) | N 不是 block 整数倍时的尾部处理 |
| 数值范围 | `negative_numbers`、`mixed_positive_negative`、`very_small_numbers`、`large_numbers` | 误用绝对值/符号相关运算 |
| 随机规模 | 32、1000、10000 | 一般化 |

这里有个**容易忽略的坑**：functional 用例里 `C` 是 `torch.zeros(n)` 分配的（第 73、88 行），
而 `all_zeros` 这条的期望输出恰好也全是 0。也就是说 —— **一个什么都没做的空 kernel，
能通过 `all_zeros` 这一例**。别指望靠跑测试用例来自证正确性，随机规模那三例才是真的在验。

**performance（1 例）**：`N = 25_000_000`，`A`、`B` 在 `[-1000, 1000)` 上均匀随机，`C` 预先置零。
这一例不计正确性，只计时。

### 2.3 判定标准

`atol = rtol = 1e-5`，即 `|C_got - C_ref| ≤ atol + rtol * |C_ref|`。

这题其实**比这个宽松得多**：参考实现是 `float32` 的逐元素加法，你的 kernel 也是逐元素加法，
每个元素**只做一次加法**，没有累加顺序的问题——所以两者应当**逐比特相同**，
`1e-5` 的容差根本用不上。

反过来说：如果这题你都跑出超过 `1e-5` 的误差，那不是精度问题，是**逻辑写错了**
（读写越界、类型错了、同步没做）。这个判断在后面的题里要反过来用：
一旦出现归约（reduce/sum/softmax），误差就变成真问题，容差才真正开始约束实现。

---

## 3. 模板代码逐行注释

### 3.1 CUDA 模板（`starter/starter.cu`）

上游模板把 kernel 体留空，只搭好骨架。下面是**带完整注释的等价代码**：

```cuda
#include <cuda_runtime.h>

// 设备端 kernel：每个线程负责一个（或若干）元素
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // 留空 —— 需要自己填
}

// 主机端入口。
// extern "C" 是关键：禁用 C++ name mangling，
// 这样评测器用 ctypes 按名字 "solve" 就能 dlsym 到它。
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    // 向上取整：让 grid 覆盖 N 个元素，即使 N 不是 256 的整数倍
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // <<<gridDim, blockDim>>>：配置并异步启动 kernel
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

    // 同步等待 kernel 结束。评测器调用 solve 后立刻就要读 C，
    // 不 sync 的话读到的是未完成的结果。
    cudaDeviceSynchronize();
}
```

三个值得停下来想的点：

- **`extern "C"` 不是可选的。** C++ 会把 `solve` 链接成 `_Z5solvePKfS0_Pfi` 之类的符号名，
  ctypes 有 `argtypes` 也没用，找不到符号直接报错。
- **`<<<...>>>` 是异步的**，`solve` 返回时 kernel 可能还没跑完。`cudaDeviceSynchronize()`
  在这里是**正确性**的一部分，不是性能优化。注意它同步的是**默认流（legacy default stream）**——
  如果你自己建了流并往里 launch 而不做同步，会静默出错。
- **为什么不直接用 `throw` / 返回值报错？** `solve` 的签名被题面锁死了，是 `void` 返回。
  出错只能靠 `printf` 或断言。

**模板留下的两个坑：**

1. `blocksPerGrid` 是 `int`，`N + 255` 在 `N` 接近 `INT_MAX` 时会溢出。本题 `N ≤ 1e8`，
   不会触发；但这是写 kernel 时要条件反射式检查的东西（另一处同类问题是 `gridDim.x` 上限
   `2^31-1`，即 `N` 超过约 5.5e11 时才需要换 grid-stride loop）。
2. **尾部元素**：`blocksPerGrid` 保证了 grid 覆盖 `N`，但**多出来的线程会访问越界地址**。
   必须在 kernel 里加 `if (i < N)`。`scalar_tail_1/2/3` 这几条用例就是专门来抓这个的。

### 3.2 Triton 模板（`starter/starter.triton.py`）

```python
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pass                       # 留空

# a, b, c 是显存上的 tensor（不是裸指针，与 CUDA 版不同）
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    # triton.cdiv = 向上取整除法，等价于 (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    # [grid] 是 Triton 的 launch 语法；BLOCK_SIZE 作为 constexpr 编译期常量传入
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
```

与 CUDA 版的差异：

- **不用 `cudaDeviceSynchronize()`。** Triton 在默认流上启动，而 PyTorch 的后续操作
  （比对时的 `.cpu()`、`torch.allclose`）会自己同步。这也是为什么 Triton 模板的
  `solve` 收的是 `torch.Tensor` 而不是裸指针。
- **`BLOCK_SIZE` 标了 `tl.constexpr`**，编译期常量，Triton 据此特化出不同版本；
  CUDA 里对应的是模板参数或宏。
- `N` 参数字面量在 kernel 里叫 `n_elements`——名字随意，位置才是契约。

---

## 4. 一份正确的实现

### 4.1 朴素正确版（先保证对）

```cuda
#include <cuda_runtime.h>

__global__ void vector_add(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           long long N) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {                 // ← 尾部保护，scalar_tail_* 用例靠它
        C[i] = A[i] + B[i];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    const int threads = 256;
    long long blocks = ((long long)N + threads - 1) / threads;
    vector_add<<<(unsigned)blocks, threads>>>(
        (const float*)A, (const float*)B, C, (long long)N);
    cudaDeviceSynchronize();
}
```

`__restrict__` 告诉编译器 A/B/C 不重叠。这题指针确实不重叠，加上它能让编译器放心地
重排 load/store 顺序，是**零成本的正确标注**。

### 4.2 带宽版（`float4` 向量化）

25M 个元素、每个元素一次 4 字节 load。如果编译不向量化，就是 5000 万次 4 字节事务。
用 `float4` 一次搬 16 字节，`LDG.128`/`STG.128` 能把访存指令数降到 1/4，
对**带宽题**来说这才是决定性的（见下一节的算账）。

```cuda
__global__ void vector_add4(const float4* __restrict__ A,
                            const float4* __restrict__ B,
                            float4* __restrict__ C,
                            long long n4) {
    // grid-stride：让固定大小的 grid 覆盖任意大的 n4，
    // 顺便避免 blocks > 2^31-1 的问题
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n4; i += stride) {
        float4 a = A[i];
        float4 b = B[i];
        C[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    const int threads = 256;
    long long n4 = N / 4;          // 能整除 4 的元素用 float4 处理
    long long tail_start = n4 * 4; // 余数部分（N % 4 个）单独处理

    if (n4 > 0) {
        // 用 blocks 上限压住 grid，靠 stride 循环覆盖全部
        long long want = (n4 + threads - 1) / threads;
        unsigned blocks = (unsigned)(want > 65535 * 32 ? 65535 * 32 : want);
        vector_add4<<<blocks, threads>>>(
            (const float4*)A, (const float4*)B, (float4*)C, n4);
    }

    // 尾部：最多 3 个元素。这里直接退回标量处理，简单且开销可忽略。
    for (int i = (int)tail_start; i < N; ++i) C[i] = A[i] + B[i];

    cudaDeviceSynchronize();
}
```

**`float4` 的对齐前提**：`reinterpret_cast` 到 `float4*` 要求地址 16 字节对齐。
PyTorch 的 caching allocator 返回的显存基址至少 256 字节对齐，所以这里成立。
但**这是个需要验证的前提**，不是能随便假设的——在别的框架/别的分配路径下，
不对齐会导致非法地址访问（misaligned address 错误，硬崩）。

### 4.3 Triton 版

```python
import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                          # 当前 block 的编号
    # 本 block 负责的下标区间
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements                        # 尾部掩码

    x = tl.load(a + offs, mask=mask)                # 越界位置不加载
    y = tl.load(b + offs, mask=mask)
    tl.store(c + offs, x + y, mask=mask)            # 越界位置不写回


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
```

Triton 的 mask 相当于一步到位的 `if (i < N)`——**读和写都要加 mask**，
只给 store 加 mask 而 load 不加，越界读一样会崩。

Triton 的向量化是编译器自己推断的：它靠 `offs` 的连续性判断能否生成 128 位访存。
如果发现没能向量化，可以用 `tl.max_contiguous(tl.multiple_of(offs, BLOCK_SIZE), BLOCK_SIZE)`
把「这段下标是连续且对齐的」显式告诉编译器。

### 4.4 本地编译运行

`starter.cu` **只有骨架**：kernel 体是空的，而且**没有 `main`**——它是个共享库，
由评测器 `ctypes.CDLL` 加载后调 `solve`。所以不能直接「编译成 exe 跑」。

更麻烦的是：**官方评测脚本 `challenge.py` 本地跑不了**，因为它依赖 CUDA 版 PyTorch
（`A.device == "cuda"`、`torch.add(out=C)`）。而 `pip install torch` 默认装的是 CPU 版，
`torch.cuda.is_available()` 返回 `False`。

所以本地练手的正确姿势是：**自己写一个 `main`，直接用 `cudaMalloc` + kernel 调用来验证**。
完整可运行的版本在
[examples/leetgpu/01_vector_add/vector_add.cu](../../examples/leetgpu/01_vector_add/vector_add.cu)，
自带正确性校验（覆盖 N=1/2/3/7/30/255/256/257 等边界）和三配置性能对比。

```bash
cd examples/leetgpu/01_vector_add
nvcc -O3 -arch=sm_120 vector_add.cu -o vector_add.exe
./vector_add.exe          # 默认 N = 25,000,000
./vector_add.exe 1000000  # 自定义 N
```

`-arch` 按自己的卡填：RTX 50 系（Blackwell）是 `sm_120`，40 系是 `sm_89`，30 系是 `sm_86`。
用 `nvidia-smi` 看型号，或用 `nvcc --list-gpu-arch` 列出全部。
写 `-arch=native` 让 nvcc 自己探测也行（要求 nvcc 版本认得这张卡）。

想确认生成的指令宽度，反汇编看 SASS：

```bash
cuobjdump -sass vector_add.exe | grep -E 'LDG|STG'
```

#### Windows 上的两个编码坑

这两个坑都会浪费半小时，而且报错信息完全指不到真正的原因。

**坑 1：`.cu` 文件必须存成「带 BOM 的 UTF-8」。**

nvcc 前端在 Windows 上默认按本地代码页（中文系统是 936/GBK）读源码。
UTF-8 的中文注释被按 GBK 拆字节后会错位，**吃掉后面的换行和代码**，
于是报出一堆莫名其妙的错：

```
error: identifier "tail" is undefined          ← 它明明就在上面几行
warning C4010: 单行注释包含行继续符
error: missing closing quote
```

加 BOM 后前端自动识别 UTF-8，全部消失。**加 `/utf-8` 编译选项没用**——
那只是传给宿主编译器 `cl.exe` 的，nvcc 自己的前端照样按 GBK 读。

```bash
# 给已有文件补 BOM
python -c "p='vector_add.cu'; b=open(p,'rb').read(); open(p,'wb').write(b'\xef\xbb\xbf'+b if not b.startswith(b'\xef\xbb\xbf') else b)"
```

**坑 2：运行时的 `printf` 字面量一律用 ASCII。**

窄字符串字面量编译时会被转成**执行字符集**，MSVC 默认就是本地代码页（936）。
于是源码里 UTF-8 的中文，到运行时变成 GBK 字节，往管道/控制台一输出全是乱码。

而用 `-Xcompiler -utf-8` 去修执行字符集，**又会反过来让 nvcc 前端解析不了这些中文字面量**
（报 `missing closing quote`，和坑 1 同样的错）——两个修法互相打架。

最省事的解法：**中文只写在注释里，`printf` 的内容全用 ASCII。**
注释有 BOM 就能正确解析，运行时输出是纯 ASCII 则与代码页无关。

> 顺带一提，Git Bash 下 `-Xcompiler /utf-8` 会被 MSYS 的路径转换吃掉，
> 变成 `-Xcompiler D:/git/Git/utf-8`，报 `c1xx: fatal error C1083: 无法打开源文件`。
> 要写就得写 `-Xcompiler -utf-8`（短横线）。不过按坑 2 的建议，你不需要这个选项。

#### Linux / WSL 下怎么搞

**上面两个编码坑在 Linux 上全部消失。** 没有代码页概念，默认就是 UTF-8，
不用加 BOM，`printf` 里直接写中文也没问题。命令行几乎一样，只是没有 `.exe` 后缀：

```bash
nvcc -O3 -arch=sm_120 vector_add.cu -o vector_add
./vector_add
```

差别只在宿主编译器：Windows 上是 `cl.exe`（MSVC），Linux 上是 `g++`。
`-Xcompiler` 的选项要跟着换（比如 `-Xcompiler -fPIC`）。

**在 WSL2 里装 CUDA 有一个必须注意的坑：绝不能装 Linux 版显卡驱动。**

WSL2 的 GPU 是透传的：Windows 侧的驱动通过 `/dev/dxg` 暴露给 Linux，
WSL 自带一个 `libcuda.so` 桩（在 `/usr/lib/wsl/lib/`，由 WSL 注入、不属于任何 dpkg 包）
把调用转发过去。一旦装了 Linux 驱动，这个桩会被覆盖，GPU 直接不可用且很难恢复。

正确做法是**只装 toolkit，不装驱动**，用 NVIDIA 官方 apt 源里的 `cuda-toolkit-*`：

```bash
# 1. 加源（Ubuntu 24.04 = noble，对应 ubuntu2404）
curl -sL -o /tmp/cuda-keyring.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update

# 2. 只装 toolkit —— 注意是 cuda-toolkit-12-8，不是 cuda-12-8
sudo apt-get install -y --no-install-recommends cuda-toolkit-12-8

# 3. PATH
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh
```

装之前务必先用 dry-run 验证一遍**依赖里没有驱动包**：

```bash
apt-get install -s --no-install-recommends cuda-toolkit-12-8 | grep '^Inst' | grep -E 'cuda-drivers|nvidia-driver'
# 必须输出为空
```

两个名字很像但完全不同的包，别搞混：

| 包名 | 是什么 | 能不能装 |
| --- | --- | --- |
| `cuda-toolkit-12-8` | 工具链（nvcc / ptxas / cudart / 头文件） | ✅ 装这个 |
| `cuda-driver-dev-12-8` | 驱动 API 的**头文件**和链接 stub | ✅ 无害，会被自动带上 |
| `cuda-drivers` / `cuda-drivers-550` | **真正的驱动**（550.90.07） | ❌ 绝不能装 |

`cuda-toolkit-12-8` 的依赖链是 compiler / libraries / libraries-dev / tools /
documentation / nvml-dev，**不含 `cuda-drivers`**（实测 86 个包，约 3.5 GB）。
`libcuda.so` 仍然来自 `/usr/lib/wsl/lib/`。

#### 两边实测对比

同一份源码，同一张卡（RTX 5060 Ti），Windows 原生 vs WSL2：

| 配置 | Windows | WSL2 |
| --- | --- | --- |
| 朴素标量版 | 760.8 µs (88.0%) | 777 µs (86.2%) |
| `float4` 铺满 grid | 760.9 µs (88.0%) | 801 µs (83.7%)¹ |
| `float4` 超订 grid | 933.1 µs (71.8%) | 975 µs (68.7%) |

¹ WSL 的三次测量是 836 / 783 / 784 µs，第一次是离群值。取中位数约 784 µs。

**两点值得注意：**

1. **WSL 整体略慢**（约 2%），这在 WSL2 上是正常的——虚拟化层会带来一些开销。
2. **「向量化不带来收益」这个结论在两边都成立。** WSL 的 naive 和 float4 差异
   落在测量噪声里（784 vs 777，不到 1%），而 grid 超订在两边都稳定慢 25% 左右。
   **这个结论是稳的，不是某一台机器的偶然。**
   完整原始数据见 [examples/leetgpu/01_vector_add/](../../examples/leetgpu/01_vector_add/)。

---

## 5. 这题在考什么：一次带宽算账

用 [01-roofline.md](../kernel-opt/01-roofline.md) 的框架算一遍。

**访存量**（`N = 25e6`）：

```
读 A : 25e6 × 4 B = 100 MB
读 B : 25e6 × 4 B = 100 MB
写 C : 25e6 × 4 B = 100 MB
                    ────────
合计  : 300 MB   （每个元素 12 字节）
```

**计算量**：每个元素 1 次加法 = 1 FLOP，总共 25 MFLOP。

**算术强度**：

```
AI = 25e6 FLOP / 300e6 Byte = 0.083 FLOP/Byte
```

对比 [01-roofline.md](../kernel-opt/01-roofline.md) 里 A100 的拐点约 **78 FLOP/Byte** ——
差了**三个数量级**。结论毫无悬念：

> **纯带宽瓶颈。算力在这里完全不是约束，一个 FLOP 对应 12 字节的搬运。**

**理论时间下限**（就等于访存量除以带宽）：

| 设备 | 带宽 | 理论最短耗时 |
| --- | --- | --- |
| A100-SXM-80G | 2039 GB/s | 300e6 / 2039e9 ≈ **147 µs** |
| H100-SXM | 3350 GB/s | ≈ **90 µs** |
| Jetson Orin（LPDDR5） | ~204 GB/s | ≈ **1.47 ms** |

**这决定了优化的全部方向**：既然墙是带宽，那就只有两件事可做——

1. **把实际带宽打到峰值附近**（这是这题的主线）；
2. **少搬字节**（这题没有余地：A、B 必须读、C 必须写，`float32` 是题面锁死的）。

第 2 条正是 [05-quantization.md](../kernel-opt/05-quantization.md) 里 INT8/INT4 的动机——
但在这题上无用武之地。所以本题**唯一**的胜负手是第 1 条。

而且注意：这题**没有数据复用**。A、B 每个字节只被读一次，C 每个字节只被写一次，
共享内存、L1、寄存器复用**全部无效**——没有任何东西可以缓存下来二次使用。
这是这题与 GEMM / Attention 的根本区别：那些题的算术强度高，才轮得到 tiling 和复用发挥。
**把它当成反面教材记住：什么情况下 tiling 是白费力气。**

### 实测结果（RTX 5060 Ti，N=25,000,000）

下面是 Windows 原生编译的实测数据（WSL2 的对比数据见 4.4 节），可运行的代码见
[examples/leetgpu/01_vector_add/vector_add.cu](../../examples/leetgpu/01_vector_add/vector_add.cu)：

| 配置 | 耗时 | 带宽 | 峰值占比 |
| --- | --- | --- | --- |
| 朴素标量版（铺满 grid） | 762.7 µs | 393.3 GB/s | 87.8% |
| `float4` 铺满 grid | 761.0 µs | 394.2 GB/s | 88.0% |
| `float4` + 超大 grid（grid-stride 空转） | 932.7 µs | 321.7 GB/s | 71.8% |

> RTX 5060 Ti 16GB：128-bit GDDR7，理论峰值 448 GB/s。

**这个结果推翻了一条流传很广的经验。** 前两行几乎完全相等——向量化**没有**带来收益，
而第三行慢了 22%，那才是真正的坑。两处都要展开说。

#### 反直觉之一：这里向量化不带来收益

反汇编确认了两种实现确实不同（`cuobjdump -sass`）：

- 朴素版：`LDG.E.CONSTANT`（32 位）+ `STG.E`（32 位）
- float4 版：`LDG.E.128.CONSTANT`（128 位）+ `STG.E.128`（128 位）

指令宽度确实差了 4 倍，但**带宽一模一样**。原因是：

> **一个 warp 的 32 个线程访问 32 个连续 `float`，本身就会被合并成 1 次 128 字节事务。
> 也就是说标量版在 DRAM 层面产生的事务数，和向量化版是一样的。**

向量化省下的是**指令发射**（4 条 `LDG` 变 1 条 `LDG.128`），不是内存事务。
而一个访存延迟完全被遮蔽的纯带宽 kernel，瓶颈在 DRAM 而不在发射带宽——
省下的指令数根本用不上。88% 的峰值占比已经接近这类 kernel 的实际天花板。

**所以「向量化能拿回 30%~50% 带宽」这个说法是有前提的**，它成立于：

- 访存**没有**被合并的时候（线程访问跨步、非连续）；
- 或者 kernel 是**指令发射受限**而非带宽受限（occupancy 很低、每线程工作量很大）。

这题两者都不是。**结论：先测量，再决定要不要向量化。**
`LDG.E.128` 比 `LDG.E` 好看，但好看不等于快。

#### 反直觉之二：grid-stride 用错了会亏 22%

第三行是 `vector_add_vec4<<<65535*4, 256>>>`，即 **6710 万个线程去干 625 万件事**。
多出来的线程因为 `i < n4` 一次循环都不执行——听起来是免费的。实际不是：

- 这些线程仍然要被**分配、调度、执行判断、回收**；
- 每个线程至少要跑一遍循环条件判断和寄存器初始化；
- 6710 万线程的调度开销，最终压在 762 µs 这个量级上就是 **+170 µs**。

用**执行模型**的语言说更清楚：这张卡每个 SM 只能同时驻留 1536 个线程，
256 线程的 block 就是 6 个 —— 一个 wave = 36 SM × 6 = 216 个 block。
262,140 个 block 里只有 24,415 个有活干，**剩下 237,725 个是纯空转，占了约 1100 个 wave**。
block / warp / wave 这套概念和推导见
[02-cuda-basics.md 第 0 节](../kernel-opt/02-cuda-basics.md)，那里用本题的三种配置做了完整对照。

**这题的 `n4` 只有 625 万，`cdiv(n4, 256) = 24415` 个 block 就正好铺满**，
根本不需要 grid-stride。4.2 里那个 `cap = 65535*32` 的写法是为
「`n4` 大到 grid 上限兜不住」准备的通用保险，**本题触发不了**。

判断规则很简单：**grid-stride 的 grid 大小要贴着工作量配，不要随便给个上限值。**
要么用 `cdiv(n, threads)` 直接铺满，要么按 SM 数量算一个合理的倍数
（`blocks ≈ SMs × waves`），而不是拍一个 `65535*32`。

#### 那这题的优化空间到底在哪

三行数据合起来说明：**88% 已经到头了**。剩下的 12% 是 DRAM 的刷新、
行激活等固有开销，不是靠改 kernel 能拿回来的。真要更好，只能：

- **少搬字节**（本题被 `float32` 锁死，没有余地）；
- 或者接受这是这道题的天花板。

查问题的顺序（如果实测远低于 88%）：

- grid/block 配置是不是把并行度压死了，或者反过来超订了？（最常见）
- 指针是不是真的不重叠？`__restrict__` 加了没？
- 用 [08-profiling.md](../kernel-opt/08-profiling.md) 里讲的
  `dram__throughput.avg.pct_of_peak_sustained_elapsed` 直接看带宽利用率。
- 最后才轮到看 SASS 里是 `LDG.E.128` 还是 `LDG.E`。

### A100 / H100 上的预期（未实测，按带宽推算）

本文档开头给出的 147 µs 是**按 2039 GB/s 理论峰值算的解析值**，不是实测。
按本机 88% 的实际达成率外推：

| 设备 | 理论峰值 | 理论下限 | 按 88% 估算的实测 |
| --- | --- | --- | --- |
| RTX 5060 Ti | 448 GB/s | 670 µs | **763 µs（已实测，87.8%）** |
| A100-SXM-80G | 2039 GB/s | 147 µs | ≈ 167 µs |
| H100-SXM | 3350 GB/s | 90 µs | ≈ 102 µs |

**这两行估算没有验证过**，手头没有对应的卡。实际达成率会随架构变化——
HBM 设备的可达峰值占比通常比 GDDR 更高一些，所以真实值可能比表里的估算更好。

---

## 6. 常见错误清单

| 症状 | 原因 |
| --- | --- |
| `undefined symbol: solve` | 忘了 `extern "C"`，符号被 C++ mangle 了 |
| `scalar_tail_*` 用例崩 | kernel 里没写 `if (i < N)`，尾部线程越界 |
| 结果全错 / 全是 0 | 写了 `C = A + B` 但没落盘；或 kernel 参数顺序传错 |
| 间歇性错误 | 少了 `cudaDeviceSynchronize()`，评测器读到了未完成的 `C` |
| `misaligned address` | `float4` 强转但指针未 16 字节对齐 |
| `all_zeros` 过、其他全挂 | 空 kernel。`all_zeros` 的期望输出全是 0，而 `C` 预置为 0，空实现恰好能过——**这条用例不构成正确性证明** |

---

## 7. 一句话小结

**这题的全部内容是：认出一个 0.083 FLOP/Byte 的纯带宽 kernel，然后把它打到带宽峰值。**

而实测告诉我们，达到峰值靠的不是向量化——**朴素标量版和 `float4` 版一样快（762 µs vs 761 µs）**，
真正会拖慢你的是 grid 配置错误（超订线程，+22%）。
向量化在这里省的是指令发射，不是内存事务，而瓶颈在 DRAM。

它的价值不在难度，在于它是后面所有题的**基准线**——GEMM、Attention 那些题里
每一个「这个优化到底有没有用」的判断，都要靠和这条基准线的对比来回答。
但**这条基准线本身也提醒你：教科书上的优化经验有前提，先测量再下结论。**
