# examples

可直接编译运行的 CUDA 示例。每个示例都是自包含的 `.cu`，自带正确性校验和性能测量，
不依赖 PyTorch 或任何外部库。

判题平台 [LeetGPU](https://leetgpu.com) 只给出函数骨架（kernel 体为空、没有 `main`），
评测器通过 `ctypes` 加载 `.so` 后调用 `solve`；而它的评测脚本 `challenge.py` 依赖
CUDA 版 PyTorch，本地不一定装得上。这里的示例补上了缺失的 `main`，方便本地练手。

---

## 快速开始

**Linux / WSL2 / Git Bash：**

```bash
cd examples
./build.sh run
```

**Windows 的 cmd / PowerShell / 资源管理器：**

```bat
build.cmd run
```

两边跑的是同一个 `build.sh`，`build.cmd` 只是个找 bash 的壳。脚本会自动找 `nvcc`、
探测本机 GPU 架构、编译所有 `.cu`、然后逐个运行。

> **在 Windows 上双击 `build.sh` 会打开编辑器，而不是执行它。** 这是预期行为，不是 bug：
> Windows 的 `PATHEXT` 里没有 `.SH`，双击走的是文件关联，`.sh` 的默认程序通常是编辑器。
> `.sh` 需要 bash 解释器才能跑——用 `build.cmd`，或者开一个 Git Bash / WSL 终端敲 `./build.sh`。
> 详见下面「Windows 上怎么运行 `build.sh`」。

> 如果报 `Permission denied`，说明 clone 下来时丢了可执行位（本仓库 `core.filemode=false`，
> git 不记录权限位）。两种解法：`chmod +x build.sh`，或者直接 `bash build.sh run`——
> 后者在任何情况下都能用。

```
==> nvcc      /usr/local/bin/nvcc
==> arch      native
==> platform  Linux
  ok leetgpu/01_vector_add/device_query.cu  ->  build/leetgpu/01_vector_add/device_query
--- 运行 device_query ---
...
==> 完成：2 个目标 -> build/
```

## 脚本用法

```bash
./build.sh                              # 只编译 examples/ 下所有 .cu
./build.sh run                          # 编译后运行
./build.sh clean                        # 清理 build/
./build.sh leetgpu/01_vector_add        # 只处理指定目录
./build.sh run leetgpu/01_vector_add    # 编译 + 运行指定目录
./build.sh path/to/foo.cu               # 也可以直接指定单个文件
./build.sh --help
```

在 cmd / PowerShell 里把上面的 `./build.sh` 换成 `build.cmd` 即可，参数完全一样
（包括反斜杠写法 `build.cmd run leetgpu\01_vector_add`，脚本内部会归一化路径分隔符）。

产物统一放在 `examples/build/`（`.gitignore` 已忽略），不污染源码目录。

两个环境变量可以覆盖默认行为：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `CUDA_ARCH` | `native` | nvcc 的 `-arch`。`native` 自动探测本机 GPU |
| `NVCC` | 从 PATH 找 | 显式指定 nvcc 路径 |

```bash
CUDA_ARCH=sm_89 ./build.sh run          # 强制按 Ada 架构编译
NVCC=/usr/local/cuda-12.8/bin/nvcc ./build.sh
```

## 手动编译

不想用脚本的话，直接调 `nvcc` 也一样：

```bash
cd examples/leetgpu/01_vector_add

# Windows（Git Bash / cmd / PowerShell）
nvcc -O3 -arch=sm_120 vector_add.cu -o vector_add.exe
./vector_add.exe

# Linux / WSL2
nvcc -O3 -arch=sm_120 vector_add.cu -o vector_add
./vector_add
```

`-arch` 按自己的卡填：**50 系是 `sm_120`**，40 系 `sm_89`，30 系 `sm_86`。
用 `-arch=native` 让 nvcc 自己探测更省事（需要 CUDA 11.5+）。
不确定的话跑一下 `device_query`，它会打印 `compute capability`。

---

## 环境准备

### Windows 原生

装 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)（本机是 12.8）和
Visual Studio（提供 MSVC，nvcc 会自动找到并调用 `cl.exe`）。装完 `nvcc` 就在 PATH 里了。

#### Windows 上怎么运行 `build.sh`

**双击 `build.sh` 会打开编辑器而不是执行它。** 如果双击后 VSCode 弹出来了、还显示一堆
启动日志，那说明**脚本压根没跑**——不是脚本里的东西打开了 VSCode。

原因有两层：

1. Windows 的 `PATHEXT` 环境变量里**没有 `.SH`**。`cmd` 和 Explorer 都靠 `PATHEXT`
   判断「这个扩展名能不能直接执行」；`.sh` 不在里面，所以不会走「运行」那条路，
   而是退回到**文件关联**——用关联程序打开它。
2. 关联程序是编辑器。查一下本机：

   ```bat
   reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\.sh\OpenWithList"
   ```

   ```
       a        REG_SZ    Code.exe
       MRUList  REG_SZ    a
   ```

   `.sh` 的打开方式 MRU 里只有 `Code.exe`，且该扩展名没有 `UserChoice`（用户显式选定的
   默认应用），于是这个 MRU 直接生效。用 Shell API 的 `AssocQueryString` 查最终解析结果，
   拿到的就是 `"C:\VSCode\Code.exe" "%1"`——把文件当文档喂给编辑器。

   > ⚠️ **别用 `assoc .sh` 判断**，它会误导你。本机 `assoc .sh` 返回 `sh_auto_file`、
   > `ftype sh_auto_file` 返回 `git-bash.exe`，看着像是「双击会跑起来」——但那是
   > Git for Windows 写的机器级 `HKCR` 映射，**优先级低于上面那个按用户记录的 MRU**。
   > 实际双击走的是编辑器。

`.sh` 需要 bash 解释器，关联不会给你变出来。三条可行路径：

```bat
build.cmd run                           :: 推荐。cmd / PowerShell / 双击都能用
```

```bash
./build.sh run                          # Git Bash 终端里
```

```bash
cd /mnt/e/github/edgevllm/examples && ./build.sh run    # WSL 里
```

`build.cmd` 就是替你找 bash 的一层壳：用 `where bash.exe` 找解释器（优先 Git Bash），
再把参数原样转给 `build.sh`。找不到就报错提示装 Git for Windows。

⚠️ **两个编码坑，改代码前务必看一下。** 这两个坑的报错信息完全指不到病根：

**1. `.cu` 文件必须存成「带 BOM 的 UTF-8」。**

nvcc 前端在 Windows 上按本地代码页（中文系统是 936/GBK）读源码，UTF-8 的中文注释
会被拆错字节、吃掉后面的换行和代码，然后报出一堆假错误：

```
error: identifier "tail" is undefined          ← 它明明就在上面几行
warning C4010: 单行注释包含行继续符
error: missing closing quote
```

`build.sh` 会在编译前检查并拦住这种情况。手动修：

```bash
python -c "p='vector_add.cu';b=open(p,'rb').read();open(p,'wb').write(b'\xef\xbb\xbf'+b)"
```

注意加 `-Xcompiler /utf-8` 编译选项**没用**——那只传给 `cl.exe`，nvcc 自己的前端照样按 GBK 读。

**2. `printf` 的字面量一律用 ASCII 写。**

窄字符串字面量会被编译成**执行字符集**，MSVC 默认就是本地代码页。源码里的 UTF-8 中文
到运行时变成 GBK 字节，往控制台/管道一输出全是乱码。而用 `-Xcompiler -utf-8` 去修
执行字符集，又会反过来让 nvcc 前端解析不了中文字面量——**两个修法互相打架**。

所以本仓库的约定是：**中文只写在注释里，运行时的 `printf` 内容全用 ASCII。**
注释有 BOM 就能正确解析，ASCII 输出则与代码页无关。

> 顺带一提，Git Bash 下 `-Xcompiler /utf-8` 会被 MSYS 的路径转换吃掉，变成
> `-Xcompiler D:/git/Git/utf-8`，报 `c1xx: fatal error C1083`。要写就得写
> `-Xcompiler -utf-8`（短横线）。不过按上面的约定，你不需要这个选项。

**上面两个坑在 Linux 上全都不存在**——没有代码页概念，默认就是 UTF-8。
同一个 `.cu` 文件在两边都能编，那份 BOM 在 Linux 上无害。

### WSL2

WSL2 的 GPU 是**透传**的：Windows 侧的驱动通过 `/dev/dxg` 暴露给 Linux，
WSL 自带一个 `libcuda.so` 桩（`/usr/lib/wsl/lib/`，由 WSL 注入、不属于任何 dpkg 包）
把调用转发过去。

⚠️ **绝不能装 Linux 版显卡驱动。** 装了会覆盖那个桩，GPU 直接不可用且很难恢复。

正确做法是**只装 toolkit，不装驱动**：

```bash
# 1. 加 NVIDIA 官方 apt 源
curl -sL -o /tmp/cuda-keyring.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update

# 2. 只装 toolkit —— 是 cuda-toolkit-12-8，不是 cuda-12-8
sudo apt-get install -y --no-install-recommends cuda-toolkit-12-8
```

装之前务必 dry-run 验证依赖里没有驱动包：

```bash
apt-get install -s --no-install-recommends cuda-toolkit-12-8 | grep '^Inst' \
  | grep -E 'cuda-drivers|nvidia-driver'      # 必须输出为空
```

三个名字很像但完全不同的包，别搞混：

| 包名 | 是什么 | 能不能装 |
| --- | --- | --- |
| `cuda-toolkit-12-8` | 工具链（nvcc / ptxas / cudart / 头文件） | ✅ 装这个 |
| `cuda-driver-dev-12-8` | 驱动 API 的**头文件**和链接 stub | ✅ 无害，会被自动带上 |
| `cuda-drivers` / `cuda-drivers-550` | **真正的驱动** | ❌ 绝不能装 |

最后配 PATH。**要配两处**，否则脚本和非登录 shell 会找不到 `nvcc`：

```bash
# 登录 / 交互式 shell
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh

# 非登录 shell（脚本、CI）—— /etc/profile.d 不生效，需要包装脚本。
# 注意不能用符号链接：nvcc 按自身路径找兄弟工具和头文件，软链到 /usr/local/bin
# 会导致 ptxas 和 cuda_runtime.h 都找不到。
for t in nvcc ptxas nvdisasm cuobjdump compute-sanitizer; do
  sudo sh -c "printf '#!/bin/sh\nexec /usr/local/cuda-12.8/bin/%s \"\$@\"\n' $t > /usr/local/bin/$t && chmod +x /usr/local/bin/$t"
done
```

验证：

```bash
bash -c 'nvcc --version'        # 非登录 shell 也要能找到
bash -c 'nvidia-smi -L'         # 应该能看到 GPU
```

---

## 目录

| 路径 | 内容 |
| --- | --- |
| [build.sh](build.sh) | 编译 / 运行脚本，Git Bash + Linux / WSL2 通用 |
| [build.cmd](build.cmd) | Windows 壳，供 cmd / PowerShell / 双击使用 |
| [leetgpu/01_vector_add/](leetgpu/01_vector_add/) | LeetGPU 第 1 题：向量加法 |
| ├ [vector_add.cu](leetgpu/01_vector_add/vector_add.cu) | 朴素版 + `float4` 向量化版，自带正确性校验和三配置性能对比 |
| ├ [device_query.cu](leetgpu/01_vector_add/device_query.cu) | 打印设备属性与执行模型限制，用来算 wave 数、寄存器预算等 |
| └ [docs/leetgpu/01-vector-add.md](../docs/leetgpu/01-vector-add.md) | 这道题的完整解读与实测分析 |

`device_query` 值得先跑一下——它会告诉你卡的 SM 数、每 SM 常驻线程数、
寄存器总量、共享内存上限、理论峰值带宽。写 kernel 时的很多判断（occupancy 够不够、
wave 数合不合理、寄存器压力多大）都要以这些数字为前提。
