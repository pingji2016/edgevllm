# 在 K3 上跑端侧 AI 推理（实测）

把这块 RISC-V 板子当 **AI 推理平台**用：能跑什么、跑多快、卡在哪。

> **本文所有数字都是 2026-10-15 在 `pingji-k3` 上实测所得**，不是规格书抄的。
> 硬件能力 → [`k3-hardware.md`](../k3-hardware/k3-hardware.md)；Android → [`k3-android-cuttlefish.md`](../k3-android/k3-android.md)；arm64 转译 → [`arm64-on-riscv64.md`](../k3-arm64-on-riscv/arm64-on-riscv64.md)

---

## 0. 一句话结论

| | 结论 |
| --- | --- |
| ✅ **多模态推理能跑** | 2.7 GB 视觉语言模型输出正确描述；**小图整轮只要 25 秒** |
| ✅ **语音识别能跑** | Whisper base 达到 **≈1× 实时**（10.1 秒音频 → 10.35 秒） |
| 🔑 **第一优化手段是「降采样」** | 像素少 **37 倍** → 图像编码快 **50 倍**（237 s → 4.7 s） |
| ⚠️ **线程数要看场景** | Whisper：4 → 8 线程提速 **2.2 倍**；VLM：`-t 8` 与 `-t 16` **无差别** |
| ❌ **一半算力够不到** | 会话被锁在 **8 个 X100 核**；带 **IME 矩阵扩展的 8 个 A100 核**，**内核层面就不给用**（§6） |
| ❌ **硬件编解码是坏的** | Linlon VPU（`stcodec`）编解码**全部段错误**，只能退回软编软解 |

**一句话：能跑，而且小图场景已经可用；但一半硬件（A100 + IME）在 Linux 下根本够不到。**

---

## 1. 环境

| 项 | 值（实测） |
| --- | --- |
| 板子 | SpaceMiT K3 Pico ITX |
| 内核 | `6.18.3-cf-k3-1.3.0`（Cuttlefish 定制内核） |
| CPU | 16 核在线：「8× X100 @2.15GHz」+「8× A100 @1.8GHz」 |
| **本会话可用核** | **仅 `0-7`（X100）** ← 见 §6 |
| 内存 | 15 GiB（无 swap） |
| ISA 扩展 | `zve64d` / `zvfh` / `zvfbfmin` / `zvfbfwma` / `zvbb` / `zvkned` / `zvkg` … |
| 视觉长度 | whisper.cpp 自报 **`RVV_VLEN = 32`**（= 256 bit，X100 的 VLEN） |

---

## 2. 板上已装的 AI 工具（不用自己编）

| 工具 | 路径 | 说明 |
| --- | --- | --- |
| **`llama.cpp-tools-spacemit`** | `/usr/bin/llama-cli` 等 | **SpaceMiT 官方魔改版** llama.cpp（`version 10050 (5496cdc9c)`），含 IME / TCM / AICPU 优化 |
| **`whisper-cli`** | `/usr/bin/whisper-cli` | whisper.cpp（RVV 加速） |
| **Sherpa-ONNX** | `~/mage-vl/sherpa/sherpa-onnx-v1.13.8-linux-riscv64-spacemit-shared` | 带 spacemit 优化的 ONNX 语音框架 |
| `llama-mtmd-cli` | `~/mage-vl/llama.cpp/build/bin/` | 多模态 CLI（VLM 用这个，**自建 vanilla 版**） |

### `llama.cpp-tools-spacemit` 的启动横幅（含金量很高）

```
CPU_RISCV64_SPACEMIT: num_cores: 16, num_perfer_cores: 8, perfer_core_arch_id: a064,
                      exclude_main_thread: 0, use_ime1: 0, use_ime2: 1,
                      mem_backend: HPAGE, cpu_mask: ff00, aicpu_id_offset: 8
CPU_RISCV64_SPACEMIT: tcm is available, blk_size: 393216, blk_num: 8, is_fake_tcm: 0
CPU_RISCV64_SPACEMIT: alloc_chunk: open(/dev/tcm_sync_mem) failed, errno=2
CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap
```

**逐条翻译：**

- `cpu_mask: ff00` → 它想把重计算固定在 **CPU 8–15**（A100 核）
- `use_ime2: 1` → **启用 IME v2 矩阵扩展**（AI 加速的关键）
- `aicpu_id_offset: 8` → 「AI CPU」从 8 号开始编号
- `TCM`（紧耦合内存）可用，8 块 × 384 KiB ≈ 3 MiB
- ⚠️ `/dev/tcm_sync_mem` **不存在** → 初始化屏障只能退回堆内存（性能损失）

> **SpaceMiT 的 AI 加速 = A100 核 + IME + TCM 三件套。**
> 而 §6 会看到，**第一件在 Linux 下就拿不到。**

---

## 3. 实测 A：多模态推理（VLM 看图说话）

### 3.1 模型

| 文件 | 大小 | 作用 |
| --- | --- | --- |
| `models/mage-vl-backbone-Q4_K_M.gguf` | **2.7 GB** | VLM 主干（文本 + 视觉融合） |
| `models/mage-vit-mmproj-Q8_0.gguf` | 352 MB | ViT 视觉投影层 |
| `models/dog.jpg` | 0.5 MB | 测试图，**1024 × 2048** |
| `models/dog_small.png` | 0.1 MB | 测试图，**168 × 336** |

### 3.2 命令（项目自带 `run.sh`）

```bash
cd ~/mage-vl
sh run.sh models/dog.jpg "Describe this image in one sentence."
```

`run.sh` 内部实际执行：

```bash
export LD_LIBRARY_PATH=$PWD/llama.cpp/build/bin
./llama.cpp/build/bin/llama-mtmd-cli \
  -m models/mage-vl-backbone-Q4_K_M.gguf \
  --mmproj models/mage-vit-mmproj-Q8_0.gguf \
  --image models/dog.jpg \
  -p "Describe this image in one sentence." \
  -n 128 -t 16 -c 4096
```

### 3.3 结果 ✅

**大图（1024×2048）：**

> A dog is sitting on a carpet in front of a wall with a rug hanging on it.

**小图（168×336）：**

> The image features a dog sitting on a patterned rug. The dog has a mix of white, black, and brown fur, with a predominantly white chest and …

两者都**正确**。

### 3.4 耗时分解（大图）

| 阶段 | 耗时 |
| --- | --- |
| 模型加载（mmap） | < 1 s（真正的页错误在计算时发生） |
| **图像编码（ViT）** | **237,474 ms ≈ 3 分 57 秒** ← 绝对瓶颈 |
| 文本生成 | 剩余（128 token 上限） |
| **整轮 wall time** | **约 6–7 分钟** |

### 3.5 分析（两个对照实验）

#### 实验 1 —— 线程数：`-t 8` vs `-t 16`

同一张小图 `models/dog_small.png`（168×336）各跑一次：

| 线程 | 整轮 wall | 图像编码 |
| --- | --- | --- |
| `-t 8` | **25 s** | 4,743 ms |
| `-t 16` | **26 s** | 4,774 ms |

差距 **4%** —— **在 8 核上把 `-t` 开成 16 不会更快**（llama.cpp 实际被可用核数限制，超订没有收益）。
`run.sh` 里写死 `-t 16`，属于无害但也没用的配置。

#### 实验 2 —— 图像尺寸：**这才是决定性的变量**

| 图片 | 分辨率 | 图像编码 | 相对 |
| --- | --- | --- | --- |
| `dog.jpg` | **1024 × 2048** | **237,474 ms** | 1× |
| `dog_small.png` | **168 × 336** | **4,743 ms** | **1/50** |

像素数少 **37 倍**，编码时间快 **50 倍**。

> ### 🔑 结论一：端侧 VLM 的图像编码耗时**基本正比于图像 token 数**（≈ 像素数）。
> **把输入降采样，是端侧多模态部署的第一优化手段** —— 远比其他调参有效。
>
> `1024×2048` 是张 200 万像素的超大图，对端侧是**异常输入**。
> 换成 168×336，**整轮只要 25 秒 —— 这块板子跑端侧 VLM 是完全可用的。**

> ### 结论二：另外一半的性能被锁在门外
> 图像编码是**向量/矩阵密集**计算 —— 正是 **A100 核 + IME 矩阵扩展**该发力的地方。
> 但实际只有 8 个 **X100** 核在算，而且它们**没有 IME**（§6）。

---

## 4. 实测 B：语音识别（Whisper）

### 4.1 配置对照（同一段 10.1 秒音频 `asr/0.wav`）

| 模型 | 线程 | 总耗时 | 相对实时 | encode | batchd |
| --- | --- | --- | --- | --- | --- |
| `ggml-base-q5_1` (60 MB) | 4 | **22.90 s** | 0.44× | 15.53 s | 6.83 s |
| `ggml-base-q5_1` (60 MB) | **8** | **10.35 s** | **0.98×** | 7.54 s | 2.40 s |
| `ggml-small-q5_1` (190 MB) | 8 | 34.50 s | 0.29× | 27.43 s | 6.40 s |

### 4.2 结论

1. **线程数 4 → 8，提速 2.2 倍**（22.9 s → 10.35 s）。这本身就是一条可复现的调优结论。
   （whisper.cpp 默认只用 4 线程，**不改就白白损失一半算力**。）
2. **base 模型 ≈ 1× 实时**：10.1 秒音频，10.35 秒出结果。**实时语音转写在这块板子上可行。**
3. 瓶颈同样是 **encode 阶段（声学编码器）**，占 73% —— 又一个矩阵计算密集段。

> ⚠️ 注意 **§3.5 与 §4.2 的对照**：VLM 加线程没用，Whisper 加线程提速 2.2 倍。
> 原因很可能是 **Whisper 默认 4 线程没吃满 8 核，而 VLM 默认 `-t 16` 已经超了**。
> **调优要先看「当前是否吃满核」，再决定加不加线程** —— 而不是无脑加。

### 4.3 转写质量

音频本身是**中英混说**，small 模型输出：

```
[00:00:00.000 --> 00:00:02.980]  昨天是 Monday
[00:00:02.980 --> 00:00:05.980]  Today is 禮拜二
[00:00:05.980 --> 00:00:09.980]  Today after tomorrow 是星期三
```

**正确。** 注意：默认语言是 `en`，**必须显式加 `-l zh`**，否则中文音频会被按英文转写成乱码。

### 4.4 命令

```bash
cd ~/mage-vl/asr
whisper-cli -m ggml-base-q5_1.bin -f 0.wav -l zh -t 8
```

---

## 5. 实测 C：视频编解码（软 vs 硬）

板上 `logs/codecbench.log`（1080p30 / 5 秒 / 150 帧）：

| 测试 | 结果 |
| --- | --- |
| 软编 H264（libx264 veryfast） | ✅ 5.413 s |
| 软编 H265（libx265 ultrafast） | ✅ 10.923 s |
| 软解 H264 | ✅ 0.577 s |
| 软解 H265 | ✅ 0.738 s |
| **硬编 H264（`h264_stcodec`）** | ❌ **段错误** + `Use pixel format (0)` |
| **硬编 H265（`hevc_stcodec`）** | ❌ **段错误** |
| **硬解 H264（`h264_v4l2m2m`）** | ❌ `frame=0` / 输出文件权限错误 |
| **硬解 H265（`hevc_stcodec`）** | ❌ **段错误** |

**结论：Linlon VPU 的硬件编解码通路没打通，只有软件编解码可用。**

> 这是**又一个明确、可量化的缺口** —— 而且正好落在「音视频 × 端侧」这个交叉点上。
> 修好后，1080p 编码从 5.4 秒（软）降到毫秒级不是幻想。

---

## 6. 关键发现：A100 核（8–15）在 Linux 下不可用

**这是本次实测最重要、也查得最透的一条。**

### 6.1 现象

```bash
$ grep Cpus_allowed_list /proc/self/status
Cpus_allowed_list:	0-7          # ← 只有 X100

$ taskset -c 0-7  true   → OK
$ taskset -c 0-8  true   → EINVAL    # ← 边界精确卡在 cpu7 / cpu8
$ taskset -c 7-8  true   → EINVAL
$ taskset -c 8    true   → EINVAL
$ taskset -c 0-15 true   → EINVAL    # ← 连"放宽"都不行

$ python3 -c "import os; os.sched_setaffinity(0,{8})"
OSError: [Errno 22] Invalid argument       # ← 内核级限制，不是 taskset 的锅
```

### 6.2 逐条排除（全部实测）

| 怀疑对象 | 检查结果 | 判定 |
| --- | --- | --- |
| **权限**（普通用户受限） | `sudo taskset -c 8` → **同样 EINVAL** | ❌ 不是 |
| **cgroup v1 cpuset** | `mount` 只有 `cgroup2`，**没有 v1** | ❌ 不是 |
| **cgroup v2 cpuset** | 根启用了（`cpuset cpu io memory pids misc`），但 `user.slice` 的 `subtree_control` **没有 cpuset**；`cpuset.cpus.effective` = `0-15` | ❌ 不是 |
| **systemd `CPUAffinity`** | `system.conf` 里是注释掉的；`init.scope` / `system.slice` / `sshd` / `user@1000.service` 全为空 | ❌ 不是 |
| **`isolcpus`** | `/sys/devices/system/cpu/isolated` **为空** | ❌ 不是 |
| **`/proc/cmdline`** | 只有常规参数（`earlycon=sbi console=ttyS0,115200 …`），**无 CPU 屏蔽** | ❌ 不是 |
| **`~/.bashrc` / profile.d 里的 `taskset`** | 无 | ❌ 不是 |
| **CPU 未上线** | `/sys/devices/system/cpu/online` = `0-15`，`cpu8/online` = `1` | ❌ 不是 |
| **设备树禁用** | `/proc/device-tree/cpus/cpu@8/status` = **`okay`**，`compatible` = `spacemit,a100riscv` | ❌ 不是 |
| **SMP 启动失败** | dmesg：**`smp: Brought up 1 node, 16 CPUs`** | ❌ 不是 |
| **热插拔坏了** | `echo 0/1 > cpu8/online` **都成功**（rc=0），但之后 `taskset -c 8` **仍然 EINVAL** | ❌ 不是 |
| **内核编译开关** | `config-6.18.3-cf-k3-1.3.0` 里 **没有任何 AICPU / A100 开关**；`CONFIG_SPACEMIT_TCM=y`、`CONFIG_HOTPLUG_CPU=y` | ❌ 不是 |

### 6.3 结论

**CPU 8–15 是「online 但不在 `cpu_active_mask`」** —— 内核主动把这 8 个 A100 核排除在调度域外。
`__sched_setaffinity()` 里对 `cpu_active_mask` 的检查，正是我们观察到的 EINVAL 的来源。

旁证（都在 dmesg / sysfs 里）：

| 证据 | 含义 |
| --- | --- |
| `OF: reserved mem: rcpu0_pwr / rcpu0_heap / **rcpu0_rsc_table**` | OpenAMP 风格的**远程核资源表** —— A100 是被"远程核"框架管理的 |
| MTD 分区里有 **`esos`**（1 MB） | ESOS（Energy Service OS）固件，跑在独立核上 |
| `/sys/class/` 里有 **`ai_dma` / `aidma_list`** | 厂商 AI DMA 接口 —— A100 的**数据通路**是 DMA，不是普通线程 |
| `/sys/module/` 里有 **`reset_spacemit`** | 厂商专用模块 |

> **也就是说：Linux 把 A100 当作「远程/卸载核」交给厂商运行时管，而不是当作普通 CPU 核。**
> 想用上 IME，得走 **TCM + ai_dma + 厂商运行时** 那条路 ——
> 这解释了为什么 `llama.cpp-tools-spacemit` 一上来就要 `/dev/tcm_sync_mem` 和 `cpu_mask: ff00`：
> **它本来就是为「另一个核簇」设计的。**

### 6.4 还有没有救

| 方案 | 可行性 | 风险 |
| --- | --- | --- |
| 从用户态打开（`taskset` / cgroup / systemd） | ❌ **已证明不可行** | — |
| **换回出厂 Bianbu 内核**试一次 | ⚠️ 未知（两个内核的相关配置看起来一样，但行为可能不同） | **板上是远程的**，改 `/boot/env_k3.txt` 的 `knl_name` / `dtb_dir` 后重启，**起不来就失联** → 需要人在现场 |
| 补齐 `/dev/tcm_sync_mem`（TCM 同步内存节点） | ⚠️ 待查（`CONFIG_SPACEMIT_TCM=y` 已内建，但节点没建出来） | 低 |
| 走 **`ai_dma` + 厂商运行时** 的正路 | 🔍 需要 SpaceMiT 的文档 / SDK | 低 |
| **直接问 SpaceMiT**（论坛 / issue / 社区） | ✅ 最省事，而且是正确姿势 | 无 |

> 💡 **换内核这件事现在就能做，但会失联风险。**
> `/boot/env_k3.txt` 里就是三行：
> ```
> knl_name=vmlinuz-6.18.3-cf-k3-1.3.0
> ramdisk_name=initrd.img-6.18.3-cf-k3-1.3.0
> dtb_dir=spacemit/6.18.3-cf-k3-1.3.0
> ```
> 而 `/boot/` 里**出厂内核 `vmlinuz-6.18.3-generic` 也在**。改回去重启就能对比 —— 但必须**人在现场**再动。

---

## 7. 结论与下一步

| 优先级 | 做什么 | 预期收益 | 风险 |
| --- | --- | --- | --- |
| **P0** | **输入降采样 + 做一次分辨率扫描**（168×336 → 256 → 512 → 1024），画出「分辨率 vs 延迟」曲线 | **最直接的可用性收益**（已有 50× 的实测证据） | 无 |
| **P1** | **问 SpaceMiT**：A100 核在 Linux 下怎么用？`/dev/tcm_sync_mem` 怎么出？ | 打开另一半硬件 | 无 |
| **P2** | 补齐 `/dev/tcm_sync_mem`，跑一次 `llama.cpp-tools-spacemit` | 吃到官方 TCM/HPAGE 优化 | 低 |
| **P3** | 修 **Linlon VPU 硬编硬解**（`stcodec` 段错误） | 视频编解码从秒级 → 毫秒级 | 低 |
| **P4** | **换回出厂内核**验证 A100 是否可用 | 一次性定性 | ⚠️ **高（可能失联，需人到现场）** |
| **P5** | Whisper：beam 数 / 量化档位扫描；VLM：mmproj 量化（Q8_0 → Q4） | 各自再挖一档 | 无 |

> **这台机器的价值恰恰在于「能跑，但没跑满」。**
> 上面每一条都是**可测量、可复现、有明确验收标准**的工程问题。

---

## 8. 复现命令

```bash
JUMP=root@111.228.40.211
Q() { ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR "$JUMP" \
  "docker exec -i k3-ttyd ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
   -o BatchMode=yes -o IdentitiesOnly=yes -o LogLevel=ERROR \
   -i /app/ssh/id_ed25519 -p 2222 -l pingji 127.0.0.1 $(printf '%q' "$1")" </dev/null; }

# 1) 可用核
Q 'grep Cpus_allowed_list /proc/self/status; nproc; nproc --all'

# 2) SpaceMiT 版 llama.cpp 的加速配置
Q 'llama-cli --version'

# 3) VLM 看图说话
#    大图（1024×2048）约 6–7 分钟
Q 'cd ~/mage-vl && nohup sh ./run.sh models/dog.jpg "Describe this image in one sentence." > /tmp/vlm.log 2>&1 & echo started'
#    小图（168×336）约 25 秒
Q 'cd ~/mage-vl && sh ./run.sh models/dog_small.png "Describe this image."'

# 4) 语音识别（约 10 秒）
Q 'cd ~/mage-vl/asr && whisper-cli -m ggml-base-q5_1.bin -f 0.wav -l zh -t 8'

# 5) 视频编解码基准
Q 'cat ~/mage-vl/logs/codecbench.log'

# 6) A100 核不可用的验证
Q 'taskset -c 8 true; echo rc=$?; grep Cpus_allowed_list /proc/1/status'
```

---

## 附：本次实测的原始数据

| 项目 | 值 |
| --- | --- |
| VLM 图像编码（1024×2048） | `mtmd batch encoding done in 237474 ms` |
| VLM 图像编码（168×336） | `mtmd batch encoding done in 4743 ms` / `4774 ms` |
| VLM 线程对照 | `-t 8` wall **25 s** ／ `-t 16` wall **26 s** |
| VLM 输出（大图） | `A dog is sitting on a carpet in front of a wall with a rug hanging on it.` |
| VLM 输出（小图） | `The image features a dog sitting on a patterned rug. The dog has a mix of white, black, and brown fur, …` |
| Whisper base / 4 线程 | `total time = 22896.50 ms`（10.1 s 音频） |
| Whisper base / 8 线程 | `total time = 10354.66 ms` |
| Whisper small / 8 线程 | `total time = 34501.75 ms` |
| whisper 自报 | `CPU : RISCV_V = 1 \| RVV_VLEN = 32 \| OPENMP = 1 \| REPACK = 1` |
| llama.cpp-spacemit | `version: 10050 (5496cdc9c)`，`built with GNU 14.3.0 for Linux riscv64` |
| 可用 CPU | `Cpus_allowed_list: 0-7`；`nproc` = 8 ／ `nproc --all` = 16 |
| CPU 拓扑 | `cpu0 cluster_cpus_list = 0-3`；`cpu8 cluster_cpus_list = 8-11` |
| 缺失设备 | `/dev/tcm_sync_mem`（errno=2）；`/dev/tcm` 存在 |
| 内核配置 | `CONFIG_SPACEMIT_TCM=y`、`CONFIG_HOTPLUG_CPU=y`、无 AICPU 开关 |
