# K3 板子的硬件能力与短板（SpaceMiT K3 Pico ITX）

把这块板子「能干什么、干不了什么」钉死，以后不用再问。

> **Android / Cuttlefish 那部分不在这里** → [`k3-android-cuttlefish.md`](../k3-android/k3-android.md)
> **访问方式（门户 / 隧道 / 终端）** → [`k3-portal.md`](../k3-portal/k3-portal.md)
> **arm64 应用跑不起来的解法** → [`arm64-on-riscv64.md`](../k3-arm64-on-riscv/arm64-on-riscv64.md)
> **端侧 AI 推理实测：能跑什么、多快、卡在哪** → [`k3-ondevice-ai.md`](../k3-ondevice-ai/k3-ondevice-ai.md)

**数据来源分两类，本文严格区分：**

| 标记 | 含义 |
| --- | --- |
| **【实测】** | 2026-10-15 直接从板子上抓的（`lscpu` / `free` / `lsblk` / `lspci` / `ip` / `/proc/device-tree`） |
| **【规格】** | 厂商公开资料（SpaceMiT K3 产品页 / datasheet / 发布会），**不一定在本板上引出** |

---

## 0. 结论速览

| 项 | 结论 |
| --- | --- |
| 是什么 | **SpaceMiT K3 Pico ITX** 开发板（芯片 = 进迭时空 K3） |
| 硬实力 | 16 核 / 16GB / 119GB NVMe / 10GbE + Wi-Fi 6 / GPU / 硬编解码 |
| **最大坑** | **「16 核」实际是 8 通用核（X100）+ 8 AI 核（A100）**，普通程序默认只用 8 个 → §1.2 |
| 通用性能 | 单核约「ARM A76 级」≈ 2018 年手机水平 → **它的定位是「能跑大模型」，不是「跑得快」** |
| 最稀缺的能力 | RVA23 合规 + 完整 KVM 虚拟化 → **全世界极少数能跑 riscv64 Android 的机器** |
| 适合 | 边缘 AI 盒子 / 机器人主控 / RISC-V 开发与移植平台 |
| 不适合 | 通用小服务器 / NAS / 软路由（同价位 x86 是碾压） |

---

## 1. 这块板子（【实测】2026-10-15）

### 1.1 实测清单

| 项 | 实测值 |
| --- | --- |
| 型号 | `SpacemiT K3 Pico ITX`（`/proc/device-tree/model`） |
| 内核 | `6.18.3-cf-k3-1.3.0`（**Cuttlefish 定制内核**，非出厂 Bianbu 内核） |
| CPU | **16 核在线**（`0-15`）：8× X100 + 8× A100 |
| 频率上限 | X100 **2150 MHz** / A100 **1800 MHz** ⚠️ 低于标称 2.4/2.0（见下注） |
| 内存 | **15 GiB**（无 swap / zram） |
| 存储 | `sda` **119.3 GB**：`sda1` 256M、`sda2` 256M→`/boot`、`sda3` 118.8G→`/` |
| 固件存储 | `mtdblock0-6`（SPI NOR，约 16MB，放 SPL/U-Boot） |
| PCIe | **3× SpaceMiT X100 PCIe Root Complex** |
| └ 挂载 1 | `0002:01:00.0` **Realtek RTL8127 10GbE 网卡** |
| └ 挂载 2 | `0004:01:00.0` **Realtek RTL8852BE Wi-Fi 6 (802.11ax) + BT** |
| 网口 | `end0`（板载 SoC GMAC，**UP**）、`enP2p1s0`（**10GbE，NO-CARRIER ← 没插线**）、`wlP4p1s0`（Wi-Fi 6，UP） |
| GPU | PowerVR：`/dev/dri/card0-2` + `renderD128` |
| 视频编解码 | Linlon VPU：`/dev/video-dec0`、`/dev/video0` |
| 其他 | `/dev/input`、`/dev/uinput` |

> 注：`lscpu` 报的 X100 2150MHz / A100 1800MHz 低于厂商标称的 2.4 / 2.0 GHz。
> 最可能是 **Cuttlefish 定制内核的调频策略不同**（换过内核，见 `k3-android-cuttlefish.md` §3.7），未深究。
> 要对比就把出厂 Bianbu 内核换回去测。

### 1.2 ⚠️ 「16 核」不是 16 核（最容易踩的坑）

X100 和 A100 **不是同构的**，这是 K3 最特别也最容易误判的地方：

| | **X100**（8 个） | **A100**（8 个） |
| --- | --- | --- |
| 定位 | 通用计算 | **AI / 向量专用** |
| 向量长度 VLEN | 256 bit | **1024 bit** |
| **Hypervisor 扩展（H）** | ✅ 有 | ❌ **没有** |
| 默认给谁用 | **普通进程** | 必须**显式绑定**才能用 |

**后果（都是实打实的）：**

1. **普通程序默认只跑在 8 个 X100 核上**。就算 X100 跑满，内核**也不会**把任务迁到 A100
   —— 因为 A100 的 ISA 是子集，跑通用指令会异常。
2. 所以 `nproc` 在 `pingji` 会话里是 **8**，不是 16（另有 cpuset 限制）。
3. **KVM 虚拟机也只能落在 X100 上** —— A100 没有 H 扩展。跑 Cuttlefish 时实际能用的核更少。

> **结论：把这块板子当「16 核」做容量规划，一定会失望；当「8 核通用 + 8 核 AI 加速器」才对。**

厂商自己的论文也承认这个异构调度是个工程难题（X100/A100 亲和性管理、跨核迁移受限），
正在往 Linux `cpuset` 机制上改。**这是 K3 架构的固有代价，不是配置问题。**

---

## 2. K3 SoC 的能力（【规格】芯片级，未必在本板引出）

### 2.1 CPU / AI

| 项 | 规格 |
| --- | --- |
| 计算核 | 8× X100（自研高性能核）+ 8× A100（自研 AI 核） |
| 通用算力 | 130 KDMIPS；大核 2.4GHz，"单核比肩 ARM A76" |
| AI 算力 | **60 TOPS**，数据类型 **BF16 / FP16 / FP8 / INT8 / INT4** |
| 大模型 | 官方称可流畅跑 **30B–80B** 参数模型 |
| 效率 | MAC 利用率 >95%，带宽利用率约 70% |
| ISA | **RVA23U64 + RVA23S64 完整合规**（RISC-V 2024 年定的 64 位基准档） |
| 可选扩展 | `Zvkng/Zvksg/Zvbc`（向量密码学）、`Zfh/Zvfh`、**BF16 系列** `Zfbfmin/Zvfbfmin/Zvfbfwma` |
| 中断控制器 | 自研 **APIC**（APLIC + ACLINT），支持 MSI |
| 周边 IP | AIA/APLIC（中断虚拟化）、IOMMU（内存虚拟化）、IOPMP（安全）、LPC/eSPI（对接 BMC） |

> 公司主线上的**新一代核 X200**：单核 **>16 分/GHz SPECint2006**、**3.3GHz @ 6nm**，面向云数据中心
> （**不是**本板上的 X100，别混淆）。
> **N200 NoC**：单 Chiplet 内最高 **640 GB/s** 访存，整系统 Cache 一致。

### 2.2 多媒体

| 项 | 规格 |
| --- | --- |
| 3D GPU | 支持 **Vulkan / OpenCL / OpenGL ES** |
| 硬编解码 | **4K@180fps 解码 / 4K@90fps 编码**（H.265 / H.264 / VP8 / VP9） |
| 显示输出 | **双路 3840×2160@60fps**（MIPI-DSI 8-lane 4.5Gbps/lane，或 DP/eDP） |
| 摄像头输入 | **4× MIPI-CSI，共 12 lane，最多 12 路** |
| 2D 加速 | 自研 **V2D** |

### 2.3 IO

| 项 | 规格 |
| --- | --- |
| PCIe | **Gen3 ×8**（8Gbps/lane），支持 **RC 和 EP 模式 + 热插拔** → 可插 NVMe / 网卡 / 甚至 AMD 显卡（`amdgpu/radeon`） |
| USB | 1× USB 3.0 Host、1× USB 3.0 DRD（Type-C）、1× USB 2.0 Host |
| 以太网 | **4× GMAC**（RGMII/RMII/MII），带 **TSN** 时间敏感网络 |
| 低速 IO | **6× SPI、2× eSPI、17× UART、10× CAN-FD、9× I²C、30× PWM** |
| 存储接口 | SPI Flash / eMMC 5.1 / **UFS 2.2** / SDIO 3.0 / NVMe over PCIe |

> 那串 **10× CAN-FD + 17× UART + 30× PWM** 是**车载 / 工业 / 机器人**级的 IO 数量。
> 这块板子是「Pico ITX 评估板」，只引出了网口 / PCIe / NVMe，**芯片能力远大于板上引出的部分**。

### 2.4 安全 / 可靠性

| 项 | 规格 |
| --- | --- |
| 安全 | 硬件加密引擎、TRNG、eFuse、**IOPMP**（RISC-V IO 内存保护） |
| 国密 | 支持 **SM2 / SM3 / SM4** |
| 功耗 / 温度 | TDP **15–25W**；工业级 **−40 ~ +85℃** |

---

## 3. 软件栈能干什么

| 能力 | 状态 |
| --- | --- |
| **虚拟化** | **KVM + VTVM**，eBPF 动态追踪（本板跑 Cuttlefish 就靠它） |
| **实时内核** | 有 **RT-Linux 专用 defconfig** → 可做硬实时控制 |
| **操作系统** | Bianbu 4.0.1（LXQt / 服务器版）；**全球首个原生适配 Ubuntu 26.04 LTS 的 RISC-V 平台**；上游 Linux 7.0 等已原生适配 |
| **AI 栈** | `spacemit-onnxruntime`、`llama.cpp-tools-spacemit`、Triton / Tilelang 算子编译 |
| **媒体栈** | FFmpeg 8.0 / GStreamer 1.28（**均带硬解硬编**）、PipeWire、MPP、Mesa 3D 24.01 |
| **桌面 / 应用** | Chromium 143、LibreOffice、VSCodium、VLC、Zed、Qt 5/6、GTK 3/4 |
| **开发工具** | gcc15、Python 3.14、OpenJDK、Node.js、OpenCV 4.14、OpenSSL 3.5 |
| **性能分析** | **Perfetto**（RISC-V 原生自编译）+ **rvtrace** 硬件指令级 trace |
| **Android** | Android 16（K1，BayLibre）/ **Android 17 riscv64**（Cuttlefish，本板） |
| **服务器侧** | Vital Stone 产品线已跑通「RISC-V + OpenHarmony + 本地大模型」全栈 |

---

## 4. 已知缺口（【实测】+ 文档）

| 缺什么 | 影响 | 出处 |
| --- | --- | --- |
| **没有 ARM 转译层** | 装不了任何带 arm64 `.so` 的 App（微信/QQ/抖音…） | → [`arm64-on-riscv64.md`](../k3-arm64-on-riscv/arm64-on-riscv64.md) |
| `VHOST_NET` 未开 | Cuttlefish 里的 **guest 上不了网**（只影响 guest） | 上游自己也没开 |
| `ANDROID_BINDER_IPC` 未开 | 容器化 Android（Waydroid/redroid）这条路走不通 | — |
| `CONFIG_KVM=m` | 开机要手动 `modprobe kvm`，否则没有 `/dev/kvm` | — |
| `/boot` 只有 256M | 加内核后余量约 90M，够但不宽裕 | — |
| **10GbE 网口没接线** | 网络能力被浪费（`enP2p1s0` NO-CARRIER） | §1.1 |
| 频率上限低于标称 | 可能是 CF 内核调频策略，未深究 | §1.1 注 |
| 生态年轻 | riscv64 Android 2026-06 才第一次真机跑通 | — |
| **厂商单点依赖** | 固件 / 驱动 / 内核全指着一家 | — |

---

## 5. 这条链子每层是谁做的

理解「谁贡献了什么」，才能知道缺口在哪、往哪使劲：

| 层 | 谁 | 做了什么 |
| --- | --- | --- |
| **Android 平台级 RISC-V 支持** | **Google / AOSP** | 把 riscv64 变成 AOSP 官方目标（`aosp_cf_riscv64_phone`）。`google/android-riscv64`：「所有 Android/riscv64 工作都直接在 AOSP 里做」 |
| **「在 K3 上真的跑起来」** | **Jun Sun（monkey-jsun）—— 一个人** | 伞项目 `aosp-cuttlefish-riscv64`：K3 定制内核、crosvm/u-boot 的 riscv64 移植、Cuttlefish 的 `dev/riscv64` 分支、host 容器 |
| **芯片 + 内核 + 虚拟化底座** | **SpacemiT（进迭时空）** | K3 硬件、Bianbu 内核基线、KVM/VTVM、GPU（PowerVR）/ VPU（Linlon）驱动 |
| **正式化 / 上游化** | **BayLibre**（+ Imagination） | Android 16 上 RISC-V、内核回移、mesa Vulkan、通用 HAL；**全 upstream、无 NDA、可重建** |

**厂商背景（为什么说它「野心大」）：**

- 成立 **2021-11**，总部杭州余杭，**50–200 人**；创始团队来自**平头哥、全志**
- 办公地：杭州 / 北京 / 上海 / 珠海 / 深圳 / 西安 / 英国
- 产品线三石：**Key Stone**（终端 K1/K3）、**Vital Stone**（服务器）、**Power Stone**（PMIC）+ **MUSE** 生态整机
- 节奏：**每年一代芯片**，每代 CPU 与 AI 算力提升 4–8 倍
- 量产成绩：**K1 出货 15 万颗**（国内 RISC-V 高算力芯片量产最多）
- K3 的四个「全球首颗」：RVA23 量产 / RVV 1024bit / FP8 原生推理 / 完整虚拟化
- 融资：A+ 轮（2024-12，数亿元）→ **B 轮（2026-01，>6 亿元 / >8600 万美元）**；股东含北京市 / 香港政府平台、中网投、农行 AIC、君联、联想创投、经纬
- 已在**北京、上海两大国家级人形机器人创新中心**完成技术验证（灵龙机器人用 K3 跑完半马）

---

## 6. 适合干什么 / 不适合干什么

| 方向 | 合适度 | 理由 |
| --- | --- | --- |
| **边缘 AI 推理盒** | ★★★★★ | 60 TOPS / 15–25W，能跑 30B 级量化模型 |
| **机器人 / 工业控制主控** | ★★★★★ | 10× CAN-FD、17× UART、30× PWM、**RT-Linux** |
| **RISC-V 软件移植 / 适配平台** | ★★★★★ | 真实 RVA23 硬件，不用 QEMU 软件模拟 |
| **Homelab / NAS** | ★★★☆☆ | 16GB + NVMe + **10GbE（还没用）**，能跑但不划算 |
| **软路由 / 防火墙** | ★★★☆☆ | 多网口 + TSN + 硬件加密，但生态不如 x86 |
| **通用开发机** | ★★☆☆☆ | 单核 A76 级，编译大项目会难受 |
| **跑国产 arm64 App** | ★☆☆☆☆ | **ABI 不通，装不上** → 见 `arm64-on-riscv64.md` |

> 提醒：**上面「合适度」是相对于这块板子的定位打的，不是绝对分数。**
> 单看通用性能，一块几百块的 x86 小主机就能把它比下去；它的价值全在「RISC-V + AI 算力 + IO」这个组合。

---

## 7. 怎么复现这些数据

板子的访问链路见 [`k3-portal.md`](../k3-portal/k3-portal.md)。一条命令抓全部：

```bash
JUMP=root@111.228.40.211
Q() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$JUMP" \
  "docker exec -i k3-ttyd ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
   -o BatchMode=yes -o IdentitiesOnly=yes -o LogLevel=ERROR \
   -i /app/ssh/id_ed25519 -p 2222 -l pingji 127.0.0.1 $(printf '%q' "$1")" </dev/null; }

Q 'cat /proc/device-tree/model; echo;
   echo ===CPU===;   lscpu | head -30;
   echo ===CORES===; grep -c processor /proc/cpuinfo;
   echo ===MEM===;   free -h;
   echo ===DISK===;  lsblk -o NAME,SIZE,TYPE,MOUNTPOINT;
   echo ===PCI===;   lspci;
   echo ===NET===;   ip -br link;
   echo ===DEV===;   ls /dev/dri /dev/video* 2>/dev/null;
   echo ===KERNEL===; uname -a'
```

单条查询（如看 GPU）：

```bash
Q 'ls -l /dev/dri'
```
