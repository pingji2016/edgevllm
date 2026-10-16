# 在 K3 上跑 Android（Cuttlefish on SpaceMiT K3）

把 `pingji-k3`（SpacemiT K3 / RISC-V）变成一台能跑 Android 的机器，浏览器里看屏幕、`adb` 连上去。
**路线是 Cuttlefish + crosvm + KVM 跑 riscv64 的 AOSP guest，不是 Google 官方那个 AVD 模拟器。**

> 本文档是**设计 + 实施手册**。落地脚本在 `deploy/k3/`。现状调查见 [`k3-portal.md`](../k3-portal/k3-portal.md)（那篇讲门户/隧道/终端，这篇讲 Android）。
>
> **本目录自成一个工作根**：下文所有 `bash deploy/k3/…`、`bash deploy/aosp/…` 都相对本文件所在目录
> （`docs/edge/k3/k3-android/`）执行，照抄即可 —— 脚本就在旁边的 `deploy/` 下。

---

## 0. 结论速览

| 项 | 结论 |
| --- | --- |
| 可行吗 | **可行**，RISC-V 社区已有的路径，且在**同一款板子**上被验证过 |
| 依据 | [`monkey-jsun/aosp-cuttlefish-riscv64`](https://github.com/monkey-jsun/aosp-cuttlefish-riscv64)，最新 **v1.3.1（2026-09-26）** |
| 已做到什么程度 | AOSP **Android 17** riscv64 guest、**PowerVR GPU 加速**渲染、**Linlon VPU 硬编 H.264**、浏览器 **WebRTC** 看屏幕、`adb` 连接、预装 F-Droid |
| 板子硬件条件 | KVM / RVA23 ISA / vhost-vsock / 内存 / 磁盘**都够** |
| 板子缺什么 | ① Docker 没装 → **已装（29.1.3）** ② 内核要换项目定制的 `6.18.3-cf-k3-*` → **已换** ③ `CONFIG_VHOST_NET`：**上游自己的内核也没开**（只影响 guest 联网，见 §3.7） |
| 网络 | 板子到 GitHub / ghcr 的 TLS 会被**中途断掉**（`git clone` 报 `GnuTLS recv error`）→ 已在板子上装**本机 Xray 代理**（见 §6） |
| 最大限制 | ① **不能跑 Google 官方镜像**（只有 x86_64/arm64）；② **app 生态**：带 arm64 `.so` 的应用**根本装不上**（实测微信 8.0.79 / QQ 9.2.65 均报 `INSTALL_FAILED_NO_MATCHING_ABIS`），只有纯 Java 应用和为 riscv64 编译的应用能跑 → 见 §5.4 |
| 若目标是 **ROM 定制开发** | **K3 只能当"测试机"，不能当"编译机"**（AOSP 不支持 riscv64 作为构建主机）→ 见 §5 |
| 主要风险 | 换内核要重启板子（会短暂中断门户/心跳）；上游是**单人维护的实验项目**（1 star） |

**当前进度**（2026-10-14）：**阶段 0 / 1 / 1.5 / 2 / 3 / 4 已完成** —— Docker `29.1.3`、内核 `6.18.3-cf-k3-1.3.0`、`/dev/kvm` 可用、板载 Xray 代理通、2.09GB 宿主镜像已拉、`cf-data` 1.8G 已预取，**Android 17（riscv64）已经跑起来**：`VIRTUAL_DEVICE_BOOT_COMPLETED`、`adb` 通、抓屏是真实锁屏界面；对外访问经 111 正向转发已开（`127.0.0.1:18443/16520`）。剩下的：GPU 加速（在刷 rutabaga 错，但屏幕能渲染）、以及 §5 的编译机。应用兼容性的实测结论见 **§5.4**。

---

## 1. 为什么是 Cuttlefish（三条路线的判定）

"Android 模拟器"有三种完全不同的东西，RISC-V 上只有第三条通。

| 路线 | K3 上 | 原因 |
| --- | --- | --- |
| **① Google 官方 Android Emulator（SDK 里的 AVD）** | ❌ | 官方系统镜像只有 **x86_64 / arm64**，没有 riscv64；`emulator` 二进制也只有 x86_64/aarch64 host 版。在 RISC-V 上跑 arm64 镜像 = **QEMU TCG 纯软件翻译**（连 MMU 都用软件模拟），慢到不可用 |
| **② Waydroid / redroid（容器里跑 Android）** | ❌ | 没有 riscv64 系统镜像；且容器方案依赖内核 binder，当前内核 `CONFIG_ANDROID_BINDER_IPC is not set` |
| **③ Cuttlefish + crosvm + KVM** | ✅ | AOSP 官方的"虚拟设备"，用 **crosvm（Rust 写的 VMM）走 KVM 硬件虚拟化**跑 riscv64 guest。K3 的 ISA 刚好满足 AOSP 的 RVA23 要求，**没有指令翻译开销** |

> 注意：这不是"模拟 Android"，是**虚拟化 Android**。guest 和 host 同为 riscv64，靠 KVM 直通执行；GPU 走 gfxstream 转发到 K3 的 PowerVR。

---

## 2. 目标架构

```
你的浏览器 / adb                                ← 8443 WebRTC（看屏幕）
    │  https://127.0.0.1:8443/  /  adb connect 127.0.0.1:6520
    ▼
【本机 Mac】ssh -N -L 8443:127.0.0.1:18443 -L 6520:127.0.0.1:16520 root@111
    │
    ▼
【111.228.40.211】127.0.0.1:18443 / 127.0.0.1:16520
    │   ← k3-android-fwd.service（111 上）用 ssh -L 维持，经 k3-ttyd 容器
    ▼
【111.228.40.211】k3-ttyd 容器（host 网络）
    │   docker exec k3-ttyd ssh -N -L ... -p 2222 -l pingji 127.0.0.1
    ▼
【pingji-k3】192.168.110.194
    │
    ├── k3-tunnel.service     ssh -NT -R 127.0.0.1:2222 -> 板子:22   （已有，不动）
    ├── k3-heartbeat.timer    每 30s 上报状态给门户                   （已有，不动）
    ├── xray-k3.service       板子本机代理 127.0.0.1:10808            （新增，见 §6）
    │
    └── docker: cf-host-k3
          └── crosvm --(KVM)--> AOSP guest (riscv64, Android 17)
                ├── gfxstream ──> PowerVR GPU (/dev/dri/renderD128)
                └── V4L2 ──────> Linlon VPU（H.264 编码）
```

**为什么不是"给现有反隧道追加 `-R`"**（原计划，实测被堵）：

板子在内网，111 是唯一出口，所以原计划复用 `k3-tunnel.service` 追加两条 `-R`。但 111 的
`/etc/ssh/sshd_config.d/99-k3.conf` 里写着：

```
PermitListen 127.0.0.1:2222
```

`PermitListen` 是**白名单**语义：sshd **只允许客户端绑 2222**，新增 `-R` 会被直接拒绝
（实测 `Error: remote port forwarding failed for listen port 18443`）。而 `k3-tunnel` 带
`ExitOnForwardFailure=yes` —— **硬改会把 2222 一起打挂**，而 2222 是进板子的**唯一通道**
（门户里的终端也走它），打挂了远程连不回去。

所以改用**完全不动板子**的方案：在 111 上做 `ssh -L` 正向转发。`k3-ttyd` 容器是
**host 网络**，容器里的 `127.0.0.1` 就是 111 的 `127.0.0.1`；转发终点由板子的 sshd 在
**板子本机视角**解析（`127.0.0.1:8443/6520`），效果与反隧道等价。**风险为 0**：不改板子、不改 111 的 sshd。

> `tunnel-add` 子命令仍保留，但它现在会先在**预检**阶段被拦下（不会碰 drop-in）。
> 真要用反隧道，就得先往 111 的 `99-k3.conf` 里加 `PermitListen 127.0.0.1:18443` / `:16520`
> 再 `systemctl reload ssh` —— 但没必要，`expose` 更安全。

---

## 3. 板子现状实测（基线，2026-10-14）

### 3.1 系统

| 项 | 值 |
| --- | --- |
| 主机名 | `pingji-k3` |
| 系统 | Bianbu 4.0.1（`ID=bianbu`，Resolute Raccoon） |
| 内核 | `6.18.3-generic` `#1.0.1.4`，`riscv64` |
| 板子 IP | `192.168.110.194/24`（WiFi `wlP4p1s0`，出口网卡同） |
| 登录 | `pingji`（在 `sudo` 组，**有 sudo 权限但要密码**，非交互会话无法提权） |

### 3.2 硬件

| 项 | 值 | 判定 |
| --- | --- | --- |
| CPU | **16 核在线**（`cpu0-7` = X100 @2.4GHz，`cpu8-15` = A100 @2.0GHz） | ✅ 远超"≥2 核"要求 |
| CPU（`pingji` 会话视角） | `nproc` = **8**（会话 cpuset 被限，起容器时注意亲和性） | ⚠️ 记一笔 |
| 内存 | 15.9 GiB，无 swap / zram | ✅ 满足"≥8 GB" |
| 磁盘 | `/dev/sda3` → `/` 117G，**可用 90G** | ✅ 满足"≥32 GB" |
| `/boot` | 224M，**可用 148M** | ⚠️ 加内核后余量约 90M，够但不宽裕 |
| GPU | `/dev/dri/card0-2` + `renderD128`（PowerVR） | ✅ 支持 gfxstream 加速 |

### 3.3 虚拟化能力（关键）

| 项 | 值 | 判定 |
| --- | --- | --- |
| H 扩展（Hypervisor） | ISA 含 `rv64imafdcvh_...` → 有 `h` | ✅ |
| `/dev/kvm` | 存在（`crw-rw---- root:kvm 10,232`） | ✅ |
| 内核 KVM 配置 | `CONFIG_KVM=y`、`CONFIG_KVM_RISCV_VTVM=y` | ✅ |
| RVA23 baseline | `zicbom/zicboz/zicond/zihintntl/zimop/zaamo/zalrsc/zawrs/zfa/zfh/(zca|zcb|zcd)/zba/zbb/zbs` + 完整 `zv*` 向量 + `smaia/ssaia/sscofpmf/sstc/svinval/svnapot/svpbmt` | ✅ 基本齐 |
| `VHOST_VSOCK` | `CONFIG_VHOST_VSOCK=m`，`/dev/vhost-vsock` 在 | ✅ |
| `VHOST_NET` | **`# CONFIG_VHOST_NET is not set`** | ❌ **缺** |
| `ANDROID_BINDER_IPC` | `# CONFIG_ANDROID_BINDER_IPC is not set` | ❌（只用 Cuttlefish 的话不需要） |
| `CONFIG_UDMABUF` | **`# CONFIG_UDMABUF is not set`** | ❌（GPU 加速 gfxstream 要） |
| `CONFIG_VIDEO_LINLON_K3` | `=y` | ✅（硬编 H.264 的驱动，Bianbu 已有） |

### 3.4 工具链

| 工具 | 状态 |
| --- | --- |
| `git` | ✅ `/usr/bin/git` |
| `curl` / `wget` / `python3` / `rsync` | ✅ |
| **`docker`** | ❌ **未安装**（`docker: 未找到命令`） |
| `adb` | ❌（系统里只有 `adbd`/`android-lib*` 这些库包） |
| `qemu-system-*` | ❌（不需要，Cuttlefish 用 crosvm） |

### 3.5 网络出口（实测）

| 目标 | 结果 | 含义 |
| --- | --- | --- |
| `https://ghcr.io/v2/` | **401**（0.64s） | ✅ **可达**（401 = 正常要鉴权）→ 预构建镜像能拉 |
| `https://registry-1.docker.io/v2/` | **超时**（8s 无响应） | ❌ Docker Hub 被墙 → `docker.io` 官方镜像要走镜像源 |
| `https://mirrors.aliyun.com/` | 301（0.09s） | ✅ apt 源可用 |
| `https://github.com/` | 200（3.7s） | ✅ **但 HTTP 能通不等于能用** ↓ |

> ⚠️ **实测补充（重要）**：`curl https://github.com/` 返回 200，但 **`git clone https://github.com/...` 会被中途掐断**：
> ```
> fatal: 无法访问 'https://github.com/monkey-jsun/cuttlefish-host-container/'：
> GnuTLS recv error (-110): The TLS connection was non-properly terminated.
> ```
> 也就是说板子的国际线路是**随机可用**的，靠它拉 2GB 镜像必然出事。**结论：先在板子上加一层本机代理**（见 §6），
> 之后 `git clone`、`docker pull`、AOSP 镜像下载全部走代理。
>
> 走代理实测（2026-10-14）：`github` 200 / 2.0s，`ghcr.io` 401 / 1.0s，出口 IP `43.155.215.x`，`git clone` ✅。

### 3.6 现有反隧道（**已被证明不能复用** —— 见 §2 的说明）

`/etc/systemd/system/k3-tunnel.service`：

```ini
[Service]
User=pingji
ExecStart=/usr/bin/ssh -NT \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o TCPKeepAlive=yes -o ConnectTimeout=10 -o BatchMode=yes -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/home/pingji/.ssh/known_hosts \
  -i /home/pingji/.ssh/k3_tunnel_ed25519 \
  -l root \
  -R 127.0.0.1:2222:127.0.0.1:22 \
  111.228.40.211
Restart=always
RestartSec=10
```

> ⚠️ `ExitOnForwardFailure=yes`：**任何一条 `-R` 建立失败，整条隧道都会退出并重启**。
> 所以新增 `-R` 之前，必须确认 111 上目标端口没被占用，否则会**连带把 K3 门户打挂**。

> 🚫 **更硬的墙：111 的 sshd 不允许你绑别的端口。** `/etc/ssh/sshd_config.d/99-k3.conf` 里是
> `PermitListen 127.0.0.1:2222` —— 白名单语义，**只有 2222 能绑**。实测加 `-R 127.0.0.1:18443:...`
> 立刻报 `Error: remote port forwarding failed for listen port 18443`。
> 因为上面那条 `ExitOnForwardFailure=yes`，**真去改 drop-in 会把 2222 一起拖死**，而 2222
> 是进板子的唯一通道（门户终端也走它）→ 远程救不回来。
> **结论：对外访问不要走反隧道，用 `deploy.sh expose`（111 上做 `ssh -L`），见 §2。**

配套：`k3-heartbeat.service` + `k3-heartbeat.timer`（30s 一次，token 在 `/etc/k3-heartbeat.token`）。

### 3.7 换内核之后的变化（2026-10-14 实测）

装了上游 `linux-image-6.18.3-cf-k3-1.3.0` 并重启后（`/boot/env_k3.txt` 由 deb 的 postinst 自动改好，旧内核保留可回滚）：

| 项 | 换之前 | 换之后 | 说明 |
| --- | --- | --- | --- |
| 内核 | `6.18.3-generic` | `6.18.3-cf-k3-1.3.0` | ✅ |
| `/dev/kvm` | ✅ 存在 | ⚠️ **消失** → `modprobe kvm` 后回来 | **坑**：新内核把 KVM 编成 **模块**（`CONFIG_KVM=m`，上游有意为之），开机不自动加载 |
| `CONFIG_KVM_RISCV_VTVM` | - | `=y` | AIA/IMSIC 相关，起 crosvm 要 |
| `CONFIG_UDMABUF` | ❌ is not set | ✅ **`=y`** | GPU 加速（gfxstream）的前提，已具备 |
| `CONFIG_VHOST_VSOCK` | `=m` | `=m` | adb over vsock |
| `CONFIG_VIDEO_LINLON_K3` | `=y` | `=y` | 硬编 H.264 |
| `CONFIG_VHOST_NET` | ❌ is not set | ❌ **仍 is not set** | 见下 |
| `/boot` 余量 | 148M | ~90M | 双内核并存 |
| `docker` | 未安装 | `29.1.3-0ubuntu4` | apt 装的 `docker.io` |

> **关于 `VHOST_NET`**：扒开上游内核 deb 里的 `/boot/config-6.18.3-cf-k3-1.3.0` 看过，
> **上游自己也没开**（同一份 deb 里 `UDMABUF=y` 却是开的）。所以上游 README 里
> "需要 `CONFIG_VHOST_NET`"这条**大概率是过期描述**（v1.1 的发布说明里写了
> "drop host-mode network for less host pollution"）。
> 影响面：**只影响 guest 自己联网**（Android 里上网）。看屏幕（WebRTC 在宿主侧）、`adb`（走 vsock）都不受影响。
> **不要为了这一项去重编内核**，先按现状跑，真发现 guest 需要网络再处理。

> **必须在 `verify-host` 里确认 `/dev/kvm` 回来了**：新内核开机后 `/dev/kvm` 是**没有的**，
> 跑一次 `deploy.sh mods`（写 `/etc/modules-load.d/k3-android.conf` + `modprobe kvm vhost_vsock`）即可，
> 而且**重启后依然有效**。这是阶段 1 最容易漏掉的一步。

---

## 4. 上游方案组件清单

| 组件 | 来源 | 大小 | 备注 |
| --- | --- | --- | --- |
| K3 定制内核 deb | [`linux-6.18-k3` releases](https://github.com/monkey-jsun/linux-6.18-k3/releases/latest) | ~50MB | `linux-image-6.18.3-cf-k3-1.3.0`，基于 `spacemit-com/linux-6.18` `k3-br-v1.0.7`（**就是 Bianbu 自己的基线**，所以 NPU/GPU 驱动不会丢） |
| Cuttlefish 宿主容器镜像 | `ghcr.io/monkey-jsun/aosp-cuttlefish-riscv64/cf-host-k3:latest` | 2.09 GB | 已实测**板子能拉到 ghcr.io** |
| `cvd_host_package_riscv64.tar.gz` | [`android-cuttlefish` releases](https://github.com/monkey-jsun/android-cuttlefish/releases/latest) | ~215 MB | 预构建镜像里已内置 URL 自动拉取 |
| AOSP guest 镜像 zip | [`aosp-cuttlefish-riscv64-addons` releases](https://github.com/monkey-jsun/aosp-cuttlefish-riscv64-addons/releases/latest) | ~830 MB+ | `aosp_cf_riscv64_phone_js-img.zip`（Android 17） |
| `cuttlefish-host-container` 脚本 | [`monkey-jsun/cuttlefish-host-container`](https://github.com/monkey-jsun/cuttlefish-host-container) | 小 | 提供 `cf-build.sh` / `cf-init.sh` / `cf-run.sh` / `cf-stop.sh` |

内核相对 Bianbu 基线补了 5 处（都跟 KVM 能不能起来直接相关）：IMSIC 原子 RMW 位宽、AIA 模拟的 `nr_sources` 上限、`vstimecmp` 软件镜像、`hmp_get_cpumask` 导出、KVM 可模块化的只读静态键。
v1.3 起还开了 `CONFIG_UDMABUF`（gfxstream 要）和 `CONFIG_VIDEO_LINLON_K3`（硬编 H.264 要）。

**⚠️ 别照抄 README 的需求清单**。实测这份 deb 里的 config 是：

```
CONFIG_KVM=m                    # 模块，开机要手动 modprobe，否则没有 /dev/kvm
CONFIG_KVM_RISCV_VTVM=y
CONFIG_VSOCKETS=m
CONFIG_VHOST_VSOCK=m            # adb over vsock
CONFIG_UDMABUF=y                # GPU 加速 ✅
CONFIG_VIDEO_LINLON_K3=y        # 硬编 ✅
# CONFIG_VHOST_NET is not set   # ❌ 上游自己也没开（README 那条要求已过期）
# CONFIG_ANDROID_BINDER_IPC is not set
```

`deploy/k3/deploy.sh kernel` 会**在装之前先把 deb 里的 config 抠出来做预检**，避免为一项错的期望白重启一次。

---

## 5. 如果目标是 ROM 定制开发：K3 的角色

**结论先说：K3 当"测试机"可以，当"编译机"不行。**

| 环节 | 在哪做 | 原因 |
| --- | --- | --- |
| 改 AOSP 源码 / 编译 | **必须是 Linux x86_64 机器** | 上游明确 *AOSP does not support riscv64 as a build host*；官方数字 **~150GB 同步、编完 ~320GB**、32GB+ 内存 |
| 跑起来 / 看效果 / 调 UI | **K3（就是本文档搭的 Cuttlefish）** | 真 RISC-V + KVM 硬件虚拟化 + PowerVR GPU 加速，比 QEMU 软件翻译快一个量级 |
| 产物形态 | K3 吃的是 **Cuttlefish 镜像** | 上游发的是 `aosp_cf_riscv64_phone_js-img.zip`（`super.img` 等），**不是刷机包**，不能刷进真机 |

工作流是**"x86_64 上编，K3 上跑"**：

```
[x86_64 Linux 编译机]                              [K3: Cuttlefish]
 repo init (上游 manifest)                          cf-run.sh 起 guest
 repo sync  (~150GB)                                ▲
 fetch-apks.sh / patches/apply.sh                   │ 换 cf-data/product/*.img 再重启
 source build/envsetup.sh                           │ （deploy.sh push-img 一条命令）
 lunch aosp_cf_riscv64_phone_js-…-userdebug         │
 m dist   (~2-4h 首次，增量几分钟)                    │
        │                                           │
        └── out/dist/aosp_cf_riscv64_phone_js-img-vX.Y.Z.zip ─┘
```

**编译机从哪来**：

| 选项 | 结论 |
| --- | --- |
| **云上按量开一台 x86_64**（已定） | ✅ 推荐。磁盘按 500GB 起、32GB+ 内存、SSD；编完就关 |
| 一台自己的 x86_64 Linux 机器 | ✅ 也行 |
| 用 111 这台服务器 | ❌ **不行**：2 核 / 1.9GB 内存 / 21G 空闲磁盘，差两个数量级 |
| 你的 Mac | ❌ **不行**（除非是 x86_64 Mac）：AOSP 构建主机只支持 Linux x86_64；Apple Silicon 上开 Linux 虚拟机也是 arm64，同样不被支持 |

### 5.1 编译机规格与选址

| 项 | 建议 | 原因 |
| --- | --- | --- |
| 架构 | **x86_64** | 上游明确：*AOSP does not support riscv64 as a build host* |
| 内存 | **32GB 起，64GB 舒服** | `soong_ui` 的链接阶段吃内存；32GB 会 swap 很慢 |
| 磁盘 | **500GB SSD**，别用 HDD | 官方数字：**~150GB 同步，编完 ~320GB**，再加 ccache |
| 系统 | Ubuntu 22.04 / 24.04 LTS | AOSP 官方支持的发行版 |
| **地区** | **选境外区域**（对 googlesource/GitHub 直连快） | 这样**不用折腾国内镜像/代理**，能省掉一大类问题；否则见下方镜像方案 |
| 计费 | 按量（编完就释放） | 全量编 2–4h，增量几分钟；盘可以留（快照） |

> 磁盘别省：`repo sync` 卡在磁盘满上是最常见的翻车方式。**先 `df -h` 确认有 500GB。**
> 机器开好之后跑什么、每步大概多久、踩过的坑，见操作手册 [`deploy/aosp/README.md`](deploy/aosp/README.md)。

### 5.2 完整编译命令（上游真实配方）

配方在 **[`monkey-jsun/aosp-cuttlefish-riscv64-addons`](https://github.com/monkey-jsun/aosp-cuttlefish-riscv64-addons)**（镜像的"家"：
产品定义 + manifest 固定 + AOSP 补丁 + 预装 APK）：

```bash
# 0) 依赖
sudo apt-get install -y git curl python3 openjdk-21-jdk  # 以及 AOSP 官方那套 build 依赖
mkdir -p ~/bin && curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo \
  && chmod a+x ~/bin/repo && export PATH=$PATH:~/bin

# 1) 取源码（manifest 由上游仓库提供，pin 住整棵 AOSP 树）
mkdir android17-release && cd android17-release
repo init --partial-clone --no-use-superproject \
    -u https://github.com/monkey-jsun/aosp-cuttlefish-riscv64-addons \
    -b v1.3.1 \
    -m manifests/aosp-cf-js-baseline.xml
    # 想追最新开发分支就把 v1.3.1 换成 main（tag 见下方说明）
repo sync -j8 -c --no-tags --optimized-fetch --retry-fetches=3

# 2) 预装 APK（WebView / F-Droid，不在 git 里，按 sha256 校验下载）
device/monkey-jsun/cuttlefish_riscv64/tools/fetch-apks.sh

# 3) 打上游对 AOSP 的补丁（幂等，每次编前都可跑）
device/monkey-jsun/cuttlefish_riscv64/patches/apply.sh

# 4) 编
source build/envsetup.sh
lunch aosp_cf_riscv64_phone_js-trunk_staging-userdebug
BUILD_NUMBER=v1.3.1 m dist

# 产物（~900MB）
#   out/dist/aosp_cf_riscv64_phone_js-img-v1.3.1.zip
```

> **别手敲这一坨。** 上面的步骤已经固化成 [`deploy/aosp/deploy.sh`](deploy/aosp/deploy.sh) 的子命令了：
> `check` / `deps` / `branches` / `mirror-on` / `init` / `sync` / `apks` / `patch` / `build` / `all` / `kernel` / `fetch` / `status` / `logs`。
> 在 Mac 上（不在编译机上）也能用 —— 加个 `HOST=root@x.x.x.x` 它会自动把脚本上传过去执行；
> 长任务加 `BG=1` 派生到后台并写日志（首次 `sync`/`build` 建议这么跑）。
> 用法与坑见 [`deploy/aosp/README.md`](deploy/aosp/README.md)。

**要复现上游某个已发布镜像**：`repo init` 的 `-b main` 换成 `-b <tag>`。
**当前上游只有两个 tag：`v1.2`、`v1.3.1`**（2026-10-14 用 `git ls-remote` 实测），
所以能复现的就是 **`-b v1.3.1`** —— 也正是板子上正在跑的那个镜像
（容器 ENV `CF_PRODUCT_IMG_URL=.../releases/download/v1.3.1/aosp_cf_riscv64_phone_js-img-v1.3.1.zip`）。

> ⚠️ **版本号别搞混，这里有两个独立的版本线**（我们踩过）：
> | 版本线 | 仓库 | 当前值 | 管什么 |
> | --- | --- | --- | --- |
> | `v1.3.x` | [`aosp-cuttlefish-riscv64-addons`](https://github.com/monkey-jsun/aosp-cuttlefish-riscv64-addons) | **v1.3.1** | **AOSP 镜像**（产品定义 + manifest + 补丁 + 预装 APK），我们 `repo init` 的那个 |
> | `cf-k3-v1.3.x` | [`android-cuttlefish`](https://github.com/monkey-jsun/android-cuttlefish) | cf-k3-**v1.3.2** | **宿主侧 cvd 包**（`CF_HOST_PACKAGE_URL`），不由我们编译 |
> | `6.18.3-cf-k3-1.3.0` | [`linux-6.18-k3`](https://github.com/monkey-jsun/linux-6.18-k3) | cf-k3-**v1.3.0** | **K3 内核** deb |
>
> 上游 README 里 `-b v1.3.2` / `BUILD_NUMBER=v1.3.2` 是**它的示例写法，那个 tag 并不存在**
> （照抄会在 `repo init` 直接失败）。`deploy/aosp/deploy.sh` 的 `init` 子命令会先做
> `git ls-remote` 预检，tag 不存在当场报错并列出可用值，不会让你白等。

**这正是我们已经在 K3 上跑的那个镜像的来源**：产品名 `aosp_cf_riscv64_phone_js` = 原封的 `aosp_cf_riscv64_phone`
+ 一个产品定义 + 预装 WebView（AOSP 不给 riscv64 出 WebView，用的是 BayLibre 的构建）+ F-Droid。

> **预装 APK 的两个下载源都在境外**（`media.githubusercontent.com` 的 145MB WebView、
> `f-droid.org` 的 12MB F-Droid），`fetch-apks.sh` 按 `prebuilts/apk-pins.tsv` 里的
> `bytes + sha256` 校验，**下不动就是这一步**——见 `deploy/aosp/README.md` 的坑 3。

> **国内网络备选**：`repo init` 本身走 GitHub（好办），但 manifest 里的 `android.googlesource.com`
> 在国内经常很慢（1085 个 project 里只有 1 个在我们这个 GitHub 仓库上，其余全在 googlesource）。
> 要么选境外云区域（推荐），要么对 googlesource 做重定向到 TUNA 的 AOSP 镜像
> （`deploy/aosp/deploy.sh mirror-on` 就是这个，**只重定向 googlesource**，别把 GitHub 也一起重定向）：
> ```bash
> git config --global url."https://mirrors.tuna.tsinghua.edu.cn/git/AOSP/".insteadOf \
>                    "https://android.googlesource.com/"
> ```

### 5.3 改完怎么上 K3（开发循环）

```bash
# 0) 在编译机上编完 → 把产物拉回本机（落在 out/aosp/）
bash deploy/aosp/deploy.sh fetch

# 1) 上板子（自动把旧 product 备份成 product.bak-<时间戳>）
bash deploy/k3/deploy.sh push-img out/aosp/aosp_cf_riscv64_phone_js-img-v1.3.1.zip
bash deploy/k3/deploy.sh stop && bash deploy/k3/deploy.sh run     # 重启 guest 用新镜像
```

`push-img` 会：把 zip 推进板子 → 把原 `cf-data/product` 备份成 `product.bak-<时间戳>` →
解压新镜像到 `cf-data/product`。容器下次起机就用新镜像（`entrypoint.sh` 是"文件在就跳过下载"）。
出问题就 `mv product.bak-<时间戳> product` 回滚。

> **`m dist` 的增量很快（几分钟）**，所以在编译机上改一行 Java → 编 → 推 → 重启，这个循环是能接受的。
> 如果只改 framework/系统层，不用重编内核和 crosvm。

**另一条更省事的定制路线（内核）**：内核**可以在 x86_64 上交叉编译，约 10 分钟**（上游配方）：
```bash
sudo apt install gcc-riscv64-linux-gnu
git clone https://github.com/monkey-jsun/linux-6.18-k3.git && cd linux-6.18-k3
git checkout cf-k3-v1.3.0
cp config-aia2-kvm-modular .config && touch .scmversion
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- olddefconfig
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j$(nproc) \
     LOCALVERSION=-cf-k3-1.3.0 KDEB_PKGVERSION=6.18.3-cf-k3-1.3.0 bindeb-pkg
```
产出的 deb 用 `deploy.sh kernel` 那套装法就能上板子（阶段 1 走的同一条路）。

> **建议的推进顺序**：先在 K3 上把**上游现成的 AOSP 17 riscv64 镜像跑通**（本文档阶段 2~4，**已经跑通**），
> 确认"能跑 + 能看 + 能 adb"这条路是通的，**再**去准备编译机改源码。
> 否则你会同时面对"编译不过"和"跑不起来"两个问题，排查成本翻倍。

> **期望管理（riscv64 Android 的成熟度）**：AOSP 对 riscv64 的支持还在推进中（到 Android 17 这一代才能跑 Cuttlefish riscv64），
> 上游是社区/单人项目。定制**自由度最大**的部分恰好是**已经能跑**的部分（系统层 / 框架层 / Java）；
> **HAL / 驱动 / native lib** 这些是最容易卡住的地方。

### 5.4 能装什么 App（riscv64 ABI 硬约束，2026-10-14 实测）

**一句话：带 arm64 / x86 native `.so` 的应用连装都装不上 —— 微信、QQ 属于这一类。**

> **这条限制有没有解？** 有方向，但上游是空白（`arm64 → riscv64` 二进制翻译）。
> 机制、三条路线、要动哪些系统组件，以及**已在 K3 上实测通过的第一步** →
> 独立成篇：[`arm64-on-riscv64.md`](../k3-arm64-on-riscv/arm64-on-riscv64.md)。
> 板子本身的能力与短板 → [`k3-hardware.md`](../k3-hardware/k3-hardware.md)。

guest 声明的 ABI 能力：

| 属性 | 值 |
| --- | --- |
| `ro.product.cpu.abi` | `riscv64` |
| `ro.product.cpu.abilist` / `abilist64` | `riscv64`（**只有这一个**，没有 arm64-v8a / armeabi-v7a / x86_64） |
| `ro.product.cpu.abilist32` | 空 |
| `ro.dalvik.vm.native.bridge` | `0` → **没有 ARM 转译层**（Google 的 `libndk_translation` 只给 x86 宿主，RISC-V 上没有对应实现，上游镜像也没带） |

实测对照（都是在板上真下载、真 `adb install` 的）：

| 应用 | APK 里的原生库 | 结果 |
| --- | --- | --- |
| **QQ** 9.2.65（官方 arm64 渠道包，382 MB） | `lib/arm64-v8a/` ×198、`lib/x86_64/` ×3，**`lib/riscv64/` ×0** | ❌ `INSTALL_FAILED_NO_MATCHING_ABIS: Failed to extract native libraries, res=-113` |
| **微信** 8.0.79（官方 arm64 渠道包，273 MB） | `lib/arm64-v8a/` ×209，**`lib/riscv64/` ×0** | ❌ 同上，一字不差 |
| **F-Droid**（guest 预装，纯 Java） | `.so` 数量 **0** | ✅ 装好且在跑 |
| **WebView**（guest 预装） | `lib/riscv64/libwebviewchromium.so`（130 MB 的 Chromium） | ✅ 装好且在跑 |

关键点是**这不是"性能差"，是"装不上"**：在包管理器的 ABI 匹配那一关就被拒（`res=-113`），
连启动的机会都没有。而且**改不了**：

- 微信 / QQ / 淘宝 / 抖音 / 支付宝…… 只发 `arm64-v8a`（少数老版本带 `x86_64`），**没有 riscv64 版本**，
  也**不可能**由我们重编（闭源）。
- RISC-V Android 上**没有** ARM→RISC-V 的 native bridge —— 不是"没打开开关"，是 AOSP 里根本没这个东西。
- 宿主 KVM **只能跑与宿主同架构的 guest**（riscv64 宿主只能跑 riscv64 guest），
  所以"在 K3 上换个 arm64 / x86_64 的 Android 镜像"**物理上不存在**。
- 唯一剩下的理论路径是 QEMU TCG **全系统软件模拟**，那是"分钟级开机、基本不可用"的性能，不具备实用性。
- 顺带一个细节：微信把图片和字体也塞进了 `lib/arm64-v8a/`（`lib/images/` ×16、`lib/fonts/` ×2）——
  国产 App 常见的"拿 native 库目录规避资源扫描"的做法，说明它对 native 目录是硬依赖。

**K3 上能装什么：**

1. **纯 Java / Kotlin 应用**（APK 里没有 `lib/`）—— F-Droid 上这类很多；
2. **为 riscv64 编译过的应用与系统组件** —— 上游预装的 WebView 就是（BayLibre 编的 riscv64 Chromium）；
3. **我们自己编的东西** —— §5.2 那棵 AOSP 树 `m dist` 出来的 riscv64 镜像、以及 riscv64 目标的 APK。
   **这正是 ROM 定制开发要的东西。**

**要真跑微信 / QQ，只能换载体**（与 K3 无关的独立选择）：

| 载体 | 微信 / QQ | 代价 |
| --- | --- | --- |
| x86_64 机器 + Android（AVD / redroid / Waydroid，靠 `libndk_translation` 跑 arm64）+ KVM | ✅ | 需要 x86_64 CPU 且开放虚拟化（云主机 / PC） |
| arm64 板子（如 RK3588）+ redroid / Waydroid 或 Cuttlefish arm64 | ✅ | 换硬件 |
| Google 官方 Cuttlefish（宿主只有 x86_64 / arm64） | ✅ | 同上，与 K3 的 riscv64 路线互斥 |

> 所以要先回答一个问题：**你要的是"在这块 RISC-V 板子上做 ROM 定制"，还是"在这块板子上用微信"？**
> 两者不冲突，但不共用同一个载体 —— 前者 K3 没问题，后者 K3 做不到。

---

## 6. 板子上的代理（Xray）——治 GitHub / ghcr 断流

**为什么需要**：板子的国际线路是"HTTP 打得到、TLS 会被掐"。实测 `git clone https://github.com/...` 直接报
`GnuTLS recv error (-110)`，而拉 2.09GB 宿主镜像、下 830MB+ AOSP 镜像全都要走国际线路。
**不先解决这个，阶段 2 起每一步都会随机失败。**

**做法**：在板子本机跑一个 Xray（SOCKS + HTTP mixed，**只监听 `127.0.0.1:10808`**，不对外暴露），
再把 `git` / `docker daemon` 指过去。

| 项 | 值 |
| --- | --- |
| 内核 | Xray-core **v26.3.27**（`Xray-linux-riscv64.zip`，静态链接 RISC-V ELF） |
| 为什么不是 v2rayN | v2rayN 是 Windows/桌面 GUI（`v2rayN-linux-riscv64.deb` 也有，是 Avalonia GUI）；**它内置的内核就是 Xray**，直接拿内核更干净、无 GUI 依赖 |
| 出口 | 用户提供的 vless + `raw`（伪装成 HTTP `Host: www.bing.com`）→ `:443` |
| geo 数据 | `geoip.dat` / `geosite.dat`（Loyalsoldier 版）。config 里用了 `geosite:google/cn/private`、`geoip:cn/private`，**缺了 Xray 直接起不来** |
| 服务 | systemd `xray-k3.service`（`xray run -confdir /etc/xray-k3`，`Restart=always`，加固：`NoNewPrivileges` / `ProtectSystem=strict` / `ProtectHome`） |
| 监听 | `127.0.0.1:10808`（mixed = SOCKS5 + HTTP 同端口） |

**落地命令**：

```bash
bash deploy/k3/deploy.sh vpn-dl          # 本机下载 Xray riscv64 + geo 数据（缓存到 deploy/k3/xray/dl/，已 gitignore）
bash deploy/k3/deploy.sh vpn-install     # 推到板子 + 装 systemd 服务 + 起
bash deploy/k3/deploy.sh vpn-test        # 验证：github / ghcr / 出口 IP / 代理下的 git clone
bash deploy/k3/deploy.sh proxy-on        # docker daemon + git 走代理（关键：docker pull 才通）
bash deploy/k3/deploy.sh vpn-status      # 状态一览
bash deploy/k3/deploy.sh proxy-off       # 撤掉 docker daemon / git 的代理
bash deploy/k3/deploy.sh vpn-rm          # 停并移除板子上的服务
```

**实测结果**（2026-10-14）：

| 检查 | 结果 |
| --- | --- |
| 直连 `https://github.com/` | ❌ 超时 8s |
| 代理 `https://github.com/` | ✅ **200 / 2.0s** |
| 代理 `https://ghcr.io/v2/` | ✅ 401（正常，要鉴权）/ 1.0s |
| 代理出口 IP | `43.155.215.x` |
| 代理下 `git clone` | ✅ |
| `docker pull hello-world`（docker.io 本来被墙） | ✅ 成功 |

**密钥处理**：

- 真实配置是 **`deploy/k3/xray/config.json`**（含服务器地址 + UUID）→ **已 gitignore，绝不入库**；
  仓库里只有 `config.json.example`（脱敏成 `<SERVER_IP>` / `<UUID>`）。
- ⚠️ 该 UUID 已在聊天记录里出现过，**建议在服务端轮换一次**；换完改 `config.json` 再 `vpn-install` 即可。
- 板子上配置在 `/etc/xray-k3/00-config.json`，权限 `0600`。

**局限（重要）**：

1. **这不是"全局 VPN"，是应用级代理**。它不会自动给所有程序翻墙，要显式指过去（docker daemon / git 已指好）。
   想要全局透明代理得再上 `tproxy` + nftables + `ip rule`，是另一件事。
2. **容器内部默认用不到它**（容器里的 `127.0.0.1` 是容器自己）。Cuttlefish 容器首次运行要下 ~1GB
   （`cvd-host` 包 + AOSP 镜像）。**解法不是给容器开代理，而是宿主侧预取**（`deploy.sh prefetch`）：
   在板子上按 `entrypoint.sh` 期望的布局把 `cf-data/{host,product}` 放好，容器一看文件在就跳过下载。
   这样容器**根本不需要联网**，比"给容器灌 `HTTP_PROXY`"更稳。
   （若将来容器内确实要联网，再加一个绑到 `docker0` 网关的 inbound + `--docker-arg --env=https_proxy=...`。）

---

## 7. 实施步骤

> 每阶段都有**验收标准**，过了才进下一阶段。脚本在 `deploy/k3/`。
> **阶段 1 起需要板子上的 sudo 密码**（`pingji` 在 sudo 组但不免密）。

### 阶段 0 · 前置与备份（不改任何东西）——✅ 已完成

- [x] **sudo 密码**：不在文档里记录 —— 存本机 `~/.k3-sudo`（`0600`，不入 git），也可用环境变量 `K3_SUDO_PASS` 覆盖。实测确认过。注意输错一次之后 `sudo` 自己有**凭证缓存**，会出现"看似也通过"的假象，别被它骗了
- [x] 现状快照：`bash deploy/k3/deploy.sh snapshot` → `deploy/k3/snapshots/20261014-095045.txt`（135 行，回滚对照用）
- [x] `/boot` 余量 148M（够：换内核后双内核并存剩 ~90M）
- [x] 111 上 `8443` / `6520` 空闲 → guest 侧用 8443/6520，**反隧道对外改用 18443/16520**（避开正式栈）
- [ ] 板子 uptime 短的疑点：用户答"不知道" → **留作观察项**；起 Cuttlefish 前顺手看一眼有没有别人在用

**验收**：✅ 已产出快照文本。

### 阶段 1 · 宿主准备（需要 sudo + 一次重启）——✅ 已完成

一步一条命令，全部走 `deploy/k3/deploy.sh`：

```bash
bash deploy/k3/deploy.sh prep        # 板子上：apt 装 docker.io + 下载内核 deb（不改内核）
bash deploy/k3/deploy.sh kernel      # 装内核前先扒 deb 里的 config 做预检，再 dpkg -i
bash deploy/k3/deploy.sh reboot      # 重启（/boot/env_k3.txt 由 deb 的 postinst 自动指向新内核）
bash deploy/k3/deploy.sh mods        # ★ 关键一步：modprobe kvm vhost_vsock + 写 modules-load.d 持久化
bash deploy/k3/deploy.sh verify-host # 验收
```

**验收（实测结果）**：

| 检查 | 结果 |
| --- | --- |
| 内核 | `6.18.3-cf-k3-1.3.0` ✅ |
| `docker` | `Docker version 29.1.3-0ubuntu4`，daemon ✅ |
| `/dev/kvm` | `crw-rw----+ root kvm` ✅ **（跑完 `mods` 才有）** |
| `/dev/vhost-vsock` | ✅ |
| `CONFIG_UDMABUF` | ✅ `=y`（GPU 加速的前提，换内核的真正收益之一） |
| `CONFIG_VHOST_NET` | ❌ 仍是 `is not set`（**上游自己也没开**，见 §3.7；不影响起机 / 看屏 / adb） |
| `k3-tunnel` / `k3-heartbeat.timer` | `active active` ✅（重启后自动重连，门户 302 + 心跳 age 9s） |

> ⚠️ **踩过的两个坑（已修进脚本）**：
> 1. **脚本经 `sudo` 跑时 `$HOME` 会变成 `/root`** → 内核 deb 被下到 `/root/k3-android/pkg/`。
>    现在脚本统一用板子上 `pingji` 的**绝对 home**（`board_home()`）。
> 2. **新内核 `CONFIG_KVM=m`，开机不会自动加载模块 → `/dev/kvm` 消失**。
>    必须跑 `mods`；它同时写 `/etc/modules-load.d/k3-android.conf`，所以**重启后依然有效**。

### 阶段 1.5 · 板子代理（必须在任何大下载之前）——✅ 已完成

```bash
bash deploy/k3/deploy.sh vpn-install     # 装 Xray（本机 127.0.0.1:10808）
bash deploy/k3/deploy.sh vpn-test        # 验证
bash deploy/k3/deploy.sh proxy-on        # docker daemon + git 指过去
```

**验收（实测）**：代理下 `github` 200 / `ghcr.io` 401 / 出口 IP `43.155.215.x` / `git clone` ✅ / `docker pull hello-world` ✅。详见 §6。

> **顺序很重要**：先装代理再进阶段 2。否则 2.09GB 宿主镜像 + 830MB+ guest 镜像的下载会随机失败（`GnuTLS recv error`）。

### 阶段 2 · 部署 Cuttlefish

```bash
# 一条命令：板子上 clone 上游脚本（失败会自动从本机中转）+ pull 2.09GB 镜像
bash deploy/k3/deploy.sh deploy
```

> 板子直连 GitHub 不稳，脚本里做了**本机 clone → tar 推过去**的回退；镜像走 §6 的代理拉。
> 用预构建镜像就**跳过 `cf-build.sh` / `cf-init.sh`**：镜像里已内置 cvd-host 和 AOSP 镜像的 URL，首次运行会自动拉并初始化 `/cf`。
> 代价是首次运行要多等一轮 ~1GB 下载。

**验收**：`docker images` 里能看到 `cf-host-k3`。

### 阶段 3 · 起机并验证

```bash
bash deploy/k3/deploy.sh prefetch      # ★ 先在宿主侧把 cvd-host 包 + AOSP 镜像下好（~1GB，走代理）
bash deploy/k3/deploy.sh run
bash deploy/k3/deploy.sh smoke
```

**为什么要 `prefetch`**：容器起来后 `entrypoint.sh` 会去下

```
CF_HOST_PACKAGE_URL=https://github.com/monkey-jsun/android-cuttlefish/releases/download/cf-k3-v1.3.2/cvd_host_package_riscv64.tar.gz
CF_PRODUCT_IMG_URL=https://github.com/monkey-jsun/aosp-cuttlefish-riscv64-addons/releases/download/v1.3.1/aosp_cf_riscv64_phone_js-img-v1.3.1.zip
```

而**容器里的 `127.0.0.1` 不是板子**（`cf-run.sh` 用的是默认 bridge 网络，没 `--network host`），
所以板子上那个代理容器里够不着。解法是**在宿主侧按 `entrypoint.sh` 期望的布局先把文件放好**：

```
k3-android/cuttlefish-host-container/cf-data/
├── host/bin/launch_cvd          ← cvd_host_package_riscv64.tar.gz 解开
└── product/*.img                ← aosp_cf_riscv64_phone_js-img-v1.3.1.zip 解开
```

`entrypoint.sh` 的逻辑是"文件在就跳过下载"，所以容器起来直接进 `launch_cvd`，**不需要联网**。

`cf-run.sh` 实际跑的容器（要点）：

```
docker run -it --rm --name cf --label cf-host=1 --privileged \
  --device /dev/net/tun --device /dev/vhost-vsock --device /dev/dri --device /dev/kvm --device /dev/video0 \
  -p 8443:8443 -p 5900:5900 -p 6520:6520 -p 15550-15599:15550-15599/udp \
  -v <cf-data>:/cf \
  cf-host-k3
```

`CF_GPU_MODE=gfxstream`（走 `/dev/dri/renderD128` + `UDMABUF`）、`CF_MEM_MB=8192`、`CF_CPUS=4`、`CF_VM_MANAGER=crosvm`（host/guest 同架构自动判定）。

**验收**：

```bash
# 板子上
docker ps                                    # cf 容器在跑
curl -k -o /dev/null -w '%{http_code}\n' https://127.0.0.1:8443   # WebRTC 端口有响应
curl -k -o /dev/null -w '%{http_code}\n' https://127.0.0.1:8443/client.html
# 从能访问板子的机器上
adb connect 192.168.110.194:6520
adb devices
adb shell getprop ro.product.cpu.abi         # → riscv64
adb shell getprop ro.build.version.release   # → 17
```

浏览器打开 `https://<板子IP>:8443` 应能看到 Android 桌面，可交互（WebRTC 是 `https` 自签证书，浏览器要放行）。

**实测结果（2026-10-14）—— 全通**：

| 检查 | 结果 |
| --- | --- |
| `launch_cvd` 起机 | ✅ `--vm_manager=crosvm`（host/guest 同架构自动选中）、`CF_GPU_MODE=gfxstream`、`CF_MEM_MB=8192`、`CF_CPUS=4` |
| 引导 | ✅ `VIRTUAL_DEVICE_BOOT_STARTED` → `NETWORK_ETHERNET_CONNECTED` → **`VIRTUAL_DEVICE_BOOT_COMPLETED`** |
| guest | ✅ `ro.product.cpu.abi=riscv64`、`ro.build.version.release=17`、`aosp_cf_riscv64_phone_js`/`vsoc_riscv64`、`userdebug` |
| adb | ✅ 容器内 `adb connect 127.0.0.1:6520` → `device` |
| 屏幕 | ✅ `adb exec-out screencap -p` 出来 **720×1280 PNG**，是真实的 Android 锁屏（"Android Virtual Operator" + 时钟 + Unlock） |
| WebRTC 端口 | ✅ `https://127.0.0.1:8443/` → 302（shim 在跑），另开 5900(VNC) / 6520(adb) / 15550-15599/udp |
| 资源 | 容器 3.6GiB / 15.6GiB 内存，CPU ~44% |
| 容器 | `cf`（label `cf-host=1`）——**注意容器名就叫 `cf`**，所以脚本里一律用 `--filter label=cf-host`，别用 `--filter name=` |

> **`CONFIG_VHOST_NET` 的疑问到此澄清**：guest 网络**通了**（`NETWORK_ETHERNET_CONNECTED`），
> 走的是 crosvm 自己的 net 后端，**不需要 vhost_net**。上游 README 那条要求确实是过期的，**别去重编内核**。

> ⚠️ **GPU 还在刷错**：`devices::virtio::gpu ERROR ... ErrRutabaga(InvalidContextId)` 持续输出，
> 起机时也有 `--gpu_mode=gfxstream was requested but the prerequisites were not detected`。
> 但**屏幕照样渲染出来了**（说明退到了软件路径）。也就是说：**能用，但不是硬件加速**。
> 若要追加速，先试 `--gpu-mode=guest_swiftshader`（明示软件渲染，更稳）再排查 rutabaga；
> 这一项**不影响 ROM 定制开发的可用性**，可以后置。

### 阶段 4 · 对外访问（在 111 上做正向转发，**不动板子**）——✅ 已完成

```bash
bash deploy/k3/deploy.sh expose        # 建 k3-android-fwd.service（111 上）+ 验证
bash deploy/k3/deploy.sh unexpose      # 撤掉
```

**它不是反隧道**。原计划的"追加 `-R`"被 111 的 `PermitListen 127.0.0.1:2222` 挡死（见 §2 / §3.6）。
`expose` 干的事是：在 **111** 上落一个 systemd 单元，内容是

```ini
# 111:/etc/systemd/system/k3-android-fwd.service
[Service]
ExecStart=/usr/bin/docker exec -i k3-ttyd ssh -NT \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o TCPKeepAlive=yes -o ConnectTimeout=10 -o BatchMode=yes -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i /app/ssh/id_ed25519 -p 2222 -l pingji 127.0.0.1 \
  -L 127.0.0.1:18443:127.0.0.1:8443 \
  -L 127.0.0.1:16520:127.0.0.1:6520
Restart=always
```

`k3-ttyd` 是 **host 网络**，所以容器里的 `127.0.0.1:18443` 就是 111 的 `127.0.0.1:18443`；
`-L` 的终点 `127.0.0.1:8443` 由**板子的 sshd**在板子本机视角解析 —— 效果与反隧道等价。

| | 反隧道（原计划） | 正向转发（实际用的） |
| --- | --- | --- |
| 改板子的 `k3-tunnel` | 要 | **不要** |
| 改 111 的 sshd | 要（加 `PermitListen`） | **不要** |
| 打挂门户的风险 | **有**（连带 `ExitOnForwardFailure`） | **无** |
| 依赖 | 板子 | 板子 + 111 上的 `k3-ttyd` 容器 |

**端口**：111 上本地用**高位端口 18443 / 16520**（不占 8443/6520，避免与 111 上别的东西撞车）。
都只绑 `127.0.0.1`，**不在公网暴露**。

本机访问（Mac 上开一个窗口常驻）：

```bash
ssh -N -L 8443:127.0.0.1:18443 -L 6520:127.0.0.1:16520 root@111.228.40.211
# 浏览器 https://127.0.0.1:8443/    （自签证书，放行即可）
# adb connect 127.0.0.1:6520
```

**验收（实测）**：

| 检查 | 结果 |
| --- | --- |
| 111 上 `ss -ltn` | ✅ `127.0.0.1:18443`、`127.0.0.1:16520` 在听 |
| 111 上 `https://127.0.0.1:18443/` | ✅ 302 |
| 本机 `https://127.0.0.1:18443/client.html` | ✅ 200（WebRTC 客户端页面能取到） |
| 本机 `adb connect 127.0.0.1:16520` | ✅ `device`，`abi=riscv64`、`release=17`、`model=Cuttlefish riscv64 phone (js)` |
| K3 门户 / 心跳 | ✅ 未受影响（全程没碰板子的隧道） |

### 阶段 5 · 纳入管理（Docker + git）

- [ ] 板子上跑的容器由 **Docker** 管理（上游的 `cf-run.sh` 本身就是 `docker run`）
- [ ] 脚本落到本仓库 `deploy/k3/`，本机 `git` 记录
- [ ] 板子侧：`deploy/` 脚本同步到 `/opt/k3-android`（或 `~/k3-android`）并建本地 git 仓库（与 111 上 `/opt/web-profile` 的做法一致）
- [ ] 密钥 / 令牌不入 git（`.gitignore`）

**验收**：`deploy/k3/deploy.sh status` 一句话看清板子上的容器、端口、隧道、内核。

---

## 8. 资源预算

| 资源 | 需求 | 板子 |
| --- | --- | --- |
| CPU | 项目要求 ≥2 核给容器；guest 默认 4 vCPU | 物理 16 核（会话可见 8） |
| 内存 | ≥8 GB；guest 默认 8 GB | 15.9 GiB，需 CPU 侧留量 |
| 磁盘 | ≥32 GB；镜像 2.09G + AOSP ~1G + 运行期 | 可用 90G |
| 带宽 | 首次约 3–4 GB 下载 | 下行实测 ~87 Mbps |
| 端口 | 板子 8443 / 6520；111 上 18443 / 16520 | 需确认空闲 |

**对现有业务的影响**：占掉板子大部分 CPU/内存。板子上还跑着 `spacemit-ai-gateway`（NPU 相关）和桌面服务（PipeWire/WirePlumber），
Cuttlefish 起满时会明显抢资源。建议：**不常开**，用完 `cf-stop.sh`。

---

## 9. 风险与回滚

| 风险 | 影响 | 概率 | 缓解 / 回滚 |
| --- | --- | --- | --- |
| **换内核后起不来** | 板子离线，门户/心跳全断 | 低 | `dpkg` 保留旧内核 + `/boot` 有旧条目；项目内核基于 Bianbu 同基线；最坏走**串口**（板子有 `ttyS0` getty）从 U-Boot 选旧内核 |
| **`VHOST_NET` 新内核仍没开** | guest 无网络，方案硬阻断 | 中 | 先只装内核 + 重启，**仅验证 config**，不继续部署；真缺就自己用 `config-aia2-kvm-modular` 重编内核（交叉编译 ~10 min） |
| **`UDMABUF` 新内核仍没开** | GPU 加速不可用，只能软件渲染 | 低 | 用 `--gpu_mode=none` 先跑通功能，再补内核 |
| **换内核打断 Bianbu 特有功能** | 桌面/NPU 异常 | 低 | 基线一致；出问题回滚内核即可 |
| **`/boot` 空间不足** | 内核装不上 | 低 | 现余 148M，够一个内核；装完别再加 |
| **改隧道把门户打挂** | 门户/终端全断，只能等板子自己重连或人去现场 | 中 | 用高位端口避免占用；改完立即 `systemctl is-active` 双验证；保留原 unit 内容便于还原 |
| **Docker Hub 被墙** | 拉不动 `docker.io` 官方镜像 | 高（已知） | 用镜像源；本项目所需镜像在 ghcr.io，**实测可达** |
| **上游单人维护** | 后续 update / 排错靠自己 | 中 | 把依赖版本 pin 进 `deploy/k3/`（内核 deb 版本、镜像 tag、AOSP 版本），别用 `latest` |
| **app 跑不了** | 达不到预期用途 | 高 | 已确认目标是 **ROM 定制开发**（见 §5）：K3 只当测试机，编译在 x86_64 上做；带 arm64 native so 的 App 在 riscv64 guest 上跑不了 |

**回滚清单**（任一阶段失败）：
```bash
./cf-stop.sh                                   # 停 Cuttlefish
sudo systemctl stop docker                     # 停 Docker
sudo rm /etc/systemd/system/k3-tunnel.service.d/android.conf
sudo systemctl daemon-reload && sudo systemctl restart k3-tunnel
sudo dpkg -i linux-image-6.18.3-generic_*.deb  # 装回旧内核（或用 apt 里保留的）
sudo reboot
```

---

## 10. 决策与待确认

**已定（2026-10-14）**：

| 问题 | 答复 | 落地 |
| --- | --- | --- |
| `pingji` 的 sudo 密码 | 不在文档里记录 | 存本机 `~/.k3-sudo`（0600，不入 git）；也可用 `K3_SUDO_PASS` 覆盖 |
| 能重启板子吗 | 可以 | 已重启一次进新内核（中断约 70s，门户/心跳自动恢复） |
| 拿 K3 Android 干什么 | **ROM 定制开发** | → **K3 是测试机，不是编译机**，见 §5（编译机还没着落，这是当前最大缺口） |
| 板子 uptime 短 | 不知道 | 留作观察项 |
| 111 上 18443 / 16520 | 未反对 | 阶段 4 用（guest 侧仍是 8443 / 6520） |
| `deploy/k3/` 纳入 Docker+git 管理 | 默认纳入 | 已在 `deploy/k3/`，随本仓库本地 git 提交 |

**还没定 / 下次要回答**：

1. **编译机从哪来**（x86_64 Linux，32GB+ 内存、400GB+ 空闲磁盘）——不定这个，ROM 定制只能停在"跑上游镜像"。
2. guest 到底要不要联网？`VHOST_NET` 缺只影响 guest 自己上网；如果只是做 ROM 定制 + 看屏幕 + adb，可以先不管（见 §3.7）。
3. 要不要做**全局透明代理**（tproxy），还是保持现在的应用级代理（docker/git 已指好）。
4. 是否轮换聊天记录里暴露的那个 UUID。

---

## 11. 参考

- 上游伞项目：[monkey-jsun/aosp-cuttlefish-riscv64](https://github.com/monkey-jsun/aosp-cuttlefish-riscv64)（v1.3.1）
- 内核：`linux-6.18-k3`（`cuttlefish-on-k3` 分支）
- crosvm 移植：`monkey-jsun/crosvm`（`riscv64` 分支，10 个补丁）
- U-Boot：`monkey-jsun/u-boot`（`riscv64` 分支，2 个补丁）
- Cuttlefish 移植：`monkey-jsun/android-cuttlefish`（`dev/riscv64` 分支）
- AOSP 官方 RISC-V 进展：[google/android-riscv64](https://github.com/google/android-riscv64)（Cuttlefish 路线，`aosp_cf_riscv64_phone`）
- 代理内核：[XTLS/Xray-core](https://github.com/XTLS/Xray-core)（`Xray-linux-riscv64.zip`，v26.3.27）
- geo 数据：[Loyalsoldier/v2ray-rules-dat](https://github.com/Loyalsoldier/v2ray-rules-dat)（`geoip.dat` / `geosite.dat`）
- 板子门户 / 隧道 / 终端：本仓库 [`k3-portal.md`](../k3-portal/k3-portal.md)
