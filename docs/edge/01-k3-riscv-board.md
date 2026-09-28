# 01 · 边缘设备接入：一根网线，一块 RISC-V 板

本篇记的是一次真机接入的完整过程 —— 从「网线插上了但它到底在哪」到「SSH 进去看清它有几个网口」。

它不属于优化笔记：[07-edge.md](../kernel-opt/07-edge.md) 讲的是**在边缘设备上怎么优化**
（带宽受限、功耗墙、KV cache 压缩），本篇讲的是**怎么先摸到那台设备**。前者是策略，
后者是第一步，而第一步往往会卡人。

**一句话结论：** 那台板子本来就接着自己的 WiFi 并且能上外网，插网线换来的不是「连通」
（它一直通），而是**延迟差两个数量级**（局域网内 WiFi 69 ms vs 网线 <1 ms）。

---

## 0. 速览

| 问题 | 结论 |
| --- | --- |
| 设备是什么 | SpacemiT K3 开发板，**RISC-V**（`riscv64`），8 核 X100，15 GiB 内存，Bianbu 4.0.1 |
| 有几个网口 | **3 个**：2 个有线（`end0` 板载 / `enP2p1s0` 空置）+ **1 个无线**（RTL8852BE） |
| 它自己能上网吗 | **能**。WiFi 已连 `39-1102_5G`，独立网段 `192.168.110.194/24`，公网可通 |
| 那网线起到什么作用 | 走 ICS 那条路 metric 更低（100 < 600），**出网实际借道电脑**；路由上它是主路，WiFi 是备份 |
| 网线真正的价值 | 延迟。同一台设备 WiFi 69–84 ms，网线 <1 ms |
| 登录方式 | 普通用户 + 密码可行；**root 用密码被拒**（Ubuntu 默认策略，非密码错） |

---

## 1. 设备是什么

```
$ ssh pingji@<设备IP> 'id; uname -a'
uid=1000(pingji) gid=1000(pingji) groups=...27(sudo)...
Linux pingji-k3 6.18.3-generic #1.0.1.4 SMP PREEMPT_DYNAMIC riscv64 GNU/Linux
```

```
$ cat /etc/os-release
Bianbu 4.0.1
```

| 项 | 值 |
| --- | --- |
| 主机名 | `pingji-k3` |
| 架构 | **`riscv64`** ← 不是 ARM，也不是 x86 |
| 内核 | `6.18.3-generic #1.0.1.4` |
| 发行版 | **Bianbu 4.0.1**（进迭时空 / SpacemiT 基于 Ubuntu 的 RISC-V 发行版） |
| CPU | SpacemiT X100，8 核 |
| 内存 | 15 GiB |
| SSH 服务端 | `OpenSSH 10.2p1 Ubuntu-2ubuntu3` |

**怎么从外部猜出它是 Linux：** 见 §3.2 的 TTL 判据。主机名里的 `k3` + `riscv64`
基本可以锁定 SoC 型号。

---

## 2. 网络：它有几个口，各自在哪

```
                    ┌────────────────────────┐
                    │   路由器 192.168.110.1  │
                    │   SSID: 39-1102(_5G)   │
                    └───────┬────────┬───────┘
                       5GHz │        │ 2.4GHz
              ┌─────────────┴──┐  ┌──┴─────────────┐
              │  K3 板          │  │  你的笔记本     │
              │  wlP4p1s0       │  │  WLAN          │
              │  192.168.110.194│  │  192.168.110.81│
              └────────┬────────┘  └───────┬────────┘
                       │                   │
                  end0 │                   │ 以太网
          192.168.137.198                   │ 192.168.137.1
                       └──── 网线直连 ──────┘
                        （ICS 共享出的 137 网段）
```

**两台机器本来就在同一个网段**（`192.168.110.0/24`，同一个路由器的两个频段），
不插网线也能互通 —— 实测 `ssh pingji@192.168.110.194` 直接就能进。

### 2.1 接口清单

在设备上遍历 `/sys/class/net/`，靠 `wireless` 子目录判无线（比 `iw` 更可靠，不依赖装工具）：

```bash
for i in /sys/class/net/*; do
    n=$(basename "$i"); [ "$n" = lo ] && continue
    [ -d "$i/wireless" ] && echo "$n 无线" || echo "$n 有线"
done
```

| 接口 | 类型 | 驱动 | 状态 | IP |
| --- | --- | --- | --- | --- |
| `end0` | 有线（板载） | `dwmac-spacemit-ethqos` | UP | `192.168.137.198/24` |
| `enP2p1s0` | 有线（第二个口） | `r8127` | DOWN / `NO-CARRIER` | 没插线 |
| `wlP4p1s0` | **无线** | `rtw89_8852be`（RTL8852BE） | UP | `192.168.110.194/24` |

另外还有蓝牙（Realtek USB 无线电 + `hci0`），以及一个空的 M.2 WWAN 槽位 —— `rfkill list`
里有 `m.2 WWAN` 这一条，说明板子留了 4G/5G 模块的位置但没装。

> **命名规则**：`en` / `wl` 开头是 systemd 的可预测网卡名，后面跟总线位置。
> `end0` 的 `d` 表示 onboard（板载），`enP2p1s0` 的 `P2p1s0` 是 PCI 位置。
> 板载口叫 `end0` 而不是 `eth0`，是 RISC-V 平台很常见的情况 —— 脚本里别写死 `eth0`。

### 2.2 两条默认路由（关键）

```
$ ip route
default via 192.168.137.1  dev end0      proto dhcp src 192.168.137.198 metric 100
default via 192.168.110.1  dev wlP4p1s0  proto dhcp src 192.168.110.194 metric 600
192.168.110.0/24 dev wlP4p1s0  proto kernel scope link src 192.168.110.194 metric 600
192.168.137.0/24 dev end0      proto kernel scope link src 192.168.137.198 metric 100
```

**同时存在两条 `default`，metric 小的赢。** 所以：

- 出网走 `192.168.137.1`（**电脑的 ICS**），metric 100
- WiFi 那条 metric 600 是**备份**，主路断了会自动接管

这解释了一个容易误判的现象：设备明明是无线连着的，但抓包看它的出网流量全在网线上。
**metric 是判断「它实际走哪条路」的唯一依据，不能靠「哪个口 UP」猜。**

实测联网（公网 IP 直连 + DNS 解析都通）：

```
223.5.5.5 (公网 IP 直连)  : 通
www.baidu.com (DNS 解析)  : 通
```

---

## 3. 接入过程：五个坑

### 3.1 ICS 的 `192.168.137.1` 绑错了接口

Windows 的「Internet 连接共享」会把自己的私有侧固定成 `192.168.137.1/24`，并在上面起
DHCP 服务（UDP 67）和 DNS 代理（UDP 53）。

**坑在于它可能绑到虚拟网卡上。** 本机装了 Hyper-V 和 VMware，`Get-NetIPAddress` 出来的
接口列表是这样的：

```
vEthernet (Default Switch)    172.20.0.1         ← Hyper-V
VMware Network Adapter VMnet8 192.168.60.1
VMware Network Adapter VMnet1 192.168.217.1
本地连接* 10                    169.254.7.174     ← 带 * 的是隐藏虚拟适配器，169.254 是 APIPA
本地连接* 9                     169.254.107.10
以太网                          192.168.137.1     ← ICS 应该在这
WLAN                          192.168.110.81
```

`192.168.137.1` 得落在**物理以太网**那一条上。如果它跑到 `vEthernet (Default Switch)`
之类的虚拟口上，网线那头就永远拿不到地址 —— **物理网口看起来插好了，但没在参与共享。**

判别方法：看物理网口的收发计数。接上设备后 `ReceivedUnicastPackets` 应该开始涨；
一直是 0 就说明包根本没从这儿进来（ARP 是二层协议，防火墙拦不住它，所以这个计数很干净）。

### 3.2 发现设备：ARP + TTL

设备拿到 DHCP 地址后，用 ARP 表找它最直接：

```powershell
Get-NetNeighbor -InterfaceAlias '以太网' | Where-Object State -eq 'Reachable'
```

**TTL 是白送的操作系统指纹**，`ping` 回来就能读：

| TTL | 大概是 |
| --- | --- |
| 64 | Linux / Android / 嵌入式 |
| 128 | Windows |
| 255 | 网络设备 |

实测 `TTL=64` → 不是 Windows。再用名字解析类的手段交叉验证（NetBIOS / mDNS 都查不到
名字），可以进一步排除 Windows。这个判断后来被 `uname` 证实了。

反过来说：**只靠 TTL 只能排除 Windows，指不出是什么。** 真正定性的那一步是登进去
跑 `uname -m` 拿到 `riscv64` —— 在那之前「是台 Linux」就是能说的全部。

> **注意**：TTL 是从初始值逐跳递减的，同网段看到的就是初始值。跨路由后要按跳数还原。

### 3.3 端口探测为什么会自相矛盾

同一个 22 端口，先报**关**，后报**开**。这种矛盾几乎总是**超时设太短**：

| 方法 | 超时 | 结果 |
| --- | --- | --- |
| `TcpClient.BeginConnect` + `WaitOne(5000)` | 5 s | 关 |
| 批量扫描 35 个端口，逐个 `WaitOne(500)` | 0.5 s | 开 |

短超时会把「慢」当成「关」。但**光调长超时也不够** —— 真正可靠的是**协议层验证**：

```bash
ssh -o BatchMode=yes -o PreferredAuthentications=none root@<IP>
# → root@<IP>: Permission denied (publickey,password).
```

这一行输出信息量极大，它同时证明了四件事：

1. TCP 通了（不然是 `Connection refused` / `timed out`）
2. SSH 服务在跑，而且完成了密钥交换（拿到了 banner）
3. 服务端**接受密码认证**（方法列表里有 `password`）
4. 它拒绝了我们 —— 但**拒绝原因不在这行字里**，见下一节

**教训：判断一个服务在不在，别用「TCP 连得上吗」，要用「跟它说它的协议」。**

### 3.4 root 登不上：两种可能，怎么区分

上一节那句 `Permission denied (publickey,password)` 有**两种完全不同的成因**：

1. 密码错了
2. 密码对了，但**策略不允许 root 用密码登录**

而这两种情况**错误信息一模一样**，从报错上分不开。这是本次最容易误判的地方。

Ubuntu 出厂配置本来就偏向第 2 种：

- `PermitRootLogin` 默认是 `prohibit-password`
- 安装时**根本不设 root 密码**（`/etc/shadow` 里是 `!`），默认走 `sudo`

关键细节：**`prohibit-password` 下，`password` 仍然会出现在服务端广播的认证方法列表里。**
所以「看到 password 在列表里」推不出「root 能用密码登」。

**怎么一次性分开：换个普通用户试。**

```bash
ssh pingji@<IP> 'id'
# → uid=1000(pingji) ... 27(sudo)     ← 成功
```

普通用户进得去，说明**密码是对的**，root 被拒纯粹是策略。再去确认策略：

```bash
sudo passwd -S root    # L = 锁定，P = 有密码
```

**结论：不要为了「能登」去开 `PermitRootLogin yes`。** 普通用户在 `sudo` 组里，
`sudo` 就够用了，而开 root 密码登录是把自己的攻击面白送出去 ——
尤其在设备接回真实局域网之后，这个配置会被忘在那儿。

### 3.5 没有 `sshpass` 怎么自动登录

自动化场景下 `ssh` 会卡在密码提示上。常见解法是 `sshpass` 或 `plink`，但**这两个都不一定装**。
OpenSSH ≥ 8.4 自带一条正规路径：

```bash
# 1. 写一个 askpass 助手（从环境变量读，密码不落盘）
cat > /tmp/.ap.sh <<'EOF'
#!/bin/sh
printf '%s\n' "$SSH_PW"
EOF
chmod +x /tmp/.ap.sh

# 2. 强制走 askpass（SSH_ASKPASS_REQUIRE=force 是关键，否则只在无 tty 时才用）
SSH_PW='<密码>' SSH_ASKPASS=/tmp/.ap.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no \
    -o NumberOfPasswordPrompts=1 -o StrictHostKeyChecking=no \
    user@host '命令'
```

要点：

- **`SSH_ASKPASS_REQUIRE=force`** 是核心。只设 `SSH_ASKPASS` 的话，ssh 有 tty 就不会用它。
- **`DISPLAY=:0` 必须设**。askpass 机制源自 X11，没有 `DISPLAY` 时部分版本会直接跳过。
- **密码走环境变量而不是写进脚本**，脚本本身可以留在磁盘上也不泄密。
- **Git Bash 里没有 `setsid`**，不需要它 —— 直接跑就行。

> **排错技巧：** 助手脚本里加一句 `echo "$1" >> /tmp/mark`，事后数一下调用了几次，
> 就能区分「ssh 压根没问密码」和「问了但密码被拒」。这两种情况的报错也是一样的。

---

## 4. 可复用的检查清单

**要判断一台陌生设备是什么、能不能进：**

- [ ] `ping` → 通不通？**TTL 是多少**（64=Linux / 128=Windows）
- [ ] `Get-NetNeighbor` → MAC 是多少，OUI 能不能查到厂商
- [ ] 端口扫描**超时要给够**（≥2 s），且只用来筛候选
- [ ] 对筛出来的端口做**协议层验证**（`ssh -o BatchMode=yes`），不要只信 TCP 握手
- [ ] 认证失败时，**换个用户名再试一次** —— 区分「密码错」和「策略拒」
- [ ] 进去之后先 `id`，看有没有 `sudo` 组，能不碰 root 就不碰

**要看一台设备的网络实况：**

- [ ] `ip -br link` → 有几个口，谁是 UP
- [ ] 靠 `/sys/class/net/*/wireless` 判无线（不依赖装 `iw`）
- [ ] `ip route` → **门路是 metric 说了算，不是 UP 说了算**
- [ ] `iw dev <iface> link` → SSID / 频段 / 信号 / 协商速率
- [ ] `rfkill list` → 有没有被软/硬 blocking 的无线电（也能看出预留了哪些槽位）

**Windows 侧的坑：**

- [ ] Git Bash 里跑 `ipconfig` / `tasklist` / `ping`，中文输出全是乱码（本地代码页）
      → 改用 PowerShell 的结构化输出（`Get-NetIPAddress` / `Get-NetRoute`），标签保持英文
- [ ] ICS 私有侧的 `192.168.137.1` **必须落在物理网口**上，别落到 Hyper-V / VMware 的虚拟口
- [ ] 虚拟网卡会污染接口列表（Hyper-V、VMware、Wi-Fi Direct 的 `本地连接*`），
      挑物理口时按名字排除，别按顺序猜

---

## 5. 这次真正学到的一件事

一开始的假设是「设备没网，我共享给它」。实际是：

> **它一直有网，插网线换来的不是连通性，是延迟。**

如果只验证「能不能上外网」（`ping 223.5.5.5` 通），这个结论**永远不会浮出来** ——
因为它全程都是通的，只是慢。要看出差别，必须去看**它实际走哪条路**（`ip route` 的 metric）
和**每条路的成色**（`iw dev link` 的协商速率、ping 的 RTT）。

这和 [08-profiling.md](../kernel-opt/08-profiling.md) 是同一个道理，只是换了层：
**别问「它工作吗」，要问「它在哪一层、以什么代价工作」。**

---

## 参考

- `man ssh` 的 `SSH_ASKPASS` / `SSH_ASKPASS_REQUIRE` 段
- `man 5 sshd_config` 的 `PermitRootLogin` 段
- [Bianbu（进迭时空）文档](https://developer.spacemit.com/) —— RISC-V 发行版
- [rtw89 驱动](https://github.com/lwfinger/rtw89) —— Realtek RTL8852BE 的开源驱动上游

**相关篇章：** [07-edge.md](../kernel-opt/07-edge.md)（拿到设备之后，怎么在上面做优化）、
[08-profiling.md](../kernel-opt/08-profiling.md)（「先测量再下结论」的同一条原则）。
