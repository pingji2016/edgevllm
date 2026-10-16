# K3 · Android（Cuttlefish）部署脚本

在 `pingji-k3`（SpaceMiT K3 / RISC-V）上把 Android 跑起来：浏览器 WebRTC 看屏幕、`adb` 连上去。
**方案设计与依据见 [`docs/k3-android-cuttlefish.md`](../../k3-android.md)**，这里只讲怎么操作。

> 一句话：这不是 Google 官方那个 AVD 模拟器，是 **Cuttlefish + crosvm + KVM** 跑 riscv64 的 AOSP guest，
> 上游是社区项目 [`monkey-jsun/aosp-cuttlefish-riscv64`](https://github.com/monkey-jsun/aosp-cuttlefish-riscv64)。

## 接入路径

板子在内网 `192.168.110.194`，本机只能经 111 进：

```
本机 --ssh--> 111.228.40.211 --docker exec k3-ttyd--> 板子(127.0.0.1:2222)
```

板子直连（同局域网）时在 `.env.k3` 里设 `K3_DIRECT=1`。

## 文件

| 文件 | 说明 |
| --- | --- |
| `deploy.sh` | 全部操作入口（见下方子命令） |
| `.env.k3.example` | 配置模板；`cp .env.k3.example .env.k3` 后改（真实文件不入 git） |
| `tunnel/android.conf.example` | 反隧道 systemd drop-in 的参考样本（**本环境用不了**，见下方坑 3；实际用 `expose`） |
| `xray/config.json.example` | 板子代理（Xray）配置模板，**脱敏**版；真实配置放同目录 `config.json`（不入 git） |
| `xray/dl/` | `vpn-dl` 下载的 Xray riscv64 二进制 + geo 数据（不入 git） |
| `snapshots/` | `snapshot` 子命令产出的板子现状快照（不入 git） |

## 前置：sudo

板子上 `pingji` 在 `sudo` 组但**不免密**。二选一：

```bash
echo '你的密码' > ~/.k3-sudo && chmod 600 ~/.k3-sudo   # 脚本自动读
# 或者不配 —— 脚本会把该执行的命令原样打印出来，你自己在板子终端里敲
```

## 用法

```bash
# —— 阶段 0：体检与留底（只读，不改任何东西）——
bash deploy/k3/deploy.sh check        # 内核/KVM/vhost/docker/资源/端口 一次看全
bash deploy/k3/deploy.sh snapshot     # 存现状快照到 snapshots/，回滚时对照

# —— 阶段 1：宿主准备（需要 sudo；会重启一次板子）——
bash deploy/k3/deploy.sh prep         # 装 Docker + 下载 K3 定制内核 deb（不动内核）
bash deploy/k3/deploy.sh kernel       # 装内核（会先扒 deb 里的 config 做预检，再问一句）
bash deploy/k3/deploy.sh reboot       # 重启，并自动等板子回来
bash deploy/k3/deploy.sh mods         # ★ 必跑：加载 kvm/vhost_vsock 并持久化（新内核 KVM 是模块，不加载就没 /dev/kvm）
bash deploy/k3/deploy.sh verify-host  # ★ 关键验收：内核 / /dev/kvm / docker / 隧道

# —— 阶段 1.5：板子代理（必须在拉镜像之前）——
bash deploy/k3/deploy.sh vpn-dl       # 本机下载 Xray riscv64 + geo 数据
bash deploy/k3/deploy.sh vpn-install  # 装到板子并起 systemd 服务（127.0.0.1:10808）
bash deploy/k3/deploy.sh vpn-test     # 验证：github / ghcr / 出口 IP / 代理下 git clone
bash deploy/k3/deploy.sh proxy-on     # docker daemon + git 走代理（不装这个 docker pull 会随机失败）
bash deploy/k3/deploy.sh vpn-status   # 代理状态一览
bash deploy/k3/deploy.sh proxy-off    # 撤掉 docker/git 代理
bash deploy/k3/deploy.sh vpn-rm       # 停并移除板子上的代理服务

# —— 阶段 2/3：部署与起机 ——
bash deploy/k3/deploy.sh deploy       # clone 上游 + pull cf-host-k3 镜像（~2.1GB）
bash deploy/k3/deploy.sh prefetch     # 宿主侧预取 cvd-host 包 + AOSP 镜像（~1GB，走代理）★建议先跑
bash deploy/k3/deploy.sh run          # 起 Cuttlefish（后台派生，日志落 cf-run.log）
bash deploy/k3/deploy.sh smoke        # 验收：WebRTC 端口 + guest ABI（容器内直连，不用隧道）
bash deploy/k3/deploy.sh shot         # 抓一张 guest 的屏回本机（不需要隧道/浏览器）
bash deploy/k3/deploy.sh push-img F.zip  # ROM 开发循环：把新编的镜像 zip 换到板子上（见文末）

# —— 阶段 4：对外访问（在 111 上做正向转发，★不动板子的隧道）——
bash deploy/k3/deploy.sh expose       # 建 111:/etc/systemd/system/k3-android-fwd.service
# 本机另开一个窗口常驻：
#   ssh -N -L 8443:127.0.0.1:18443 -L 6520:127.0.0.1:16520 root@111.228.40.211
#   浏览器 https://127.0.0.1:8443/     adb connect 127.0.0.1:6520

# —— 日常 ——
bash deploy/k3/deploy.sh status       # 容器 / 端口 / 隧道 一句话
bash deploy/k3/deploy.sh logs         # Cuttlefish 日志（容器 + cf-run.log）
bash deploy/k3/deploy.sh stop         # 停 Cuttlefish（省资源）
bash deploy/k3/deploy.sh unexpose     # 撤掉 111 上的正向转发
bash deploy/k3/deploy.sh rollback     # 停 Cuttlefish + 停 docker
```

## 阶段与验收

| 阶段 | 命令 | 通过标准 |
| --- | --- | --- |
| 0 | `check` / `snapshot` | 快照落盘；`/boot` 余量够；111 上 18443/16520 空闲 |
| 1 | `prep` → `kernel` → `reboot` → `mods` → `verify-host` | `uname -r` = `6.18.3-cf-k3-*`；**`/dev/kvm` 存在**（靠 `mods`）；`docker info` 可用 |
| 1.5 | `vpn-install` → `vpn-test` → `proxy-on` | 代理下 `github` 200 / `git clone` 成功 / `docker pull hello-world` 成功 |
| 2 | `deploy` → `prefetch` → `run` | `docker ps` 有 `cf`（label `cf-host`）；`VIRTUAL_DEVICE_BOOT_COMPLETED` |
| 3 | `smoke` / `shot` | `abi=riscv64` / `release=17`；`shot` 抓到 720×1280 的真实锁屏图 |
| 4 | `expose` | 111 上 `127.0.0.1:18443/16520` 有监听、`https://127.0.0.1:18443/client.html` 200、**且 K3 门户 `:3000` 不受影响** |

## ⚠️ 四个必须知道的坑

1. **`CONFIG_KVM=m` → 开机没有 `/dev/kvm`。** 新内核把 KVM 编成模块（上游有意为之），
   **每次重启后都要 `modprobe kvm`**。跑一次 `deploy.sh mods` 即可（它写 `/etc/modules-load.d/k3-android.conf`，之后开机自动加载）。
   `verify-host` 里 `/dev/kvm` 是 `❌` 就说明忘了这步。
2. **别照抄上游 README 的内核要求。** 上游 README 写"需要 `CONFIG_VHOST_NET`"，但扒开**它自己发布的**内核 deb 看，
   `VHOST_NET` 也是 `is not set`。实测 `UDMABUF=y`（GPU 加速 ✅）、`KVM_RISCV_VTVM=y`、`VSOCKETS=m`。
   `VHOST_NET` 缺**只影响 guest 自己联网**，不影响起机 / WebRTC 看屏 / `adb`（走 vsock）。
   **不要为这一项去重编内核。**
3. **反隧道在这套环境里注定失败，用 `expose`。** 111 的 `/etc/ssh/sshd_config.d/99-k3.conf` 里是
   `PermitListen 127.0.0.1:2222` —— **白名单**，sshd 只准客户端绑 2222。加任何 `-R` 都会报
   `remote port forwarding failed for listen port 18443`；而 `k3-tunnel` 带 `ExitOnForwardFailure=yes`，
   硬改会把 **2222（进板子的唯一通道，门户终端也走它）一起打挂**，远程救不回来。
   → `expose` 改成**在 111 上做 `ssh -L`**（`k3-ttyd` 是 host 网络），**不动板子、不动 sshd、风险为 0**。
   （`tunnel-add` 保留但加了预检，会在动手前就失败退出。）
4. **板子的国际线路是"HTTP 通、TLS 被掐"。** 实测 `git clone https://github.com/...` 报
   `GnuTLS recv error (-110)`，而 `curl https://github.com/` 却返回 200。拉 2GB 镜像 / 下 AOSP 镜像必踩。
   → **先做阶段 1.5 的代理**，并 `proxy-on` 把 docker daemon 也指过去（Docker Hub `registry-1.docker.io` 本来就被墙）。

## 回滚

```bash
bash deploy/k3/deploy.sh rollback            # 停 Cuttlefish + 停 docker
bash deploy/k3/deploy.sh unexpose            # 撤掉 111 上的正向转发（k3-android-fwd）
# 换回原内核（板子上，注意 dpkg 会把旧内核留着）
sudo dpkg -i ~/k3-android/pkg/linux-image-6.18.3-generic_*.deb   # 或 apt install --reinstall
sudo reboot
```

最坏情况（新内核起不来）：用**串口**（板子有 `ttyS0` getty）或在 U-Boot 里选回旧内核。
板子的引导配置在 `/boot/env_k3.txt`。

## 注意

- **`snapshot` 的内容含板子网络/单元文件信息，不入 git**（`.gitignore` 已忽略 `snapshots/`）。
- sudo 密码、`.env.k3` 都不入 git。
- **代理的真实配置 `xray/config.json` 含服务器地址 + UUID，不入 git**（仓库里只有 `config.json.example`）。
- **容器首次运行要从 GitHub 下 ~1GB**（`cvd-host` 包 + AOSP 镜像）。容器里的 `127.0.0.1` 不是板子，
  用不到板子的代理 → **先在宿主侧跑 `prefetch`**，按 `entrypoint.sh` 期望的布局把文件放进
  `k3-android/cuttlefish-host-container/cf-data/{host,product}`，容器一看文件在就直接起 guest。
- 上游是**单人维护的实验项目**；`.env.k3` 里建议把内核 deb 版本和镜像 tag **pin 死**，别跟 `latest`。
- 起满 Cuttlefish 会明显抢占板子资源（板子上还有 `spacemit-ai-gateway` 和桌面那一套），**用完 `stop`**。

## ROM 定制开发（编译在别处，K3 只当测试机）

**AOSP 不能在 riscv64 上编，K3 当不了编译机。** 编译必须在一台 **x86_64 Linux**（32GB+ 内存、
500GB SSD）上做；完整配方与命令见 [`docs/k3-android-cuttlefish.md`](../../k3-android.md) §5。

开发循环：

```bash
# 在 x86_64 编译机上 —— 别手敲，用专门的驱动（在 Mac 上跑也行，它会 ssh 过去）
HOST=root@x.x.x.x BG=1 bash deploy/aosp/deploy.sh all
HOST=root@x.x.x.x bash deploy/aosp/deploy.sh fetch      # 产物拉回 out/aosp/
#   -> out/aosp/aosp_cf_riscv64_phone_js-img-v1.3.1.zip

# 拿到本机后，换到板子上
bash deploy/k3/deploy.sh push-img out/aosp/aosp_cf_riscv64_phone_js-img-v1.3.1.zip
bash deploy/k3/deploy.sh stop && bash deploy/k3/deploy.sh run
```

> 上游 addons 仓库只有 `v1.2` / `v1.3.1` 两个 tag（`v1.3.1` = 板子上现在跑的这版），
> 上游 README 里示例的 `-b v1.3.2` **并不存在**；`deploy/aosp/deploy.sh init` 会先预检。
> 编译机怎么开、每一步多久、三个坑，见 [`deploy/aosp/README.md`](../aosp/README.md)。

`push-img` 会把旧的 `cf-data/product` 备份成 `product.bak-<时间戳>`，出问题 `mv` 回去就能回滚。

> 只改 framework / 系统层的话**内核和 crosvm 都不用重编**（`m dist` 增量几分钟）。
> 内核单独定制也不贵：x86_64 上交叉编译约 10 分钟，装法同阶段 1。
