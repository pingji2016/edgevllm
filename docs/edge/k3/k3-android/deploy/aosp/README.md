# AOSP（riscv64 / Cuttlefish on K3）· 云上编译机脚本

在**一次性**的 x86_64 Linux 云主机上编出能跑在 K3 上的 riscv64 镜像，然后交给 `deploy/k3/` 推上板子。
**方案、规格与配方依据见 [`docs/k3-android-cuttlefish.md`](../../k3-android.md) §5.1~§5.4**，这里只讲怎么操作。

> 一句话：**K3 只能当测试机，编不了自己**（上游明确 *AOSP does not support riscv64 as a build host*），
> 所以编译机必须是 x86_64，编完就把镜像搬回板子。

## 文件

| 文件 | 说明 |
| --- | --- |
| `deploy.sh` | 全部操作入口（见下方子命令）。**同一个脚本两种跑法**：在 Mac 上跑会自动 ssh 到编译机，在编译机上跑就本地执行 |
| `.env.aosp.example` | 配置模板；`cp .env.aosp.example .env.aosp` 后改（真实文件不入 git） |

`fetch` 拉回的产物默认落在 `<repo>/out/aosp/`（已被 `.gitignore` 的 `dist` 规则覆盖之外，注意别误提交）。

## 第 1 步：开一台机器

| 项 | 要求 | 备注 |
| --- | --- | --- |
| 架构 | **x86_64** | AOSP 不支持 riscv64 当构建主机 |
| 内存 | **32GB 起，64GB 舒服** | `soong_ui` 链接阶段吃内存，不够就疯狂 swap |
| 磁盘 | **500GB SSD**（别用 HDD） | 同步 + 构建合计 **约 400GB**（两个口径见「坑 1」），还要给 `ccache` 留。云盘选 **ESSD PL1**：同为 500GB，PL1 ≈ 26,800 IOPS / 350MB/s，PL0 只有 ≈ 7,800 IOPS / 180MB/s —— 编译是海量小文件 IO |
| 系统 | **Ubuntu 24.04 LTS**（22.04 也行，但 22.04 没 `openjdk-21-jdk`） | AOSP 官方支持的发行版 |
| 地区 | **优先境外**（对 googlesource / GitHub 直连快） | 省掉一整类网络问题；国内区域见下方「坑 3」。**入方向（下载）流量免费**，所以境内/境外在流量费上没有差别 |
| 带宽 | **按使用流量**，峰值拉到 **100Mbps** | ⚠️ 见下方「坑 5」：下载不要钱，但**速度被出网带宽锁死** |
| 计费 | **按量**，编完释放 | 盘只开**一块 500GB 系统盘**（不开数据盘）；复用走「手动快照 → 自定义镜像」，见「坑 6」 |

**开机后第一件事是 `check`，不是直接编** —— 磁盘不够会在 sync 跑到一半时炸，白等一小时：

```bash
bash deploy/aosp/deploy.sh check          # 本地跑：会提示你设 HOST
HOST=root@1.2.3.4 bash deploy/aosp/deploy.sh check
```

`check` 会打印架构 / 系统 / CPU / 内存 / 磁盘 / JDK / `repo` / **四个关键地址的连通性**，并给结论。
只读，不改任何东西。

## 第 2 步：一条龙

```bash
# 装依赖 + repo init + sync + 预装 APK + 打补丁 + m dist + 列产物
HOST=root@1.2.3.4 BG=1 bash deploy/aosp/deploy.sh all

# 看着它编（BG=1 的日志在编译机 ~/.aosp-deploy/logs/）
HOST=root@1.2.3.4 bash deploy/aosp/deploy.sh logs
```

**`BG=1` 强烈建议**：首次 `sync` 1~2 小时、首次 `m dist` 2~4 小时。
脚本把自身按**内容指纹**存到编译机（`~/.aosp-deploy/deploy-<哈希>.sh`），所以你在本地改脚本、
再 `BG=1` 起新任务，都**不会**影响正在跑的那个任务（这是踩过的坑：bash 是按字节偏移增量读脚本的）。
也可以给单个子命令加 `--bg`：`... deploy.sh sync --bg`。

只想分步走（推荐第一次这么干，好定位问题）：

```bash
HOST=root@1.2.3.4 bash deploy/aosp/deploy.sh branches   # 先看上游有哪些 branch / tag，决定 BRANCH
HOST=root@1.2.3.4 bash deploy/aosp/deploy.sh deps       # 依赖 + repo + git 身份
HOST=root@1.2.3.4 bash deploy/aosp/deploy.sh init       # repo init（会预检 BRANCH 存不存在）
HOST=root@1.2.3.4 BG=1 bash deploy/aosp/deploy.sh sync  # ~150GB
HOST=root@1.2.3.4 bash deploy/aosp/deploy.sh apks       # WebView / F-Droid（sha256 校验）
HOST=root@1.2.3.4 bash deploy/aosp/deploy.sh patch      # 上游对 AOSP 的补丁（幂等）
HOST=root@1.2.3.4 BG=1 bash deploy/aosp/deploy.sh build # lunch + m dist
HOST=root@1.2.3.4 bash deploy/aosp/deploy.sh artifacts  # 产物 + sha256
```

`BRANCH` 默认 **`v1.3.1`** —— 跟板子上正在跑的那个镜像同源（改起来有对照）。
要追最新开发分支就 `BRANCH=main`。

> ⚠️ **上游只有 `v1.2` / `v1.3.1` 两个 tag**（2026-10-14 实测）。上游 README 里示例写的
> `-b v1.3.2` **是不存在的**，照抄会在 `repo init` 直接失败 —— 所以 `init` 子命令先做
> `git ls-remote` 预检，tag 不对会当场列出可用值。
>
> 另外**有三条独立的版本线**，别混：
>
> | 版本线 | 仓库 | 当前值 | 管什么 | 谁编 |
> | --- | --- | --- | --- | --- |
> | `v1.3.x` | `aosp-cuttlefish-riscv64-addons` | **v1.3.1** | AOSP 镜像 | **我们（这个脚本）** |
> | `cf-k3-v1.3.x` | `android-cuttlefish` | cf-k3-**v1.3.2** | 宿主侧 cvd 包 | 上游发布，直接用 |
> | `6.18.3-cf-k3-1.3.0` | `linux-6.18-k3` | cf-k3-**v1.3.0** | K3 内核 | 我们（`kernel` 子命令） |

## 第 3 步：把镜像搬回板子

```bash
# 1) 拉回本机
HOST=root@1.2.3.4 bash deploy/aosp/deploy.sh fetch

# 2) 上板子并重启 guest（deploy/k3/ 那边的事）
bash deploy/k3/deploy.sh push-img out/aosp/aosp_cf_riscv64_phone_js-img-v1.3.1.zip
bash deploy/k3/deploy.sh stop && bash deploy/k3/deploy.sh run

# 3) 验收
bash deploy/k3/deploy.sh smoke
bash deploy/k3/deploy.sh shot
```

`push-img` 会把板子上原来的 `cf-data/product` 备份成 `product.bak-<时间戳>`，出问题 `mv` 回去就行。
**增量编译只要几分钟**，所以「改一行 → `m dist` → fetch → push-img → 重启」这个循环是能接受的。

## 另一条线：内核

内核**可以在同一个 x86_64 编译机上交叉编译，约 10 分钟**，不用动 AOSP 那棵树：

```bash
HOST=root@1.2.3.4 bash deploy/aosp/deploy.sh kernel
```

产出 `linux-image-6.18.3-cf-k3-1.3.0-*.deb`，用 `deploy/k3/deploy.sh kernel` 那套装法上板子（阶段 1 走的就是这条路）。
**板子上旧内核会保留**，`/boot/env_k3.txt` 指回去就能回滚。

## 子命令速查

| 子命令 | 干什么 | 支持 `BG=1` |
| --- | --- | --- |
| `check` | 体检：架构 / 系统 / 内存 / 磁盘 / JDK / 网络 | |
| `deps` | 装 AOSP 构建依赖 + `repo` + `ccache` + git 身份（幂等） | ✓ |
| `branches` | 列出上游可用 branch / tag | |
| `mirror-on` / `mirror-off` | 只把 `android.googlesource.com` 重定向到 TUNA | |
| `init` | `repo init`（预检 BRANCH 存在性） | |
| `sync` | `repo sync`（~150GB） | ✓ |
| `apks` | 下载预装 APK（WebView / F-Droid） | |
| `patch` | 打上游对 AOSP 的补丁（幂等） | |
| `build` | `lunch` + `m dist` | ✓ |
| `all` | 上面全跑一遍 | ✓ |
| `artifacts` | 列产物 + sha256 | |
| `kernel` | 交叉编译 K3 定制内核 deb | ✓ |
| `status` | 树 / 分支 / 上次产物 / 磁盘 / 后台任务 | |
| `logs` | 打并跟最新一条后台日志 | |
| `fetch` | （Mac 侧）把产物 zip 拉回本机 | |
| `clean` | 删源码树 / 内核树（要确认） | |

## 四个坑

1. **磁盘别省。** `repo sync` 卡在「磁盘满」上是最经典的翻车方式。`check` 里 <350GB 直接判死刑是有理由的。
   **两个口径不一致，别卡着下限开**：
   - 上游 addons 的 README 给的是「同步 ~150GB、编完 ~320GB」；
   - **AOSP 官方规格页给的是「至少 400GB 可用空间 = 250GB 检出 + 150GB 构建」**
     （`source.android.com/docs/setup/start/requirements`，检出含 `.repo` 里的 git 对象）。

   所以**按 500GB 给余量**，`check` 会实测告诉你这台机器上的真实占用。宁可盘大一点，
   或者编完立刻把产物 `fetch` 走、把盘做快照再释放机器。

2. **JDK 要 21。** Android 16+ 的 `trunk_staging` 要 `openjdk-21-jdk`。Ubuntu 24.04 源里就有，
   22.04 源里没有（要自己加源）—— 所以**优先开 24.04**。`check` 会告诉你当前是不是 21。

3. **国内区域才需要折腾网络，而且只折腾一半。**
   - `repo init` 走的 manifest 仓库在 **GitHub**；manifest 里的 AOSP 项目在 **`android.googlesource.com`**
     （1085 个 project 里只有 1 个在 GitHub 上，其余全在 googlesource）。
   - 国内通常是 googlesource 慢/不通，GitHub 还行 → 用 `mirror-on`（**只**重定向 googlesource 到 TUNA，
     **绝不**把 GitHub 一起重定向，manifest 里有 GitHub 上的项目）。
   - **`apks` 那一步的两个下载源也在境外**：`media.githubusercontent.com`（145MB WebView）、
     `f-droid.org`（12MB F-Droid）。`fetch-apks.sh` 按 `prebuilts/apk-pins.tsv` 的 `bytes + sha256`
     校验，下不动就卡在这里。
   - 所以两个都慢 → 给 `PROXY=socks5h://...`（脚本会把 `http_proxy/https_proxy/all_proxy` 和 git 的
     `http.proxy` 都设上；`git ls-remote`、`repo`、`curl` 全都跟着走）。
   - 最省事的仍然是**直接选境外区域**：`check` 四个地址全绿就什么都不用配。

4. **`--bg` 的日志在编译机上，不在你本机。** `logs` 子命令会自动 ssh 过去打最新的那条。
   日志目录 `~/.aosp-deploy/logs/`。想实时跟就直接 `ssh` 上去 `tail -f`。

5. **带宽别选小 —— 这条最隐蔽，它直接决定「能不能编」。**
   - 云厂商口径（阿里云官方原文）：**只对出方向流量计费，入方向不计费** ——「在 ECS 实例内部从公网下载资源」
     正是入方向。我们那 150GB 全是**下载**，所以**不花一分钱**。CDT 另有免费额度
     （中国内地 20GB/月、非中国内地 200GB/月，**仅按出方向计算**），而我们的出方向只有 apt 元数据
     加上 `fetch` 拉回的 ~1GB 产物，基本落在免费额度内。
   - **但是**：入网带宽的上限**受出网带宽钳制** —— 出网带宽 <10Mbps 时，入网带宽最大也只有 10Mbps。
     购买页默认常是「按固定带宽 1M / 5M」，那样下载上限只有 1.25MB/s，**150GB 要下 33 小时**。
   - 所以下单时：**带宽计费选「按使用流量」，带宽峰值拉到 100Mbps**。下载免费且能跑 ~12.5MB/s（150GB ≈ 3.3 小时）。
   - 顺带纠正一个直觉：**挂代理并不能省流量费**。经代理下载，数据到编译机网卡上仍然是**入方向**，一样免费；
     走代理真正的代价是**你那台出口 VPS 要出方向吐 150GB**，吃它的月流量和带宽。所以大树交给 `mirror-on`，
     代理只兜境外那 ~160MB。

6. **盘怎么开：一块 500GB 系统盘就够，不用买数据盘。**
   - **够用**：同步 + 构建 ~400GB，500GB 系统盘即可（ESSD PL1 单盘 20~65,536 GiB，500 完全合法）。
   - **复用走「手动快照 → 自定义镜像」**：自定义镜像会为源实例的**每块云盘（系统盘 + 数据盘）**各建一份快照，
     所以镜像里 **OS + apt 依赖 + `repo` 工具 + 源码树 + `out/`** 全都在 —— 比数据盘方案更彻底。
     下次用镜像建实例 = 环境原样回来，直接 `sync` + `build` 增量。
   - **系统盘能卸下来挂给新机器吗？不能。** 阿里云**支持**离线卸载系统盘（实例需先停止），但卸载后只能挂回
     **原实例**（要求它处于「无系统盘」状态），或作为**数据盘**挂到别的实例做迁移/修复 ——
     **不能**挂给另一台新机器当系统盘启动。而且系统盘默认开启「随实例释放」，源实例一释放它照样被释放。
   - **数据盘什么时候才值得加**：只有你打算**频繁换机型/配置**（8C32G → 16C64G → 抢占式多核）时才值。
     **单独购买**的按量数据盘能在**同一可用区**内自由卸载/挂载（注意是**可用区**，不是地域）。代价三条：
     ① 卸载前必须先把 `/etc/fstab` 里那行注释掉，否则「实例重启后你将无法连接实例」；用镜像建实例时
     fstab 里若留着数据盘挂载项，新实例会**无法启动**；
     ② **随实例创建**的数据盘默认「随实例释放」，只有**单独购买**的默认不随释放；
     ③ 挂上之后要手动分区 + 格式化才能用。
   - **两个别忘**：① 用**手动快照** —— 手动快照不受云盘释放影响，而**自动**快照会被「自动快照随云盘释放」捎带删掉；
     ② **云盘不支持缩容**，别一次性开太大（开小了以后可以在线扩容）。

## 能不能复用别人编好的系统盘 / 环境？

**不能 —— 唯一能复用的是你自己留下的那份。** 推理链：

- **官方 CI 有 riscv64 的 target**：`ci.android.com` 上的 `aosp_cf_riscv64_phone-trunk_staging-userdebug`。
  但它发布的只有 **`*-img-<buildid>.zip`（~830MB）+ `cvd-host_package.tar.gz`（~215MB）** 这两样 —— 是**产物**，
  不是**编译环境**。上游 `monkey-jsun` 发布的也是同样两样（**板子上现在跑的就是那份**）。
  拿这些能跳过编译，但它们恰恰是你要**改**的对象。
- **没有「预装 AOSP 编译环境」的盘可下载**：环境 = 源码树（150~250GB）+ `out/`（150~320GB），合计约 400GB，
  没有分发渠道，也没有人会给你传。
- **Docker 版的「AOSP 编译环境」镜像省不了你要的时间**：`Trumeet/AOSP-Build-Environment-Docker`、
  `jftr/aosp-build-env`、`mrenouf/android_build` 这类镜像只有 **~334MB**，装的只是**依赖 + `repo` 工具**，
  源码与 `out/` 必须挂容器外。**只省 `deps` 那 ~10 分钟**，省不了 sync（150GB）和 build（2~4h）。
- **`out/` 与源码树的绝对路径绑定**：AOSP 官方 best practice **明令禁止**把绝对路径写进 ninja 文件或中间产物
  （原文：*Don't encode absolute paths in build intermediates or outputs. This would make it difficult to
  reproduce builds on other machines.*）—— 也就是说它**想**让你能移动树。但实践上仍有地雷：移动后
  `bootable/bootloader/edk2/Conf/BuildEnv.sh` 这类自动生成、写死旧路径的文件要手工删掉。
  所以就算真拿到别人的 `out/`，也得放在**完全相同的路径**下才可能增量（本脚本 `TREE` 相对 `$HOME`，
  固定 `root` + `android17-release` 就是为了这个）。

**所以「第一次」没有捷径，但可以压缩**：`mirror-on` 扛 sync（省最多）、带宽按使用流量 + 峰值 100Mbps、
核数与内存给足、开 `ccache`（对**第二次起**有效）。真正的复用是**把 `out/` 留在盘上**（快照 / 数据盘 /
自定义镜像），之后每次都是增量几分钟。

## 与其它脚本的关系

```
deploy/aosp/   ← 编（x86_64 云机，产镜像 zip / 内核 deb）
      │  fetch
      ▼
deploy/k3/     ← 跑（板子上的 Cuttlefish）push-img / stop / run / smoke / shot
```

`deploy/aosp/` 不碰板子，`deploy/k3/` 不编东西，两边只通过一个 zip（或 deb）交接。
另外 `deploy/k3/deploy.sh` 那条链路的网络问题（板子到 GitHub 的 TLS 被掐）跟这里无关：
编译机自己拉源码，不经过板子所在网络。
