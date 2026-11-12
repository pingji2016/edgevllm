# pingji-MS-7B89 环境说明

> 主机：`pingji-MS-7B89` · Ubuntu 26.04.1 LTS · 内核 7.0.0-30-generic · x86_64
> CPU：AMD Ryzen 5 3600（6 核 / 12 线程）· 内存 30.3 GB · 系统盘 `/dev/sda2` 233 G
> 访问方式：111 反隧道 `127.0.0.1:22022` → `ssh -p 22022 pingji@127.0.0.1`
> 本文档最后更新：2026-11-12

---

> 📁 **桌面文档索引**
> - `⚠️重要提示-磁盘安全.md` — NVMe 只读红线（先看这个）
> - `环境说明.md` — 本文件，全部环境/版本/路径
> - `3D重建使用说明.md` — 3DGS 详细用法与踩坑
> - `3DGS-乐高场景/` — 示例产物（转台视频、预览图、.ply）
> - `ComfyUI/` — 出图输出目录
> - `启动ComfyUI.desktop` — 双击启动出图服务


## 0. ⚠️ 磁盘红线（先看这个）

**两块 NVMe 盘全是 Windows 的数据，只能读、绝对不能写。**

| 设备 | 内容 | 允许 |
| --- | --- | --- |
| `/dev/sda`（sda1 1 G + sda2 237 G） | Linux 系统盘 | ✅ **唯一可读写** |
| `/dev/nvme0n1`（16 M / 232 G / 812 M） | Windows 系统盘 | ❌ 只读 |
| `/dev/nvme1n1p1`（300 G） | Windows 数据盘 | ❌ 只读 |
| `/dev/nvme1n1p2`（631 G） | 数据盘，只读挂 `/mnt/v2` | ❌ 只读 |

禁止：`mkfs.*` `parted` `fdisk` `sgdisk` `wipefs` `dd of=/dev/nvme*`
`ntfsfix` `ntfsresize`、不带 `-o ro` 的 `mount`、以及文件管理器里对 `/mnt/v2`
的删除/移动/改名（都会写盘）。

正确姿势：`sudo mount -o ro /dev/nvme1n1p2 /mnt/v2` → `cp` 到 `~/` 再用 → `sudo umount /mnt/v2`。

> 详细版（原因、误写后果、自查方法）见 `~/桌面/⚠️重要提示-磁盘安全.md`（副本在 `~/docs/`）

---

## 1. 磁盘

| 设备 | 大小 | 文件系统 | 挂载点 | 说明 |
| --- | ---: | --- | --- | --- |
| `/dev/sda1` | 1 G | vfat | `/boot/efi` | EFI |
| `/dev/sda2` | 237 G | ext4 | `/` | 系统盘，剩 ~195 G |
| `/dev/nvme0n1p2` | 232 G | ntfs | 未挂载 | Windows C: |
| `/dev/nvme1n1p1` | 300 G | ntfs | 未挂载 | Windows 数据 |
| `/dev/nvme1n1p2` | 631 G | ntfs | `/mnt/v2`（**只读**） | 99% 满，剩 12 G |

> `/mnt/v2` 是手动只读挂载，重启后失效：
> `sudo mount -o ro /dev/nvme1n1p2 /mnt/v2`

### 内存 / swap

| 项 | 值 |
| --- | --- |
| 物理内存 | 30.3 GB |
| swap | **16 G 总计**：`/swap.img` 8 G（系统安装时自带）+ `/swapfile` 8 G（本次新增） |
| swappiness | **10**（`/etc/sysctl.d/99-swap.conf`） |
| fstab | `/swapfile none swap sw,pri=10 0 0` |

> 只想要 8 G 的话：`sudo swapoff /swapfile && sudo rm /swapfile`，并删掉 fstab 里那一行。

---

## 2. 基础软件

| 软件 | 版本 | 位置 |
| --- | --- | --- |
| git | `1:2.53.0-1ubuntu1.1` | `/usr/bin/git` |
| wget | `1.25.0-2ubuntu4.4` | `/usr/bin/wget` |
| curl | `8.18.0-1ubuntu2.8` | `/usr/bin/curl` |
| unzip | `6.0-29ubuntu1` | `/usr/bin/unzip` |
| jq | `1.8.1-4ubuntu2` | `/usr/bin/jq` |
| openssh-server | `1:10.2p1-2ubuntu3.8` | `/usr/sbin/sshd` |
| fonts-noto-cjk | `1:20240730+repack1-1build1` | `/usr/share/fonts/opentype/noto/` |
| ibus-libpinyin | `1.16.5-2` | ibus 拼音引擎 |
| build-essential | `12.12ubuntu2.26.04.2` | gcc **15.2.0** / g++ / make |
| pkg-config | `2.5.1-4` | `/usr/bin/pkg-config` |
| python3-pip | `25.1.1+dfsg-1ubuntu2` | 系统 Python 3.14 的 pip |
| python3-venv | `3.14.3-0ubuntu2` | `python3 -m venv` |
| python3-dev | `3.14.3-0ubuntu2` | Python 扩展头文件 |
| tmux | `3.6a-2ubuntu0.2` | `/usr/bin/tmux` |
| htop | `3.4.1-5build2` | `/usr/bin/htop` |
| ncdu | `1.22-1build1` | `/usr/bin/ncdu` |
| ripgrep | `15.1.0-1ubuntu1` | `/usr/bin/rg` |
| fd-find | `10.3.0-2ubuntu1` | `/usr/bin/fdfind`（注意不是 `fd`） |
| tree | `2.3.1-1` | `/usr/bin/tree` |
| rsync | 已装 | `/usr/bin/rsync` |
| lm-sensors | `1:3.6.2-2build1` | `/usr/bin/sensors` |
| nvme-cli | `2.16-1` | `/usr/sbin/nvme` |
| smartmontools | `7.5-2` | `/usr/sbin/smartctl` |
| net-tools | `2.10-2ubuntu1` | `ifconfig` 等 |
| dnsutils / mtr-tiny | 已装 | `dig` / `mtr` |

### 时区与时间同步

| 项 | 值 |
| --- | --- |
| 时区 | `Asia/Shanghai`（CST, +0800） |
| NTP | **chrony.service**（active/running），时钟已同步 |
| 系统时钟 | `System clock synchronized: yes` |

> `systemd-timesyncd` inactive 属正常：Ubuntu 桌面板默认用 chrony，二选一。

---

## 3. Python 环境（uv）

用 **uv** 管多版本 Python 和项目依赖，不往系统 Python 里装包（PEP 668 保护）。

| 项 | 值 |
| --- | --- |
| uv 版本 | **0.13.7**（`x86_64-unknown-linux-gnu`） |
| 二进制 | `~/.local/bin/uv`（全局软链 `/usr/local/bin/uv`、`/usr/local/bin/uvx`） |
| PATH 配置 | `/etc/profile.d/uv.sh` |
| 已装 Python | **3.12.16**、**3.11.18**（uv 托管） |
| Python 存放 | `~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/` |
| 系统 Python | `3.14.4`（`/usr/bin/python3`，**不要往里装包**） |
| uv 缓存 | `~/.cache/uv` |

```bash
cd <项目>
uv venv --python 3.12          # 有 pyproject.toml 时直接 uv sync
uv pip install -r requirements.txt
uv run python main.py          # 不用手动 activate
uv python install 3.13         # 需要别的版本随时加
uvx ruff check .               # 临时跑工具
```

冒烟测试：`uv venv --python 3.12` + `uv pip install numpy` → `python 3.12.16 | numpy 2.5.4 | 60 ms`。

> 在 `/tmp`（tmpfs）建 venv 会提示 `UV_LINK_MODE=copy`（跨文件系统无法硬链接），
> 项目放 `~/` 下没这个问题。

---

## 4. Java 环境

| 项 | 值 |
| --- | --- |
| JDK 21（默认） | **21.0.13.0.3** · `/usr/lib/jvm/java-21-openjdk-amd64` |
| JDK 17 | **17.0.21.0.3** · `/usr/lib/jvm/java-17-openjdk-amd64` |
| 默认 `java` | 21（`update-alternatives` 自动模式） |
| 切换脚本 | `/etc/profile.d/java.sh`（导出 `JAVA_HOME` + `jdk` 函数） |
| sdkman | **5.24.0 / native 0.8.0** 已装（`~/.sdkman`），**暂未用它装 JDK** |

```bash
jdk 17        # 切到 17（当前 shell 生效）
jdk 21        # 切回 21
jdk           # 不带参数则列出可用版本
```

> 为什么不用 sdkman 装 JDK：sdkman 的启动健康检查不走代理（会报
> `INTERNET NOT REACHABLE`），且实际下载走 curl 时常被中断
> （`OpenSSL SSL_read: unexpected eof`）。本机 JDK 改由 apt 装，稳。
> sdkman 已装好备用（`sdkman_healthcheck_enable=false` 已写进
> `~/.sdkman/etc/config`），网络好时可用 `sdk install java <id>`。

---

## 5. VPN / 代理（Xray）

用 v2rayN 安装包自带的 Xray 内核跑**无界面**服务（v2rayN 本体是 GUI，
只有登录桌面才能开，SSH 场景用不了）。

| 项 | 值 |
| --- | --- |
| v2rayN 桌面版 | `7.19.5`（`/opt/v2rayN/v2rayN`，菜单名 v2rayN，需图形会话） |
| 安装包来源 | `/mnt/v2/do/v2rayN-linux-64.deb`（64 MB） |
| **Xray 内核** | **26.2.6**（`/opt/v2rayN/bin/xray/xray`） |
| 配置文件 | `/etc/xray/config.json`（root 600） |
| 服务单元 | `/etc/systemd/system/xray.service` |
| 监听 | `127.0.0.1:10808`（mixed = SOCKS5 + HTTP 同端口） |
| 出口节点 | `43.155.215.179:443`（VLESS + raw/http 伪装，Host: www.bing.com） |
| 分流 | `geosite:google` 走代理；`geoip:cn` / `geosite:cn` / 私有地址直连 |

> ⚠️ `/etc/xray/config.json` 含 VLESS UUID，**不要提交到公开仓库**。

```bash
sudo systemctl status xray         # 状态
sudo systemctl restart xray        # 改配置后重启
sudo journalctl -u xray -n 50      # 日志
```

### shell 里用代理

`/etc/profile.d/proxy.sh`：`proxy_on` / `proxy_off` / `proxy_ip`。

```bash
proxy_on && proxy_ip     # 输出 43.155.215.179
```

单条命令：`https_proxy=http://127.0.0.1:10808 wget -qO- https://api.ipify.org`

---

## 6. 已知网络问题（重要）

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `apt install` 报「网络不可达 / 连接超时」 | 解析到 IPv6 地址，本机 IPv6 不通 | 加 `-o Acquire::ForceIPv4=true` |
| 装 `security.ubuntu.com` 上的包超时 | 该域名国内直连不通 | 加 `-o Acquire::http::Proxy=http://127.0.0.1:10808 -o Acquire::https::Proxy=http://127.0.0.1:10808` |
| sdkman 报 `INTERNET NOT REACHABLE` | 健康检查不走代理 | 已关掉健康检查；下载仍建议开代理 |
| 反隧道偶发断连（111 上 22022/2222 同时消失） | 111 侧网络抖动 | 10–30 s 自动恢复，重试即可 |

推荐的 apt 完整写法：

```bash
sudo apt-get -o Acquire::ForceIPv4=true \
  -o Acquire::http::Proxy=http://127.0.0.1:10808 \
  -o Acquire::https::Proxy=http://127.0.0.1:10808 \
  install -y <包名>
```

---

## 7. 能力清单 / 待办

### 已装好、可直接用

| 能力 | 位置 | 实测 |
| --- | --- | --- |
| 本地大模型（llama.cpp） | `~/apps/llama-vulkan` + `~/models` | Qwen3-8B Q4 **74 tok/s** |
| 出图（ComfyUI） | `~/apps/comfyui` | SDXL 10.3 s/张，Flux **7.8 s/张** |
| 图生图 / 局部重绘 | `~/apps/comfyui` + `inpaint.py` | **8.4 s**，遮罩外像素差 0 |
| 3D 重建（3DGS） | `~/apps/3dgs` | 1 万迭代 **约 2 分钟** |
| Python 多版本 | `uv` + 3.11/3.12 | — |
| Java | JDK 17 / 21（`jdk 17`） | — |

### 待办 / 未装

| 项目 | 说明 |
| --- | --- |
| Docker / Podman | 用户暂缓 |
| `tailscale` | 可选，远程访问比反隧道省心 |
| `conda/miniforge` | 需要 cudatoolkit 等非 Python 依赖时 |
| 语音全链路（Whisper/CosyVoice） | 评估过可行，未安装 |
| 视频生成（LTX-Video 2B） | 评估过可行，未安装 |
| LLM LoRA 微调 | 16 GB 勉强可行，未尝试 |

---

## 8. 备注

- `sudo` 需要密码（未配置 NOPASSWD）。
- 111 上的便捷入口：`ub '<命令>'`（免密，走 `/root/.ssh/id_ub_pingji`）。
  - 注意：在一个脚本里连续调用 `ub` 时要加 `< /dev/null`，
    否则内层 ssh 会吃掉脚本剩余内容。
- 111 上的密码兜底脚本：`/tmp/ub.sh`（`SSH_ASKPASS`），重启会丢。
- **非登录 shell**（`ssh host cmd`）不读 `/etc/profile.d`，
  `uv` / `jdk` / `proxy_on` 这些需要 `bash -lc '...'` 或绝对路径。
- v2rayN 的 Windows 配置在 `/mnt/v2/wg2/v2rayN-windows-64`；
  Linux 版配置目录 `~/.local/share/v2rayN`（首次启动 GUI 后生成）。

---

## 9. GPU 与驱动

| 项 | 值 |
| --- | --- |
| GPU | **NVIDIA GeForce RTX 5060 Ti 16 GB**（Blackwell GB206，sm_120，技嘉） |
| 驱动 | `nvidia-driver-615-open` **615.71.09**（KMD 615.71.09 / CUDA UMD **13.4**） |
| 当前内核 | `7.0.0-30-generic` |
| 已加载模块 | `nvidia` `nvidia_drm` `nvidia_modeset` `nvidia_uvm` |
| 验证命令 | `nvidia-smi` |
| 空闲功耗/温度 | 6 W / 33 °C |

> ⚠️ **不要改回 GRUB 默认项 0**（那是 `7.0.0-43`，驱动未验证，可能黑屏）。
> 已固定：`/etc/default/grub` → `GRUB_DEFAULT="1>2"`，备份 `/etc/default/grub.bak.*`。

---

## 10. 本地大模型（llama.cpp + Vulkan）

| 项 | 值 |
| --- | --- |
| 引擎 | **llama.cpp b6500 预编译 Vulkan 版** |
| 目录 | `~/apps/llama-vulkan/build/bin/`（`llama-cli` `llama-bench` `llama-server`，26 MB） |
| 为什么用 Vulkan | Ubuntu 源 CUDA toolkit 只有 **12.4**，编不了 sm_120；官方也**没有** Linux CUDA 预编译包 |
| 识别结果 | `RTX 5060 Ti \| fp16:1 bf16:1 \| matrix cores: NV_coopmat2` |
| 模型目录 | `~/models/` |
| 已下模型 | `Qwen3-8B-Q4_K_M.gguf`（4.7 GB）、`Qwen3-0.6B-Q8_0.gguf`（610 MB） |
| 服务脚本 | `~/apps/llama-server-run.sh [模型路径] [ctx]` → `127.0.0.1:8080` |

### 实测性能（`llama-bench -ngl 99`，全部在显存里）

| 模型 | prefill (pp512) | decode (tg128) |
| --- | ---: | ---: |
| **Qwen3-8B Q4_K_M** | **3086 tok/s** | **74.1 tok/s** |
| Qwen3-0.6B Q8_0 | 21321 tok/s | 283.3 tok/s |

> 对照 K3（RISC-V CPU）：0.6B 是 28.7 / 18.3 tok/s → GPU 的 prefill 快 **740×**、decode 快 **15×**。

### 用法

```bash
# 1) 命令行聊天
~/apps/llama-vulkan/build/bin/llama-cli \
  -m ~/models/Qwen3-8B-Q4_K_M.gguf -ngl 99 -cnv

# 2) OpenAI 兼容 API（后台）
~/apps/llama-server-run.sh ~/models/Qwen3-8B-Q4_K_M.gguf 8192 &

# 3) 调 API
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"你好"}],
       "max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

> **Qwen3 默认开「思考模式」**：会先输出 `reasoning_content` 把 token 吃光，
> `content` 可能是空的。做普通聊天/接口时务必加
> `"chat_template_kwargs": {"enable_thinking": false}`
> （或 prompt 结尾加 `/no_think`）。

---

## 11. 救援通道（隧道断了怎么进去）

本机与 K3 之间有一根**直连网线**（K3 `end0` ↔ 本机 `enp34s0`），
可以完全绕开 111 和隧道：

| 项 | 值 |
| --- | --- |
| 本机链路本地地址 | `fe80::2d8:61ff:fed0:3fb`（EUI-64，来自 enp34s0 MAC `00:d8:61:d0:03:fb`） |
| K3 侧接口 | `end0`（千兆，carrier=1） |
| K3 sudo 密码 | `123456` |
| 私钥 | 111 上 `/root/.ssh/id_ub_pingji` |

链路抖动后 K3 可能丢掉 IPv6 地址，先补一个：

```bash
# 在 K3 上
echo 123456 | sudo -S ip -6 addr add fe80::1/64 dev end0
ping -6 -c2 -I end0 ff02::1          # 确认能看到对端
# 然后从 K3 连本机
ssh -i /tmp/k_box pingji@fe80::2d8:61ff:fed0:3fb%end0
```

> 注意：Ubuntu 26.04 默认用 stable-privacy，**地址有时会变**；
> 用 `avahi-resolve -6 -n pingji-MS-7B89.local` 可以按主机名找到。

---

## 12. 111 侧的 sshd 修复记录

`PermitListen` 是**单值选项**，写两行只有第一行生效，导致 K3 的 2222 转发一直被拒：

```bash
# 错误写法（第二行被忽略）
PermitListen 127.0.0.1:22022
PermitListen 127.0.0.1:2222

# 正确写法（一行内空格分隔）
PermitListen 127.0.0.1:22022 127.0.0.1:2222
```

已修改 `/etc/ssh/sshd_config.d/99-k3.conf` 并 `systemctl reload ssh`，
备份在 `/root/99-k3.conf.bak.*`。

---

## 13. 图像生成（ComfyUI）

| 项 | 值 |
| --- | --- |
| ComfyUI | 最新版（`~/apps/comfyui/ComfyUI`，git `4d64a8b`） |
| 运行环境 | `~/apps/comfyui/venv`（Python 3.12 + **torch 2.15.0+cu132**，sm_120 已验证 `cap (12,0)`） |
| 启动脚本 | `~/apps/comfyui/start.sh`（桌面有图标 `启动ComfyUI.desktop`） |
| 服务地址 | `http://127.0.0.1:8188` |
| **出图目录** | **`~/桌面/ComfyUI/`**（直接显示在桌面上） |
| 模型目录 | `~/apps/comfyui/ComfyUI/models/checkpoints/` |
| 已装模型 | `sd_xl_base_1.0.safetensors`（6.5 GB，含 VAE） |
| 显存占用 | 加载后 **~7 GB**（SDXL 1024²，总 16 GB 富余） |

### 实测

| 任务 | 用时 |
| --- | --- |
| SDXL 1024×1024 / 28 步 / dpmpp_2m karras | **10.3 s**（首次含加载 31 s） |
| 输出文件 | `~/桌面/ComfyUI/k3_sdxl_test_00002_.png`（1024×1024，1.5 MB） |

### 用法

```bash
# 启动（桌面图标也可以）
~/apps/comfyui/start.sh &

# 浏览器打开
#   http://127.0.0.1:8188

# 命令行出图（API）：见 ~/apps/comfyui/test_sdxl.json
curl -X POST http://127.0.0.1:8188/prompt -H 'Content-Type: application/json' \
     --data @/tmp/wf_run.json
```

> 注意：ComfyUI 的 `requirements.txt` 里 `comfyui-workflow-templates` 依赖的
> `...-media-assets-02` 在清华源缺失，安装时已过滤掉该包（只影响 UI 内置示例模板）。

### Flux.1-schnell（第二个模型，质量更高）

| 项 | 值 |
| --- | --- |
| 模型 | `flux1-schnell-fp8.safetensors`（**17.2 GB**，fp8 全量：UNet+T5+CLIP+VAE 一体） |
| 位置 | `~/apps/comfyui/ComfyUI/models/checkpoints/` |
| 工作流 | `~/apps/comfyui/wf_flux.json`（4 步 / cfg 1.0 / euler + ModelSamplingFlux） |
| 显存 | 加载后 **12.6 GB** / 16.3 GB |

| 任务 | 用时 |
| --- | ---: |
| 1024×1024 / **4 步** | **7.8 s**（连续两次 7.80 / 7.79） |
| 首张（含加载 17 GB） | 18.15 s |

> 显存吃紧：Flux + SDXL 同时加载会 OOM。切模型前最好重启 ComfyUI。
> 跑 Flux 时已把 llama-server 停掉（它占 6 GB）；要再开就
> `~/apps/llama-server-run.sh &`。

### 通用出图脚本

```bash
# 任意工作流 + 随机 seed，打印耗时/文件名
bash /tmp/run_wf.sh /home/pingji/apps/comfyui/wf_flux.json rand
bash /tmp/run_wf.sh /home/pingji/apps/comfyui/test_sdxl.json rand
```

### 装依赖踩的坑（重要）

| 问题 | 解决 |
| --- | --- |
| torch cu130 大 wheel 反复超时 | 用 `--index-strategy unsafe-best-match` + `--index-url pytorch/cu130 --extra-index-url 清华` |
| 装到一半多个 uv 抢 venv 锁死锁 | 脚本加 `flock`，装之前 `pkill -9 -f 'uv pip'` 并 `rm venv/.lock` |
| pypi.org 元数据解析极慢 | 只用清华源；实在缺包再加 `pypi.org` 并把代理打开 |

---

## 14. 3D 重建（3D Gaussian Splatting）

视频/多图 → 相机位姿 → 3D 高斯 → 可实时浏览的 `.ply`。

| 项 | 值 |
| --- | --- |
| 环境 | `~/apps/3dgs/venv`（Python 3.12 + **torch 2.15.0+cu132**） |
| 框架 | **nerfstudio**（`ns-train` / `ns-export`）+ **gsplat 1.4.0** |
| gsplat CUDA 核 | 已为 **sm_120** 编译好，缓存在 `~/.cache/torch_extensions/py312_cu132/gsplat_cuda/` |
| CUDA toolkit | `/usr/local/cuda-13.0`（nvcc 13.0.88），`/usr/local/bin/nvcc` 软链 |
| 编译器 | `gcc-14` / `g++-14`（CUDA 13 不支持 gcc 15） |
| 其他 | `colmap`、`ffmpeg`（视频→数据集、抽帧、编码） |
| 脚本 | `~/apps/3dgs/bin/`：`env.sh` `train.sh` `export.sh` `render.sh` `prepare_transforms.py` |
| 示例数据 | `~/data/hf/nerf_synthetic/lego`（136 帧 800×800） |
| 产物 | `~/apps/3dgs/{outputs,exports,renders}` |

### 实测（lego 场景）

| 项 | 值 |
| --- | ---: |
| 训练 10,000 迭代 | **约 2 分钟** |
| 显存峰值 | **13.3 GB / 16.3 GB** |
| 导出 .ply | 18 MB |
| 渲染 96 帧视频 | 约 20 秒 |

### 用法

```bash
# 自己的视频 → 数据集（COLMAP，5–15 分钟）
source ~/apps/3dgs/bin/env.sh
ns-process-data video --data ~/我的视频.mp4 --output-dir ~/data/myscene

# 训练 → 导出 → 渲染
~/apps/3dgs/bin/train.sh  ~/data/myscene 15000 myscene   # 打印 CONFIG=...
~/apps/3dgs/bin/export.sh <CONFIG>
~/apps/3dgs/bin/render.sh <CONFIG> ~/桌面/我的转台.mp4 96
```

> 详细说明（含 10 个踩坑记录、能力边界、拍摄要点）见
> **`~/桌面/3D重建使用说明.md`**；示例产物在 `~/桌面/3DGS-乐高场景/`。

### 3D 方向常见坑（一句话版）

| 现象 | 原因 |
| --- | --- |
| `gsplat: No CUDA toolkit found` | 预编译核不含 sm_120，必须用 CUDA ≥12.8 现编 |
| `Ninja is required` | torch 用 `which ninja` 找，**venv/bin 必须在 PATH** |
| 训练时又说没 toolkit | 启动脚本忘了 `export PATH=$CUDA_HOME/bin:$PATH` |
| `transforms.json`/`fl_x` 缺失 | NeRF 原版格式与 nerfstudio 不兼容，需转换 |
| `weights_only` UnpicklingError | torch ≥2.6 默认严格模式，加 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` |
| `ns-render spiral` 断言失败 | splatfacto 用 `FullImageDatamanager`，官方 spiral 不支持 |
| 渲染全黑 | `rgb[0]` 误切高度维（返回 H×W×3，无 batch 维） |
