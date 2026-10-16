#!/usr/bin/env bash
# K3 (pingji-k3, SpaceMiT K3 / RISC-V) 上跑 Android —— Cuttlefish 部署驱动
#
# 设计与依据见 docs/k3-android-cuttlefish.md
#
# 用法：
#   bash deploy/k3/deploy.sh check          # 只读体检：内核 / KVM / vhost / docker / 资源 / 端口
#   bash deploy/k3/deploy.sh snapshot       # 存一份「现状快照」到 deploy/k3/snapshots/（回滚对照）
#   bash deploy/k3/deploy.sh prep           # 板子上：装 Docker + 下载 K3 定制内核 deb（不改内核）
#   bash deploy/k3/deploy.sh kernel         # 板子上：dpkg -i 内核 deb（之后手动 reboot）
#   bash deploy/k3/deploy.sh reboot         # 重启板子（会短暂中断门户/心跳）
#   bash deploy/k3/deploy.sh mods           # 加载 kvm/vhost_vsock 模块并持久化（内核是 CONFIG_KVM=m，不加载就没 /dev/kvm）
#   bash deploy/k3/deploy.sh verify-host    # 重启后验收：内核 / vhost_net / docker / 隧道
#   bash deploy/k3/deploy.sh vpn-install    # 板子上装本机代理（Xray riscv64）——治 GitHub/ghcr 断流
#   bash deploy/k3/deploy.sh vpn-test       # 测代理：github / ghcr / 出口 IP / 代理下 git clone
#   bash deploy/k3/deploy.sh vpn-status     # 代理服务与 docker/git 代理配置一览
#   bash deploy/k3/deploy.sh proxy-on       # docker daemon + git 走本机代理（拉 ghcr 镜像更稳）
#   bash deploy/k3/deploy.sh proxy-off      # 撤掉 docker daemon / git 的代理
#   bash deploy/k3/deploy.sh vpn-rm         # 停并移除板子上的 xray-k3 服务
#   bash deploy/k3/deploy.sh deploy         # 板子上：clone 上游 + pull 宿主容器镜像
#   bash deploy/k3/deploy.sh prefetch       # 宿主侧预取 cvd-host 包 + AOSP 镜像（走代理），容器起来就不用联网
#   bash deploy/k3/deploy.sh run            # 起 Cuttlefish
#   bash deploy/k3/deploy.sh smoke          # 验收：WebRTC 端口 + guest ABI（容器内直连）
#   bash deploy/k3/deploy.sh shot           # 抓一张 guest 的屏回本机（不需要隧道）
#   bash deploy/k3/deploy.sh push-img F.zip # ROM 开发循环：把新编的镜像 zip 换到板子上
#   bash deploy/k3/deploy.sh status         # 一句话看清容器 / 端口 / 隧道
#   bash deploy/k3/deploy.sh logs           # Cuttlefish 容器日志
#   bash deploy/k3/deploy.sh stop           # 停 Cuttlefish
#   bash deploy/k3/deploy.sh expose         # ★ 对外访问：在 111 上建正向转发（18443 WebRTC / 16520 adb）
#   bash deploy/k3/deploy.sh unexpose       # 撤掉 111 上的正向转发
#   bash deploy/k3/deploy.sh tunnel-add     # 反隧道（★ 本环境会被 sshd 的 PermitListen 拒，用 expose）
#   bash deploy/k3/deploy.sh tunnel-rm      # 撤销反隧道追加
#   bash deploy/k3/deploy.sh rollback       # 停 Cuttlefish + 停 docker + 撤隧道（不自动换回内核）
#
# 访问路径：本机 --ssh--> 111.228.40.211 --docker exec k3-ttyd--> 板子(127.0.0.1:2222)
#   板子在 192.168.110.194 内网，只能这样进。板子直连时设置 K3_DIRECT=1。
#
# sudo：板子上 pingji 在 sudo 组但**不免密**。需 root 的操作会先把脚本传到板子上再执行：
#   有密码  → 自动跑（密码来自 K3_SUDO_PASS 或 ~/.k3-sudo）
#   没密码  → 打印一条干净的 `sudo bash <路径>`，你自己在板子终端里敲
# 这样避免把长脚本塞进 ssh 命令行（引号/中文都会被转义成一团）。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

# ---- 默认配置（可被 deploy/k3/.env.k3 覆盖）-------------------------------
JUMP="${JUMP:-root@111.228.40.211}"     # 跳板机（公网）
TTYD="${TTYD:-k3-ttyd}"                 # 跳板机上持有板子密钥的容器
KEY="${KEY:-/app/ssh/id_ed25519}"       # 容器内的密钥路径
BOARD_PORT="${BOARD_PORT:-2222}"        # 跳板机上板子反隧道的本地端口
BOARD_USER="${BOARD_USER:-pingji}"
BOARD_IP="${BOARD_IP:-192.168.110.194}" # 板子内网 IP（展示 / 直连模式用）
K3_DIRECT="${K3_DIRECT:-0}"             # 1 = 本机可直接 ssh 到板子（局域网）

CF_REPO="${CF_REPO:-https://github.com/monkey-jsun/cuttlefish-host-container}"
CF_DIR="${CF_DIR:-k3-android/cuttlefish-host-container}"   # 板子上（相对 $HOME）
PKG_DIR="${PKG_DIR:-k3-android/pkg}"                       # 板子上（相对 $HOME）
SCRIPTS_DIR="${SCRIPTS_DIR:-k3-android/scripts}"           # 板子上（相对 $HOME）
CF_IMAGE="${CF_IMAGE:-ghcr.io/monkey-jsun/aosp-cuttlefish-riscv64/cf-host-k3:latest}"
CF_IMAGE_TAG="${CF_IMAGE_TAG:-cf-host-k3}"
KERNEL_DEB_URL="${KERNEL_DEB_URL:-}"    # 留空则自动取 upstream releases/latest

# ---- 代理（Xray）----------------------------------------------------------
# 板子到 GitHub/ghcr.io 的 TLS 会被中途断掉（实测 git clone 报 GnuTLS recv error）。
# 这里在板子本机起一个 Xray（SOCKS+HTTP mixed，只监听 127.0.0.1），
# 再把 docker daemon / git 指过去。真实配置放 deploy/k3/xray/config.json（不入 git）。
XRAY_VER="${XRAY_VER:-v26.3.27}"
XRAY_ASSET="${XRAY_ASSET:-Xray-linux-riscv64.zip}"
XRAY_URL="${XRAY_URL:-https://github.com/XTLS/Xray-core/releases/download/$XRAY_VER/$XRAY_ASSET}"
XRAY_GEO_URL="${XRAY_GEO_URL:-https://github.com/Loyalsoldier/v2ray-rules-dat/releases/latest/download}"
XRAY_PORT="${XRAY_PORT:-10808}"          # 必须与 config.json 的 inbound 端口一致
XRAY_DIR="${XRAY_DIR:-k3-android/xray}"  # 板子上的中转目录（相对 $HOME）
XRAY_ETC="${XRAY_ETC:-/etc/xray-k3}"     # 板子上的配置目录（xray run -confdir）
XRAY_SHARE="${XRAY_SHARE:-/usr/local/share/xray}"
XRAY_BIN="${XRAY_BIN:-/usr/local/bin/xray}"
XRAY_UNIT="${XRAY_UNIT:-/etc/systemd/system/xray-k3.service}"

# 111 上对外的转发端口（避开 8443/6520，降低占用冲突连带打挂隧道的概率）
RW_WEBRTC="${RW_WEBRTC:-18443}"
RW_ADB="${RW_ADB:-16520}"
# guest 侧端口
GW_WEBRTC="${GW_WEBRTC:-8443}"
GW_ADB="${GW_ADB:-6520}"

SUDO_PASS="${K3_SUDO_PASS:-}"
if [ -z "$SUDO_PASS" ] && [ -f "$HOME/.k3-sudo" ]; then
  SUDO_PASS="$(cat "$HOME/.k3-sudo")"
fi

[ -f "$HERE/.env.k3" ] && . "$HERE/.env.k3"

# ---- 小工具 ---------------------------------------------------------------
say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!  %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
ask() { printf '\033[1;33m?  %s [y/N] \033[0m' "$*"; read -r r; [ "$r" = y ] || [ "$r" = Y ]; }

# ---- 板子访问层 -----------------------------------------------------------
_inner() {  # 生成「跳板机上的 docker exec ... ssh 板子 <命令>」这一串
  local opts="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o IdentitiesOnly=yes -o LogLevel=ERROR"
  printf 'docker exec -i %s ssh %s -i %s -p %s -l %s 127.0.0.1 %s' \
    "$TTYD" "$opts" "$KEY" "$BOARD_PORT" "$BOARD_USER" "$(printf '%q' "$1")"
}

# 在板子上执行一条命令。默认不占用本地 stdin；bssh_in 把本地 stdin 透传过去。
bssh() {
  if [ "$K3_DIRECT" = 1 ]; then
    ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR -p "$BOARD_PORT" "$BOARD_USER@$BOARD_IP" "$1" </dev/null
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR "$JUMP" "$(_inner "$1")" </dev/null
  fi
}
bssh_in() {
  if [ "$K3_DIRECT" = 1 ]; then
    ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR -p "$BOARD_PORT" "$BOARD_USER@$BOARD_IP" "$1"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR "$JUMP" "$(_inner "$1")"
  fi
}
# 在板子上执行一段脚本（脚本走 stdin）
bscript() { bssh_in 'bash -s'; }

# 把本地脚本传到板子 ~/$SCRIPTS_DIR/<name>.sh，输出板子上的绝对路径
push_script() {
  local name="$1"
  bssh_in "mkdir -p \"\$HOME/$SCRIPTS_DIR\" && cat > \"\$HOME/$SCRIPTS_DIR/$name.sh\" && chmod 700 \"\$HOME/$SCRIPTS_DIR/$name.sh\" && echo \"\$HOME/$SCRIPTS_DIR/$name.sh\""
}

# 带 sudo 执行（有密码就自动；没有就把「sudo bash <路径>」打印出来让你手动跑）
# 用法： bsudo "systemctl reboot"
#        bsudo_file kernel <<'EOS'  ...  EOS
bsudo() {
  local cmd="$1"
  if [ -n "$SUDO_PASS" ]; then
    bssh "printf '%s\\n' $(printf '%q' "$SUDO_PASS") | sudo -S -p '' bash -c $(printf '%q' "$cmd")"
  else
    warn "没有 sudo 密码，请在板子上手动执行："
    printf '\n\033[1;37m  sudo %s\033[0m\n\n' "$cmd"
    return 9
  fi
}
bsudo_file() {
  local name="$1" path
  path="$(push_script "$name")" || die "上传脚本到板子失败"
  echo "  脚本已上传：$path"
  if [ -n "$SUDO_PASS" ]; then
    bssh "printf '%s\\n' $(printf '%q' "$SUDO_PASS") | sudo -S -p '' bash \"$path\""
  else
    warn "没有 sudo 密码，请在板子上执行："
    printf '\n\033[1;37m  sudo bash %s\033[0m\n\n' "$path"
    return 9
  fi
}

need_board() {
  bssh 'echo ok' >/dev/null 2>&1 || die "连不上板子（${JUMP} -> ${TTYD} -> 127.0.0.1:${BOARD_PORT}）。先查 111 上 k3-tunnel 是否在跑。"
}

# 板子上 pingji 的绝对 home。脚本被 sudo 拉起时 $HOME 会变成 /root，
# 所以凡是走 sudo 的脚本里必须用这个绝对路径，不能用 $HOME。
BOARD_HOME_CACHE=""
board_home() {
  [ -n "$BOARD_HOME_CACHE" ] || BOARD_HOME_CACHE="$(bssh 'echo $HOME')"
  printf '%s' "$BOARD_HOME_CACHE"
}

# ---- 板子侧公共片段 -------------------------------------------------------
# 取「当前运行内核」的 config（/boot/config-<uname -r> 优先，退回 /proc/config.gz）
KCFG='
kcfg() {
  if [ -r "/boot/config-$(uname -r)" ]; then cat "/boot/config-$(uname -r)"
  elif [ -r /proc/config.gz ]; then zcat /proc/config.gz
  else echo ""; fi
}
# 逐项判定内核开关：✅ 开 / ❌ is not set / ⚠️ 未出现
kcheck() {
  for key in "$@"; do
    v=$(kcfg | grep -E "^(# )?$key(=| is not set)" | head -1)
    if [ -z "$v" ]; then
      printf "  %-26s ⚠️  未在 config 中出现\n" "$key"
    elif printf "%s" "$v" | grep -q "is not set"; then
      printf "  %-26s ❌ %s\n" "$key" "$v"
    else
      printf "  %-26s ✅ %s\n" "$key" "$v"
    fi
  done
}
'
KEY_OPTS="CONFIG_VHOST_NET CONFIG_VHOST_VSOCK CONFIG_KVM CONFIG_UDMABUF CONFIG_VIDEO_LINLON_K3"

# ---- 子命令：check --------------------------------------------------------
cmd_check() {
  need_board
  say "板子体检（只读）"
  bscript <<EOS
$KCFG
echo "--- 系统 ---"
echo "hostname : \$(hostname)"
echo "os       : \$(. /etc/os-release; echo \$PRETTY_NAME)"
echo "kernel   : \$(uname -r)  (\$(uname -m))"
echo "uptime   : \$(uptime -p 2>/dev/null || uptime)"
echo "--- 资源 ---"
echo "cpu(hw)  : \$(grep -c ^processor /proc/cpuinfo) 核   会话可见: \$(nproc)"
free -m | sed -n '2p' | awk '{printf "mem      : %s MB 总 / %s MB 可用\n",\$2,\$7}'
df -h / /boot | awk 'NR>1{printf "disk     : %-8s %s 可用\n",\$6,\$4}'
echo "--- 虚拟化 ---"
[ -e /dev/kvm ] && echo "kvm      : /dev/kvm 存在" || echo "kvm      : ❌ 缺 /dev/kvm"
grep -m1 ^isa /proc/cpuinfo | grep -q 'vh_' && echo "isa      : ✅ 含 H 扩展" || echo "isa      : ❌ 未见 H 扩展"
[ -e /dev/vhost-vsock ] && echo "vsock    : ✅ /dev/vhost-vsock 存在" || echo "vsock    : ❌ 缺 /dev/vhost-vsock"
kcheck $KEY_OPTS
echo "--- 工具 ---"
for b in git docker adb rsync curl; do printf "%-9s: %s\n" "\$b" "\$(command -v \$b || echo '-')"; done
echo "--- GPU ---"
ls /dev/dri/ 2>/dev/null | tr '\n' ' '; echo
echo "--- 隧道 / 心跳 ---"
systemctl is-active k3-tunnel k3-heartbeat.timer 2>/dev/null | tr '\n' ' '; echo
systemctl show k3-tunnel -p ExecStart --value 2>/dev/null | grep -oE '\-R [0-9.:]+' | sed 's/^/  /'
echo "--- guest 端口占用（应空闲）---"
ss -ltn 2>/dev/null | awk 'NR>1{print \$4}' | grep -E ':(${GW_WEBRTC}|${GW_ADB})\$' && echo "  ↑ 已被占用，起 Cuttlefish 前需清" || echo "  $GW_WEBRTC / $GW_ADB 空闲"
EOS
  say "跳板机 111 侧端口（反隧道要用的）"
  ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR "$JUMP" \
    "ss -ltn | awk 'NR>1{print \$4}' | grep -E ':(${RW_WEBRTC}|${RW_ADB})\$' && echo '  ↑ 已被占用，tunnel-add 前必须腾出来' || echo '  ${RW_WEBRTC} / ${RW_ADB} 空闲'"
}

# ---- 子命令：snapshot -----------------------------------------------------
cmd_snapshot() {
  need_board
  mkdir -p "$HERE/snapshots"
  local out="$HERE/snapshots/$(date '+%Y%m%d-%H%M%S').txt"
  say "保存现状快照 -> $out"
  {
    echo "# pingji-k3 现状快照  $(date '+%Y-%m-%d %H:%M:%S')"
    bscript <<EOS
echo "## uname"; uname -a
echo "## os-release"; cat /etc/os-release
echo "## 内核包"; dpkg -l 'linux-image-*' 'linux-headers-*' 2>/dev/null | grep '^ii' || true
echo "## /boot"; ls -la /boot
echo "## /boot 空间"; df -h /boot
echo "## 引导相关"; ls -la /boot/spacemit 2>/dev/null; cat /boot/env_k3.txt 2>/dev/null
echo "## systemd 自定义单元"; ls /etc/systemd/system/*.service /etc/systemd/system/*.timer 2>/dev/null
echo "## k3-tunnel 全文"; cat /etc/systemd/system/k3-tunnel.service
echo "## k3-tunnel drop-in"; ls -la /etc/systemd/system/k3-tunnel.service.d/ 2>/dev/null || echo "(无)"
echo "## 网络"; ip -br addr; ip route
echo "## 监听"; ss -ltnp
echo "## cpu isa"; grep -m1 ^isa /proc/cpuinfo
echo "## mem/disk"; free -m; df -h
EOS
  } > "$out"
  echo "  已写入（$(wc -l < "$out") 行）"
}

# ---- 子命令：prep ---------------------------------------------------------
# 装 Docker + 把定制内核 deb 下下来（不安装、不重启）
cmd_prep() {
  need_board
  local url="$KERNEL_DEB_URL"
  if [ -z "$url" ]; then
    say "解析上游最新 K3 内核 deb"
    url="$(bssh "curl -fsSL https://api.github.com/repos/monkey-jsun/linux-6.18-k3/releases/latest | python3 -c \"import sys,json;d=json.load(sys.stdin);print([a['browser_download_url'] for a in d['assets'] if a['name'].startswith('linux-image')][0])\"" || true)"
    [ -n "$url" ] || die "拿不到内核 deb 地址（GitHub API 可能被限流）。在 deploy/k3/.env.k3 里手动设 KERNEL_DEB_URL"
  fi
  echo "  内核 deb: $url"

  local BH; BH="$(board_home)"
  say "板子上：装 docker.io + 下载内核 deb"
  bsudo_file prep <<EOS
set -euo pipefail
mkdir -p "$BH/$PKG_DIR"
echo "[1/3] apt 源"
grep -rhE '^deb ' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null | head -5 || true
echo "[2/3] 安装 docker.io"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
systemctl enable --now docker
usermod -aG docker $BOARD_USER || true
echo "[3/3] 下载内核 deb"
cd "$BH/$PKG_DIR"
curl -fL --retry 3 -o linux-image-cf-k3.deb "$url"
chown $BOARD_USER:$BOARD_USER linux-image-cf-k3.deb
ls -l
echo
echo "  完成。下一步： bash $0 kernel"
EOS
}

# ---- 子命令：kernel -------------------------------------------------------
# 装之前先从 deb 里把 kernel config 抠出来做预检（省一次白重启）
cmd_kernel() {
  need_board
  local BH; BH="$(board_home)"

  say "预检：读取 deb 内的 kernel config（不改动系统）"
  bscript <<EOS
$KCFG
deb="$BH/$PKG_DIR/linux-image-cf-k3.deb"
[ -f "\$deb" ] || { echo "✗ 找不到 \$deb，先跑 deploy.sh prep"; exit 1; }
echo "--- deb ---"
dpkg-deb -f "\$deb" Package Version Architecture 2>/dev/null | sed 's/^/  /'
echo "--- deb 内 config 的关键项 ---"
cfg=\$(dpkg-deb -c "\$deb" | awk '{print \$NF}' | grep -E '/config-' | head -1)
kcfg() { dpkg-deb --fsys-tarfile "\$deb" | tar -xO "\$cfg" 2>/dev/null; }
kcheck $KEY_OPTS
echo "--- /boot 余量（装内核要写这里）---"
df -h /boot | tail -1
EOS
  echo
  warn "注意：上游自己也把 CONFIG_VHOST_NET 关掉了（README 里那条要求可能过期）。"
  warn "VHOST_NET ❌ 只影响 guest 联网；KVM/UDMABUF/VSOCK OK 的话 Cuttlefish 照样能起、能看屏幕、能 adb。"
  ask "继续安装内核？" || { echo "  已取消"; return 0; }

  bsudo_file kernel <<EOS
set -euo pipefail
cd "$BH/$PKG_DIR"
ls -l linux-image-cf-k3.deb
dpkg -i linux-image-cf-k3.deb
echo
echo "--- 装后内核包 ---"
dpkg -l 'linux-image-*' | grep '^ii' || true
echo "--- /boot ---"
ls -la /boot
echo "--- 新内核的 KVM/vhost 配置（关键验收点）---"
for c in /boot/config-*; do
  echo "[\$c]"
  grep -E '^(# )?CONFIG_(VHOST_NET|VHOST_VSOCK|KVM|UDMABUF|VIDEO_LINLON_K3)(=| is not set)' "\$c" || true
done
echo
echo "  下一步： bash $0 reboot  然后  bash $0 verify-host"
EOS
}

# ---- 子命令：reboot ------------------------------------------------------
cmd_reboot() {
  need_board
  warn "重启会中断 K3 门户/心跳几十秒；k3-tunnel 会自动重连。"
  ask "确认重启板子？" || { echo "  已取消"; return 0; }
  bsudo "systemctl reboot" || { warn "没有 sudo 密码，请手动执行： sudo systemctl reboot"; return 0; }
  say "等待板子回来（最多 5 分钟）"
  for _ in $(seq 1 60); do
    sleep 5
    if bssh 'echo up' >/dev/null 2>&1; then
      echo "  已恢复：$(bssh 'uname -r')"
      return 0
    fi
    printf '.'
  done
  echo
  warn "5 分钟没等到；可能卡在新内核。用串口（ttyS0）或 U-Boot 选回旧内核，见 docs/k3-android-cuttlefish.md §7"
  return 1
}

# ---- 子命令：mods --------------------------------------------------------
# 本项目内核 CONFIG_KVM=m（可模块化，上游特意这么做的），开机不自动加载 →
# 不 modprobe 就没有 /dev/kvm。这里加载并写 modules-load.d 持久化。
cmd_mods() {
  need_board
  say "加载并持久化 Cuttlefish 需要的内核模块"
  bsudo_file k3-mods <<'EOS'
set -euo pipefail
cat > /etc/modules-load.d/k3-android.conf <<'CONF'
# 由 deploy/k3/deploy.sh mods 生成 —— Cuttlefish 需要的模块
# 本项目内核 CONFIG_KVM=m，开机默认不加载，不 modprobe 就没有 /dev/kvm
kvm
vhost_vsock
CONF
modprobe kvm
modprobe vhost_vsock
echo "--- /etc/modules-load.d/k3-android.conf ---"
cat /etc/modules-load.d/k3-android.conf
echo "--- lsmod ---"
lsmod | grep -E '^(kvm|vhost|vsock)' || true
echo "--- 设备节点 ---"
ls -l /dev/kvm /dev/vhost-vsock
EOS
}

# ---- 子命令：verify-host -------------------------------------------------
cmd_verify_host() {
  need_board
  say "宿主验收"
  bscript <<EOS
$KCFG
k=\$(uname -r)
echo "kernel           : \$k"
case "\$k" in
  *cf-k3*) echo "  判定           : ✅ 已是定制内核" ;;
  *)       echo "  判定           : ❌ 仍是原内核（换内核/重启没生效）" ;;
esac
echo "docker           : \$(docker --version 2>/dev/null || echo '❌ 未安装')"
if docker info >/dev/null 2>&1; then
  echo "docker daemon    : ✅ 可用"
else
  echo "docker daemon    : ❌ 不可用（可能还没加 docker 组，重登一次会话）"
fi
[ -e /dev/kvm ] && echo "/dev/kvm         : ✅" || echo "/dev/kvm         : ❌"
[ -e /dev/vhost-vsock ] && echo "/dev/vhost-vsock : ✅" || echo "/dev/vhost-vsock : ❌"
echo "--- 关键内核开关（项目硬性要求）---"
kcheck $KEY_OPTS
echo "--- 隧道 / 心跳 ---"
systemctl is-active k3-tunnel k3-heartbeat.timer | tr '\n' ' '; echo
EOS
  echo
  warn "若 CONFIG_VHOST_NET / CONFIG_UDMABUF 仍是 ❌，是硬阻断：需要请上游开，或用上游的 config-aia2-kvm-modular 自己重编内核。"
}

# 把一个本地目录打成 tar 推到板子上（用于板子连不上 GitHub 时中转）
push_tar() {  # push_tar <本地目录> <板子上的父目录>
  local src="$1" parent="$2"
  COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs \
    --exclude='._*' --exclude='.DS_Store' \
    -C "$(dirname "$src")" -czf - "$(basename "$src")" \
    | bssh_in "mkdir -p '$parent' && tar -xzf - -C '$parent'"
}

# ---- 子命令：deploy ------------------------------------------------------
cmd_deploy() {
  need_board
  local BH; BH="$(board_home)"
  local CF_ABS="$BH/k3-android/cuttlefish-host-container"

  if bssh "[ -d '$CF_ABS/.git' ]"; then
    say "上游脚本已在板子上，跳过 clone"
  else
    say "板子上 clone 上游脚本"
    if ! bscript <<EOS
set -e
mkdir -p "$BH/k3-android"; cd "$BH/k3-android"
git clone --depth 1 $CF_REPO
EOS
    then
      warn "板子连 GitHub 不稳（TLS 被断），改从本机 clone 再推过去"
      local tmp; tmp="$(mktemp -d)"
      git clone --depth 1 "$CF_REPO" "$tmp/cuttlefish-host-container"
      push_tar "$tmp/cuttlefish-host-container" "$BH/k3-android"
      rm -rf "$tmp"
      bssh "ls -d '$CF_ABS' && ls '$CF_ABS' | head"
    fi
  fi

  say "拉宿主容器镜像（约 2.1GB，首次较慢）"
  bscript <<EOS
set -euo pipefail
docker pull $CF_IMAGE
docker tag $CF_IMAGE $CF_IMAGE_TAG
echo "--- 镜像 ---"
docker images | grep -E 'REPOSITORY|cf-host' || true
echo
echo "  下一步： bash $0 run"
EOS
}

# ---- 子命令：prefetch -----------------------------------------------------
# 容器首次运行要从 GitHub 下 cvd-host 包(~215MB) + AOSP 镜像(~830MB)。
# 容器里的 127.0.0.1 不是板子，用不到板子的代理；所以**在宿主侧先下好并按
# entrypoint.sh 期望的布局解开**，容器一看文件都在就直接起 guest，不用联网。
# 布局（容器里是 /cf）：cf-data/host/bin/launch_cvd、cf-data/product/*.img
cmd_prefetch() {
  need_board
  local BH; BH="$(board_home)"
  say "板子宿主侧预取 cvd-host 包 + AOSP 镜像（走板子上的代理，约 1GB）"
  bscript <<EOS
set -uo pipefail
P=$XRAY_PORT
IMG=$CF_IMAGE_TAG
CF_DATA="\$HOME/$CF_DIR/cf-data"

urls=\$(docker inspect "\$IMG" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null)
HOST_URL=\$(printf '%s\n' "\$urls" | sed -n 's|^CF_HOST_PACKAGE_URL=||p')
PROD_URL=\$(printf '%s\n' "\$urls" | sed -n 's|^CF_PRODUCT_IMG_URL=||p')
echo "  cvd-host 包 : \$HOST_URL"
echo "  AOSP 镜像   : \$PROD_URL"
command -v unzip >/dev/null || { echo "✗ 缺 unzip，先装： sudo apt-get install -y unzip"; exit 1; }

mkdir -p "\$CF_DATA/host" "\$CF_DATA/product"

if [ -x "\$CF_DATA/host/bin/launch_cvd" ]; then
  echo "  [1/2] cvd-host 包已就绪，跳过"
else
  echo "  [1/2] 下载并解开 cvd-host 包 ..."
  curl -fL --retry 3 --proxy socks5h://127.0.0.1:\$P "\$HOST_URL" | tar -xzf - -C "\$CF_DATA/host" \
    || { echo "✗ cvd-host 包下载/解开失败"; exit 1; }
fi

if compgen -G "\$CF_DATA/product/*.img" >/dev/null || [ -f "\$CF_DATA/product/android-info.txt" ]; then
  echo "  [2/2] AOSP 镜像已就绪，跳过"
else
  echo "  [2/2] 下载并解开 AOSP 镜像（~830MB，慢）..."
  rm -f "\$CF_DATA/.product.zip"
  curl -fL --retry 3 --proxy socks5h://127.0.0.1:\$P -o "\$CF_DATA/.product.zip" "\$PROD_URL" \
    || { echo "✗ AOSP 镜像下载失败"; exit 1; }
  unzip -n -q "\$CF_DATA/.product.zip" -d "\$CF_DATA/product" \
    || { echo "✗ 解压失败"; exit 1; }
  rm -f "\$CF_DATA/.product.zip"
fi

echo "--- 结果 ---"
ls -l "\$CF_DATA/host/bin/launch_cvd" 2>/dev/null || echo "  ⚠️ 没看到 host/bin/launch_cvd"
ls -1 "\$CF_DATA/product" | head -12 | sed 's/^/  /'
du -sh "\$CF_DATA" 2>/dev/null
df -h / | tail -1
echo
echo "  下一步： bash $0 run"
EOS
}

# ---- 子命令：run / stop / logs / status ----------------------------------
cmd_run() {
  need_board
  say "起 Cuttlefish（cf-host-k3）—— 后台派生，日志落在板子上"
  bscript <<EOS
set -euo pipefail
cd "\$HOME/$CF_DIR"
[ -x ./cf-run.sh ] || { echo "✗ 没有 cf-run.sh，先跑 deploy.sh deploy"; exit 1; }
[ -e /dev/kvm ]    || { echo "✗ 没有 /dev/kvm —— 先跑 deploy.sh mods"; exit 1; }

if docker ps -q --filter label=cf-host | grep -q .; then
  echo "  cf 容器已在跑（deploy.sh logs 看日志）"; exit 0
fi

LOG="\$HOME/$CF_DIR/cf-run.log"
: > "\$LOG"
# cf-run.sh 是「docker run -it」（前台附着）——直接跑会把 ssh 会话一直挂住。
# 用 script 造一个 pty 喂给 docker 的 -t，再用 setsid 把它从本会话脱离，日志落文件。
setsid script -q -c "./cf-run.sh -i $CF_IMAGE_TAG" /dev/null > "\$LOG" 2>&1 < /dev/null &
echo "  已派生后台进程，日志： \$LOG"
EOS

  say "等容器起来（最多 3 分钟）"
  bscript <<EOS
for i in \$(seq 1 18); do
  st=\$(docker ps --filter label=cf-host --format '{{.Status}}' | head -1)
  if [ -n "\$st" ]; then echo "  [\$((i*10))s] 容器已起： \$st"; break; fi
  sleep 10
done
echo "--- cf-run.log 尾部 ---"
tail -25 "\$HOME/$CF_DIR/cf-run.log" 2>/dev/null || echo "  (还没日志)"
EOS
  cmd_status
}

cmd_stop() {
  need_board
  say "停 Cuttlefish"
  bscript <<EOS
cd "\$HOME/$CF_DIR" 2>/dev/null || exit 0
if [ -x ./cf-stop.sh ]; then ./cf-stop.sh; else docker ps --filter label=cf-host -q | xargs -r docker stop; fi
EOS
}

cmd_status() {
  need_board
  say "K3 Android 状态"
  bscript <<EOS
echo "--- 容器 ---"
docker ps --filter label=cf-host --format '  {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null | grep . || echo "  (无 cf 容器)"
echo "--- 端口 ---"
ss -ltn 2>/dev/null | awk 'NR>1{print \$4}' | grep -E ':(${GW_WEBRTC}|${GW_ADB})\$' | sed 's/^/  /' || echo "  $GW_WEBRTC / $GW_ADB 未监听"
echo "--- 隧道 / 心跳 / 内核 ---"
printf '  k3-tunnel=%s  heartbeat=%s  kernel=%s\n' "\$(systemctl is-active k3-tunnel)" "\$(systemctl is-active k3-heartbeat.timer)" "\$(uname -r)"
EOS
}

cmd_logs() {
  need_board
  bscript <<'EOS'
echo "--- cf 容器日志（最后 120 行）---"
cid=$(docker ps -a --filter label=cf-host -q | head -1)
if [ -n "$cid" ]; then docker logs --tail 120 "$cid" 2>&1 | tail -120; else echo "  (没有 cf 容器)"; fi
echo "--- cf-run.log（最后 60 行）---"
tail -60 "$HOME/k3-android/cuttlefish-host-container/cf-run.log" 2>/dev/null || echo "  (没有 cf-run.log)"
EOS
}

# ---- 子命令：smoke -------------------------------------------------------
cmd_smoke() {
  need_board
  say "冒烟：WebRTC 端口 + guest ABI（容器内直连，不需要隧道）"
  bscript <<EOS
printf '  板子 https://127.0.0.1:${GW_WEBRTC} -> '
curl -sk -o /dev/null -w '%{http_code}\n' -m 10 https://127.0.0.1:${GW_WEBRTC}/ || echo FAIL
CID=\$(docker ps -q --filter label=cf-host | head -1)
if [ -z "\$CID" ]; then echo "  cf 容器没在跑（deploy.sh run）"; exit 0; fi
echo "  --- 容器内 adb ---"
docker exec "\$CID" sh -c 'export PATH=/cf/host/bin:\$PATH; adb connect 127.0.0.1:${GW_ADB} >/dev/null 2>&1; adb devices | tail -n +2; printf "abi="; adb -s 127.0.0.1:${GW_ADB} shell getprop ro.product.cpu.abi; printf "release="; adb -s 127.0.0.1:${GW_ADB} shell getprop ro.build.version.release' 2>&1 | sed 's/^/  /'
EOS
  if command -v adb >/dev/null 2>&1; then
    if (exec 3<>"/dev/tcp/127.0.0.1/${GW_ADB}") 2>/dev/null; then
      say "本机 adb（127.0.0.1:${GW_ADB} 通）"
      adb connect "127.0.0.1:${GW_ADB}" >/dev/null 2>&1 || true
      echo "  abi: $(adb -s "127.0.0.1:${GW_ADB}" shell getprop ro.product.cpu.abi 2>/dev/null || echo 连不上)"
      echo "  devices: $(adb devices | awk 'NR>1{printf "%s=%s ", $1, $2}')"
    else
      warn "本机 127.0.0.1:${GW_ADB} 没通 —— 要本机 adb 就先 tunnel-add，再："
      echo "     ssh -N -L ${GW_ADB}:127.0.0.1:${RW_ADB} ${JUMP}"
    fi
  else
    warn "本机没有 adb；板子内已验过（见上）"
  fi
}

# ---- 子命令：shot --------------------------------------------------------
# 抓一张 guest 的屏回本机（不需要隧道 / 不需要浏览器）
cmd_shot() {
  need_board
  local BH out tmp
  BH="$(board_home)"
  out="${1:-$HERE/snapshots/$(date +%Y%m%d-%H%M%S)-screen.png}"
  tmp="$BH/k3-android/shot.png"
  mkdir -p "$(dirname "$out")"
  say "抓屏 -> $out"
  bscript <<EOS
set -e
CID=\$(docker ps -q --filter label=cf-host | head -1)
[ -n "\$CID" ] || { echo "✗ 没有 cf 容器在跑（deploy.sh run）"; exit 1; }
docker exec "\$CID" sh -c 'export PATH=/cf/host/bin:\$PATH; adb -s 127.0.0.1:$GW_ADB exec-out screencap -p' > "$tmp"
ls -l "$tmp"
EOS
  bssh "base64 -w0 '$tmp'" | base64 -d > "$out"
  file "$out"
  ls -l "$out"
}

# ---- 子命令：push-img（ROM 开发循环的最后一步）----------------------------
# 把编译机产出的 out/dist/aosp_cf_riscv64_phone_js-img-*.zip 换到板子上：
# 备份旧 product → 展开新镜像 → 下次起机就用它（entrypoint.sh 是「文件在就跳过下载」）。
cmd_push_img() {
  local zip="${1:-}"
  [ -n "$zip" ] || die "用法： bash $0 push-img <本地 aosp_cf_riscv64_phone_js-img-*.zip>"
  [ -f "$zip" ] || die "找不到文件： $zip"
  need_board
  local BH remote
  BH="$(board_home)"
  remote="$BH/$CF_DIR/cf-data/.img-incoming.zip"

  say "推送镜像 zip（$(du -h "$zip" | cut -f1)）→ 板子（经隧道，慢，耐心等）"
  bssh_in "cat > '$remote'" < "$zip"

  say "板子上：备份旧 product，展开新镜像"
  bscript <<EOS
set -e
CF_DATA="\$HOME/$CF_DIR/cf-data"
[ -s "$remote" ] || { echo "✗ 传过去是空的"; exit 1; }
ts=\$(date +%Y%m%d-%H%M%S)
if [ -d "\$CF_DATA/product" ]; then
  mv "\$CF_DATA/product" "\$CF_DATA/product.bak-\$ts"
  echo "  旧镜像已备份为 cf-data/product.bak-\$ts"
fi
mkdir -p "\$CF_DATA/product"
unzip -n -q "$remote" -d "\$CF_DATA/product"
rm -f "$remote"
echo "--- 新 product ---"
ls -1 "\$CF_DATA/product" | head -14 | sed 's/^/  /'
if compgen -G "\$CF_DATA/product/*.img" >/dev/null; then
  echo "  ✅ 有 .img"
else
  echo "  ⚠️ 没看到顶层 .img —— zip 结构可能跟 entrypoint.sh 期望的不一样"
fi
echo
echo "  生效： bash $0 stop && bash $0 run"
echo "  回滚： 板子上  mv \$HOME/$CF_DIR/cf-data/product.bak-\$ts \$HOME/$CF_DIR/cf-data/product"
EOS
}

# ---- 子命令：tunnel-add / tunnel-rm --------------------------------------
DROPIN="/etc/systemd/system/k3-tunnel.service.d/android.conf"

cmd_tunnel_add() {
  need_board
  warn "★ 改隧道有连带风险：k3-tunnel 带 ExitOnForwardFailure=yes，任一 -R 建不起来整条隧道会退出重连，"
  warn "  连带的 2222（K3 门户）也会断。所以先确认 111 上 ${RW_WEBRTC}/${RW_ADB} 空闲。"
  ssh -o BatchMode=yes -o LogLevel=ERROR "$JUMP" "ss -ltn | awk 'NR>1{print \$4}' | grep -E ':(${RW_WEBRTC}|${RW_ADB})\$'" \
    && die "跳板机上端口已被占用，换 RW_WEBRTC/RW_ADB 再试"

  # ★ 预检：这条隧道是我们进板子的唯一通道（111 -> k3-ttyd -> 127.0.0.1:2222），
  #   一旦改坏，门户和终端也一起断，远程救不回来。所以先拿一条**独立** ssh 只试这两条 -R：
  #   建得住（timeout 打到 124）才继续，否则原样返回、不碰 drop-in。
  say "预检：独立 ssh 试绑这两条 -R（不碰 2222）"
  bscript <<EOS || die "预检没过：新的 -R 建不起来，**没有改动任何东西**。多半是 111 的 sshd 用 PermitListen 只放行了 2222 —— 改用： bash $0 expose"
set -o pipefail
timeout 8 ssh -NT -o ExitOnForwardFailure=yes -o BatchMode=yes -o IdentitiesOnly=yes \\
  -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/home/pingji/.ssh/known_hosts \\
  -i /home/pingji/.ssh/k3_tunnel_ed25519 -l root \\
  -R 127.0.0.1:${RW_WEBRTC}:127.0.0.1:${GW_WEBRTC} \\
  -R 127.0.0.1:${RW_ADB}:127.0.0.1:${GW_ADB} \\
  111.228.40.211
rc=\$?
if [ "\$rc" = 124 ]; then
  echo "  ✅ 两条 -R 都建住了（8 秒内没退出）"
else
  echo "  ✗ 试绑失败（rc=\$rc）"
  exit 1
fi
EOS

  ask "继续追加反隧道？" || { echo "  已取消"; return 0; }

  bsudo_file tunnel-add <<EOS
set -euo pipefail
mkdir -p /etc/systemd/system/k3-tunnel.service.d
cat > $DROPIN <<'CONF'
# 由 deploy/k3/deploy.sh tunnel-add 生成：在原有 -R 基础上追加 WebRTC / adb 转发
# 注意：drop-in 里 ExecStart= 单独一行是「清空原值」语义，所以这里是完整命令
[Service]
ExecStart=
ExecStart=/usr/bin/ssh -NT \\
  -o ExitOnForwardFailure=yes \\
  -o ServerAliveInterval=30 \\
  -o ServerAliveCountMax=3 \\
  -o TCPKeepAlive=yes \\
  -o ConnectTimeout=10 \\
  -o BatchMode=yes \\
  -o IdentitiesOnly=yes \\
  -o StrictHostKeyChecking=accept-new \\
  -o UserKnownHostsFile=/home/pingji/.ssh/known_hosts \\
  -i /home/pingji/.ssh/k3_tunnel_ed25519 \\
  -l root \\
  -R 127.0.0.1:2222:127.0.0.1:22 \\
  -R 127.0.0.1:${RW_WEBRTC}:127.0.0.1:${GW_WEBRTC} \\
  -R 127.0.0.1:${RW_ADB}:127.0.0.1:${GW_ADB} \\
  111.228.40.211
CONF
cat $DROPIN
systemctl daemon-reload
systemctl restart k3-tunnel
sleep 3
echo "--- 板子侧验证 ---"
systemctl is-active k3-tunnel
ss -ltn | awk 'NR>1{print \$4}' | grep -E ':(2222|${RW_WEBRTC}|${RW_ADB})\$' || true
EOS

  say "跳板机侧验证"
  ssh -o BatchMode=yes -o LogLevel=ERROR "$JUMP" \
    "ss -ltn | grep -E ':(2222|${RW_WEBRTC}|${RW_ADB})\$' || echo '  ⚠ 没看到监听'"
  echo
  echo "  本机访问（开一个窗口保持）："
  echo "    ssh -N -L ${GW_WEBRTC}:127.0.0.1:${RW_WEBRTC} -L ${GW_ADB}:127.0.0.1:${RW_ADB} root@${JUMP}"
  echo "    浏览器 https://127.0.0.1:${GW_WEBRTC}/     adb connect 127.0.0.1:${GW_ADB}"
  warn "务必确认 K3 门户（:3000）还能开，再收工。"
}

cmd_tunnel_rm() {
  need_board
  bsudo "rm -f $DROPIN && systemctl daemon-reload && systemctl restart k3-tunnel && sleep 3 && systemctl is-active k3-tunnel k3-heartbeat.timer" \
    || warn "没有 sudo 密码：在板子上手动 sudo rm -f $DROPIN && sudo systemctl daemon-reload && sudo systemctl restart k3-tunnel"
}

# ---- 子命令：expose / unexpose（★ 实际用的对外访问方案）--------------------
# 为什么**不用**反隧道（tunnel-add 在本环境注定失败）：
#   111 的 /etc/ssh/sshd_config.d/99-k3.conf 里写着
#       PermitListen 127.0.0.1:2222
#   也就是 sshd **只允许客户端绑 2222 这一个端口**，任何新增的 -R 都会被拒：
#       Error: remote port forwarding failed for listen port 18443
#   而 k3-tunnel 带 ExitOnForwardFailure=yes —— 硬改会把 2222（门户 + 我进板子的
#   唯一通道）一起打挂，远程都救不回来。
#
# 所以改成**完全不动板子**的做法：在 111 上做 ssh -L 正向转发。k3-ttyd 容器是
# host 网络，容器里的 127.0.0.1 就是 111 的 127.0.0.1；转发最终交给板子的 sshd 去连
# 127.0.0.1:8443/6520（在板子本机视角解析），效果与反隧道等价。脚在 111 上，风险为 0。
FWD_UNIT="${FWD_UNIT:-k3-android-fwd}"

cmd_expose() {
  need_board
  say "在 111 上建正向转发：${RW_WEBRTC}->板子 ${GW_WEBRTC}(WebRTC)、${RW_ADB}->板子 ${GW_ADB}(adb)"
  ssh -o BatchMode=yes -o LogLevel=ERROR "$JUMP" "ss -ltn | awk 'NR>1{print \$4}' | grep -E ':(${RW_WEBRTC}|${RW_ADB})\$'" \
    && die "111 上 ${RW_WEBRTC}/${RW_ADB} 已被占用"
  ssh -o BatchMode=yes -o LogLevel=ERROR "$JUMP" "docker ps --format '{{.Names}}' | grep -qx '$TTYD'" \
    || die "111 上没有在跑的容器 $TTYD"

  say "写 systemd 单元 /etc/systemd/system/${FWD_UNIT}.service"
  ssh -o BatchMode=yes -o LogLevel=ERROR "$JUMP" "cat > /etc/systemd/system/${FWD_UNIT}.service" <<UNIT
[Unit]
Description=K3 Android 正向转发（${TTYD} -> 板子 ${GW_WEBRTC}/${GW_ADB}，只绑 111 的 127.0.0.1）
# 不用反隧道：本机 sshd 的 PermitListen 只放行 2222（/etc/ssh/sshd_config.d/99-k3.conf）
After=docker.service
Requires=docker.service

[Service]
Type=simple
ExecStart=/usr/bin/docker exec -i $TTYD ssh -NT \\
  -o ExitOnForwardFailure=yes \\
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes \\
  -o ConnectTimeout=10 -o BatchMode=yes -o IdentitiesOnly=yes \\
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \\
  -i $KEY -p $BOARD_PORT -l $BOARD_USER 127.0.0.1 \\
  -L 127.0.0.1:$RW_WEBRTC:127.0.0.1:$GW_WEBRTC \\
  -L 127.0.0.1:$RW_ADB:127.0.0.1:$GW_ADB
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

  say "启动并验证"
  ssh -o BatchMode=yes -o LogLevel=ERROR "$JUMP" \
    "systemctl daemon-reload && systemctl enable --now ${FWD_UNIT} >/dev/null && sleep 3 && echo -n '  unit: ' && systemctl is-active ${FWD_UNIT} && ss -ltn | grep -E ':(${RW_WEBRTC}|${RW_ADB})\$' | sed 's/^/  /'" \
    || warn "没起来。看日志： ssh $JUMP journalctl -u ${FWD_UNIT} -n 30 --no-pager"
  ssh -o BatchMode=yes -o LogLevel=ERROR "$JUMP" \
    "curl -sk -o /dev/null -w '  111 上 https://127.0.0.1:${RW_WEBRTC} -> %{http_code}\n' -m 10 https://127.0.0.1:${RW_WEBRTC}/ || true"
  echo
  echo "  本机（Mac）开一个窗口保持："
  echo "    ssh -N -L ${GW_WEBRTC}:127.0.0.1:${RW_WEBRTC} -L ${GW_ADB}:127.0.0.1:${RW_ADB} ${JUMP}"
  echo "  然后：浏览器 https://127.0.0.1:${GW_WEBRTC}/   （自签证书，放行即可）"
  echo "        adb connect 127.0.0.1:${GW_ADB}"
  echo
  echo "  没动板子的 k3-tunnel，所以 K3 门户（111:3000）不受影响；只有 127.0.0.1:${RW_WEBRTC}/${RW_ADB} 对外可见（仍不在公网）。"
}

cmd_unexpose() {
  say "撤掉 111 上的正向转发"
  ssh -o BatchMode=yes -o LogLevel=ERROR "$JUMP" \
    "systemctl disable --now ${FWD_UNIT} >/dev/null 2>&1 || true; rm -f /etc/systemd/system/${FWD_UNIT}.service; systemctl daemon-reload; sleep 1; ss -ltn | grep -E ':(${RW_WEBRTC}|${RW_ADB})\$' || echo '  ${RW_WEBRTC}/${RW_ADB} 已释放'"
}

# ---- 子命令：rollback ----------------------------------------------------
cmd_rollback() {
  need_board
  warn "回滚 = 停 Cuttlefish + 停 docker + 撤隧道追加 + 撤 111 上的正向转发。**不会自动换回内核**（那步要手动，见下）。"
  ask "继续？" || { echo "  已取消"; return 0; }
  bscript <<EOS
cd "\$HOME/$CF_DIR" 2>/dev/null && [ -x ./cf-stop.sh ] && ./cf-stop.sh || true
EOS
  bsudo "systemctl stop docker && rm -f $DROPIN && systemctl daemon-reload && systemctl restart k3-tunnel" \
    || warn "没有 sudo 密码，手动执行： sudo systemctl stop docker; sudo rm -f $DROPIN; sudo systemctl daemon-reload; sudo systemctl restart k3-tunnel"
  cmd_unexpose || true
  echo
  echo "  若还要换回原内核（板子上执行）："
  echo "    dpkg -l 'linux-image-*'                                     # 找到 6.18.3-generic 的包名"
  echo "    sudo apt-get install --reinstall -y linux-image-6.18.3-generic"
  echo "    sudo reboot"
}

# ---- 子命令：vpn-*（板子本机代理，治 GitHub/ghcr 断流）--------------------
# 本机下载 Xray riscv64 + geo 数据到 deploy/k3/xray/dl/（已 gitignore）
cmd_vpn_dl() {
  local dl="$HERE/xray/dl"
  mkdir -p "$dl/bin"
  if [ -x "$dl/bin/xray" ] && [ -s "$dl/bin/geoip.dat" ]; then
    echo "  本机已有 Xray + geo 数据（要重下： rm -rf ${dl}）"
  else
    say "本机下载 Xray-core riscv64（${XRAY_VER}）"
    curl -fL --retry 3 --progress-bar -o "$dl/$XRAY_ASSET" "$XRAY_URL"
    unzip -o -q "$dl/$XRAY_ASSET" -d "$dl/bin"
    say "本机下载 geo 数据（Loyalsoldier，含 geosite:google/cn/private 与 geoip:cn/private）"
    curl -fL --retry 3 --progress-bar -o "$dl/bin/geoip.dat" "$XRAY_GEO_URL/geoip.dat"
    curl -fL --retry 3 --progress-bar -o "$dl/bin/geosite.dat" "$XRAY_GEO_URL/geosite.dat"
  fi
  file "$dl/bin/xray" | sed 's/^/  /'
  ls -l "$dl/bin" | sed 's/^/  /'
}

cmd_vpn_install() {
  cmd_vpn_dl
  [ -s "$HERE/xray/config.json" ] || die "缺 $HERE/xray/config.json（真实配置，不入 git）。照 $HERE/xray/config.json.example 填好 UUID/服务器地址。"
  need_board

  local BH; BH="$(board_home)"
  local stage; stage="$(mktemp -d)"
  mkdir -p "$stage/xray"
  cp "$HERE/xray/dl/bin/xray" "$stage/xray/xray"
  cp "$HERE/xray/dl/bin/geoip.dat" "$HERE/xray/dl/bin/geosite.dat" "$stage/xray/"
  cp "$HERE/xray/config.json" "$stage/xray/config.json"

  say "推送到板子 $BH/$XRAY_DIR/xray/（压缩后约十几 MB）"
  push_tar "$stage/xray" "$BH/$XRAY_DIR"
  rm -rf "$stage"

  say "板子上：装二进制/geo/配置 + systemd 服务"
  bsudo_file vpn-install <<EOS
set -euo pipefail
SRC="$BH/$XRAY_DIR/xray"
install -D -m 0755 "\$SRC/xray"       "$XRAY_BIN"
install -D -m 0644 "\$SRC/geoip.dat"  "$XRAY_SHARE/geoip.dat"
install -D -m 0644 "\$SRC/geosite.dat" "$XRAY_SHARE/geosite.dat"
mkdir -p "$XRAY_ETC"
install -m 0600 "\$SRC/config.json" "$XRAY_ETC/00-config.json"

cat > "$XRAY_UNIT" <<'UNIT'
[Unit]
Description=Xray (K3 local proxy for GitHub/ghcr)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=XRAY_LOCATION_ASSET=/usr/local/share/xray
ExecStart=/usr/local/bin/xray run -confdir /etc/xray-k3
Restart=always
RestartSec=3
LimitNOFILE=65535
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX AF_NETLINK

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now xray-k3
sleep 2
echo "--- 状态 ---"
systemctl is-active xray-k3 || true
echo "--- 监听（应看到 127.0.0.1:${XRAY_PORT}）---"
ss -ltnp 2>/dev/null | grep -E ":$XRAY_PORT\\b" || echo "  ⚠️ 没看到 $XRAY_PORT 监听，看下面的日志"
echo "--- 日志 ---"
journalctl -u xray-k3 -n 20 --no-pager || true
EOS
  echo
  echo "  下一步： bash $0 vpn-test"
}

cmd_vpn_test() {
  need_board
  say "从板子上测代理（端口 ${XRAY_PORT}）"
  bscript <<EOS
P=$XRAY_PORT
echo "--- 服务 ---"
systemctl is-active xray-k3 2>/dev/null || echo "  (未运行)"
echo "--- 直连 ---"
curl -s -o /dev/null -m 8 -w "  direct  github   http=%{http_code}  %{time_total}s\n" https://github.com/ || echo "  direct  github   失败（预期，TLS 被断）"
echo "--- 走代理 ---"
curl -s -o /dev/null -m 20 --proxy socks5h://127.0.0.1:\$P -w "  proxy   github   http=%{http_code}  %{time_total}s\n" https://github.com/ || echo "  proxy   github   失败"
curl -s -o /dev/null -m 20 --proxy socks5h://127.0.0.1:\$P -w "  proxy   ghcr.io  http=%{http_code}  %{time_total}s\n" https://ghcr.io/v2/ || echo "  proxy   ghcr.io  失败"
printf "  proxy   出口 IP  "; curl -s -m 20 --proxy socks5h://127.0.0.1:\$P https://api.ipify.org || echo -n "(取不到)"; echo
echo "--- 走代理 git clone ---"
rm -rf /tmp/gt; if git -c http.proxy=socks5h://127.0.0.1:\$P clone --depth 1 -q https://github.com/XTLS/Xray-core /tmp/gt 2>/tmp/gt.err; then echo "  ✅ clone OK"; else echo "  ✗ clone 失败： \$(tail -1 /tmp/gt.err)"; fi
rm -rf /tmp/gt /tmp/gt.err
EOS
}

cmd_vpn_status() {
  need_board
  say "板子上代理状态"
  bscript <<EOS
echo "unit     : \$([ -f $XRAY_UNIT ] && echo 已安装 || echo 未安装)"
echo "active   : \$(systemctl is-active xray-k3 2>/dev/null || echo -)"
echo "enabled  : \$(systemctl is-enabled xray-k3 2>/dev/null || echo -)"
echo "binary   : \$(ls -l $XRAY_BIN 2>/dev/null || echo '-')"
echo "geo      : \$(ls -l $XRAY_SHARE/*.dat 2>/dev/null | tr '\n' ' ')"
echo "version  : \$($XRAY_BIN version 2>/dev/null | head -1)"
echo "listen   :"; ss -ltn 2>/dev/null | grep -E ":$XRAY_PORT\\b" | sed 's/^/  /' || echo "  -"
echo "confdir  : \$(ls $XRAY_ETC 2>/dev/null | tr '\n' ' ')"
echo "docker代理: \$(grep -h PROXY /etc/systemd/system/docker.service.d/http-proxy.conf 2>/dev/null | sed 's/^/  /' || echo '  (未配)')"
echo "git代理  : \$(git config --global --get-regexp proxy 2>/dev/null | sed 's/^/  /' || echo '  (未配)')"
EOS
}

cmd_vpn_rm() {
  need_board
  warn "停掉并移除板子上的 xray-k3 服务（保留 $XRAY_BIN 与 geo 数据）"
  ask "继续？" || { echo "  已取消"; return 0; }
  bsudo_file vpn-rm <<EOS
set -euo pipefail
systemctl disable --now xray-k3 2>/dev/null || true
rm -f "$XRAY_UNIT"
rm -rf "$XRAY_ETC"
systemctl daemon-reload
echo "  已移除。\$(systemctl is-active xray-k3 2>/dev/null || echo '服务已停')"
EOS
}

# ---- 子命令：proxy-on / proxy-off -----------------------------------------
# 把 docker daemon 与 pingji 的 git 指向上面的本机代理
cmd_proxy_on() {
  need_board
  say "docker daemon + git 走本机代理 127.0.0.1:$XRAY_PORT"
  bsudo_file proxy-on <<EOS
set -euo pipefail
mkdir -p /etc/systemd/system/docker.service.d
cat > /etc/systemd/system/docker.service.d/http-proxy.conf <<'CONF'
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:$XRAY_PORT"
Environment="HTTPS_PROXY=http://127.0.0.1:$XRAY_PORT"
Environment="ALL_PROXY=socks5h://127.0.0.1:$XRAY_PORT"
Environment="NO_PROXY=localhost,127.0.0.1,::1,192.168.0.0/16,10.0.0.0/8,172.16.0.0/12,mirrors.aliyun.com"
CONF
systemctl daemon-reload
systemctl restart docker
sleep 2
systemctl is-active docker
echo "--- dockerd 环境 ---"
systemctl show docker -p Environment --value | tr ' ' '\n' | grep -i proxy | sed 's/^/  /' || true
echo "--- 用 dockerd 的代理试拉一个小镜像 ---"
docker pull hello-world 2>&1 | tail -3
EOS
  echo
  bssh "git config --global http.https://github.com/.proxy socks5h://127.0.0.1:$XRAY_PORT && git config --global --get-regexp proxy" | sed 's/^/  /'
  echo
  echo "  提示：交互式 shell 想用代理就 export ALL_PROXY=socks5h://127.0.0.1:$XRAY_PORT"
}

cmd_proxy_off() {
  need_board
  say "撤掉 docker daemon 的代理配置（git 的也清掉）"
  bsudo_file proxy-off <<'EOS'
set -euo pipefail
rm -f /etc/systemd/system/docker.service.d/http-proxy.conf
systemctl daemon-reload
systemctl restart docker
sleep 1
systemctl is-active docker
echo "  已撤掉 docker 代理"
EOS
  bssh "git config --global --unset-all http.https://github.com/.proxy 2>/dev/null; echo '  已清掉 git 代理'" || true
}

# ---- 分发 -----------------------------------------------------------------
case "${1:-check}" in
  check)       cmd_check ;;
  snapshot)    cmd_snapshot ;;
  prep)        cmd_prep ;;
  kernel)      cmd_kernel ;;
  reboot)      cmd_reboot ;;
  mods)        cmd_mods ;;
  verify-host) cmd_verify_host ;;
  vpn-dl)      cmd_vpn_dl ;;
  vpn-install) cmd_vpn_install ;;
  vpn-test)    cmd_vpn_test ;;
  vpn-status)  cmd_vpn_status ;;
  vpn-rm)      cmd_vpn_rm ;;
  proxy-on)    cmd_proxy_on ;;
  proxy-off)   cmd_proxy_off ;;
  deploy)      cmd_deploy ;;
  prefetch)    cmd_prefetch ;;
  run)         cmd_run ;;
  smoke)       cmd_smoke ;;
  shot)        cmd_shot ;;
  push-img)    cmd_push_img ;;
  status)      cmd_status ;;
  logs)        cmd_logs ;;
  stop)        cmd_stop ;;
  tunnel-add)  cmd_tunnel_add ;;
  tunnel-rm)   cmd_tunnel_rm ;;
  expose)      cmd_expose ;;
  unexpose)    cmd_unexpose ;;
  rollback)    cmd_rollback ;;
  # 帮助：只打印文件开头的注释块
  *) awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/,""); print }' "$0" ;;
esac
