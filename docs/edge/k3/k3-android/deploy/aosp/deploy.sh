#!/usr/bin/env bash
# AOSP（riscv64 / Cuttlefish on K3）—— 云上编译机驱动
#
# 设计依据与配方： docs/k3-android-cuttlefish.md §5.1~§5.3
# 上游仓库：       https://github.com/monkey-jsun/aosp-cuttlefish-riscv64-addons
#
# 这台机器是**一次性的**：x86_64 + 32GB 内存起 + 500GB SSD + Ubuntu 22.04/24.04。
# 同一个脚本两种跑法：
#   · 在 Mac 上跑   —— 自动 ssh 到编译机执行（HOST=root@1.2.3.4）
#   · 在编译机上跑  —— 直接本地执行（Linux 自动判定；也可 LOCAL=1 强制）
#
# 用法：
#   HOST=root@1.2.3.4 bash deploy/aosp/deploy.sh check      # 体检：架构/内存/磁盘/JDK/网络连通性
#                   ... deps       # 装 AOSP 构建依赖 + repo + ccache（幂等）
#                   ... branches   # 列出上游可选 branch/tag（用来决定 BRANCH）
#                   ... mirror-on  # 只把 android.googlesource.com 重定向到 TUNA（国内区域才需要）
#                   ... mirror-off # 撤掉上面的重定向
#                   ... init       # repo init（默认 pin BRANCH=v1.3.1，与板子上跑的镜像同源）
#                   ... sync       # repo sync（~150GB，首次 1~2h）
#                   ... apks       # 下载预装 APK（WebView / F-Droid，带 sha256 校验）
#                   ... patch      # 打上游对 AOSP 的补丁（幂等）
#                   ... build      # lunch + m dist（首次 2~4h，产物 ~900MB）
#                   ... all        # deps → init → sync → apks → patch → build → artifacts
#                   ... artifacts  # 列出 out/dist 产物 + sha256
#                   ... kernel     # 另一条线：交叉编译 K3 定制内核 deb（约 10 分钟）
#                   ... status     # 树/分支/上次产出/磁盘/后台任务
#                   ... logs       # 打印并跟最新一条后台日志
#                   ... fetch      # （Mac 侧）把产物 zip 拉回本机，供 deploy/k3/deploy.sh push-img 用
#                   ... clean      # 删掉源码树/内核树（危险，需确认）
#
# 开关（也可以写进 deploy/aosp/.env.aosp）：
#   HOST=root@1.2.3.4      编译机；远端模式下必填
#   LOCAL=1                在 Linux 上强制本地执行
#   BG=1 / --bg            长任务派生到后台 + 写日志（**首次 sync / build 强烈建议这么跑**）
#   PROXY=socks5h://127.0.0.1:10808   国内区域给 git/curl 挂代理
#   JOBS=8                 repo sync 并发数
#   BUILD_NUMBER=v1.3.1    产物名里那个编号（自己改的版本记得换个号，别跟官方 release 混）
#   USE_CCACHE=1           启用 ccache（增量编更快，但吃盘；默认关）
#
# 典型一天：
#   1) bash deploy/aosp/deploy.sh check                 # 确认规格够、网络通
#   2) HOST=root@x.x.x.x BG=1 bash deploy/aosp/deploy.sh all
#   3) bash deploy/aosp/deploy.sh logs                  # 看着它编
#   4) bash deploy/aosp/deploy.sh fetch                 # 拉回产物
#   5) bash deploy/k3/deploy.sh push-img <zip> && bash deploy/k3/deploy.sh stop && bash deploy/k3/deploy.sh run
set -euo pipefail

# ---------------------------------------------------------------- 双模式判定
MODE=local
if [ "$(uname -s)" = "Darwin" ] && [ "${LOCAL:-0}" != "1" ]; then
  MODE=remote
fi

SELF="${BASH_SOURCE[0]:-}"
if [ -n "$SELF" ] && [ -f "$SELF" ]; then
  HERE="$(cd "$(dirname "$SELF")" && pwd)"
else
  HERE="$PWD"
fi

# ---------------------------------------------------------------- 配置
# 相对路径一律相对于**编译机上**的 $HOME（这样 Mac 和远端不会各算各的）
TREE="${TREE:-android17-release}"                 # 源码树目录（相对 $HOME 或绝对路径）
KERNEL_DIR="${KERNEL_DIR:-linux-6.18-k3}"         # 内核树目录

ADDONS_REPO="${ADDONS_REPO:-https://github.com/monkey-jsun/aosp-cuttlefish-riscv64-addons}"
BRANCH="${BRANCH:-v1.3.1}"                        # pin 住，复现板子上正在跑的那版；要最新用 main
MANIFEST="${MANIFEST:-manifests/aosp-cf-js-baseline.xml}"
PRODUCT="${PRODUCT:-aosp_cf_riscv64_phone_js-trunk_staging-userdebug}"
BUILD_NUMBER="${BUILD_NUMBER:-v1.3.1}"            # 只决定产物名；自己改的版本记得换个编号，别跟官方 release 混

KERNEL_REPO="${KERNEL_REPO:-https://github.com/monkey-jsun/linux-6.18-k3.git}"
KERNEL_BRANCH="${KERNEL_BRANCH:-cf-k3-v1.3.0}"
KERNEL_DEFCONFIG="${KERNEL_DEFCONFIG:-config-aia2-kvm-modular}"
KERNEL_LOCALVERSION="${KERNEL_LOCALVERSION:--cf-k3-1.3.0}"
KERNEL_PKGVERSION="${KERNEL_PKGVERSION:-6.18.3-cf-k3-1.3.0}"

JOBS="${JOBS:-8}"
PROXY="${PROXY:-}"
USE_CCACHE="${USE_CCACHE:-0}"
BG="${BG:-0}"
FETCH_PATTERN="${FETCH_PATTERN:-*-img-*.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

REPO_URL="${REPO_URL:-https://storage.googleapis.com/git-repo-downloads/repo}"
REPO_URL_CN="${REPO_URL_CN:-https://mirrors.tuna.tsinghua.edu.cn/git/git-repo/repo}"
MIRROR_AOSP="${MIRROR_AOSP:-https://mirrors.tuna.tsinghua.edu.cn/git/AOSP/}"
GOOGLESOURCE="${GOOGLESOURCE:-https://android.googlesource.com/}"

GIT_NAME="${GIT_NAME:-aosp-builder}"
GIT_EMAIL="${GIT_EMAIL:-aosp-builder@localhost}"
SSH_KEY="${SSH_KEY:-}"

[ -f "$HERE/.env.aosp" ] && . "$HERE/.env.aosp"

# ---------------------------------------------------------------- 小工具
say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
info() { printf '\033[0;37m    %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m    ✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!  %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
ask()  { printf '\033[1;33m?  %s [y/N] \033[0m' "$*"; read -r r; [ "$r" = y ] || [ "$r" = Y ]; }
usage() { awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/,""); print }' "$0"; }

tree_abs()   { case "$TREE" in /*) printf '%s' "$TREE" ;; *) printf '%s/%s' "$HOME" "$TREE" ;; esac; }
kernel_abs() { case "$KERNEL_DIR" in /*) printf '%s' "$KERNEL_DIR" ;; *) printf '%s/%s' "$HOME" "$KERNEL_DIR" ;; esac; }
tree_ok()    { [ -d "$(tree_abs)/.repo" ]; }

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR)
[ -n "$SSH_KEY" ] && SSH_OPTS+=(-i "$SSH_KEY")

remote() { ssh "${SSH_OPTS[@]}" "$HOST" "$1" </dev/null; }
rcopy_in() { ssh "${SSH_OPTS[@]}" "$HOST" "$1"; }   # 本地 stdin 透传给远端
asroot() { if [ "$(id -u)" = 0 ]; then "$@"; else sudo "$@"; fi; }

# 把配置原样送到远端（值里可能有空格/引号，用 %q 转义）
FORWARD_KEYS=(TREE KERNEL_DIR ADDONS_REPO BRANCH MANIFEST PRODUCT BUILD_NUMBER
  KERNEL_REPO KERNEL_BRANCH KERNEL_DEFCONFIG KERNEL_LOCALVERSION KERNEL_PKGVERSION
  JOBS PROXY USE_CCACHE BG FETCH_PATTERN OUTPUT_DIR
  REPO_URL REPO_URL_CN MIRROR_AOSP GOOGLESOURCE GIT_NAME GIT_EMAIL)
env_string() {
  local out="" k
  for k in "${FORWARD_KEYS[@]}"; do out+=" $k=$(printf '%q' "${!k}")"; done
  printf '%s' "$out"
}

# 脚本自身的指纹：远端按指纹存成不同文件，**绝不覆盖正在跑的脚本**
self_hash() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$SELF" | cut -c1-8
  else sha256sum "$SELF" | cut -c1-8; fi
}

# ---------------------------------------------------------------- 远端模式
remote_dispatch() {
  [ -n "${HOST:-}" ] || die "远端模式需要指定编译机： HOST=root@1.2.3.4 bash $0 $*"
  local args="" a
  for a in "$@"; do args+=" $(printf '%q' "$a")"; done
  local h; h="$(self_hash)"
  local rdir='$HOME/.aosp-deploy'
  local rfile="$rdir/deploy-$h.sh"

  info "编译机： $HOST"
  info "脚本指纹： $h"
  # 1) 落地脚本（内容没变就跳过写）
  rcopy_in "mkdir -p $rdir && cat > $rfile.tmp && mv -f $rfile.tmp $rfile && chmod +x $rfile" < "$SELF"
  # 2) 执行
  remote "env$(env_string) LOCAL=1 bash $rfile$args"
}

# ---------------------------------------------------------------- 体检
cmd_check() {
  say "编译机体检"
  local arch; arch="$(uname -m)"
  printf '    %-22s %s\n' "架构" "$arch"
  if [ "$arch" = "x86_64" ]; then ok "架构正确（上游明确：AOSP 不支持 riscv64 当构建主机）"
  else die "必须是 x86_64，当前是 $arch"; fi

  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    printf '    %-22s %s\n' "系统" "${PRETTY_NAME:-未知}"
    case "${VERSION_ID:-}" in
      22.04|24.04) ok "AOSP 官方支持的发行版" ;;
      *) warn "AOSP 官方支持 22.04 / 24.04，当前 ${VERSION_ID:-未知}，可能踩依赖坑" ;;
    esac
  else
    warn "没有 /etc/os-release（不是 Linux？）"
  fi

  local nc pu memg freeg
  nc="$(nproc 2>/dev/null || echo 0)"
  printf '    %-22s %s\n' "CPU" "$nc 核"
  memg="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')" || memg=0
  printf '    %-22s %s GB\n' "内存" "${memg:-0}"
  if [ "${memg:-0}" -ge 60 ]; then ok "内存舒服（≥60GB）"
  elif [ "${memg:-0}" -ge 30 ]; then ok "内存够用（≥30GB）"
  else warn "内存 <30GB，soong 链接阶段会很痛苦（建议 32GB 起 / 64GB 舒服）"; fi

  local ta; ta="$(tree_abs)"
  local probe="$HOME"; [ -d "$ta" ] && probe="$ta"
  printf '    %-22s %s\n' "源码树" "$ta $([ -d "$ta" ] && echo '(已存在)' || echo '(还没建)')"
  printf '    %-22s %s\n' "构建目录所在盘" "$probe"
  df -h "$probe" | awk 'NR==1{print "    "$0} NR==2{gsub(/[0-9]+%/,"&"); print "    "$0}'
  freeg="$(df -BG --output=avail "$probe" 2>/dev/null | tail -1 | tr -dc '0-9')" || freeg=""
  if [ -n "${freeg:-}" ]; then
    printf '    %-22s %s GB\n' "可用空间" "$freeg"
    if [ "$freeg" -ge 500 ]; then ok "磁盘充足（官方数字：同步 ~150GB、编完 ~320GB）"
    elif [ "$freeg" -ge 350 ]; then warn "可用 ${freeg}GB，够但没余量（编完约 320GB）"
    else die "可用 ${freeg}GB —— 太小，repo sync 会撞在磁盘满上"; fi
  fi

  if command -v java >/dev/null 2>&1; then
    local jv; jv="$(java -version 2>&1 | head -1)"
    printf '    %-22s %s\n' "JDK" "$jv"
    case "$jv" in *'"21'*|*' 21'*) ok "JDK 21（Android 16+ 需要）" ;; *) warn "不是 JDK 21，trunk_staging 会要 JDK 21" ;; esac
  else
    warn "没装 JDK —— 跑一次 deps"
  fi

  command -v repo >/dev/null 2>&1 && ok "repo 已就位（$(command -v repo)）" || warn "没有 repo —— 跑一次 deps"
  command -v git  >/dev/null 2>&1 && ok "git  $(git --version | awk '{print $3}')" || die "没装 git"

  say "网络连通性（决定要不要 mirror-on / PROXY）"
  local url code t rc=0
  for url in \
    "https://github.com/monkey-jsun/aosp-cuttlefish-riscv64-addons/info/refs?service=git-upload-pack" \
    "https://android.googlesource.com/" \
    "https://storage.googleapis.com/git-repo-downloads/repo" \
    "https://mirrors.tuna.tsinghua.edu.cn/git/AOSP/" ; do
    code="$(curl -sSL -o /dev/null -m 20 -w '%{http_code}' "$url" 2>/dev/null)" || code="000"
    t="$(curl -sSL -o /dev/null -m 20 -w '%{time_total}' "$url" 2>/dev/null)" || t="-"
    if [ "$code" = "200" ] || [ "$code" = "401" ]; then
      ok "$(printf '%-52s' "$(echo "$url" | cut -c1-52)") HTTP $code  ${t}s"
    else
      printf '\033[1;31m    ✗ %-52s HTTP %s\033[0m\n' "$(echo "$url" | cut -c1-52)" "$code"
      rc=1
    fi
  done
  if [ "$rc" = 1 ]; then
    warn "有不通的地址："
    info "· GitHub 不通 / 很慢 → 换境外区域，或 PROXY=socks5h://… 挂代理"
    info "· googlesource 不通   → bash $0 mirror-on（只重定向 googlesource 到 TUNA）"
  else
    ok "四个关键地址都通 —— 不用折腾镜像和代理"
  fi

  [ -n "$PROXY" ] && info "已配 PROXY=$PROXY（git 与 curl 都会走它）"

  say "体检结论"
  if [ "$rc" = 0 ]; then ok "这台机器可以直接跑 all"
  else warn "先把上面的网络问题解决，再跑 all"; fi
}

# ---------------------------------------------------------------- 依赖
cmd_deps() {
  say "apt 基础依赖"
  asroot apt-get update -qq
  # 注：libc6-dev-i386 / lib32z1-dev 是给 32 位宿主工具的，Android 16+ 不需要，
  #     且在 24.04 上还要先 dpkg --add-architecture i386，所以这里不装。
  asroot apt-get install -y --no-install-recommends \
    git git-core gnupg flex bison build-essential zip unzip curl wget rsync \
    zlib1g-dev libc6-dev libncurses-dev libxml2-utils xsltproc \
    libgl1-mesa-dev fontconfig python3 python3-pip python3-venv \
    ccache tmux jq bc file \
    libssl-dev libelf-dev libdw-dev fakeroot dpkg-dev

  say "JDK 21"
  if apt-cache policy openjdk-21-jdk 2>/dev/null | grep -q Candidate; then
    asroot apt-get install -y --no-install-recommends openjdk-21-jdk
  else
    warn "源里没有 openjdk-21-jdk —— Ubuntu 22.04 需要手动加源，或直接用 24.04 镜像"
  fi

  say "repo 工具"
  mkdir -p "$HOME/bin"
  if command -v repo >/dev/null 2>&1; then
    ok "已有 repo： $(command -v repo)"
  else
    if curl -fsSL -m 60 "$REPO_URL" -o "$HOME/bin/repo" 2>/dev/null; then
      ok "从 Google Storage 取到 repo"
    elif curl -fsSL -m 60 "$REPO_URL_CN" -o "$HOME/bin/repo" 2>/dev/null; then
      ok "从 TUNA 镜像取到 repo"
    else
      asroot apt-get install -y repo && ok "用 apt 装了 repo（版本可能偏旧）"
    fi
    chmod a+x "$HOME/bin/repo" 2>/dev/null || true
  fi
  case ":$PATH:" in *":$HOME/bin:"*) ;; *) export PATH="$HOME/bin:$PATH"; warn "记得把 ~/bin 加进 PATH（本进程已临时加）";; esac

  say "git 身份（repo / 补丁脚本要求）"
  git config --global user.name  "$GIT_NAME"
  git config --global user.email "$GIT_EMAIL"
  git config --global color.ui false
  ok "user.name=$GIT_NAME  user.email=$GIT_EMAIL"

  if [ -n "$PROXY" ]; then
    git config --global http.proxy "$PROXY"
    git config --global https.proxy "$PROXY"
    ok "git 已挂代理 $PROXY"
  fi

  say "版本"
  info "java : $(java -version 2>&1 | head -1)"
  info "python: $(python3 --version 2>&1)"
  info "repo : $(repo --version 2>/dev/null | head -1 || echo '?')"
  ok "依赖就绪。下一步： bash $0 branches（挑 BRANCH）或直接 bash $0 init"
}

# ---------------------------------------------------------------- 镜像加速
cmd_mirror_on() {
  say "把 android.googlesource.com 重定向到 TUNA"
  git config --global "url.$MIRROR_AOSP.insteadOf" "$GOOGLESOURCE"
  ok "已设置： $GOOGLESOURCE  →  $MIRROR_AOSP"
  git config --global --get-regexp '^url\.' | sed 's/^/    /'
  warn "只动 googlesource，**不碰 GitHub**（manifest 里有 GitHub 上的项目）"
  info "撤销： bash $0 mirror-off"
}

cmd_mirror_off() {
  say "撤掉 TUNA 重定向"
  git config --global --remove-section "url.$MIRROR_AOSP" 2>/dev/null || warn "本来就没有这条配置"
  git config --global --get-regexp '^url\.' | sed 's/^/    /' || info "（已无任何 url.*.insteadOf）"
  ok "已撤销"
}

# ---------------------------------------------------------------- 上游分支
cmd_branches() {
  say "上游 $ADDONS_REPO 的 branch / tag"
  local out
  if [ -n "$PROXY" ]; then
    out="$(git -c http.proxy="$PROXY" ls-remote --heads --tags "$ADDONS_REPO" 2>/dev/null)" || out=""
  else
    out="$(git ls-remote --heads --tags "$ADDONS_REPO" 2>/dev/null)" || out=""
  fi
  [ -n "$out" ] || die "拉不到 $ADDONS_REPO（网络问题？先跑 bash $0 check，或给 PROXY=…）"
  echo "$out" | awk '{print "    "$0}'
  echo
  info "heads 用 BRANCH=<名字>；镜像版本用 tag 名（去掉 refs/tags/ 与 ^{}）"
  info "例： BRANCH=v1.3.1（已发布镜像） / BRANCH=main（最新开发分支）"
}

# ---------------------------------------------------------------- repo init
cmd_init() {
  need_repo_tool
  local ta; ta="$(tree_abs)"
  say "repo init  树=$ta  BRANCH=$BRANCH  MANIFEST=$MANIFEST"

  # branch/tag 存在性预检（省得 repo init 报一堆看不懂的错；一次网络调用搞定）
  say "预检 $BRANCH 是否存在"
  local ls_out
  if [ -n "$PROXY" ]; then
    ls_out="$(git -c http.proxy="$PROXY" ls-remote --heads --tags "$ADDONS_REPO" 2>/dev/null)" || ls_out=""
  else
    ls_out="$(git ls-remote --heads --tags "$ADDONS_REPO" 2>/dev/null)" || ls_out=""
  fi
  [ -n "$ls_out" ] || die "连不上 $ADDONS_REPO（网络？先跑 bash $0 check，或给 PROXY=…）"
  case "$ls_out" in
    *"refs/heads/$BRANCH"*) info "$BRANCH 是 branch" ;;
    *"refs/tags/$BRANCH"*)  info "$BRANCH 是 tag" ;;
    *)
      warn "$BRANCH 既不是 branch 也不是 tag；可用的有："
      echo "$ls_out" | awk '{print "      "$0}'
      die "换个 BRANCH 再试（或 bash $0 branches）" ;;
  esac

  mkdir -p "$ta"; cd "$ta"
  [ -n "$PROXY" ] && export http_proxy="$PROXY" https_proxy="$PROXY" all_proxy="$PROXY"
  export GIT_TERMINAL_PROMPT=0
  repo init --partial-clone --no-use-superproject \
    -u "$ADDONS_REPO" -b "$BRANCH" -m "$MANIFEST"
  ok "repo init 完成"
  info "下一步： bash $0 sync     （首次 ~150GB，1~2 小时；建议 BG=1）"
}

# ---------------------------------------------------------------- repo sync
cmd_sync() {
  need_tree
  local ta; ta="$(tree_abs)"
  say "repo sync -j$JOBS   （~150GB，耐心）"
  cd "$ta"
  [ -n "$PROXY" ] && export http_proxy="$PROXY" https_proxy="$PROXY" all_proxy="$PROXY"
  export GIT_TERMINAL_PROMPT=0
  repo sync -j"$JOBS" -c --no-tags --optimized-fetch --retry-fetches=3
  ok "sync 完成"
  info "磁盘： $(df -h "$ta" | awk 'NR==2{print $3" 已用 / "$4" 可用"}')"
}

# ---------------------------------------------------------------- 预装 APK + 补丁
cmd_apks() {
  need_tree
  local ta; ta="$(tree_abs)"
  local s="$ta/device/monkey-jsun/cuttlefish_riscv64/tools/fetch-apks.sh"
  [ -x "$s" ] || [ -f "$s" ] || die "找不到 $s（sync 没跑完？）"
  say "下载预装 APK（WebView / F-Droid，按 sha256 校验）"
  cd "$ta"
  [ -n "$PROXY" ] && export http_proxy="$PROXY" https_proxy="$PROXY" all_proxy="$PROXY"
  bash "$s"
  ok "APK 就绪"
}

cmd_patch() {
  need_tree
  local ta; ta="$(tree_abs)"
  local s="$ta/device/monkey-jsun/cuttlefish_riscv64/patches/apply.sh"
  [ -f "$s" ] || die "找不到 $s（sync 没跑完？）"
  say "打上游对 AOSP 的补丁（幂等）"
  cd "$ta"
  bash "$s"
  ok "补丁应用完成"
}

# ---------------------------------------------------------------- 编译
cmd_build() {
  need_tree
  local ta; ta="$(tree_abs)"
  say "lunch $PRODUCT"
  cd "$ta"
  [ -n "$PROXY" ] && export http_proxy="$PROXY" https_proxy="$PROXY" all_proxy="$PROXY"
  export GIT_TERMINAL_PROMPT=0
  if [ "$USE_CCACHE" = "1" ]; then
    export USE_CCACHE=1
    export CCACHE_EXEC="$(command -v ccache)"
    export CCACHE_DIR="$HOME/.ccache"
    export CCACHE_SIZE="${CCACHE_SIZE:-200G}"
    info "ccache 已启用： $CCACHE_DIR ($CCACHE_SIZE)"
  fi
  local jh; jh="$(ls -d /usr/lib/jvm/java-21-openjdk-* 2>/dev/null | head -1 || true)"
  [ -n "$jh" ] && export ANDROID_JAVA_HOME="$jh" && info "ANDROID_JAVA_HOME=$jh"

  # envsetup.sh 里有未定义变量引用，必须临时关掉 -u
  set +u
  # shellcheck disable=SC1091
  . build/envsetup.sh
  set -u
  lunch "$PRODUCT"
  say "m dist   （首次 2~4 小时；BUILD_NUMBER=$BUILD_NUMBER）"
  BUILD_NUMBER="$BUILD_NUMBER" m dist
  ok "编译完成"
  cmd_artifacts
}

cmd_artifacts() {
  local ta; ta="$(tree_abs)"
  say "产物 out/dist"
  [ -d "$ta/out/dist" ] || die "还没有 $ta/out/dist"
  ls -lh "$ta/out/dist" | sed 's/^/    /'
  local f
  for f in "$ta"/out/dist/*-img-"$BUILD_NUMBER".zip "$ta"/out/dist/*-img-*.zip; do
    [ -f "$f" ] || continue
    info "sha256 $(basename "$f")"
    ( cd "$ta/out/dist" && sha256sum "$(basename "$f")" | sed 's/^/      /' )
    break
  done
  echo
  info "拉回本机： bash $0 fetch"
  info "上板子：   bash deploy/k3/deploy.sh push-img <本机 zip> && bash deploy/k3/deploy.sh stop && bash deploy/k3/deploy.sh run"
}

# ---------------------------------------------------------------- 内核（另一条线）
cmd_kernel() {
  local kd; kd="$(kernel_abs)"
  say "交叉编译 K3 定制内核（x86_64 上编 riscv64，约 10 分钟）"
  asroot apt-get install -y --no-install-recommends gcc-riscv64-linux-gnu \
    bison flex libssl-dev libelf-dev bc fakeroot dpkg-dev

  if [ -d "$kd/.git" ]; then
    info "已有内核树： $kd"
    cd "$kd"
    git fetch --depth 1 origin "$KERNEL_BRANCH" || die "拉不到 $KERNEL_BRANCH（网络？试 PROXY=…）"
    git checkout -f FETCH_HEAD
  else
    git clone --depth 1 -b "$KERNEL_BRANCH" "$KERNEL_REPO" "$kd"
    cd "$kd"
  fi
  [ -f "$KERNEL_DEFCONFIG" ] || die "找不到 defconfig： $kd/$KERNEL_DEFCONFIG"
  cp "$KERNEL_DEFCONFIG" .config && touch .scmversion

  make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- olddefconfig
  say "make bindeb-pkg   （LOCALVERSION=$KERNEL_LOCALVERSION）"
  make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j"$(nproc)" \
       LOCALVERSION="$KERNEL_LOCALVERSION" KDEB_PKGVERSION="$KERNEL_PKGVERSION" bindeb-pkg

  say "内核产物"
  ls -lh "$kd"/../linux-image-*.deb "$kd"/../linux-headers-*.deb 2>/dev/null | sed 's/^/    /' || true
  echo
  info "上板子： 让 deploy/k3/deploy.sh 去装（阶段 1 的同一条路）"
  info "回滚：   板上 /boot/env_k3.txt 指回旧内核（旧内核保留着）"
}

# ---------------------------------------------------------------- 状态 / 日志
cmd_status() {
  say "编译机状态（$(hostname 2>/dev/null || echo '?')  $(uname -m)）"
  local ta; ta="$(tree_abs)"
  printf '    %-16s %s %s\n' "源码树" "$ta" "$(tree_ok && echo '(repo 树就在)' || echo '(还没 init)')"
  if tree_ok; then
    cd "$ta"
    printf '    %-16s %s\n' "分支" "$(git -C .repo/manifests rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
    printf '    %-16s %s\n' "manifest" "$(readlink .repo/manifest.xml 2>/dev/null | sed 's|manifests/||' || echo '?')"
    printf '    %-16s %s\n' "占用" "$(du -sh --exclude=out "$ta" 2>/dev/null | cut -f1)"
    if [ -d out/dist ]; then
      printf '    %-16s %s\n' "上次产物" "$(ls -1t out/dist 2>/dev/null | head -1)"
      printf '    %-16s %s\n' "产物时间" "$(stat -c %y out/dist 2>/dev/null | cut -d. -f1)"
    fi
  fi
  local kd; kd="$(kernel_abs)"
  [ -d "$kd" ] && printf '    %-16s %s\n' "内核树" "$kd ($(git -C "$kd" describe --tags --always 2>/dev/null || echo '?'))"
  printf '    %-16s %s\n' "磁盘" "$(df -h "$HOME" | awk 'NR==2{print $3" 已用 / "$4" 可用 ("$5")"}')"
  printf '    %-16s %s\n' "ccache" "$([ "$USE_CCACHE" = 1 ] && du -sh "$HOME/.ccache" 2>/dev/null | cut -f1 || echo '未启用')"

  say "后台任务"
  local pids; pids="$(pgrep -fa 'aosp-deploy/deploy-.*\.sh' 2>/dev/null | grep -v pgrep || true)"
  if [ -n "$pids" ]; then echo "$pids" | sed 's/^/    /'
  else info "没有在跑的后台任务"; fi
}

cmd_logs() {
  local d="$HOME/.aosp-deploy/logs"
  [ -d "$d" ] || die "还没有日志（没跑过 BG=1 的任务）"
  local f; f="$(ls -1t "$d"/*.log 2>/dev/null | head -1 || true)"
  [ -n "$f" ] || die "$d 里没有 .log"
  say "最新日志： $f"
  tail -n 60 "$f"
  echo
  info "实时跟： tail -f $f"
  info "全部：   ls -lt $d"
}

# ---------------------------------------------------------------- 拉回产物（Mac 侧）
cmd_fetch() {
  [ "$MODE" = remote ] || die "fetch 要在 Mac 上跑（它负责把远端产物 scp 回来）"
  [ -n "${HOST:-}" ] || die "需要 HOST=root@1.2.3.4"
  local rh; rh="$(remote 'printf %s "$HOME"')"; [ -n "$rh" ] || die "连不上 $HOST"
  local rtree; case "$TREE" in /*) rtree="$TREE" ;; *) rtree="$rh/$TREE" ;; esac
  local out="${OUTPUT_DIR:-$HERE/../../out/aosp}"
  mkdir -p "$out"
  say "把 $HOST:$rtree/out/dist/$FETCH_PATTERN 拉回 $out"
  scp "${SSH_OPTS[@]}" "$HOST:$rtree/out/dist/$FETCH_PATTERN" "$out/"
  ls -lh "$out" | sed 's/^/    /'
  echo
  info "下一步： bash deploy/k3/deploy.sh push-img $(ls -1t "$out"/*.zip 2>/dev/null | head -1 || echo '<zip>')"
}

# ---------------------------------------------------------------- 一条龙
cmd_all() {
  cmd_deps
  cmd_init
  cmd_sync
  cmd_apks
  cmd_patch
  cmd_build
  cmd_artifacts
  say "全流程结束"
  info "拉回本机： bash $0 fetch"
}

# ---------------------------------------------------------------- 清理
cmd_clean() {
  local ta; ta="$(tree_abs)"
  local kd; kd="$(kernel_abs)"
  say "要删的东西"
  [ -d "$ta" ] && info "源码树  $ta  （$(du -sh "$ta" 2>/dev/null | cut -f1)）" || info "源码树  不存在"
  [ -d "$kd" ] && info "内核树  $kd  （$(du -sh "$kd" 2>/dev/null | cut -f1)）" || info "内核树  不存在"
  warn "这是**不可恢复**的：sync 要重来（1~2h）"
  info "只是想省钱的话：确认产物已 fetch 回本机，然后直接释放这台机器（盘做快照更划算）"
  ask "确认删除？" || die "已放弃"
  [ -d "$ta" ] && rm -rf "$ta" && ok "删了 $ta"
  [ -d "$kd" ] && rm -rf "$kd" && ok "删了 $kd"
  ok "清理完成"
}

# ---------------------------------------------------------------- 公共前置
need_repo_tool() {
  case ":$PATH:" in *":$HOME/bin:"*) ;; *) export PATH="$HOME/bin:$PATH" ;; esac
  command -v repo >/dev/null 2>&1 || die "没有 repo —— 先跑 bash $0 deps"
}
need_tree() {
  need_repo_tool
  tree_ok || die "源码树还没 init： $(tree_abs)  → 先跑 bash $0 init"
}

# ---------------------------------------------------------------- 分发
ARGS=()
for a in "$@"; do
  case "$a" in
    --bg) BG=1 ;;
    *)    ARGS+=("$a") ;;
  esac
done
CMD="${ARGS[0]:-help}"

# 未知子命令：本地打用法就够，别先抱怨 HOST
KNOWN=" check deps branches mirror-on mirror-off init sync apks patch build all artifacts kernel status logs fetch clean help -h --help "
case "$KNOWN" in
  *" $CMD "*) ;;
  *) warn "不认识的子命令： $CMD"; usage; exit 1 ;;
esac

# help 永远本地处理（远端模式下不该因为缺 HOST 就看不了解法）
if [ "$MODE" = remote ] && [ "$CMD" != "help" ] && [ "$CMD" != "-h" ] && [ "$CMD" != "--help" ]; then
  remote_dispatch "$@"
  exit 0
fi

# 长任务：派生到后台（BG=1）。子进程里 BG=0，避免递归。
case "$CMD" in
  sync|build|kernel|all|deps)
    if [ "$BG" = "1" ]; then
      [ -n "$SELF" ] || die "--bg 需要以文件形式运行（不能从 stdin 管道）"
      _bg_log="$HOME/.aosp-deploy/logs/$CMD-$(date +%Y%m%d-%H%M%S).log"
      mkdir -p "$(dirname "$_bg_log")"
      BG=0 LOCAL=1 nohup bash "$SELF" "$CMD" >"$_bg_log" 2>&1 &
      ok "已在后台跑： $CMD （pid $!）"
      echo "    日志： $_bg_log"
      echo "    跟日志： bash $0 logs"
      exit 0
    fi ;;
esac

case "$CMD" in
  help|-h|--help) usage ;;
  check)      cmd_check ;;
  deps)       cmd_deps ;;
  branches)   cmd_branches ;;
  mirror-on)  cmd_mirror_on ;;
  mirror-off) cmd_mirror_off ;;
  init)       cmd_init ;;
  sync)       cmd_sync ;;
  apks)       cmd_apks ;;
  patch)      cmd_patch ;;
  build)      cmd_build ;;
  all)        cmd_all ;;
  artifacts)  cmd_artifacts ;;
  kernel)     cmd_kernel ;;
  status)     cmd_status ;;
  logs)       cmd_logs ;;
  fetch)      cmd_fetch ;;
  clean)      cmd_clean ;;
  *)          warn "不认识的子命令： $CMD"; usage; exit 1 ;;
esac
