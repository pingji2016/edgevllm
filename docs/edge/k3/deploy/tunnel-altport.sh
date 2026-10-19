#!/bin/sh
# tunnel-altport.sh —— 把 K3 反向隧道从默认 22 换到备用端口（默认 8443）
#
# 为什么需要（2026-10-19 事故）：
#   板子 → 111:22 的 SSH 会话**死在 banner 交换阶段**（111 发 SSH-2.0 包，板子一个 ACK 都没有），
#   而同一个公网 IP → 111:3000 的 HTTP 心跳完全正常。这是典型的
#   「中间设备按端口 22 / SSH 特征掐流」（家用路由器禁止 SSH 外连，或运营商 DPI）。
#   换到 8443（HTTPS-alt）即可绕开。
#
# 用法（在板子上跑）：
#   ./tunnel-altport.sh                       # 用 8443
#   ./tunnel-altport.sh 8443                  # 指定端口
#   ./tunnel-altport.sh --revert              # 回滚所有备份
#
# ★ 前提：111 上 sshd 已监听 8443，且云安全组已放行 TCP 8443。
# ★ 不能用 2222：反向转发自己就要绑 127.0.0.1:2222，会和 sshd 的 *:2222 冲突。
# ★ 会同时改主 unit 和 drop-in：drop-in 里的 `ExecStart=` 会整体清空原值，
#   只改主 unit 是无效的（见 deploy/k3/tunnel/android.conf.example 的注释）。

set -eu

UNIT=/etc/systemd/system/k3-tunnel.service
DROPIN_DIR=$UNIT.d
PORT=8443
REVERT=0

case "${1:-}" in
    --revert) REVERT=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    "") ;;
    *) PORT=$1 ;;
esac

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

[ -f "$UNIT" ] || { echo "找不到 $UNIT（这台不是 K3 板子？）" >&2; exit 1; }

# ---------- 回滚 ----------
if [ "$REVERT" = "1" ]; then
    n=0
    for b in "$UNIT".bak.* "$DROPIN_DIR"/*.bak.*; do
        [ -f "$b" ] || continue
        orig=$(echo "$b" | sed -E 's/\.bak\.[0-9-]+$//')
        echo "回滚 $orig  ←  $b"
        $SUDO cp -a "$b" "$orig"
        n=$((n+1))
    done
    [ "$n" -gt 0 ] || { echo "没有备份可回滚" >&2; exit 1; }
    $SUDO systemctl daemon-reload
    $SUDO systemctl restart k3-tunnel
    sleep 4
    systemctl is-active k3-tunnel
    exit 0
fi

# ---------- 收集所有可能定义 ExecStart 的 fragment ----------
set -- "$UNIT"
if [ -d "$DROPIN_DIR" ]; then
    for f in "$DROPIN_DIR"/*.conf; do [ -f "$f" ] && set -- "$@" "$f"; done
fi

TARGETS=""
HOST=""
for f in "$@"; do
    # 匹配「只有 user@host 一行」的 fragment（可能已带 -p N）
    line=$(grep -hE "^[[:space:]]+(-p [0-9]+ )?[^[:space:]]+@[^[:space:]]+[[:space:]]*$" "$f" 2>/dev/null | head -1)
    [ -n "$line" ] || continue
    TARGETS="$TARGETS $f"
    [ -z "$HOST" ] && HOST=$(echo "$line" | sed -E 's/.*@//; s/[[:space:]]*$//')
done

if [ -z "$TARGETS" ]; then
    echo "没在 $UNIT 或 $DROPIN_DIR/*.conf 里找到 \"user@host\" 行，请手工检查：" >&2
    systemctl cat k3-tunnel 2>/dev/null | sed -n '/ExecStart/,$p' | head -20 >&2
    exit 1
fi

echo "将修改：$TARGETS"
echo "目标主机：${HOST:-（未知）}  端口：$PORT"

# ---------- 连通性预检 ----------
if [ -n "$HOST" ] && command -v timeout >/dev/null 2>&1; then
    if timeout 6 bash -c "</dev/tcp/$HOST/$PORT" 2>/dev/null; then
        echo "预检：$HOST:$PORT 可连 ✅"
    else
        echo "预检：$HOST:$PORT **连不上** —— 先确认云安全组放行了 $PORT。" >&2
        [ "${FORCE:-0}" = "1" ] || { echo "      确认无误后加 FORCE=1 再跑。" >&2; exit 1; }
    fi
fi

# ---------- 备份并改端口 ----------
STAMP=$(date +%Y%m%d-%H%M%S)
for f in $TARGETS; do
    $SUDO cp -a "$f" "$f.bak.$STAMP"
    $SUDO sed -i -E "s#^([[:space:]]+)(-p [0-9]+ )?([^[:space:]]+@[^[:space:]]+)[[:space:]]*\$#\1-p $PORT \3#" "$f"
    echo "  改好 $f"
done

echo "=== 改后 ExecStart（生效值）==="
systemctl show k3-tunnel -p ExecStart --value 2>/dev/null | tr ';' '\n' | grep -E "ssh|@|-p " | head -5

# ---------- 重启并验证 ----------
$SUDO systemctl daemon-reload
$SUDO systemctl restart k3-tunnel
sleep 5

echo "=== 状态 ==="
systemctl is-active k3-tunnel || true
echo "=== 最近日志 ==="
journalctl -u k3-tunnel -n 10 --no-pager 2>/dev/null | tail -10

echo
echo "在 111 上验证： ss -ltn | grep 2222   （应看到 127.0.0.1:2222）"
echo "回滚： $0 --revert"
