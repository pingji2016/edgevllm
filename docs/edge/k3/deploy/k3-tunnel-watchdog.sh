#!/bin/bash
# k3-tunnel-watchdog.sh —— 反向隧道「卡死」自愈
#
# 解决的事故（2026-10-19，3 小时失联）：
#   板子 → 111:22 的 ssh 连上了 TCP，但 SSH 握手被中间设备掐断。
#   此时：
#     - ConnectTimeout=10 早就不管了（它只管 TCP connect）
#     - ServerAliveInterval=30 **只在认证成功后才生效**
#   于是 ssh 永远挂在 banner 交换里既不退出、也不超时，
#   systemd 的 Restart=always 自然永远不触发 —— 隧道"活着但死了"。
#
# 判据：隧道那条 ESTAB socket 的 Send-Q 持续 > 0（本端有数据一直没被对端 ACK）。
#   连续两次（间隔 60s）命中 → 重启 k3-tunnel。
#   健康但空闲的隧道 Send-Q 为 0，不会被误杀。
#
# 安装：
#   sudo install -m755 k3-tunnel-watchdog.sh /usr/local/bin/
#   sudo install -m644 k3-tunnel-watchdog.service /etc/systemd/system/
#   sudo install -m644 k3-tunnel-watchdog.timer   /etc/systemd/system/
#   sudo systemctl daemon-reload && sudo systemctl enable --now k3-tunnel-watchdog.timer

set -u
STAMP=/run/k3-tunnel-watchdog.stamp

PID=$(systemctl show -p MainPID --value k3-tunnel 2>/dev/null)
case "${PID:-0}" in
    0|"") rm -f "$STAMP"; exit 0 ;;
esac

# ss -tnp 列：State Recv-Q Send-Q Local Peer Process
SQ=$(ss -tnp 2>/dev/null \
     | awk -v pat="pid=${PID}," '$1=="ESTAB" && index($0, pat) {print $3; exit}')

if [ -z "${SQ:-}" ]; then
    rm -f "$STAMP"
    exit 0
fi

if [ "$SQ" -gt 0 ] 2>/dev/null; then
    if [ -f "$STAMP" ]; then
        logger -t k3-tunnel-watchdog "隧道 Send-Q=${SQ} 持续未消（疑似 SSH 握手被掐），重启 k3-tunnel"
        systemctl restart k3-tunnel
        rm -f "$STAMP"
    else
        date +%s > "$STAMP"
        logger -t k3-tunnel-watchdog "隧道 Send-Q=${SQ}，再观察一个周期"
    fi
else
    rm -f "$STAMP"
fi
