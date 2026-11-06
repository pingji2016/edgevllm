#!/bin/bash
# net-baseline.sh —— 端侧 vs 云端要用的三个网络数字
#
# 用法：./net-baseline.sh [上传测试地址]
#   默认往 111 的站点 POST 8 MiB（随便一个路径，看的是传完的速度，不看响应）
#
# 实测（2026-10-20，K3 / WiFi wlP4p1s0）：
#   RTT 到 223.5.5.5：101 ms 平均（66–147 ms）
#   上行：288 KB/s ≈ 2.3 Mbps
#   → 10 秒语音（320 KB）上传要 1.11 s，云端语音"上传+往返+推理" ≈ 1.5–2.2 s
#   → 1080p H.264 一路 7.7 Mbps，远超上行能力：视频传不上去

set -u
URL=${1:-http://111.228.40.211/bbs/nonexistent-upload-probe}

echo "== RTT =="
ping -c 5 -W 2 223.5.5.5 | tail -2

echo "== 上行 =="
dd if=/dev/zero bs=1M count=8 2>/dev/null \
  | curl -s -o /dev/null -m 120 -w 'upload=%{speed_upload} B/s\n' --data-binary @- "$URL"

echo "== 下行 =="
curl -s -o /dev/null -m 120 -w 'download=%{speed_download} B/s\n' \
  http://111.228.40.211/bbs/img/k3-stack.png
