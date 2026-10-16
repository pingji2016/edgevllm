#!/usr/bin/env bash
# =============================================================================
#  K3 VPU(Linlon v5276) 硬件编码 —— 一键脚本
#
#  背景：板子镜像默认能用硬编，但普通用户直接跑 ffmpeg 会「段错误」。
#        根因不是硬件/内核，而是权限：MPP 的编码插件
#        libv4l2_linlonv5v7_codec2.so 会逐个探测 /dev/video0..31，
#        而 /dev/video0 是 root:video 0660。用户不在 video 组时探测全失败，
#        插件返回 NULL，ffmpeg 随即段错误。加进 video 组即可。
#
#  用法：
#    ./hw-encode.sh fix            # 一次性修权限（需要 sudo，把当前用户加进 video 组）
#    ./hw-encode.sh check          # 自检：能否硬编
#    ./hw-encode.sh in.mp4 out.mp4 # 硬编转码（H.264 或 HEVC，自动判断）
#    CODEC=hevc ./hw-encode.sh in.mp4 out.h265
#
#  可用编码器（ffmpeg -encoders）：h264_stcodec / hevc_stcodec / mjpeg_stcodec
#                                   vp8_stcodec / vp9_stcodec
# =============================================================================
set -uo pipefail

WARN='\033[33m'; OK='\033[32m'; ERR='\033[31m'; NC='\033[0m'
info() { printf "${OK}==>${NC} %s\n" "$*"; }
warn() { printf "${WARN}!! ${NC}%s\n" "$*"; }
die()  { printf "${ERR}xx ${NC}%s\n" "$*" >&2; exit 1; }

need_video_group() {
  id -nG | tr ' ' '\n' | grep -qx video
}

case "${1:-}" in
  fix)
    info "把用户 $USER 加进 video 组（MPP 编码插件需要能打开 /dev/video0）"
    sudo usermod -aG video "$USER" || die "usermod 失败"
    info "完成。重新登录（或 newgrp video）后生效。"
    exit 0 ;;
  check)
    info "检查 /dev/video0 可读性"
    if [ -r /dev/video0 ]; then info "/dev/video0 可读 ✓"; else
      warn "/dev/video0 不可读 —— 硬编会段错误。跑：$0 fix"; fi
    info "video 组成员资格：$(need_video_group && echo 是 || echo 否)"
    info "可用硬件编码器：" ; ffmpeg -hide_banner -encoders 2>/dev/null | grep -E "_stcodec" | sed 's/^/    /'
    info "自检：编 2 秒彩条"
    ffmpeg -y -v error -f lavfi -i testsrc2=size=1280x720:rate=30 -t 2 \
      -c:v h264_stcodec -b:v 2M /tmp/hwenc_check.h264 2>/tmp/hwenc_check.log \
      && info "硬编成功 ✓ $(stat -c%s /tmp/hwenc_check.h264) 字节" \
      || { warn "硬编失败，日志："; grep -v MPP-DEBUG /tmp/hwenc_check.log | head -5 | sed 's/^/    /'; }
    exit 0 ;;
  ""|-h|--help)
    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

IN="$1"; OUT="${2:?用法: $0 输入 输出}"
[ -f "$IN" ] || die "找不到输入 $IN"
[ -r /dev/video0 ] || die "/dev/video0 不可读 → 先跑 $0 fix"

CODEC="${CODEC:-h264}"
case "$CODEC" in
  h264|avc) ENC=h264_stcodec ;;
  hevc|h265) ENC=hevc_stcodec ;;
  mjpeg) ENC=mjpeg_stcodec ;;
  *) die "不支持的 CODEC=$CODEC（h264|hevc|mjpeg）" ;;
esac

info "硬编：$IN → $OUT  编码器=$ENC  码率=${BITRATE:-4M}"
T0=$(date +%s.%N)
ffmpeg -y -v error -i "$IN" -c:v "$ENC" -b:v "${BITRATE:-4M}" -pix_fmt nv12 "$OUT" 2>"${OUT}.log"
rc=$?
T1=$(date +%s.%N)
[ $rc -eq 0 ] || { warn "失败（退出码 $rc），日志："; grep -v MPP-DEBUG "${OUT}.log" | head -5 | sed 's/^/    /'; exit $rc; }
info "完成，用时 $(awk "BEGIN{printf \"%.2f\", $T1-$T0}")s，产物 $(stat -c%s "$OUT") 字节"
ffprobe -v error -count_frames -show_entries stream=codec_name,width,height,nb_read_frames -of default=nw=1 "$OUT" | sed 's/^/    /'
