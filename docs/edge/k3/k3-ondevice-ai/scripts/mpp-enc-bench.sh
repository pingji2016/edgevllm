#!/bin/sh
# mpp-enc-bench.sh —— 验证「MPP 原生编码的码率控制真的生效」
#
# 背景（报告 §5.1 / §11-1）：ffmpeg 的 `*_stcodec` 封装没把 `-b:v` 映射到 MPP
# 的码率控制，硬编码率不可调。mpp_enc.c 直接调 VENC API 设置 eRcMode/bitrate，
# 这个脚本用同一段 NV12 扫多个码率，看输出码率是否跟着走，并和 ffmpeg 对照。
#
# 用法：
#   ./mpp-enc-bench.sh [源视频] [帧数] [输出目录]
#   默认：~/mage-vl/codec/src.mp4  150 帧  /tmp/mppenc
#
# 前置：用户需在 video 组（./hw-encode.sh fix），且 mpp_enc.c 与 k3aix.c 同目录。

set -u
DIR=$(cd "$(dirname "$0")" && pwd)
SRC=${1:-$HOME/mage-vl/codec/src.mp4}
N=${2:-150}
OUT=${3:-/tmp/mppenc}
BIN="$OUT/mpp_enc"

mkdir -p "$OUT" || exit 1

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

command -v ffmpeg >/dev/null || die "缺 ffmpeg"
[ -f "$SRC" ] || die "找不到源视频: $SRC"

say "0) 编译 mpp_enc"
if [ ! -x "$BIN" ] || [ "$DIR/mpp_enc.c" -nt "$BIN" ]; then
    cc -O2 -Wall -o "$BIN" "$DIR/mpp_enc.c" -lmpp || die "编译失败（确认有 libmpp：ldconfig -p | grep -i mpp）"
fi
echo "OK  $BIN"

say "1) 探测源视频并用 ffmpeg 解成 NV12"
W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of csv=p=0 "$SRC")
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$SRC")
[ -n "$W" ] && [ -n "$H" ] || die "ffprobe 取不到分辨率"
echo "源: $SRC  ${W}x${H}  取 $N 帧"

ffmpeg -v error -y -i "$SRC" -frames:v "$N" -pix_fmt nv12 -f rawvideo "$OUT/in.nv12" \
    || die "ffmpeg 解 NV12 失败"
ls -l "$OUT/in.nv12" | awk '{printf "   NV12 原始大小 %.1f MB\n", $5/1048576}'

say "2) mpp_enc 码率扫描（CBR，$N 帧 @30fps）"
printf '%-8s %12s %12s %8s %8s\n' "目标" "实际字节" "实际Mbps" "偏离" "编码fps"
printf '%s\n' "------------------------------------------------------------"
for BPS in 1000000 2000000 4000000 8000000; do
    T0=$(date +%s.%N)
    "$BIN" -i "$OUT/in.nv12" -w "$W" -h "$H" -n "$N" -o "$OUT/cbr_$BPS.h264" \
           -c h264 -b "$BPS" --rc cbr --fps 30 --gop 30 2>"$OUT/enc_$BPS.log" \
        || { echo "  $BPS 失败："; tail -3 "$OUT/enc_$BPS.log"; continue; }
    T1=$(date +%s.%N)

    SZ=$(stat -c %s "$OUT/cbr_$BPS.h264" 2>/dev/null || stat -f %z "$OUT/cbr_$BPS.h264")
    DUR=$(awk "BEGIN{print $N/30}")
    MBPS=$(awk "BEGIN{printf \"%.2f\", $SZ*8/$DUR/1000000}")
    DEV=$(awk "BEGIN{printf \"%+.1f%%\", ($SZ*8/$DUR - $BPS)*100/$BPS}")
    FPS=$(awk "BEGIN{printf \"%.1f\", $N/($T1-$T0)}")
    printf '%-8s %12s %12s %8s %8s\n' \
        "$(awk "BEGIN{printf \"%.0fM\", $BPS/1000000}")" "$SZ" "$MBPS" "$DEV" "$FPS"
done

say "3) 对照：ffmpeg 的 h264_stcodec（-b:v 不生效的现场证据）"
for BPS in 1000000 8000000; do
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_stcodec; then
        ffmpeg -v error -y -f rawvideo -pix_fmt nv12 -s "${W}x${H}" -r 30 -i "$OUT/in.nv12" \
               -c:v h264_stcodec -b:v "$BPS" -frames:v "$N" "$OUT/ff_$BPS.h264" 2>/dev/null
        SZ=$(stat -c %s "$OUT/ff_$BPS.h264" 2>/dev/null || stat -f %z "$OUT/ff_$BPS.h264" 2>/dev/null)
        echo "  ffmpeg -b:v $BPS → $SZ 字节"
    else
        echo "  (本板 ffmpeg 无 h264_stcodec，跳过)"
        break
    fi
done
echo "  ↑ 两个 -b:v 输出大小若基本相同，就证明 ffmpeg 那条路没把码率传下去。"

say "4) 产物"
ls -l "$OUT"/*.h264 2>/dev/null | awk '{printf "  %-28s %10d B\n", $9, $5}'
echo
echo "播放/校验：ffprobe -v error -show_entries stream=width,height,bit_rate -of default=nw=1 $OUT/cbr_4000000.h264"
