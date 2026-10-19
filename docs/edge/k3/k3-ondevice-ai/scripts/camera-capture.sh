#!/bin/sh
# camera-capture.sh —— 摄像头探测 + 抓帧（可接 vlm-announce.sh 播报）
#
# 背景（§8.5 / §12.7）：/dev/video0 是 **VPU（Linlon, platform:mvx）的 V4L2 节点**，
# 不是摄像头。真摄像头（USB UVC 或 MIPI-CSI 的 OV5647/OV2735）会枚举成 /dev/video1+。
# 本脚本自动把 VPU 节点过滤掉，所以「板上没有摄像头」时会给出明确诊断而不是乱抓。
#
# 用法：
#   ./camera-capture.sh --list               # 只列出真摄像头
#   ./camera-capture.sh                      # 从第一个真摄像头抓 1 帧
#   ./camera-capture.sh --frames 3           # 抓 3 帧
#   ./camera-capture.sh --dev /dev/video2 --size 1280x720 --frames 1
#   ./camera-capture.sh --frames 1 --vlm     # 抓完调用 vlm-announce.sh 播报
#
# 输出：$OUT（默认 /tmp/cam）/frame_000.jpg ...

set -u
DIR=$(cd "$(dirname "$0")" && pwd)
SYSFS=/sys/class/video4linux
DEV=""
SIZE="1280x720"
FRAMES=1
OUT=${OUT:-/tmp/cam}
DO_LIST=0
DO_VLM=0

while [ $# -gt 0 ]; do
    case "$1" in
        --list) DO_LIST=1; shift ;;
        --dev) DEV=$2; shift 2 ;;
        --size) SIZE=$2; shift 2 ;;
        --frames) FRAMES=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --vlm) DO_VLM=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
done

# 返回该节点是不是 VPU（不是摄像头）
is_vpu() {
    n=$(basename "$1")
    nm=$(cat "$SYSFS/$n/name" 2>/dev/null)
    case "$nm" in
        *Linlon*|*mvx*|*VPU*|*vpu*) return 0 ;;
    esac
    # sysfs 里带上游设备名
    up=$(readlink -f "$SYSFS/$n/device" 2>/dev/null)
    case "$up" in
        *mvx*) return 0 ;;
    esac
    return 1
}

found=""
for d in /dev/video*; do
    [ -e "$d" ] || continue
    if is_vpu "$d"; then
        [ "$DO_LIST" = "1" ] && echo "  ✗ $d  $(cat "$SYSFS/$(basename "$d")/name" 2>/dev/null)  ← VPU 节点，不是摄像头"
        continue
    fi
    [ "$DO_LIST" = "1" ] && echo "  ✓ $d  $(cat "$SYSFS/$(basename "$d")/name" 2>/dev/null)"
    [ -z "$found" ] && found="$d"
done

if [ "$DO_LIST" = "1" ]; then
    [ -z "$found" ] && {
        echo
        echo "没有发现真摄像头。板子当前状态（§12.7）："
        echo "  · /dev/video0 是 VPU（Linlon, platform:mvx）的 V4L2 节点"
        echo "  · 内核侧 OV5647/OV2735 驱动与 libvi_k3_cam_plugin.so 都已就绪"
        echo "  · 接上 USB UVC 摄像头（免驱）或 MIPI-CSI 模组后会枚举成 /dev/video1+"
    }
    exit 0
fi

[ -n "$DEV" ] && found="$DEV"
if [ -z "$found" ]; then
    echo "没有可用摄像头（见 ./camera-capture.sh --list）。" >&2
    exit 1
fi

command -v ffmpeg >/dev/null || { echo "缺 ffmpeg" >&2; exit 1; }
mkdir -p "$OUT" || exit 1

echo "摄像头: $found   $SIZE   抓 $FRAMES 帧 → $OUT"
i=0
while [ "$i" -lt "$FRAMES" ]; do
    f=$(printf '%s/frame_%03d.jpg' "$OUT" "$i")
    if ffmpeg -v error -y -f v4l2 -video_size "$SIZE" -i "$found" -frames:v 1 -q:v 2 "$f" 2>/tmp/cam_err.log; then
        echo "  ✓ $f  ($(stat -c %s "$f" 2>/dev/null || stat -f %z "$f") B)"
    else
        echo "  ✗ 抓帧失败：" >&2
        tail -3 /tmp/cam_err.log >&2
        echo "    提示：先用 v4l2-ctl --list-formats-ext -d $found 看支持的尺寸/格式" >&2
        exit 1
    fi
    i=$((i+1))
    [ "$i" -lt "$FRAMES" ] && sleep 0.3
done

if [ "$DO_VLM" = "1" ]; then
    if [ -x "$DIR/vlm-announce.sh" ]; then
        echo
        echo "== 交给 Mage-VL 播报第一帧 =="
        "$DIR/vlm-announce.sh" "$OUT/frame_000.jpg"
    else
        echo "找不到 $DIR/vlm-announce.sh，跳过 VLM" >&2
    fi
fi
