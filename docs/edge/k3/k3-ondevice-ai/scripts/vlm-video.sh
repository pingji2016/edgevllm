#!/bin/sh
# vlm-video.sh —— 视频理解：抽帧 → Mage-VL 逐帧描述 → 时间线 / 拼图总览
#
# 板上没有摄像头（§8.5/§12.7），所以「视频理解」用文件输入验证；接上 USB/MIPI
# 摄像头后，把 `--camera N` 换成 getframe 抓的若干帧即可（见 camera-capture.sh）。
#
# 用法：
#   ./vlm-video.sh 视频.mp4                 # 默认抽 6 帧 + 拼图总览
#   ./vlm-video.sh 视频.mp4 --frames 4      # 只抽 4 帧逐帧描述
#   ./vlm-video.sh 视频.mp4 --grid-only     # 只做拼图总览（1 次 VLM，最省）
#   ./vlm-video.sh 视频.mp4 --no-vlm        # 只抽帧，不跑模型（看抽帧是否正确）
#
# 环境变量：
#   BACKBONE / MMPROJ / MTMD   模型与可执行文件路径
#   VLM_TOKENS=128  VLM_THREADS=8
#
# ★ Mage-VL 一次只能吃一张图，所以「逐帧」= 每帧一次推理（4B 模型单帧约 1 分钟，
#   6 帧就是 ~6 分钟）。要快就用 --grid-only：把 N 帧拼成一张 3x2 拼图，一次问完。

set -u
DIR=$(cd "$(dirname "$0")" && pwd)

VIDEO=""
FRAMES=6
GRID_ONLY=0
NO_VLM=0
GRID=1

while [ $# -gt 0 ]; do
    case "$1" in
        --frames) FRAMES=$2; shift 2 ;;
        --grid-only) GRID_ONLY=1; shift ;;
        --no-vlm) NO_VLM=1; shift ;;
        --no-grid) GRID=0; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        -*) echo "未知参数: $1" >&2; exit 2 ;;
        *) VIDEO=$1; shift ;;
    esac
done

[ -n "$VIDEO" ] || { echo "用法: $0 视频.mp4 [--frames N] [--grid-only] [--no-vlm]" >&2; exit 2; }
[ -f "$VIDEO" ] || { echo "找不到视频: $VIDEO" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "缺 ffmpeg" >&2; exit 1; }

BACKBONE=${BACKBONE:-$HOME/mage-vl/models/mage-vl-backbone-Q4_K_M.gguf}
MMPROJ=${MMPROJ:-$HOME/mage-vl/models/mage-vit-mmproj-Q8_0.gguf}
MTMD=${MTMD:-$(command -v llama-mtmd-cli || echo "$HOME/mage-vl/llama-mtmd-cli")}
VLM_TOKENS=${VLM_TOKENS:-128}
VLM_THREADS=${VLM_THREADS:-8}

WORK=${WORK:-/tmp/vlmv}
mkdir -p "$WORK" || exit 1

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO" 2>/dev/null)
W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of csv=p=0 "$VIDEO" 2>/dev/null)
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$VIDEO" 2>/dev/null)
[ -n "$DUR" ] || { echo "ffprobe 读不到时长" >&2; exit 1; }
echo "视频: $VIDEO  ${W}x${H}  ${DUR}s"

# ---------- 1) 均匀抽帧 ----------
echo
echo "== 抽帧（$FRAMES 帧）=="
STEP=$(awk "BEGIN{d=$DUR/$FRAMES; if(d<0.1)d=0.1; printf \"%.3f\", d}")
i=0
while [ "$i" -lt "$FRAMES" ]; do
    T=$(awk "BEGIN{printf \"%.3f\", ($i+0.5)*$STEP}")
    ffmpeg -v error -y -ss "$T" -i "$VIDEO" -frames:v 1 -q:v 2 "$WORK/f$(printf %02d $i).jpg" 2>/dev/null
    if [ -f "$WORK/f$(printf %02d $i).jpg" ]; then
        i=$((i+1))
    else
        break   # 到了片尾
    fi
done
N=$(ls "$WORK"/f*.jpg 2>/dev/null | wc -l)
[ "$N" -gt 0 ] || { echo "抽帧失败" >&2; exit 1; }
echo "  抽到 $N 帧 → $WORK"

if [ "$NO_VLM" = "1" ]; then
    ls -l "$WORK"/f*.jpg | awk '{printf "  %s  %d B\n", $9, $5}'
    exit 0
fi

[ -x "$MTMD" ] || { echo "找不到 llama-mtmd-cli（设 MTMD=...）" >&2; exit 1; }
[ -f "$BACKBONE" ] || { echo "找不到 backbone: $BACKBONE" >&2; exit 1; }
[ -f "$MMPROJ" ] || { echo "找不到 mmproj: $MMPROJ" >&2; exit 1; }

vlm_ask() {   # $1=图片  $2=提示词
    "$MTMD" -m "$BACKBONE" --mmproj "$MMPROJ" --image "$1" \
        -p "$2" -n "$VLM_TOKENS" -t "$VLM_THREADS" -c 2048 2>/dev/null \
      | sed -e 's/\x1b\[[0-9;]*m//g' | awk 'BEGIN{p=0} /^\/ *assistant/{next} {print}'
}

# ---------- 2) 逐帧描述（除非 --grid-only） ----------
if [ "$GRID_ONLY" = "0" ]; then
    echo
    echo "== 逐帧描述（每帧一次 VLM 推理，耗时长）=="
    k=0
    for f in "$WORK"/f*.jpg; do
        k=$((k+1))
        T=$(awk "BEGIN{printf \"%.1f\", ($k-0.5)*$STEP}")
        echo
        echo "--- [$T s] $(basename "$f") ---"
        vlm_ask "$f" "用一句中文描述这张画面里有什么：主体、动作、场景。不要超过 30 字。"
    done
fi

# ---------- 3) 拼图总览 ----------
if [ "$GRID" = "1" ]; then
    COLS=3
    ROWS=$(( (N + COLS - 1) / COLS ))
    [ "$ROWS" -lt 1 ] && ROWS=1
    ffmpeg -v error -y -pattern_type glob -i "$WORK/f*.jpg" \
        -filter_complex "scale=480:-1,tile=${COLS}x${ROWS}:padding=4:margin=4:color=white" \
        -frames:v 1 "$WORK/grid.jpg" 2>/dev/null
    if [ -f "$WORK/grid.jpg" ]; then
        echo
        echo "== 拼图总览（${COLS}x${ROWS}，一次推理）=="
        vlm_ask "$WORK/grid.jpg" \
            "这是一段视频按时间顺序抽出的 ${N} 帧拼图（从左到右、从上到下）。请用 2-3 句中文概括视频里发生了什么、场景如何变化。"
    fi
fi

echo
echo "帧文件与拼图在 $WORK"
