#!/usr/bin/env bash
# =============================================================================
#  K3「看图说话 → 语音播报」闭环
#
#  图像 →(Mage-VL 4B 本地视觉理解)→ 中文描述 →(Matcha TTS)→ 音频 → 喇叭
#  全程离线。VLM 走 ~/mage-vl 里的 llama-mtmd-cli，TTS 走 SpaceMiT AI Gateway。
#
#  用法：
#    ./vlm-announce.sh photo.jpg                       # 看图说话
#    ./vlm-announce.sh photo.jpg "图里有几个人？"        # 自定义提问
#    ./vlm-announce.sh --camera 3                       # 从 /dev/video0 抓一帧再播报（需接摄像头）
#    NOPLAY=1 ./vlm-announce.sh photo.jpg              # 只出 wav 不出声
#
#  注：板子目前没接摄像头。K3 的 4×MIPI-CSI 在芯片级是有的（内核带 VI/ISP 与
#      libvi_k3_cam_plugin.so），插上 CSI 摄像头后 /dev/video0 会出现，--camera 即可用。
# =============================================================================
set -uo pipefail

GW="${GW:-http://127.0.0.1:18790}"
MV="${MV:-$HOME/mage-vl}"
OUT="${OUT:-$HOME/k3-ai/out}"
SPK="${SPK:-hw:0,0}"
NTOK="${NTOK:-48}"
mkdir -p "$OUT"

stage() { printf '  [%d] %-30s' "$1" "$2"; STEP_T0=$(date +%s.%N); }
done_stage() { awk "BEGIN{printf \"%6.2fs  %s\n\", $(date +%s.%N) - $STEP_T0, \"${1:-}\"}"; }
die() { echo "错误：$*" >&2; exit 1; }

MODE="file"; IMG=""; PROMPT="用一句中文描述这张图片里有什么，不超过30个字。"
case "${1:-}" in
  --camera)
    MODE="camera"; SECS="${2:-3}"
    [ -e /dev/video0 ] || die "没有 /dev/video0：板子上没接 CSI/USB 摄像头"
    IMG="$OUT/cam.jpg" ;;
  ""|-h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) IMG="$1"; [ -f "$IMG" ] || die "找不到图片 $IMG"
     [ -n "${2:-}" ] && PROMPT="$2" ;;
esac

echo "K3 看图说话 → 语音播报"
echo "────────────────────────────────────────────────────────────"

if [ "$MODE" = "camera" ]; then
  stage 1 "从摄像头抓 ${SECS}s（取最后一帧）"
  rm -f "$OUT"/cam*.jpg
  ffmpeg -y -v error -f v4l2 -i /dev/video0 -t "$SECS" -frames:v 1 "$IMG" || die "抓帧失败"
  done_stage "$(stat -c%s "$IMG") 字节 → $IMG"
else
  echo "  [1] 输入图片                     $IMG"
fi

stage 2 "Mage-VL 看图（本地 4B）"
[ -x "$MV/llama.cpp/build/bin/llama-mtmd-cli" ] || die "找不到 llama-mtmd-cli"
DESC=$(cd "$MV" && LD_LIBRARY_PATH="$MV/llama.cpp/build/bin" \
  ./llama.cpp/build/bin/llama-mtmd-cli \
    -m models/mage-vl-backbone-Q4_K_M.gguf \
    --mmproj models/mage-vit-mmproj-Q8_0.gguf \
    --image "$IMG" -p "$PROMPT" \
    -n "$NTOK" -t 8 -c 4096 --no-warmup 2>"$OUT/vlm.err" \
  | grep -vE '^\s*$|^>|^\.\.\.|loading|WARN|encoding|^$' | tail -3 | tr '\n' ' ' | sed 's/  */ /g; s/^ //; s/ $//')
done_stage "「$DESC」"
[ -n "$DESC" ] || { echo "  VLM 没输出，看 $OUT/vlm.err"; exit 2; }

stage 3 "Matcha TTS 合成"
curl -s --max-time 180 -X POST "$GW/v1/tts/synthesize" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1], "response_format":"wav"}))' "$DESC")" \
  -o "$OUT/announce.wav" || die "TTS 失败"
ffmpeg -y -v error -i "$OUT/announce.wav" -ac 2 -ar 48000 "$OUT/announce_spk.wav" || die "转立体声失败"
done_stage "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/announce.wav" | cut -c1-4)s 语音"

if [ "${NOPLAY:-0}" = "1" ]; then
  echo "  [4] 播放                        跳过（NOPLAY=1）"
else
  stage 4 "播放到 $SPK"
  aplay -D "$SPK" "$OUT/announce_spk.wav" >/dev/null 2>&1 && done_stage "OK" || done_stage "失败（喇叭没接？）"
fi

echo "────────────────────────────────────────────────────────────"
echo "图片：$IMG"
echo "播报：$DESC"
