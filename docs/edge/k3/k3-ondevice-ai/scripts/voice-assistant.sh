#!/usr/bin/env bash
# =============================================================================
#  K3 全离线语音助手（RISC-V / SpaceMiT K3）
#
#  链路：语音 →(SenseVoice ASR)→ 文本 →(Qwen3.5-2B LLM)→ 回复 →(Matcha TTS)→ 音频 → 喇叭
#  全部在本机完成，不联网。推理统一走板子自带的 SpaceMiT AI Gateway（127.0.0.1:18790）。
#
#  用法：
#    ./voice-assistant.sh ask.wav                 # 文件模式：一段语音进去
#    ./voice-assistant.sh --text "今天天气怎么样"   # 文本模式：跳过 ASR
#    ./voice-assistant.sh --mic 6                 # 麦克风模式：录 6 秒（需要插麦）
#    NOPLAY=1 ./voice-assistant.sh ask.wav        # 不出声，只产出 wav
#
#  环境变量：
#    GW     网关地址  默认 http://127.0.0.1:18790
#    MODEL  LLM 模型  默认 qwen3.5-2b-local（也可用 qwen3-0.6b-q4_0 等）
#    OUT    产物目录  默认 ~/k3-ai/out
#    SPK    声卡设备  默认 hw:0,0（板载 ES8326，只支持立体声）
# =============================================================================
set -uo pipefail

GW="${GW:-http://127.0.0.1:18790}"
MODEL="${MODEL:-qwen3.5-2b-local}"
OUT="${OUT:-$HOME/k3-ai/out}"
SPK="${SPK:-hw:0,0}"
SYS_PROMPT="${SYS_PROMPT:-你是一个运行在 RISC-V 开发板上的中文语音助手。回答要口语化、简短，控制在40个字以内，不要使用 Markdown 或表情符号。}"
MAX_TOKENS="${MAX_TOKENS:-96}"

mkdir -p "$OUT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# 计时助手：横杠进度 + 毫秒
step=0
stage() { step=$((step+1)); printf '  [%d] %-34s' "$step" "$1"; STEP_T0=$(date +%s.%N); }
done_stage() {
  local t; t=$(awk "BEGIN{printf \"%.2f\", $(date +%s.%N) - $STEP_T0}")
  printf '%6ss  %s\n' "$t" "${1:-}"
}

die() { echo "错误：$*" >&2; exit 1; }

command -v ffmpeg >/dev/null || die "缺 ffmpeg"
curl -s --max-time 5 "$GW/" >/dev/null || die "连不上 AI Gateway ($GW)，先确认 spacemit-ai-gateway.service 在跑"

# ---------------------------------------------------------------- 0. 取输入
MODE="file"; INPUT=""
case "${1:-}" in
  --text) MODE="text"; INPUT="${2:-}"; [ -n "$INPUT" ] || die "--text 后面要跟一句话" ;;
  --mic)  MODE="mic";  SECS="${2:-5}" ;;
  ""|-h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *)      INPUT="$1"; [ -f "$INPUT" ] || die "找不到文件 $INPUT" ;;
esac

echo "K3 离线语音助手  (模型 $MODEL, 网关 $GW)"
echo "────────────────────────────────────────────────────────────"

# ---------------------------------------------------------------- 1. 音频输入
if [ "$MODE" = "mic" ]; then
  stage "录音 ${SECS}s（板载 ES8326）"
  arecord -D "$SPK" -f S16_LE -c 2 -r 16000 -d "$SECS" "$TMP/in.wav" 2>/dev/null \
    || die "录音失败（声卡 $SPK）"
  ASR_IN="$TMP/in.wav"
  done_stage "$(stat -c%s "$TMP/in.wav") 字节"
else
  ASR_IN="$INPUT"          # 文件模式：直接用传入的音频
fi

# ---------------------------------------------------------------- 2. ASR
if [ "$MODE" != "text" ]; then
  stage "SenseVoice ASR"
  # 网关要 16k 单声道；板载录音是立体声，统一用 ffmpeg 规整
  ffmpeg -y -v error -i "$ASR_IN" -ac 1 -ar 16000 "$TMP/asr.wav" || die "重采样失败"
  ASR_JSON=$(curl -s --max-time 120 -X POST "$GW/v1/asr/recognize" \
      -F "file=@$TMP/asr.wav" -F "language=auto") || die "ASR 请求失败"
  QUESTION=$(printf '%s' "$ASR_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("text","").strip())' 2>/dev/null)
  done_stage "「${QUESTION:-（空）}」"
  [ -n "$QUESTION" ] || { echo "  → 没听清内容（麦克风没插？环境太吵？）"; exit 2; }
else
  QUESTION="$INPUT"
  echo "  [1] 输入文字                         ——  「$QUESTION」"
fi

# ---------------------------------------------------------------- 3. LLM
stage "LLM 生成"
REPLY_JSON=$(curl -s --max-time 300 -X POST "$GW/v1/llm/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(python3 - "$MODEL" "$SYS_PROMPT" "$QUESTION" "$MAX_TOKENS" <<'PY'
import json, sys
model, sysp, q, mt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
print(json.dumps({"model": model, "stream": False, "max_tokens": mt, "temperature": 0.6,
                  "messages": [{"role":"system","content":sysp},{"role":"user","content":q}]}))
PY
)") || die "LLM 请求失败"
REPLY=$(printf '%s' "$REPLY_JSON" | python3 -c 'import sys,json
d=json.load(sys.stdin)
print(d.get("choices",[{}])[0].get("message",{}).get("content","").strip())' 2>/dev/null)
TPS=$(printf '%s' "$REPLY_JSON" | python3 -c 'import sys,json
t=json.load(sys.stdin).get("timings",{})
print("%.1f t/s 生成了 %d token" % (t.get("predicted_per_second",0), t.get("predicted_n",0)))' 2>/dev/null)
done_stage "$TPS"
[ -n "$REPLY" ] || die "LLM 没返回内容"

# ---------------------------------------------------------------- 4. TTS
stage "Matcha TTS 合成"
curl -s --max-time 180 -X POST "$GW/v1/tts/synthesize" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1], "response_format":"wav"}))' "$REPLY" 2>/dev/null \
      || printf '{"text":"%s","response_format":"wav"}' "$REPLY")" \
  -o "$TMP/tts.wav" || die "TTS 请求失败"
# 声卡只吃立体声 48k，顺手规整（顺带留档）
ffmpeg -y -v error -i "$TMP/tts.wav" -ac 2 -ar 48000 "$OUT/reply.wav" || die "转立体声失败"
done_stage "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$TMP/tts.wav" | cut -c1-4)s 语音 → $OUT/reply.wav"

# ---------------------------------------------------------------- 5. 出声
if [ "${NOPLAY:-0}" = "1" ]; then
  echo "  [5] 播放                            跳过（NOPLAY=1）"
else
  stage "播放到 $SPK"
  aplay -D "$SPK" "$OUT/reply.wav" >/dev/null 2>&1 && done_stage "OK" || done_stage "失败（喇叭没接？）"
fi

echo "────────────────────────────────────────────────────────────"
echo "你  ：$QUESTION"
echo "助手：$REPLY"
