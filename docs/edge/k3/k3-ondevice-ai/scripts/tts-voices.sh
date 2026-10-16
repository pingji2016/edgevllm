#!/bin/sh
# TTS 音色 / 语速验证 —— 跑在板子上（SpaceMiT AI Gateway :18790）
#
# 用法:
#   ./tts-voices.sh                 # 默认中文句子
#   ./tts-voices.sh "自定义文本"
#
# ★ 本脚本会暴露一个厂方 bug（报告 §12.5）：
#   voice_id 和 speed 参数**完全没生效** —— 同文本下不同音色/语速产出的 wav md5 一模一样，
#   只有 model 生效。所以默认用「全新文本」跑，排除网关按文本缓存造成的假象。
#
# 实测结论：
#   matcha_zh   5.4s 语音 0.84s  → RTF 0.183（5.5× 实时，唯一适合实时）
#   kokoro      5.4s 语音 22–32s → RTF 3.6–5.3（0.2–0.3×，太慢）
set -u

G=${GW:-http://127.0.0.1:18790}
OUT=${OUT:-$HOME/k3-ai/out/tts}
TXT=${1:-"这是一句从没合成过的全新句子，用来排除网关按文本缓存造成的干扰。"}
mkdir -p "$OUT"

echo "文本: $TXT"
echo "输出目录: $OUT"
echo
printf '%-26s %10s %7s %8s  %s\n' 配置 耗时 时长 字节 md5
printf -- '------------------------------------------------------------------------\n'

# 模型:音色:语速
for spec in \
    "matcha_zh:default:1.0" \
    "matcha_zh_en:default:1.0" \
    "kokoro:zf_xiaobei:1.0" \
    "kokoro:zm_yunxi:1.0" \
    "kokoro:af_heart:1.0" \
    "kokoro:zf_xiaobei:1.5" \
    "kokoro:zf_xiaobei:0.7" \
    "matcha_zh:default:1.5" \
    "matcha_zh:default:0.7" \
    ; do
    m=${spec%%:*}; rest=${spec#*:}; v=${rest%%:*}; sp=${rest#*:}
    f="$OUT/${m}_${v}_${sp}.wav"

    # 先确保模型加载（字段是 model_id）
    curl -s --max-time 600 -X POST "$G/v1/tts/models/load" \
        -H 'Content-Type: application/json' -d "{\"model_id\":\"$m\"}" >/dev/null

    t=$(curl -s -w '%{time_total}' --max-time 300 -o "$f" \
        -X POST "$G/v1/tts/synthesize" -H 'Content-Type: application/json' \
        -d "{\"text\":\"$TXT\",\"voice_id\":\"$v\",\"model\":\"$m\",\"speed\":$sp,\"response_format\":\"wav\"}")

    dur=$(python3 -c "import wave;w=wave.open('$f');print('%.2f'%(w.getnframes()/w.getframerate()))" 2>/dev/null || echo '?')
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    md5=$(md5sum "$f" 2>/dev/null | cut -c1-12)

    printf '%-26s %9ss %6ss %8s  %s\n' "$m/$v/sp=$sp" "$t" "$dur" "$sz" "$md5"
done

echo
echo "判读：同一 model 下不同音色/语速的 md5 若完全相同 → 参数被丢弃（当前确实如此）。"
echo "      matcha_zh 的 speed 0.7/1.5 时长都是 4.49s（不变）→ 语速参数无效。"
