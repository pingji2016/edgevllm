#!/bin/sh
# ASR 长度-吞吐曲线 —— 跑在板子上（SpaceMiT AI Gateway :18790）
#
# 用法:
#   ./asr-length.sh a5.wav a10.wav a20.wav a45.wav
#   ./asr-length.sh $(ls ~/k3-ai/out/a*.wav)
#
# ★ 实测结论（报告 §12.6）：SenseVoice 的处理时间随音频长度**近似二次增长**，
#   5s 音频 RTF 0.15，40s 音频 RTF 0.81 —— 长音频整段丢进去会退化到"1× 实时"。
#   正确姿势：先用 /v1/vad/segments（34× 实时）切段，再逐段 recognize。
#
# 造测试音频（板上现成稿件）:
#   ffmpeg -v error -y -i 0.wav -ar 16000 -ac 1 a10.wav
#   ffmpeg -v error -y -i a10.wav -i a10.wav -filter_complex \
#       "[0:a][1:a]concat=n=2:v=0:a=1" -ar 16000 -ac 1 a20.wav
set -u

G=${GW:-http://127.0.0.1:18790}
JSON=/tmp/asrlen.json

printf '%-22s %9s %11s %8s  %s\n' 文件 音频s 处理s RTF 文本前40字
printf -- '-----------------------------------------------------------------------------------\n'

for f in "$@"; do
    [ -f "$f" ] || { printf '%-22s (缺文件)\n' "$f"; continue; }

    dur=$(python3 -c "import wave;w=wave.open('$f');print('%.2f'%(w.getnframes()/w.getframerate()))")
    a=$(date +%s.%N)
    curl -s --max-time 600 -X POST "$G/v1/asr/recognize" -F "file=@$f" -o "$JSON"
    b=$(date +%s.%N)

    python3 - "$JSON" "$f" "$dur" "$a" "$b" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
elapsed = float(sys.argv[5]) - float(sys.argv[4])
rtf = d.get("rtf") or (elapsed / float(sys.argv[3]))
txt = (d.get("text") or "")[:40]
print("%-22s %9s %11.2f %8.3f  %s" % (sys.argv[2], sys.argv[3], elapsed, rtf, txt))
PY
    sleep 2   # ★ 别并发，见报告 §12.8
done

echo
echo "提示：RTF 随长度上升 → 长音频请先 VAD 切段："
echo "  curl -s -X POST $G/v1/vad/segments -F file=@长音频.wav"
