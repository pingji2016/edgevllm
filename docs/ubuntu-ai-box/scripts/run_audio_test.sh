#!/bin/bash
set -u
B=~/apps/llama-omni/build/bin
D=~/.cache/modelscope/models/OpenBMB--MiniCPM-o-4_5-gguf/snapshots/master
export LD_LIBRARY_PATH="$B"

echo "=== 等 tts 下载 ==="
for i in $(seq 1 40); do
  if [ -f "$D/tts/MiniCPM-o-4_5-tts-F16.gguf" ]; then break; fi
  sleep 10
done
ls -lh "$D/tts/" 2>/dev/null | awk '{print "  "$5, $9}'

echo
echo "=== 切音频为 1s 块（16k 单声道）==="
rm -rf /tmp/aud && mkdir -p /tmp/aud
ffmpeg -v error -i ~/apps/CosyVoice/asset/cross_lingual_prompt.wav \
  -ar 16000 -ac 1 -f segment -segment_time 1 /tmp/aud/aud%04d.wav 2>/dev/null
ls /tmp/aud | wc -l | awk '{print "  块数: "$1}'

echo
echo "=== 单工音频测试（模型听完后一次性回复）==="
N=$(ls /tmp/aud/*.wav | wc -l)
t0=$(date +%s)
timeout 1800 "$B/llama-omni-single-test-audio" -m "$D/MiniCPM-o-4_5-Q4_K_M.gguf" \
  --test /tmp/aud/aud "$N" -ngl 99 2>&1 \
  | grep -vE "^llama_|^load_|^print_info|^system_info|^0\.[0-9]" | tail -18
echo "  [总耗时 $(( $(date +%s) - t0 ))s]"
