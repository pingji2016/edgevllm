"""语音助手 v2：LLM 换成 vLLM + CUDA Graph，ASR 增加"整段"对照

对比 v1（transformers LLM + 分段 ASR）：
  ASR 2342ms / LLM 888ms / TTS 100ms = 3331ms
"""
import glob
import json
import os
import subprocess
import sys
import time

WAV = os.path.expanduser("~/apps/CosyVoice/asset/cross_lingual_prompt.wav")
KOKORO_ROOT = open(os.path.expanduser("~/models/.kokoro_done")).read().strip()
MD06 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
LORA = os.path.expanduser("~/apps/infer-research/lora_llm")
OUT = os.path.expanduser("~/apps/infer-research/voice_agent_v2.json")
ASR_PY = "/home/pingji/apps/bench/yolo-venv/bin/python"
LIBS = ":".join(glob.glob(os.path.expanduser(
    "~/apps/minimind/.venv/lib/python3.12/site-packages/nvidia/*/lib")))

ASR_CODE = r'''
import json, os, time
import numpy as np, soundfile as sf, torch
from silero_vad import get_speech_timestamps, load_silero_vad
from faster_whisper import WhisperModel

WAV = os.environ["VA_WAV"]
d, sr = sf.read(WAV, dtype="float32")
if d.ndim > 1: d = d.mean(1)
if sr != 16000:
    idx = np.linspace(0, len(d)-1, int(len(d)*16000/sr))
    d = np.interp(idx, np.arange(len(d)), d).astype("float32")

vad = load_silero_vad()
m = WhisperModel(os.path.expanduser("~/.cache/modelscope/models/Systran--faster-whisper-small/snapshots/master"),
                 device="cuda", compute_type="float16")

# 模式 A：分段（流式形态）
stamps = get_speech_timestamps(torch.from_numpy(d), vad, sampling_rate=16000,
                               return_seconds=True, threshold=0.4)
t0 = time.time()
seg_texts = []
for s in stamps:
    seg = d[int(s["start"]*16000):int(s["end"]*16000)]
    segs, _ = m.transcribe(seg, beam_size=1, language="zh")
    seg_texts.append("".join(x.text for x in segs).strip())
t_seg = time.time() - t0

# 模式 B：整段一次
t0 = time.time()
segs, _ = m.transcribe(d, beam_size=1, language="zh")
one_text = "".join(x.text for x in segs).strip()   # 必须消费生成器，否则计时不含解码
t_one = time.time() - t0

print(json.dumps(dict(seg_text="".join(seg_texts), seg_ms=round(t_seg*1000, 1),
                      one_text=one_text, one_ms=round(t_one*1000, 1),
                      n_seg=len(stamps), duration=round(len(d)/16000, 2)),
                 ensure_ascii=False))
'''


TTS_CODE = r'''
import glob, json, os, time
import numpy as np, torch
ROOT = open(os.path.expanduser("~/models/.kokoro_done")).read().strip()
from kokoro import KPipeline
pth = sorted(glob.glob(os.path.join(ROOT, "*.pth")))[0]
pipe = KPipeline(lang_code="z", repo_id="hexgrad/Kokoro-82M-v1.1-zh", model=pth)
voice = torch.load(os.path.join(ROOT, "voices", "zf_001.pt"), map_location="cpu")
text = os.environ["VA_TTS_TEXT"][:60]
list(pipe(text, voice=voice))          # 预热（G2P 词典）
t0 = time.time()
chunks = [x for _, _, x in pipe(text, voice=voice) if x is not None]
dt = time.time() - t0
wav = np.concatenate([c.numpy() if hasattr(c, "numpy") else np.asarray(c) for c in chunks])
print(json.dumps(dict(ms=round(dt*1000, 1), audio_s=round(len(wav)/24000, 2))))
'''


def main():
    res = {}

    # ---- A. ASR：分段 vs 整段 ----
    env = {**os.environ, "LD_LIBRARY_PATH": LIBS, "VA_WAV": WAV}
    t0 = time.time()
    p = subprocess.run([ASR_PY, "-c", ASR_CODE], capture_output=True, text=True, env=env)
    a_proc = time.time() - t0
    if p.returncode != 0:
        print("ASR 失败:", p.stderr[-400:]); return
    a = json.loads(p.stdout.strip().splitlines()[-1])
    print(f"[A] 分段 {a['seg_ms']:.0f} ms（{a['n_seg']} 段）| 整段 {a['one_ms']:.0f} ms "
          f"| 子进程总 {a_proc:.1f}s")
    print(f"    分段文本: {a['seg_text'][:50]}")
    print(f"    整段文本: {a['one_text'][:50]}")
    res["asr"] = a

    user_text = a["one_text"] or "你好"

    # ---- B. LLM：vLLM + CUDA Graph（带 LoRA）----
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print("\n[B] vLLM 引擎加载中…", flush=True)
    t0 = time.time()
    llm = LLM(model=MD06, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.45, disable_log_stats=True,
              enable_lora=True, max_lora_rank=16, max_loras=2)
    load_s = time.time() - t0
    print(f"    引擎就绪 {load_s:.0f}s")

    prompts = [f"<|im_start|>user\n{user_text}\n请简短回答。<|im_end|>\n<|im_start|>assistant\n"]
    sp = SamplingParams(max_tokens=48, temperature=0.0)
    lr = LoRARequest("sft", 1, LORA)

    def gen(use_lora):
        kw = dict(lora_request=lr) if use_lora else {}
        llm.generate(prompts, sp, **kw)  # 预热
        torch.cuda.synchronize()
        t0 = time.time()
        outs = llm.generate(prompts, sp, **kw)
        torch.cuda.synchronize()
        return (time.time() - t0) * 1000, outs[0].outputs[0].text.strip()

    llm_ms, reply = gen(True)
    print(f"    LLM（LoRA + CUDA Graph）{llm_ms:.0f} ms → {reply[:50]}")
    res["llm"] = dict(load_s=round(load_s, 1), gen_ms=round(llm_ms, 1), reply=reply[:200], backend="vllm+cudagraph+lora")
    del llm
    torch.cuda.empty_cache()

    # ---- C. TTS（子进程，bench venv）----
    env2 = {**os.environ, "VA_TTS_TEXT": (reply or "你好")}
    t0 = time.time()
    p2 = subprocess.run([ASR_PY, "-c", TTS_CODE], capture_output=True, text=True, env=env2)
    tts_proc = time.time() - t0
    if p2.returncode != 0:
        print("TTS 失败:", p2.stderr[-300:])
        tts_ms, audio_s = 0, 0
    else:
        c = json.loads(p2.stdout.strip().splitlines()[-1])
        tts_ms, audio_s = c["ms"], c["audio_s"]
        print(f"[C] TTS {tts_ms:.0f} ms → {audio_s:.2f}s 音频（子进程含加载 {tts_proc:.1f}s）")
    res["tts"] = dict(gen_ms=tts_ms, audio_s=audio_s)

    # ---- 汇总：两种 ASR 配置 ----
    print("\n=== 端到端对比（稳态）===")
    for tag, asr_key in (("分段 ASR", "seg_ms"), ("整段 ASR", "one_ms")):
        total = a[asr_key] + llm_ms + tts_ms
        print(f"  {tag}: ASR {a[asr_key]:.0f} + LLM {llm_ms:.0f} + TTS {tts_ms:.0f} = {total:.0f} ms")
        res[asr_key + "_total"] = round(total, 1)
    print("\n对比 v1（transformers LLM + 分段 ASR）: ASR 2342 + LLM 888 + TTS 100 = 3331 ms")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
