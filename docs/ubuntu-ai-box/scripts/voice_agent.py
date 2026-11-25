"""语音助手全链路：VAD → ASR → LLM → TTS，测端到端与各段延迟

cu12(whisper) 与 cu13(torch) 冲突 -> ASR 放子进程，计进程启动开销。
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
OUT = os.path.expanduser("~/apps/infer-research/voice_agent.json")
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
    idx = np.linspace(0, len(d)-1, int(len(d)*16000/sr)); d = np.interp(idx, np.arange(len(d)), d).astype("float32")

t0 = time.time()
vad = load_silero_vad()
stamps = get_speech_timestamps(torch.from_numpy(d), vad, sampling_rate=16000,
                               return_seconds=True, threshold=0.4)
t_vad_load = time.time() - t0

t0 = time.time()
m = WhisperModel(os.path.expanduser("~/.cache/modelscope/models/Systran--faster-whisper-small/snapshots/master"),
                 device="cuda", compute_type="float16")
t_asr_load = time.time() - t0

t0 = time.time()
texts = []
for s in stamps:
    seg = d[int(s["start"]*16000):int(s["end"]*16000)]
    segs, _ = m.transcribe(seg, beam_size=1, language="zh")
    texts.append("".join(x.text for x in segs).strip())
t_asr = time.time() - t0

print(json.dumps(dict(text="".join(texts), n_seg=len(stamps),
                      vad_load_ms=round(t_vad_load*1000, 1),
                      asr_load_ms=round(t_asr_load*1000, 1),
                      asr_ms=round(t_asr*1000, 1),
                      duration=round(len(d)/16000, 2)), ensure_ascii=False))
'''


def main():
    res = {"audio": os.path.basename(WAV)}

    # ---------- 阶段 A：VAD + ASR（子进程，cu12） ----------
    t0 = time.time()
    env = {**os.environ, "LD_LIBRARY_PATH": LIBS, "VA_WAV": WAV}
    p = subprocess.run([sys.executable, "-c", ASR_CODE], capture_output=True, text=True, env=env)
    a_total = time.time() - t0
    if p.returncode != 0:
        print("ASR 失败:", p.stderr[-500:])
        return
    a = json.loads(p.stdout.strip().splitlines()[-1])
    print(f"[A] VAD+ASR 子进程 {a_total:.2f}s（含启动）| VAD 加载 {a['vad_load_ms']}ms | "
          f"ASR 模型加载 {a['asr_load_ms']}ms | 转写 {a['asr_ms']}ms | {a['n_seg']} 段")
    print(f"    转写: {a['text'][:60]}")
    res["asr"] = a
    res["asr_process_total_s"] = round(a_total, 2)

    # ---------- 阶段 B：LLM 回复 ----------
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MD06)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(MD06, dtype=torch.bfloat16, device_map={"": 0}).eval()
    llm = PeftModel.from_pretrained(llm, LORA).eval()
    llm_load = time.time() - t0

    user_text = a["text"] or "你好"
    msgs = [{"role": "user", "content": f"{user_text}\n请简短回答。"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    enc = tok(prompt, return_tensors="pt").to("cuda")
    n_in = enc["input_ids"].size(1)

    with torch.inference_mode():
        llm.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)  # 预热
        torch.cuda.synchronize()
        t0 = time.time()
        out = llm.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        llm_ms = (time.time() - t0) * 1000
    reply = tok.decode(out[0, n_in:], skip_special_tokens=True).strip()
    print(f"[B] LLM 加载 {llm_load:.1f}s | 生成 {llm_ms:.0f} ms（48 token）")
    print(f"    回复: {reply[:60]}")
    res["llm"] = dict(load_s=round(llm_load, 1), gen_ms=round(llm_ms, 1), reply=reply[:200])
    del llm
    torch.cuda.empty_cache()

    # ---------- 阶段 C：TTS ----------
    from kokoro import KPipeline
    pth = sorted(glob.glob(os.path.join(KOKORO_ROOT, "*.pth")))[0]
    t0 = time.time()
    pipe = KPipeline(lang_code="z", repo_id="hexgrad/Kokoro-82M-v1.1-zh", model=pth)
    voice = torch.load(os.path.join(KOKORO_ROOT, "voices", "zf_001.pt"), map_location="cpu")
    tts_load = time.time() - t0

    tts_text = reply[:60] if reply else "你好"
    list(pipe(tts_text, voice=voice))  # 预热（G2P 词典）
    t0 = time.time()
    chunks = [x for _, _, x in pipe(tts_text, voice=voice) if x is not None]
    tts_ms = (time.time() - t0) * 1000
    import numpy as np
    wav = np.concatenate([c.numpy() if hasattr(c, "numpy") else np.asarray(c) for c in chunks])
    dur = len(wav) / 24000
    import soundfile as sf
    os.makedirs(os.path.expanduser("~/apps/infer-research/voice_agent"), exist_ok=True)
    p_out = os.path.expanduser("~/apps/infer-research/voice_agent/reply.wav")
    sf.write(p_out, wav, 24000)
    print(f"[C] TTS 加载 {tts_load:.1f}s | 生成 {tts_ms:.0f} ms → {dur:.2f}s 音频")
    res["tts"] = dict(load_s=round(tts_load, 1), gen_ms=round(tts_ms, 1), audio_s=round(dur, 2))

    # ---------- 汇总（稳态，不含加载） ----------
    steady = a["asr_ms"] + llm_ms + tts_ms
    print(f"\n=== 稳态端到端（VAD/ASR 已加载）===")
    print(f"  ASR {a['asr_ms']:.0f} ms + LLM {llm_ms:.0f} ms + TTS {tts_ms:.0f} ms "
          f"= {steady:.0f} ms（{steady/1000:.2f}s）")
    print(f"  冷启动（含全部模型加载）: {a_total + llm_load + tts_load:.1f}s")
    res["steady_ms"] = round(steady, 1)
    res["cold_start_s"] = round(a_total + llm_load + tts_load, 1)
    print(f"  产物: {p_out}")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
