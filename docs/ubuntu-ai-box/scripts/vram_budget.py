"""多模型共存：把 VLM 服务之外的四类模型同时装进显存，测占用与并发延迟"""
import json
import os
import subprocess
import time

import numpy as np
import soundfile as sf
import torch

OUT = os.path.expanduser("~/apps/infer-research/vram_budget.json")
IMG = "/tmp/face.png"


def nvsmi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    used, total = [int(x) for x in out.strip().split(",")]
    return used, total


def main():
    used0, total = nvsmi()
    print(f"起始：{used0} MiB / {total} MiB（含 vlm-server）")
    rows = []

    # 1. Qwen3-0.6B + LoRA（推理态）
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
    tok = AutoTokenizer.from_pretrained(MD)
    m = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16, device_map={"": 0}).eval()
    m = PeftModel.from_pretrained(m, os.path.expanduser("~/apps/infer-research/lora_llm")).eval()
    used1 = nvsmi()[0]
    rows.append(dict(name="Qwen3-0.6B + LoRA", delta_mib=used1 - used0))
    print(f"  + Qwen3-0.6B+LoRA：{used1} MiB（+{used1-used0}）")

    # 2. faster-whisper small
    from faster_whisper import WhisperModel
    w = WhisperModel(os.path.expanduser(
        "~/.cache/modelscope/models/Systran--faster-whisper-small/snapshots/master"),
        device="cuda", compute_type="float16")
    used2 = nvsmi()[0]
    rows.append(dict(name="faster-whisper-small", delta_mib=used2 - used1))
    print(f"  + whisper-small：{used2} MiB（+{used2-used1}）")

    # 3. SAM2 tiny
    from transformers import Sam2Model, Sam2Processor
    sp = Sam2Processor.from_pretrained(os.path.expanduser(
        "~/.cache/modelscope/models/facebook--sam2.1-hiera-tiny/snapshots/master"))
    sam = Sam2Model.from_pretrained(os.path.expanduser(
        "~/.cache/modelscope/models/facebook--sam2.1-hiera-tiny/snapshots/master"),
        dtype=torch.bfloat16).cuda().eval()
    used3 = nvsmi()[0]
    rows.append(dict(name="SAM2 hiera-tiny", delta_mib=used3 - used2))
    print(f"  + SAM2-tiny：{used3} MiB（+{used3-used2}）")

    # 4. Depth-Anything small
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    dpath = os.path.expanduser(
        "~/.cache/modelscope/models/depth-anything--Depth-Anything-V2-Small-hf/snapshots/master")
    ip = AutoImageProcessor.from_pretrained(dpath)
    dep = AutoModelForDepthEstimation.from_pretrained(dpath, dtype=torch.bfloat16,
                                                      device_map={"": 0}).eval()
    used4 = nvsmi()[0]
    rows.append(dict(name="Depth-Anything-V2-S", delta_mib=used4 - used3))
    print(f"  + Depth-Anything：{used4} MiB（+{used4-used3}）")

    print(f"\n四模型合计 {(used4-used0)/1024:.2f} GiB；全卡占用 {used4/total*100:.0f}%")

    # ---- 并发跑一遍，测互相影响 ----
    from PIL import Image
    img = Image.open(IMG).convert("RGB")
    audio = sf.read(os.path.expanduser("~/apps/CosyVoice/asset/zero_shot_prompt.wav"),
                    dtype="float32")[0]
    if audio.ndim > 1:
        audio = audio.mean(1)

    lat = {}

    def timed(name, fn):
        try:
            fn()  # 预热
            torch.cuda.synchronize()
            t0 = time.time()
            fn()
            torch.cuda.synchronize()
            lat[name] = round((time.time() - t0) * 1000, 1)
        except Exception as e:
            lat[name] = f"{type(e).__name__}"

    timed("llm", lambda: m.generate(**tok("你好", return_tensors="pt").to("cuda"),
                                    max_new_tokens=8, pad_token_id=tok.eos_token_id))
    timed("asr", lambda: "".join(x.text for x in w.transcribe(audio, beam_size=5, language="zh")[0]))
    timed("sam2", lambda: sam(**sp(images=img, input_points=[[[[256, 256]]]],
                                   return_tensors="pt").to("cuda")))
    timed("depth", lambda: dep(**ip(images=img, return_tensors="pt").to("cuda")))

    print("\n共存状态下的单次延迟：")
    for k, v in lat.items():
        print(f"  {k:6s} {v if isinstance(v, str) else f'{v:8.1f} ms'}")

    json.dump(dict(start_mib=used0, total_mib=total, after_all_mib=used4,
                   rows=rows, latency_ms=lat), open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
