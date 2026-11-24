"""CosyVoice2 诊断 2：精度统一 + LLM 输出长度可观测"""
import os
import sys

sys.path.append("/home/pingji/apps/CosyVoice")
sys.path.append("/home/pingji/apps/CosyVoice/third_party/Matcha-TTS")

import librosa
import soundfile as sf
import torch

MODEL = "/home/pingji/models/cosyvoice2"
REF = "/home/pingji/apps/CosyVoice/asset/zero_shot_prompt.wav"
REF_TEXT = "希望你以后能够做的比我还好呦。"
TEXT = "你好，今天天气不错。"


def postprocess(speech, top_db=60, hop_length=220, win_length=440, max_val=0.8):
    arr = speech.squeeze(0).numpy() if isinstance(speech, torch.Tensor) else speech
    arr, _ = librosa.effects.trim(arr, top_db=top_db, frame_length=win_length, hop_length=hop_length)
    out = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    if out.abs().max() > max_val:
        out = out / out.abs().max() * max_val
    return out


def main():
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    cv = CosyVoice2(MODEL, load_jit=False, load_trt=False, fp16=False)
    print("=== 各模块参数精度 ===")
    for name in ("llm", "flow", "hift"):
        mod = getattr(cv.model, name, None)
        if mod is None:
            continue
        dtypes = {p.dtype for p in mod.parameters()}
        print(f"  {name:5s} {dtypes}")

    prompt16 = postprocess(load_wav(REF, 16000))
    print(f"\nprompt: {tuple(prompt16.shape)} {prompt16.dtype}")

    # 统一成 float32（与 prompt 一致），排除 dtype 干扰
    for name in ("llm", "flow", "hift"):
        mod = getattr(cv.model, name, None)
        if mod is not None:
            mod.float()
    print("已把所有模块转为 float32")

    print("\n=== LLM 阶段输出 ===")
    mi = cv.frontend.frontend_zero_shot(TEXT, REF_TEXT, prompt16, cv.sample_rate, '')
    try:
        tokens = []
        for out in cv.model.llm.inference(**{k: v for k, v in mi.items()
                                             if k in ("text", "text_len", "prompt_text", "prompt_text_len",
                                                      "llm_prompt_speech_token", "llm_prompt_speech_token_len",
                                                      "llm_embedding")}):
            tokens.append(out["tts_speech_token"])
        toks = torch.cat(tokens, dim=1)
        print(f"  LLM 生成 speech token: {tuple(toks.shape)}（期望约 {len(TEXT)*4}~{len(TEXT)*6}）")
    except Exception as e:
        import traceback
        print("  LLM 阶段失败:")
        traceback.print_exc(limit=3)

    print("\n=== 完整 tts 流程 ===")
    try:
        chunks = [o["tts_speech"] for o in cv.model.tts(**mi, stream=False)]
        wav = torch.cat(chunks, dim=1)
        dur = wav.shape[1] / cv.sample_rate
        p = "/tmp/cv2_diag/fixed.wav"
        os.makedirs("/tmp/cv2_diag", exist_ok=True)
        sf.write(p, wav[0].float().numpy(), cv.sample_rate)
        print(f"  ✅ 合成 {dur:.2f}s -> {p}")
    except Exception as e:
        import traceback
        print("  ❌ 失败:")
        traceback.print_exc(limit=4)


if __name__ == "__main__":
    main()
