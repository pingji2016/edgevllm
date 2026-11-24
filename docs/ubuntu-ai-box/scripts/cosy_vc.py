"""最终 bisect：用 VC 分支（source_speech_token）跳过 LLM，验证 flow+hift 是否正常"""
import os
import sys

sys.path.append("/home/pingji/apps/CosyVoice")
sys.path.append("/home/pingji/apps/CosyVoice/third_party/Matcha-TTS")

import difflib

import librosa
import numpy as np
import soundfile as sf
import torch

MODEL = "/home/pingji/models/cosyvoice2"
REF = "/home/pingji/apps/CosyVoice/asset/zero_shot_prompt.wav"
REF_TEXT = "希望你以后能够做的比我还好呦。"


def postprocess(speech, top_db=60, hop_length=220, win_length=440, max_val=0.8):
    arr = speech.squeeze(0).numpy() if isinstance(speech, torch.Tensor) else speech
    arr, _ = librosa.effects.trim(arr, top_db=top_db, frame_length=win_length, hop_length=hop_length)
    out = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    if out.abs().max() > max_val:
        out = out / out.abs().max() * max_val
    return out


def asr(path):
    from faster_whisper import WhisperModel
    d, sr = sf.read(path, dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    if sr != 16000:
        idx = np.linspace(0, len(d) - 1, int(len(d) * 16000 / sr))
        d = np.interp(idx, np.arange(len(d)), d).astype("float32")
    m = WhisperModel(os.path.expanduser(
        "~/.cache/modelscope/models/Systran--faster-whisper-small/snapshots/master"),
        device="cpu", compute_type="int8")
    segs, _ = m.transcribe(d, beam_size=5)
    return "".join(s.text for s in segs).strip()


def main():
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    cv = CosyVoice2(MODEL, load_jit=False, load_trt=False, fp16=False)
    prompt16 = postprocess(load_wav(REF, 16000))
    mi = cv.frontend.frontend_zero_shot("", REF_TEXT, prompt16, cv.sample_rate, '')

    print("=== VC 分支：prompt 的 speech token → flow → hift（跳过 LLM）===")
    chunks = []
    for o in cv.model.tts(
            text=torch.zeros(1, 0, dtype=torch.int32),
            flow_embedding=mi["flow_embedding"], llm_embedding=mi["llm_embedding"],
            prompt_text=mi["prompt_text"],
            llm_prompt_speech_token=mi["llm_prompt_speech_token"],
            flow_prompt_speech_token=mi["flow_prompt_speech_token"],
            prompt_speech_feat=mi["prompt_speech_feat"],
            source_speech_token=mi["flow_prompt_speech_token"],
            stream=False):
        chunks.append(o["tts_speech"])
    wav = torch.cat(chunks, dim=1)
    dur = wav.shape[1] / cv.sample_rate
    p = "/tmp/cv2_bisect/vc.wav"
    os.makedirs("/tmp/cv2_bisect", exist_ok=True)
    sf.write(p, wav[0].float().numpy(), cv.sample_rate)
    heard = asr(p)
    ratio = difflib.SequenceMatcher(None, REF_TEXT, heard).ratio()
    print(f"音频 {dur:.2f}s（原 prompt 3.48s）")
    print(f"期望（=prompt 内容）: {REF_TEXT}")
    print(f"听到: {heard}")
    print(f"相似度: {ratio*100:.0f}%")


if __name__ == "__main__":
    main()
