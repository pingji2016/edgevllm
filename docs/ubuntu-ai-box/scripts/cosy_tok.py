"""tokenizer 自检：文本 token 能否解码回原文"""
import sys

sys.path.append("/home/pingji/apps/CosyVoice")
sys.path.append("/home/pingji/apps/CosyVoice/third_party/Matcha-TTS")

import torch

MODEL = "/home/pingji/models/cosyvoice2"
TEXTS = ["你好，今天天气不错。", "希望你以后能够做的比我还好呦。"]


def main():
    from cosyvoice.cli.cosyvoice import CosyVoice2
    cv = CosyVoice2(MODEL, load_jit=False, load_trt=False, fp16=False)
    tok = cv.frontend.tokenizer
    print(f"tokenizer 类型: {type(tok).__name__}")
    print(f"词表大小: {getattr(tok, 'vocab_size', '?')}")
    print(f"特殊 token: bos={getattr(tok, 'bos_id', '?')} eos={getattr(tok, 'eos_id', '?')} "
          f"sos={getattr(tok, 'sos_id', '?')} task={getattr(tok, 'task_id', '?')}\n")

    for text in TEXTS:
        ids = tok.encode(text)
        back = tok.decode(ids)
        ok = back.strip().replace(" ", "") == text.replace(" ", "")
        print(f"原文: {text}")
        print(f"token: {ids[:16]}{'...' if len(ids) > 16 else ''}  (共 {len(ids)})")
        print(f"解码: {back!r}  -> {'✅ 一致' if ok else '❌ 不一致'}")
        print()

    # 再看 prompt 侧：frontend 用的文本 token 是否也一致
    from cosyvoice.utils.file_utils import load_wav
    prompt16 = load_wav("/home/pingji/apps/CosyVoice/asset/zero_shot_prompt.wav", 16000)
    mi = cv.frontend.frontend_zero_shot(TEXTS[0], "希望你以后能够做的比我还好呦。",
                                        prompt16, cv.sample_rate, '')
    print("frontend 产生的 text tokens:", mi["text"][0][:16].tolist())
    print("复算 token:", tok.encode(TEXTS[0])[:16])
    print("一致:", mi["text"][0][:16].tolist() == tok.encode(TEXTS[0])[:16])


if __name__ == "__main__":
    main()
