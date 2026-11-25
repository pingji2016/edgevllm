"""Kokoro 分阶段耗时 + 文本长度扫描（用 pipeline 总时长减去 G2P）"""
import glob
import json
import os
import time

import numpy as np
import torch

ROOT = open(os.path.expanduser("~/models/.kokoro_done")).read().strip()
OUT = os.path.expanduser("~/apps/infer-research/kokoro_bench.json")
TEXTS = {
    "短 (7 字)": "端侧推理很快。",
    "中 (28 字)": "端侧推理部署说明：解码速度五十五点三，每秒钟六十二个 token。",
    "长 (61 字)": "端侧推理部署说明：解码速度五十五点三，每秒钟六十二个 token。"
                 "在实际工程中，工程师往往先测量显存带宽，再决定量化位宽与批大小，最后才调整算子实现。",
}


def main():
    from kokoro import KPipeline

    pth = sorted(glob.glob(os.path.join(ROOT, "*.pth")))[0]
    t0 = time.time()
    pipe = KPipeline(lang_code="z", repo_id="hexgrad/Kokoro-82M-v1.1-zh", model=pth)
    print(f"管线加载 {time.time()-t0:.1f}s | 权重 {os.path.basename(pth)}")

    km = pipe.model
    n_param = sum(p.numel() for p in km.parameters())
    print(f"声学模型参数 {n_param/1e6:.1f} M")

    voice = torch.load(os.path.join(ROOT, "voices", "zf_001.pt"), map_location="cpu")

    rows = []
    for tag, text in TEXTS.items():
        # G2P 单独计时（音素化 + 中文数字/多音字处理）
        t0 = time.time()
        g2p = list(pipe.g2p(text))
        g2p_s = time.time() - t0

        # 完整合成
        t0 = time.time()
        chunks = [a for _, _, a in pipe(text, voice=voice, speed=1.0) if a is not None]
        total_s = time.time() - t0

        wav = np.concatenate([c.numpy() if hasattr(c, "numpy") else np.asarray(c) for c in chunks])
        dur = len(wav) / 24000
        model_s = max(total_s - g2p_s, 1e-6)
        rows.append(dict(tag=tag, chars=len(text), g2p_ms=round(g2p_s * 1000, 1),
                         model_ms=round(model_s * 1000, 1), total_ms=round(total_s * 1000, 1),
                         audio_s=round(dur, 2), rtf=round(total_s / dur, 3),
                         x_realtime=round(dur / total_s, 2),
                         ms_per_char=round(total_s / len(text) * 1000, 1)))
        print(f"  {tag:11s} G2P {g2p_s*1000:6.1f} ms | 模型+声码 {model_s*1000:7.1f} ms | "
              f"音频 {dur:5.2f}s | RTF {total_s/dur:.3f} ({dur/total_s:.1f}×) | "
              f"{total_s/len(text)*1000:.0f} ms/字", flush=True)

    json.dump(rows, open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
