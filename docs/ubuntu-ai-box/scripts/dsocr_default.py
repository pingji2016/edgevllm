"""DeepSeek-OCR 最小可用验证（不传 mm_processor_kwargs，用模型默认配置）"""
import difflib
import json
import os
import re
import time

from PIL import Image

MD = os.path.expanduser("~/.cache/modelscope/models/deepseek-ai--DeepSeek-OCR/snapshots/master")
IMG = "/tmp/ocr_doc.png"
REF = "/tmp/ocr_ref.txt"
OUT = os.path.expanduser("~/apps/infer-engines/dsocr_default.json")


def norm(s):
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。、；：！？,.;:!?\-—·()（）\[\]【】\"'“”‘’]", "", s)
    return s


def main():
    from vllm import LLM, SamplingParams

    ref = norm(open(REF, encoding="utf-8").read())
    img = Image.open(IMG).convert("RGB")
    print(f"参考文本 {len(ref)} 字符\n")

    t0 = time.time()
    llm = LLM(model=MD, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85, enforce_eager=True,
              disable_log_stats=True, trust_remote_code=True, moe_backend="triton")
    print(f"引擎就绪 {time.time()-t0:.0f}s", flush=True)

    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    sp = SamplingParams(max_tokens=1024, temperature=0.0)
    req = {"prompt": prompt, "multi_modal_data": {"image": img}}

    llm.generate([req], sp)  # 预热
    t0 = time.time()
    outs = llm.generate([req], sp)
    dt = time.time() - t0

    text = outs[0].outputs[0].text
    n_prompt = len(outs[0].prompt_token_ids or [])
    n_out = len(outs[0].outputs[0].token_ids)
    acc = difflib.SequenceMatcher(None, ref, norm(text)).ratio()
    print(f"prompt {n_prompt} tok | 输出 {n_out} tok | {dt:.2f}s | 精度 {acc*100:.1f}%")
    print(f"\n=== 输出前 400 字 ===\n{text[:400]}")
    json.dump(dict(prompt_tokens=n_prompt, out_tokens=n_out, seconds=round(dt, 2),
                   accuracy=round(acc, 3), text=text), open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
