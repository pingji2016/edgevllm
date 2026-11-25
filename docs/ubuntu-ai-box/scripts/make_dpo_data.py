"""从 SFT 数据合成 DPO 偏好对
chosen   = 原始问答
rejected = 把答案换成另一个样本的答案（张冠李戴，明显更差）
产出：dpo_train.jsonl（3000 对）、dpo_eval.jsonl（200 对，两侧不重叠）
"""
import json
import os
import random

SRC = os.path.expanduser("~/apps/minimind/dataset/sft_t2t_mini.jsonl")
OUT_DIR = os.path.expanduser("~/apps/minimind/dataset")

random.seed(42)


def load(n):
    samples = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            if len(samples) >= n * 3:
                break
            try:
                d = json.loads(line)
                conv = d.get("conversations") or []
            except Exception:
                continue
            if len(conv) < 2 or conv[-1].get("role") != "assistant":
                continue
            ans = (conv[-1].get("content") or "").strip()
            if not (10 <= len(ans) <= 400):
                continue
            msgs = [{"role": c.get("role"), "content": (c.get("content") or "")} for c in conv]
            if any(not m["content"] for m in msgs):
                continue
            samples.append(msgs)
    return samples


def main():
    total = 3200
    samples = load(total)
    print(f"候选样本 {len(samples)} 条")
    random.shuffle(samples)

    train_n = 3000
    pairs_train, pairs_eval = [], []
    for i, msgs in enumerate(samples):
        ans = msgs[-1]["content"]
        pool = [s[-1]["content"] for j, s in enumerate(samples) if j != i and len(s[-1]["content"]) > 10]
        if not pool:
            continue
        wrong = random.choice(pool)
        pair = {
            "chosen": msgs,
            "rejected": msgs[:-1] + [{"role": "assistant", "content": wrong}],
        }
        (pairs_train if len(pairs_train) < train_n else pairs_eval).append(pair)
        if len(pairs_train) >= train_n and len(pairs_eval) >= 200:
            break

    for name, data in (("dpo_train.jsonl", pairs_train), ("dpo_eval.jsonl", pairs_eval)):
        p = os.path.join(OUT_DIR, name)
        with open(p, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} 对 → {p}")

    print("\n示例（chosen 与 rejected 的答案应完全不同）：")
    ex = pairs_train[0]
    print("  提问:", ex["chosen"][0]["content"][:50])
    print("  chosen:", ex["chosen"][-1]["content"][:50])
    print("  rejected:", ex["rejected"][-1]["content"][:50])


if __name__ == "__main__":
    main()
