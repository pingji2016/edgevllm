import os
import sys
import time

sys.path.insert(0, "/tmp/nano-vllm")
import torch

P = "/tmp/tiny-qwen3"
PROMPT = [10, 20, 30, 40, 50, 60]
NTOK = 6

if not os.path.exists(P + "/model.safetensors"):
    from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=512, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=32, max_position_embeddings=512,
                      tie_word_embeddings=False, attention_bias=False)
    Qwen3ForCausalLM(cfg).save_pretrained(P)
    vocab = {f"<t{i}>": i for i in range(512)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<t0>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<t1>", unk_token="<t0>",
                            pad_token="<t0>").save_pretrained(P)
    print("built tiny model", flush=True)

from nanovllm import LLM, SamplingParams

llm = LLM(P, enforce_eager=True, max_model_len=256, max_num_batched_tokens=64,
          num_kvcache_blocks=32, gpu_memory_utilization=0.5)
t = time.time()
out = llm.generate([PROMPT], SamplingParams(temperature=1e-6, max_tokens=NTOK, ignore_eos=True),
                   use_tqdm=False)
dt = time.time() - t
got = out[0]["token_ids"]
print(f"nano-vllm: {got}  ({dt*1000:.0f} ms, {len(got)/dt:.1f} tok/s)", flush=True)
llm.exit()

from transformers import Qwen3ForCausalLM
m = Qwen3ForCausalLM.from_pretrained(P, dtype=torch.float32)
ids = list(PROMPT)
ref = []
with torch.no_grad():
    for _ in range(NTOK):
        lg = m(torch.tensor([ids])).logits[0, -1]
        nxt = int(lg.argmax())
        ref.append(nxt)
        ids.append(nxt)
print(f"HF greedy: {ref}", flush=True)
print("MATCH" if got[:len(ref)] == ref else "MISMATCH", flush=True)
