"""加载刚训好的 LoRA 出图，验证端到端可用"""
import os
import time

import torch
from diffusers import StableDiffusionXLPipeline

CKPT = os.path.expanduser("~/apps/comfyui/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors")
LORA = os.path.expanduser("~/apps/infer-research/lora_sdxl_min")
PROMPT = "a cinematic fantasy scene, dramatic lighting, highly detailed"

t0 = time.time()
pipe = StableDiffusionXLPipeline.from_single_file(CKPT, torch_dtype=torch.bfloat16)
print(f"加载 {time.time()-t0:.1f}s", flush=True)
pipe.load_lora_weights(LORA)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
img = pipe(PROMPT, num_inference_steps=12, guidance_scale=6.0, width=768, height=768).images[0]
torch.cuda.synchronize()
dt = time.time() - t0
out = os.path.expanduser("~/桌面/lora_demo.png")
img.save(out)
print(f"出图 {dt:.1f}s | 峰值显存 {torch.cuda.max_memory_allocated()/2**20:.0f} MiB | 已存 {out}")
