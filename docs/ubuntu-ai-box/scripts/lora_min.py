"""极简 SDXL LoRA 微调（验证链路，非追求质量）

流程：预计算 latent + 文本 embedding → 释放 VAE/文本编码器 → 只训 UNet 上的 LoRA
用法：STEPS=40 RES=512 python lora_min.py
"""
import glob
import json
import os
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

CKPT = os.path.expanduser("~/apps/comfyui/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors")
SDXL = None  # 用单文件加载，无需 diffusers 目录
STEPS = int(os.environ.get("STEPS", "40"))
RES = int(os.environ.get("RES", "512"))
CAPTION = os.environ.get("CAPTION", "a cinematic fantasy scene, dramatic lighting")
OUT = os.path.expanduser("~/apps/infer-research")


def main():
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    from peft import LoraConfig

    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_single_file(CKPT, torch_dtype=torch.bfloat16)
    unet, vae = pipe.unet.cuda(), pipe.vae.cuda()
    te1, te2 = pipe.text_encoder.cuda(), pipe.text_encoder_2.cuda()
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    print(f"管线加载 {time.time()-t0:.1f}s", flush=True)

    # ---- 预计算文本 embedding ----
    with torch.no_grad():
        ids = tok1(CAPTION, padding="max_length", max_length=tok1.model_max_length,
                   truncation=True, return_tensors="pt").input_ids.cuda()
        e1 = te1(ids, output_hidden_states=True).hidden_states[-2]
        ids2 = tok2(CAPTION, padding="max_length", max_length=tok2.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.cuda()
        o2 = te2(ids2, output_hidden_states=True)
        emb = torch.cat([e1, o2.hidden_states[-2]], dim=-1).to(torch.bfloat16)
        pooled = o2[0].to(torch.bfloat16)
        add_time_ids = torch.tensor([[RES, RES, 0, 0, RES, RES]],
                                    dtype=torch.bfloat16, device="cuda")

    # ---- 预计算 latent ----
    tf = transforms.Compose([transforms.Resize(RES), transforms.CenterCrop(RES),
                             transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    files = sorted(glob.glob(os.path.expanduser("~/桌面/ComfyUI/*.png")))[:8]
    latents = []
    with torch.no_grad():
        for p in files:
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0).cuda().to(torch.bfloat16)
            lat = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
            latents.append(lat)
    print(f"预计算 {len(latents)} 张图完成", flush=True)

    # ---- 释放不需要的模块 ----
    sched = pipe.scheduler
    del vae, te1, te2, pipe
    torch.cuda.empty_cache()

    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0,
                                target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    unet.enable_gradient_checkpointing()
    unet.train()
    n_train = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"可训练参数 {n_train/1e6:.2f} M（LoRA r=8）| 显存 "
          f"{torch.cuda.memory_allocated()/2**20:.0f} MiB", flush=True)

    opt = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=1e-4)
    losses, times = [], []
    for step in range(STEPS):
        t0 = time.time()
        lat = latents[step % len(latents)]
        noise = torch.randn_like(lat)
        t = torch.randint(0, sched.config.num_train_timesteps, (1,), device="cuda").long()
        noisy = sched.add_noise(lat, noise, t)
        pred = unet(noisy, t, encoder_hidden_states=emb,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids}).sample
        loss = F.mse_loss(pred.float(), noise.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        losses.append(loss.item())
        if step % 10 == 0 or step == STEPS - 1:
            print(f"  step {step+1:3d}/{STEPS} | loss {loss.item():.4f} | "
                  f"{times[-1]*1000:.0f} ms/step | 显存 {torch.cuda.max_memory_allocated()/2**20:.0f} MiB",
                  flush=True)

    avg = sum(times[3:]) / max(len(times[3:]), 1)
    print(f"\n平均 {avg*1000:.0f} ms/step → 300 步约 {avg*300/60:.1f} 分钟")
    print(f"loss: 首 {losses[0]:.3f} → 末 {sum(losses[-5:])/5:.3f}")

    save_dir = os.path.join(OUT, "lora_sdxl_min")
    os.makedirs(save_dir, exist_ok=True)
    saved = False
    for fn in ("save_lora_adapter", "save_attn_procs"):
        if hasattr(unet, fn):
            try:
                getattr(unet, fn)(save_dir)
                saved, how = True, fn
                break
            except Exception as e:
                print(f"  {fn} 失败: {type(e).__name__}")
    if not saved:
        from peft import get_peft_model_state_dict
        torch.save(get_peft_model_state_dict(unet), os.path.join(save_dir, "lora_state.pt"))
        how = "peft state_dict"
    print(f"LoRA 已保存({how}): {save_dir}")
    print("  文件:", sorted(os.listdir(save_dir))[:5])
    json.dump(dict(steps=STEPS, res=RES, ms_per_step=round(avg * 1000),
                   loss_first=round(losses[0], 4), loss_last=round(sum(losses[-5:]) / 5, 4),
                   trainable_m=round(n_train / 1e6, 2),
                   peak_vram_mib=round(torch.cuda.max_memory_allocated() / 2**20)),
              open(os.path.join(OUT, "lora_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
