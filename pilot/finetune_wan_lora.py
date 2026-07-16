"""
LoRA fine-tuning of Wan2.1-T2V-1.3B for time-series temporal outpainting.

Fixes the two failure modes diagnosed in the zero-shot phase:
  - "copy machine" behavior: flow-matching loss is computed ONLY on the future
    latent region, with the context region noised at the same sigma exactly as
    the RePaint inference procedure does -> the model is trained to *predict*
    the future from noisy context, matching inference step-for-step.
  - resolution OOD: trains at 240x416, making the cheap resolution
    in-distribution (4x faster inference than 480x832).

Data: infinite synthetic series (same generator as pretrain_vmae_ts.py),
rendered with run_wan.render_video using the TRUE future (y_est=ground truth).

Run (prograph env, 1 GPU):
  python pilot/finetune_wan_lora.py --steps 4000 \
      --out /nyx-storage1/hanliu/wm4ts/wan_lora
"""

import argparse
import math
import os
import numpy as np
import torch
import torch.nn.functional as F

import run_pilot as rp
import run_wan as rw
from pretrain_vmae_ts import synth_series

DEVICE = "cuda"
DTYPE = torch.bfloat16
H, W, FPP = 240, 416, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from diffusers import WanPipeline
    from peft import LoraConfig, get_peft_model

    pipe = WanPipeline.from_pretrained(rw.MODEL, torch_dtype=DTYPE)
    pipe.text_encoder.to(DEVICE)
    pe, _ = pipe.encode_prompt(prompt=rw.PROMPT, negative_prompt=None,
                               do_classifier_free_guidance=False, device=DEVICE)
    pe = pe.to(DEVICE, DTYPE)
    del pipe.text_encoder
    torch.cuda.empty_cache()

    vae = pipe.vae.to(DEVICE).eval().requires_grad_(False)
    lm = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(DEVICE, DTYPE)
    ls = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(DEVICE, DTYPE)

    tf = pipe.transformer
    tf.enable_gradient_checkpointing()
    lcfg = LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                      target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    tf = get_peft_model(tf, lcfg).to(DEVICE)
    tf.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in tf.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(7)
    n_known = 1 + (rp.CTX_P * FPP + 1 - 1) // 4           # 13 context latent frames

    tf.train()
    for step in range(1, args.steps + 1):
        P = int(rng.integers(16, 145))
        rp.configure(P)
        xs, ys = [], []
        for _ in range(args.batch):
            s = synth_series(rng, rp.CTX_P + rp.PRED_P, P)
            xs.append(s[:rp.CONTEXT])
            ys.append(s[rp.CONTEXT:])
        xb = torch.from_numpy(np.stack(xs)).float()
        yb = torch.from_numpy(np.stack(ys)).float()
        vid, mu, sd, n_ctx_f, _ = rw.render_video(xb, FPP, H, W, y_est=yb)

        with torch.no_grad():
            lat = vae.encode(vid.to(DEVICE, DTYPE)).latent_dist.mode()
            lat = ((lat - lm) / ls).float()                # [B, C, 17, h, w]

        u = float(rng.uniform(0.02, 0.98))
        sigma = args.shift * u / (1 + (args.shift - 1) * u)
        eps = torch.randn_like(lat)
        x_t = (1 - sigma) * lat + sigma * eps
        target_v = eps - lat                               # FM velocity

        ts = torch.full((lat.shape[0],), sigma * 1000.0, device=DEVICE)
        out = tf(hidden_states=x_t.to(DTYPE), timestep=ts,
                 encoder_hidden_states=pe.expand(lat.shape[0], -1, -1),
                 return_dict=False)[0].float()
        loss = F.mse_loss(out[:, :, n_known:], target_v[:, :, n_known:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in tf.parameters()
                                        if p.requires_grad], 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f} sigma {sigma:.2f}",
                  flush=True)
        if step % args.save_every == 0 or step == args.steps:
            d = os.path.join(args.out, f"step_{step}")
            tf.save_pretrained(d)
            print(f"[ckpt] {d}", flush=True)


if __name__ == "__main__":
    main()
