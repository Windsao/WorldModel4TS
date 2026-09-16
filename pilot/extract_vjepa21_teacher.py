"""Extract the V-JEPA 2.1 ViT-G/16 target encoder from the 30 GB release checkpoint into a bf16 state dict (~4 GB).

usage: python pilot/extract_vjepa21_teacher.py <vjepa2_1_vitG_384.pt> <out.pt>
Uses the EMA (target) encoder when present, as V-JEPA's own targets do; prints which key was used."""
import sys, torch

src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
print("top-level keys:", list(sd.keys()))
key = next(k for k in ("ema_encoder", "target_encoder", "encoder") if k in sd)
enc = {k.replace("module.backbone.", "", 1): v.to(torch.bfloat16) for k, v in sd[key].items()}
n = sum(v.numel() for v in enc.values()) / 1e9
torch.save(enc, dst)
print(f"saved {len(enc)} tensors, {n:.2f}B params from '{key}' -> {dst}")
