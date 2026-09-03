import sys, json, torch
FS = "/lambda/nfs/wm4ts-fs"
sys.path.insert(0, FS + "/pilot")
import run_field as RF
from preprocess_renderers import RENDERERS, input_stats, contact_sheet
rows = {}
for ds in ["ETTh2", "electricity"]:
    data, borders = RF.load_mv(ds, FS + "/data", 112)
    P = RF.P; ctx = 16 * P
    X, _ = RF.windows(data, borders, "train", ctx, 96, 1, 200)
    Xu = X.transpose(0, 2, 1).reshape(-1, ctx, 1)[:128]
    xb = torch.from_numpy(Xu)
    for name, fn in RENDERERS.items():
        vid, mu, sd = fn(xb, P)
        assert torch.isfinite(vid).all(), name + " produced NaN/Inf"
        rows[(ds, name)] = input_stats(vid)
        contact_sheet(vid[0], "%s/results/preprocessing/previews/%s_%s.png" % (FS, name, ds))
hdr = ("dataset", "renderer", "const_tub", "tub_std", "spat_rms", "temp_rms", "t/s")
print("%-12s %-18s %9s %8s %9s %9s %6s" % hdr)
for (ds, n), s in rows.items():
    print("%-12s %-18s %9.3f %8.4f %9.4f %9.4f %6.2f" % (
        ds, n, s["const_tubelet_frac"], s["tubelet_pixel_std"],
        s["spatial_grad_rms"], s["temporal_diff_rms"], s["temporal_over_spatial"]))
json.dump({"%s|%s" % k: v for k, v in rows.items()},
          open(FS + "/results/preprocessing/stageA_input_stats.json", "w"), indent=2)
print("\npreviews written to results/preprocessing/previews/")
