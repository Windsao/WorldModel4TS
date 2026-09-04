"""Final table: does a zero-shot method CONTAINING a video backbone beat VisionTS?

Two questions are answered separately and must not be conflated:
  (1) does the whole pipeline beat VisionTS?          -> arm / visionts, per dataset and Q_all
  (2) is the video backbone WHY?                      -> pretrained arm / its random-init twin
"""
import argparse, glob, json, math, os, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="pilot/results_field/zeroshot_loop")
a = ap.parse_args()
DS = ["ETTh1", "ETTh2", "ETTm2", "electricity", "traffic", "solar"]
ARMS = [("blend", "prior blend (no video)"),
        ("evidence_pt", "blend + video, evidence-weighted"),
        ("evidence_rand", "blend + random-init video, evidence-weighted"),
        ("blend_plus_pt", "blend + video, fixed 50/50"),
        ("blend_plus_rawL2", "blend + raw-L2 retrieval (no video)"),
        ("pt_only", "video retrieval alone")]

res = {}
for d in DS:
    for p in glob.glob(os.path.join(a.root, "**", f"{d}_videomae.json"), recursive=True) + \
             glob.glob(os.path.join(a.root, "**", f"{d}.json"), recursive=True):
        if "zl130" not in p and "v2" not in p:
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        if j.get("candidate_id") != "ZL-130":
            continue
        res[d] = j["results"]
        break

have = [d for d in DS if d in res]
print(f"datasets with results: {len(have)}/6  {have}\n")
if not have:
    raise SystemExit("no ZL-130 results yet")

w = max(len(k) for k, _ in ARMS) + 2
print("=== arm / VisionTS, per dataset (paired on identical origins) ===\n")
print(f"{'arm':{w}s}" + "".join(f"{d[:11]:>12s}" for d in have) + f"{'Q_all':>9s}")
Q = {}
for k, _lab in ARMS:
    cells, ls = [], []
    for d in have:
        v = res[d].get(f"{k}_over_visionts")
        if v is None:
            cells.append(f"{'-':>12s}")
            continue
        ci = res[d].get(f"{k}_ci", [0, 9])
        s = "*" if ci[1] < 1 else ("!" if ci[0] > 1 else " ")
        cells.append(f"{v:11.4f}{s}")
        ls.append(math.log(v))
    Q[k] = math.exp(st.mean(ls)) if len(ls) == len(have) else None
    q = f"{Q[k]:9.4f}" if Q[k] else f"{'(part)':>9s}"
    print(f"{k:{w}s}" + "".join(cells) + q)
print("\n  * paired CI entirely below 1 (beats VisionTS)   ! entirely above 1")

print("\n=== is the video backbone WHY? ===\n")
print(f"{'dataset':12s} {'evid/blend':>11s} {'evid/rand-twin':>15s} {'video weight':>13s} "
      f"{'fixed50/blend':>14s} {'pt/rand in it':>14s}")
for d in have:
    r = res[d]
    print(f"{d:12s} {r.get('evidence_pt_over_blend', float('nan')):11.4f} "
          f"{r.get('evidence_pt_over_rand', float('nan')):15.4f} "
          f"{r.get('video_weight_pt', float('nan')):13.3f} "
          f"{r.get('video_adds_over_blend', float('nan')):14.4f} "
          f"{r.get('pretrained_over_random_in_ensemble', float('nan')):14.4f}")

if Q.get("evidence_pt") and Q.get("blend"):
    print(f"\n  Q_all(video method / VisionTS) = {Q['evidence_pt']:.4f}"
          f"   -> {'BEATS VisionTS' if Q['evidence_pt'] < 1 else 'does NOT beat VisionTS'}")
    print(f"  Q_all(no-video blend / VisionTS) = {Q['blend']:.4f}")
    if Q.get("evidence_rand"):
        print(f"  Q_all(same method, RANDOM-INIT backbone) = {Q['evidence_rand']:.4f}")
        att = Q["evidence_pt"] / Q["evidence_rand"]
        print(f"  attributable to PRETRAINING: {att:.4f} "
              f"-> {'pretraining helps' if att < 1 else 'pretraining does not help'}")
