"""Upload the paper's evaluated checkpoints to a HuggingFace repository.

Uploads only the checkpoints the paper actually evaluates: the 60k VideoMAE-B main model and the two
5k V-JEPA arms from the initialisation study. Nothing is copied locally; files stream from the
cluster paths.

Needs a write token: set HF_TOKEN, or run `huggingface-cli login` first.

usage:
    python release/upload_to_hf.py --repo <user-or-org>/<name>          # real run
    python release/upload_to_hf.py --repo <...> --private              # unlisted
    python release/upload_to_hf.py --repo <...> --dry-run              # list what would go
"""
import argparse
import os
import sys

ROOT = "/nyx-storage1/hanliu/wm4ts"
FILES = [
    # (local path, path inside the repo)
    (f"{ROOT}/route_b/vmae_full_f32_savl_l2sp_v2_60k/step_60000/model.safetensors",
     "videomae_b_60k/model.safetensors"),
    (f"{ROOT}/route_b/vmae_full_f32_savl_l2sp_v2_60k/step_60000/config.json",
     "videomae_b_60k/config.json"),
    (f"{ROOT}/route_b/vmae_full_f32_savl_l2sp_v2_60k/run_config.json",
     "videomae_b_60k/run_config.json"),
    (f"{ROOT}/route_j/vjepa2_wm_f32_savl_l2sp_v2_5k/step_5000/world_model.pt",
     "vjepa2_l_5k/world_model.pt"),
    (f"{ROOT}/route_j/vjepa2_wm_f32_savl_l2sp_v2_5k/step_5000/route_j.json",
     "vjepa2_l_5k/route_j.json"),
    (f"{ROOT}/route_j/vjepa21b_wm_f32_savl_l2sp_v2_5k/step_5000/world_model.pt",
     "vjepa2_1_b_5k/world_model.pt"),
    (f"{ROOT}/route_j/vjepa21b_wm_f32_savl_l2sp_v2_5k/step_5000/route_j.json",
     "vjepa2_1_b_5k/route_j.json"),
]
CARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MODEL_CARD.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="target repo id, e.g. anon-user/wm4ts-checkpoints")
    ap.add_argument("--private", action="store_true", help="create the repo unlisted")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    missing = [p for p, _ in FILES if not os.path.exists(p)]
    total = sum(os.path.getsize(p) for p, _ in FILES if os.path.exists(p))
    for p, dst in FILES:
        mark = "MISSING " if p in missing else ""
        size = "" if p in missing else f"{os.path.getsize(p)/1e6:7.1f} MB"
        print(f"  {mark}{size}  {dst}")
    print(f"total {total/1e9:.2f} GB into {a.repo}" + (" (private)" if a.private else " (public)"))
    if missing:
        print("\nrefusing to upload: some files are not readable from here")
        return 1
    if a.dry_run:
        print("\ndry run, nothing uploaded")
        return 0
    if not (os.environ.get("HF_TOKEN") or os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))):
        print("\nno HuggingFace token found. Set HF_TOKEN or run `huggingface-cli login`.")
        return 1

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, private=a.private, exist_ok=True)
    api.upload_file(path_or_fileobj=CARD, path_in_repo="README.md", repo_id=a.repo)
    for p, dst in FILES:
        print("uploading", dst, flush=True)
        api.upload_file(path_or_fileobj=p, path_in_repo=dst, repo_id=a.repo)
    print("done:", f"https://huggingface.co/{a.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
