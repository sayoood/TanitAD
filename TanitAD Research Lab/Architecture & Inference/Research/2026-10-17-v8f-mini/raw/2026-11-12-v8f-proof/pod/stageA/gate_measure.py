"""Quantify the byte-gate mismatch on this pod: pod-built v2ep clip against the banked copy on HF, pixel level.
Prints statistics only (no token, no address). Usage: python gate_measure.py <clip-id-prefix> [<clip-id-prefix> ...]"""
import glob
import io
import json
import os
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

TOK = open("/root/.hf_token").read().strip()
REPOS = ["Sayood/tanitad-physicalai-w120-256x640cyl", "Sayood/tanitad-v7-training-corpus"]
OUT = "/workspace/v8proof/hierrep/gate_measure.json"


def frames(o):
    lens = o["jpeg_len"].to(torch.int64)
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), lens.cumsum(0)])
    buf = o["jpeg_buf"]
    out = []
    for j in range(len(lens)):
        b = bytes(buf[int(offs[j]):int(offs[j + 1])].numpy().tobytes())
        out.append(np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.int16))
    return np.stack(out)


def main():
    api = HfApi(token=TOK)
    listing = {}
    for r in REPOS:
        try:
            listing[r] = [f for f in api.list_repo_files(r, repo_type="dataset") if f.endswith(".v2ep.pt")]
        except Exception as e:  # noqa: BLE001
            listing[r] = []
            print("list failed", r.split("/")[-1], type(e).__name__)
    res = {}
    for cid in sys.argv[1:]:
        loc = [p for p in glob.glob("/workspace/v8proof/data/**/" + cid + "*.v2ep.pt", recursive=True)]
        hit = [(r, f) for r in REPOS for f in listing[r] if os.path.basename(f).startswith(cid)]
        if not loc or not hit:
            res[cid] = {"status": "missing", "n_local": len(loc), "n_banked": len(hit)}
            print(cid, res[cid])
            continue
        r, f = hit[0]
        bp = hf_hub_download(r, f, repo_type="dataset", token=TOK, local_dir="/workspace/v8proof/data/_gate_banked")
        a = torch.load(loc[0], map_location="cpu", weights_only=False)
        b = torch.load(bp, map_location="cpu", weights_only=False)
        keys_equal = sorted(a.keys()) == sorted(b.keys())
        same = {}
        for k in sorted(set(a) & set(b)):
            if k in ("jpeg_buf", "jpeg_len"):
                continue
            va, vb = a[k], b[k]
            if torch.is_tensor(va) and torch.is_tensor(vb):
                same[k] = bool(va.shape == vb.shape and torch.equal(va, vb))
            else:
                same[k] = bool(va == vb)
        fa, fb = frames(a), frames(b)
        d = fa - fb if fa.shape == fb.shape else None
        if d is None:
            res[cid] = {"status": "shape differs", "pod": list(fa.shape), "banked": list(fb.shape)}
        else:
            ad = np.abs(d)
            per_frame = (ad > 0).reshape(len(ad), -1).mean(1)
            res[cid] = {"status": "ok", "repo": r.split("/")[-1], "n_frames": int(len(fa)), "shape": list(fa.shape[1:]),
                        "keys_equal": keys_equal, "non_image_fields_equal": same, "all_non_image_equal": bool(all(same.values())),
                        "frac_values_differ": float((ad > 0).mean()), "max_abs_diff_levels": int(ad.max()),
                        "frac_abs_diff_gt1": float((ad > 1).mean()), "signed_mean_levels": float(d.mean()),
                        "frames_with_any_diff": int((per_frame > 0).sum()), "per_frame_frac_min_max": [float(per_frame.min()), float(per_frame.max())]}
        print(cid, json.dumps(res[cid]))
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
