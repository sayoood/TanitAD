"""align_control.py -- CONTROL 2 (frame alignment): the camera image we overlay IS the frame the cached tokens came from.

For every cache step j of each rendered clip (all 101 steps, so every frame the video can show):
  frame_j  = native frame 2j of `<clip>.v2ep.pt` (PNG decode)  -- the production 256x640 cylindrical frame the overlay uses
  tokens_j = DINOv3-L/16 re-encoded on CPU from frame_j (same recipe as dinov3_fp8_encode_ship.encode_episode: /255, ImageNet mean/std,
             drop the 5 special tokens, 16x40 patch field) pooled to 4x10 with pull_encode_pooled's reshape(4,4,10,4,D).mean((1,3))
  compared (mean per-token cosine) with the CACHED pool4x10[j'] for j' in {j-2 .. j+2}.
PASS iff for every frame the cached step j' = j has the HIGHEST cosine (argmax over the 5 neighbours is j itself), i.e. the pixels we overlay
at step j are the pixels the tokens of step j were computed from, and an off-by-one frame would have been caught.
Also: PSNR / max|d| of the overlay frame (rebuild dir `refav1-eval141-256rb`) against the BANKED eval v2ep (`refav1-eval141/eps`) for each frame.

  python align_control.py --clips a,b,c,d --out align_control.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_extract import *  # noqa: E402,F401,F403
import torchvision  # noqa: E402

REBUILD = "C:/Users/Admin/tanitad-data/refav1-eval141-256rb"
BANKED_EPS = "C:/Users/Admin/tanitad-data/refav1-eval141/eps"
HF = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def decode_frames(path: str, idx):
    o = torch.load(path, map_location="cpu", weights_only=False)
    buf, lens = o["jpeg_buf"], o["jpeg_len"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    fr = torch.stack([torchvision.io.decode_png(buf[offs[j]:offs[j + 1]].clone()) for j in idx])   # [n, 3, 256, 640] uint8
    return fr, int(len(lens))


def load_encoder():
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(HF, local_files_only=True)
    model = AutoModel.from_pretrained(HF, dtype=torch.float32, local_files_only=True).eval()
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
    return model, mean, std, 1 + getattr(model.config, "num_register_tokens", 0)


def pool4x10(f):                                   # f [n, 640, D]
    n, _, d = f.shape
    return f.reshape(n, 4, 4, 10, 4, d).mean(dim=(2, 4)).reshape(n, 40, d)


def psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10 * torch.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", default=f"{OUT}/align_control.json")
    ap.add_argument("--cache", default=f"{WORK}/cache_viz/eval/clips")
    ap.add_argument("--step", type=int, default=1, help="encode every k-th cache step (1 = all)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    model, mean, std, nsp = load_encoder()
    res = {"encoder": HF, "n_special_dropped": nsp, "recipe": "float32 CPU re-encode; fp8 cache => cosine < 1 expected", "clips": {}}
    for clip in a.clips.split(","):
        cache = torch.load(f"{a.cache}/{clip}.pt", map_location="cpu", weights_only=True)["pool4x10"].float()      # [T, 40, 1024]
        T = cache.shape[0]
        steps = list(range(0, T, a.step))
        if a.limit:
            steps = steps[:a.limit]
        fr, n_native = decode_frames(f"{REBUILD}/{clip}.v2ep.pt", [2 * j for j in steps])
        frb, _ = decode_frames(f"{BANKED_EPS}/{clip}.v2ep.pt", [2 * j for j in steps])
        ps = psnr(fr, frb).numpy()
        mx = (fr.int() - frb.int()).abs().amax(dim=(1, 2, 3)).numpy()
        cos = np.zeros((len(steps), 5), np.float32)                          # cosine to cached step j-2 .. j+2 (nan outside)
        t0 = time.time()
        with torch.inference_mode():
            for b0 in range(0, len(steps), a.bs):
                x = fr[b0:b0 + a.bs].float() / 255.0
                h = model(pixel_values=(x - mean) / std).last_hidden_state[:, nsp:, :]
                assert h.shape[1:] == (640, 1024)
                tok = pool4x10(h)                                            # [b, 40, 1024]
                for k in range(tok.shape[0]):
                    j = steps[b0 + k]
                    for d in range(-2, 3):
                        jj = j + d
                        cos[b0 + k, d + 2] = float(torch.nn.functional.cosine_similarity(tok[k], cache[jj], dim=-1).mean()) if 0 <= jj < T else np.nan
        am = np.nanargmax(np.where(np.isnan(cos), -9, cos), axis=1) - 2
        res["clips"][clip] = {
            "n_native_frames": n_native, "n_cache_steps": T, "steps_checked": len(steps),
            "cos_matched_mean": float(cos[:, 2].mean()), "cos_matched_min": float(cos[:, 2].min()),
            "cos_best_other_mean": float(np.nanmax(np.where(np.arange(5)[None] == 2, np.nan, cos), axis=1).mean()),
            "n_argmax_is_self": int((am == 0).sum()), "n_argmax_not_self": int((am != 0).sum()),
            "argmax_offsets_not_self": {int(o): int((am == o).sum()) for o in set(am.tolist()) if o != 0},
            "psnr_overlay_vs_banked_min": float(ps.min()), "psnr_overlay_vs_banked_mean": float(ps.mean()),
            "maxabs_overlay_vs_banked_max": int(mx.max()), "seconds": round(time.time() - t0, 1)}
        np.savez(f"{a.out[:-5]}_{clip}.npz", steps=np.array(steps), cos=cos, psnr=ps, maxabs=mx)
        print(clip, json.dumps(res["clips"][clip]), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    res["PASS"] = all(v["n_argmax_not_self"] == 0 for v in res["clips"].values())
    json.dump(res, open(a.out, "w"), indent=1)
    print("PASS" if res["PASS"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
